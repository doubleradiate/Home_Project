#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bbox_center_3d_node.py

功能：
  讀取 /grounding_dino/bboxes
  選一個 bbox，例如 highest_score
  取 bbox center pixel: cx, cy
  從 /realsense/depth 取深度
  用 /realsense/camera_info 反投影成相機座標系 3D 點

輸入：
  /grounding_dino/bboxes      std_msgs/String, JSON
  /realsense/depth            sensor_msgs/Image
  /realsense/camera_info      sensor_msgs/CameraInfo

輸出：
  /grounding_dino/center_3d       geometry_msgs/PointStamped
  /grounding_dino/center_3d_json  std_msgs/String

注意：
  這個輸出是 camera frame，不是 base_link。
"""

from __future__ import annotations

import json
import math
from typing import Optional, Tuple, Dict, Any, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped


class BBoxCenter3DNode(Node):
    def __init__(self):
        super().__init__("bbox_center_3d_node")

        self.declare_parameter("bboxes_topic", "/grounding_dino/bboxes")
        self.declare_parameter("depth_topic", "/realsense/depth")
        self.declare_parameter("camera_info_topic", "/realsense/camera_info")

        self.declare_parameter("point3d_topic", "/grounding_dino/center_3d")
        self.declare_parameter("point3d_json_topic", "/grounding_dino/center_3d_json")

        self.declare_parameter("depth_window_radius", 5)
        self.declare_parameter("depth_scale", 0.001)  # 16UC1 mm -> meter
        self.declare_parameter("select_mode", "highest_score")  # first / highest_score / largest_area
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 10.0)

        self.bboxes_topic = str(self.get_parameter("bboxes_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.point3d_topic = str(self.get_parameter("point3d_topic").value)
        self.point3d_json_topic = str(self.get_parameter("point3d_json_topic").value)

        self.depth_window_radius = int(self.get_parameter("depth_window_radius").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.select_mode = str(self.get_parameter("select_mode").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.last_depth_msg: Optional[Image] = None
        self.last_camera_info: Optional[CameraInfo] = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        normal_qos = QoSProfile(depth=10)

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            sensor_qos,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )

        self.bboxes_sub = self.create_subscription(
            String,
            self.bboxes_topic,
            self.bboxes_callback,
            normal_qos,
        )

        self.point_pub = self.create_publisher(PointStamped, self.point3d_topic, 10)
        self.json_pub = self.create_publisher(String, self.point3d_json_topic, 10)

        self.get_logger().info("BBoxCenter3DNode ready.")
        self.get_logger().info(f"bboxes_topic      : {self.bboxes_topic}")
        self.get_logger().info(f"depth_topic       : {self.depth_topic}")
        self.get_logger().info(f"camera_info_topic : {self.camera_info_topic}")
        self.get_logger().info(f"point3d_topic     : {self.point3d_topic}")
        self.get_logger().info(f"json_topic        : {self.point3d_json_topic}")
        self.get_logger().info(f"select_mode       : {self.select_mode}")

    def depth_callback(self, msg: Image):
        self.last_depth_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        self.last_camera_info = msg

    def bboxes_callback(self, msg: String):
        if self.last_depth_msg is None:
            self.get_logger().warn("No depth image received yet.")
            return

        if self.last_camera_info is None:
            self.get_logger().warn("No camera_info received yet.")
            return

        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Failed to parse bbox JSON: {repr(e)}")
            return

        bboxes = payload.get("bboxes", [])
        if not bboxes:
            self.get_logger().info("No bboxes in GroundingDINO result.")
            return

        det = self.select_detection(bboxes)
        if det is None:
            self.get_logger().warn("No valid detection selected.")
            return

        uv = self.get_bbox_center(det)
        if uv is None:
            self.get_logger().warn("Selected bbox has no valid center.")
            return

        u, v = uv

        depth_m = self.get_depth_at_pixel(self.last_depth_msg, u, v)
        if depth_m is None:
            self.get_logger().warn(
                f"No valid depth around bbox center u={u:.1f}, v={v:.1f}"
            )
            return

        xyz = self.deproject_pixel_to_3d(u, v, depth_m, self.last_camera_info)
        if xyz is None:
            self.get_logger().warn("Failed to deproject pixel to 3D.")
            return

        x, y, z = xyz

        frame_id = self.last_camera_info.header.frame_id
        if not frame_id:
            frame_id = self.last_depth_msg.header.frame_id

        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = frame_id
        point_msg.point.x = float(x)
        point_msg.point.y = float(y)
        point_msg.point.z = float(z)
        self.point_pub.publish(point_msg)

        out = {
            "frame_id": frame_id,
            "class": det.get("class", ""),
            "score": float(det.get("score", 0.0)),
            "pixel": {
                "u": float(u),
                "v": float(v),
            },
            "depth_m": float(depth_m),
            "point_3d": {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            },
            "bbox": det,
        }

        json_msg = String()
        json_msg.data = json.dumps(out, ensure_ascii=False)
        self.json_pub.publish(json_msg)

        self.get_logger().info(
            f"3D center: class={out['class']} "
            f"score={out['score']:.3f} "
            f"pixel=({u:.1f}, {v:.1f}) "
            f"xyz=({x:.3f}, {y:.3f}, {z:.3f}) "
            f"frame={frame_id}"
        )

    def select_detection(self, bboxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        valid = []

        for d in bboxes:
            try:
                if self.get_bbox_center(d) is not None:
                    valid.append(d)
            except Exception:
                continue

        if not valid:
            return None

        if self.select_mode == "highest_score":
            valid.sort(key=lambda d: float(d.get("score", 0.0)), reverse=True)
        elif self.select_mode == "largest_area":
            valid.sort(key=lambda d: float(d.get("area", 0.0)), reverse=True)
        else:
            pass

        return valid[0]

    def get_bbox_center(self, det: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        if "cx" in det and "cy" in det:
            return float(det["cx"]), float(det["cy"])

        if all(k in det for k in ["x1", "y1", "x2", "y2"]):
            u = (float(det["x1"]) + float(det["x2"])) / 2.0
            v = (float(det["y1"]) + float(det["y2"])) / 2.0
            return u, v

        if "center" in det and isinstance(det["center"], dict):
            c = det["center"]
            if "x" in c and "y" in c:
                return float(c["x"]), float(c["y"])

        return None

    def depth_image_to_array(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)

        if encoding in ["16uc1", "mono16"]:
            dtype = np.uint16
            bytes_per_pixel = 2
        elif encoding in ["32fc1"]:
            dtype = np.float32
            bytes_per_pixel = 4
        else:
            raise ValueError(f"Unsupported depth encoding: {msg.encoding}")

        data = np.frombuffer(msg.data, dtype=dtype)

        expected_step = width * bytes_per_pixel
        if step == expected_step:
            depth = data.reshape((height, width))
        else:
            row_elems = step // bytes_per_pixel
            depth = data.reshape((height, row_elems))[:, :width]

        return depth

    def get_depth_at_pixel(self, msg: Image, u: float, v: float) -> Optional[float]:
        try:
            depth = self.depth_image_to_array(msg)
        except Exception as e:
            self.get_logger().error(f"Depth convert failed: {repr(e)}")
            return None

        h, w = depth.shape[:2]

        px = int(round(u))
        py = int(round(v))

        if px < 0 or px >= w or py < 0 or py >= h:
            self.get_logger().warn(
                f"Pixel out of depth image range: u={u:.1f}, v={v:.1f}, depth_size={w}x{h}"
            )
            return None

        r = max(0, self.depth_window_radius)

        x1 = max(0, px - r)
        x2 = min(w, px + r + 1)
        y1 = max(0, py - r)
        y2 = min(h, py + r + 1)

        patch = depth[y1:y2, x1:x2]

        if msg.encoding.lower() in ["16uc1", "mono16"]:
            patch_m = patch.astype(np.float32) * self.depth_scale
        else:
            patch_m = patch.astype(np.float32)

        valid = patch_m[
            np.isfinite(patch_m)
            & (patch_m > self.min_depth_m)
            & (patch_m < self.max_depth_m)
        ]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def deproject_pixel_to_3d(
        self,
        u: float,
        v: float,
        depth_m: float,
        camera_info: CameraInfo,
    ) -> Optional[Tuple[float, float, float]]:
        k = camera_info.k

        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn("Invalid camera intrinsics: fx/fy is zero.")
            return None

        z = float(depth_m)
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy

        if not all(math.isfinite(a) for a in [x, y, z]):
            return None

        return x, y, z


def main(args=None):
    rclpy.init(args=args)
    node = BBoxCenter3DNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
