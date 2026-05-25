#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bbox_all_3d_marker_node.py

功能：
  讀取 /grounding_dino/bboxes 裡所有 detection
  每個 bbox 取中心點 cx, cy
  用 /realsense/depth + /realsense/camera_info 反投影成 camera frame 3D
  再用 TF 轉到 base_link
  發布 MarkerArray 給 RViz2 顯示

輸入：
  /grounding_dino/bboxes      std_msgs/String, JSON
  /realsense/depth            sensor_msgs/Image
  /realsense/camera_info      sensor_msgs/CameraInfo
  /tf                         TF

輸出：
  /grounding_dino/objects_3d_json   std_msgs/String
  /visualization_marker_array       visualization_msgs/MarkerArray

RViz2：
  Fixed Frame: base_link
  Add -> MarkerArray
  Topic: /visualization_marker_array
"""

from __future__ import annotations

import json
import math
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class BBoxAll3DMarkerNode(Node):
    def __init__(self):
        super().__init__("bbox_all_3d_marker_node")

        self.declare_parameter("bboxes_topic", "/grounding_dino/bboxes")
        self.declare_parameter("depth_topic", "/realsense/depth")
        self.declare_parameter("camera_info_topic", "/realsense/camera_info")

        self.declare_parameter("objects_3d_json_topic", "/grounding_dino/objects_3d_json")
        self.declare_parameter("marker_topic", "/visualization_marker_array")

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("camera_frame_override", "")

        self.declare_parameter("depth_window_radius", 5)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 10.0)

        self.declare_parameter("marker_lifetime_s", 1.0)
        self.declare_parameter("sphere_scale", 0.08)
        self.declare_parameter("text_scale", 0.18)
        self.declare_parameter("text_z_offset", 0.14)

        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("max_objects", 100)

        self.bboxes_topic = str(self.get_parameter("bboxes_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.objects_3d_json_topic = str(self.get_parameter("objects_3d_json_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)

        self.target_frame = str(self.get_parameter("target_frame").value)
        self.camera_frame_override = str(self.get_parameter("camera_frame_override").value)

        self.depth_window_radius = int(self.get_parameter("depth_window_radius").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.marker_lifetime_s = float(self.get_parameter("marker_lifetime_s").value)
        self.sphere_scale = float(self.get_parameter("sphere_scale").value)
        self.text_scale = float(self.get_parameter("text_scale").value)
        self.text_z_offset = float(self.get_parameter("text_z_offset").value)

        self.min_score = float(self.get_parameter("min_score").value)
        self.max_objects = int(self.get_parameter("max_objects").value)

        self.last_depth_msg: Optional[Image] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_marker_count = 0

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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

        self.json_pub = self.create_publisher(String, self.objects_3d_json_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.get_logger().info("BBoxAll3DMarkerNode ready.")
        self.get_logger().info(f"bboxes_topic          : {self.bboxes_topic}")
        self.get_logger().info(f"depth_topic           : {self.depth_topic}")
        self.get_logger().info(f"camera_info_topic     : {self.camera_info_topic}")
        self.get_logger().info(f"objects_3d_json_topic : {self.objects_3d_json_topic}")
        self.get_logger().info(f"marker_topic          : {self.marker_topic}")
        self.get_logger().info(f"target_frame          : {self.target_frame}")
        self.get_logger().info(f"camera_frame_override : {self.camera_frame_override}")

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
            self.publish_delete_old_markers()
            self.publish_objects_json([])
            self.get_logger().info("No bboxes in GroundingDINO result.")
            return

        if self.max_objects > 0:
            bboxes = bboxes[: self.max_objects]

        source_frame = self.get_source_camera_frame()
        if not source_frame:
            self.get_logger().warn("Cannot determine camera frame.")
            return

        objects_3d: List[Dict[str, Any]] = []

        for idx, det in enumerate(bboxes):
            obj = self.convert_one_detection(det, idx, source_frame)
            if obj is not None:
                objects_3d.append(obj)

        self.publish_objects_json(objects_3d)
        self.publish_markers(objects_3d)

        self.get_logger().info(
            f"Published {len(objects_3d)}/{len(bboxes)} objects in {self.target_frame}"
        )

    def get_source_camera_frame(self) -> str:
        if self.camera_frame_override:
            return self.camera_frame_override

        if self.last_camera_info is not None and self.last_camera_info.header.frame_id:
            return self.last_camera_info.header.frame_id

        if self.last_depth_msg is not None and self.last_depth_msg.header.frame_id:
            return self.last_depth_msg.header.frame_id

        return ""

    def convert_one_detection(
        self,
        det: Dict[str, Any],
        idx: int,
        source_frame: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            score = float(det.get("score", 0.0))
            if score < self.min_score:
                return None

            class_name = str(det.get("class", "object"))

            uv = self.get_bbox_center(det)
            if uv is None:
                return None

            u, v = uv

            depth_m = self.get_depth_at_pixel(self.last_depth_msg, u, v)
            if depth_m is None:
                return None

            camera_xyz = self.deproject_pixel_to_3d(
                u,
                v,
                depth_m,
                self.last_camera_info,
            )
            if camera_xyz is None:
                return None

            target_xyz = self.transform_point_to_target(
                camera_xyz,
                source_frame,
                self.target_frame,
            )
            if target_xyz is None:
                return None

            return {
                "id": idx,
                "class": class_name,
                "score": score,
                "source_frame": source_frame,
                "target_frame": self.target_frame,
                "pixel": {
                    "u": float(u),
                    "v": float(v),
                },
                "depth_m": float(depth_m),
                "camera_point": {
                    "x": float(camera_xyz[0]),
                    "y": float(camera_xyz[1]),
                    "z": float(camera_xyz[2]),
                },
                "base_point": {
                    "x": float(target_xyz[0]),
                    "y": float(target_xyz[1]),
                    "z": float(target_xyz[2]),
                },
                "bbox": det,
            }

        except Exception as e:
            self.get_logger().warn(f"Failed to convert detection {idx}: {repr(e)}")
            return None

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
                f"Pixel out of depth range: u={u:.1f}, v={v:.1f}, depth_size={w}x{h}"
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

    def transform_point_to_target(
        self,
        xyz: Tuple[float, float, float],
        source_frame: str,
        target_frame: str,
    ) -> Optional[Tuple[float, float, float]]:
        point = PointStamped()
        point.header.stamp = rclpy.time.Time().to_msg()
        point.header.frame_id = source_frame
        point.point.x = float(xyz[0])
        point.point.y = float(xyz[1])
        point.point.z = float(xyz[2])

        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )

            transformed = do_transform_point(point, tf)

            return (
                float(transformed.point.x),
                float(transformed.point.y),
                float(transformed.point.z),
            )

        except Exception as e:
            self.get_logger().warn(
                f"TF transform failed: {source_frame} -> {target_frame}: {repr(e)}"
            )
            return None

    def publish_objects_json(self, objects_3d: List[Dict[str, Any]]):
        msg = String()
        msg.data = json.dumps(
            {
                "target_frame": self.target_frame,
                "count": len(objects_3d),
                "objects": objects_3d,
            },
            ensure_ascii=False,
        )
        self.json_pub.publish(msg)

    def publish_delete_old_markers(self):
        marker_array = MarkerArray()

        for i in range(self.last_marker_count):
            for offset in [0, 10000]:
                m = Marker()
                m.header.frame_id = self.target_frame
                m.header.stamp = self.get_clock().now().to_msg()
                m.ns = "grounding_dino_objects"
                m.id = i + offset
                m.action = Marker.DELETE
                marker_array.markers.append(m)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

        self.last_marker_count = 0

    def publish_markers(self, objects_3d: List[Dict[str, Any]]):
        marker_array = MarkerArray()

        now = self.get_clock().now().to_msg()

        # 刪掉上一輪多餘 marker，避免 RViz 殘留
        for i in range(len(objects_3d), self.last_marker_count):
            for offset in [0, 10000]:
                m = Marker()
                m.header.frame_id = self.target_frame
                m.header.stamp = now
                m.ns = "grounding_dino_objects"
                m.id = i + offset
                m.action = Marker.DELETE
                marker_array.markers.append(m)

        for i, obj in enumerate(objects_3d):
            x = float(obj["base_point"]["x"])
            y = float(obj["base_point"]["y"])
            z = float(obj["base_point"]["z"])

            cls = str(obj["class"])
            score = float(obj["score"])

            r, g, b = self.color_from_class(cls)

            sphere = Marker()
            sphere.header.frame_id = self.target_frame
            sphere.header.stamp = now
            sphere.ns = "grounding_dino_objects"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = self.sphere_scale
            sphere.scale.y = self.sphere_scale
            sphere.scale.z = self.sphere_scale
            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = 0.9
            sphere.lifetime = Duration(seconds=self.marker_lifetime_s).to_msg()
            marker_array.markers.append(sphere)

            text = Marker()
            text.header.frame_id = self.target_frame
            text.header.stamp = now
            text.ns = "grounding_dino_objects"
            text.id = i + 10000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = z + self.text_z_offset
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_scale
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"{cls} {score:.2f}\n({x:.2f}, {y:.2f}, {z:.2f})"
            text.lifetime = Duration(seconds=self.marker_lifetime_s).to_msg()
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)
        self.last_marker_count = len(objects_3d)

    def color_from_class(self, class_name: str) -> Tuple[float, float, float]:
        h = hashlib.md5(class_name.encode("utf-8")).hexdigest()

        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0

        # 避免顏色太暗
        r = 0.35 + 0.65 * r
        g = 0.35 + 0.65 * g
        b = 0.35 + 0.65 * b

        return float(r), float(g), float(b)


def main(args=None):
    rclpy.init(args=args)
    node = BBoxAll3DMarkerNode()

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
