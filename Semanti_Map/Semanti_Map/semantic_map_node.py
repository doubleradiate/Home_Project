#!/usr/bin/env python3
"""
Semantic Map Node — TF 投影層
------------------------------
職責：接收 GroundingDINO 偵測結果，結合深度影像與 TF 將 2D pixel 投影為
      map frame 3D 座標，轉發給下游地圖整合節點。

訂閱:
  - /grounding_dino/detections  (std_msgs/String, JSON) : GroundingDINO 偵測
  - /realsense/depth             (sensor_msgs/Image)     : 深度影像 (16UC1/32FC1)
  - /realsense/camera_info       (sensor_msgs/CameraInfo): 相機內參
發布:
  - /semantic_map/raw_detections (std_msgs/String, JSON) : map frame 3D 偵測（未去重）
"""

import json
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

import tf2_ros
from tf2_geometry_msgs import do_transform_point


class SemanticMapNode(Node):
    def __init__(self):
        super().__init__('semantic_map_node')

        # ---------- 參數 ----------
        self.declare_parameter('detection_topic', '/grounding_dino/detections')
        self.declare_parameter('depth_topic', '/realsense/depth')
        self.declare_parameter('camera_info_topic', '/realsense/camera_info')
        self.declare_parameter('raw_detections_topic', '/semantic_map/raw_detections')
        self.declare_parameter('camera_frame', 'Camera_OmniVision_OV9782_Color')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('min_confidence', 0.35)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_window', 2)

        self.detection_topic = self.get_parameter('detection_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.raw_detections_topic = self.get_parameter('raw_detections_topic').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.depth_window = int(self.get_parameter('depth_window').value)

        # ---------- 內部狀態 ----------
        self._depth_image = None
        self._camera_info = None
        self._lock = Lock()

        # ---------- TF2 ----------
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- Subscribers ----------
        self.create_subscription(String, self.detection_topic, self._det_callback, 10)
        self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._info_callback, 10)

        # ---------- Publisher ----------
        self.raw_pub = self.create_publisher(String, self.raw_detections_topic, 10)

        use_sim_time = self.get_parameter('use_sim_time').value
        self.get_logger().info(
            f"SemanticMapNode (TF 投影層) 已啟動。\n"
            f"  use_sim_time   = {use_sim_time}\n"
            f"  camera_frame   = {self.camera_frame}\n"
            f"  map_frame      = {self.map_frame}\n"
            f"  → {self.raw_detections_topic}"
        )

    # ------------------------------------------------------------------
    # Depth & CameraInfo
    # ------------------------------------------------------------------
    def _depth_callback(self, msg: Image):
        enc = msg.encoding.lower()
        h, w = msg.height, msg.width
        try:
            if enc in ('16uc1', '16uc'):
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((h, w))
                img = arr.astype(np.float32) * self.depth_scale
            elif enc in ('32fc1', '32fc'):
                img = np.frombuffer(msg.data, dtype=np.float32).reshape((h, w)).copy()
            else:
                self.get_logger().warn(
                    f"不支援的深度影像格式: {msg.encoding}", once=True
                )
                return
            with self._lock:
                self._depth_image = img
        except Exception as e:
            self.get_logger().warn(f"深度影像解碼失敗: {e}")

    def _info_callback(self, msg: CameraInfo):
        with self._lock:
            self._camera_info = msg

    def _get_depth_at(self, u: float, v: float, img: np.ndarray) -> float:
        h, w = img.shape
        ui, vi = int(round(u)), int(round(v))
        r = self.depth_window
        patch = img[
            max(0, vi - r):min(h, vi + r + 1),
            max(0, ui - r):min(w, ui + r + 1),
        ]
        valid = patch[(patch > 0.05) & np.isfinite(patch)]
        return float(np.median(valid)) if valid.size > 0 else 0.0

    # ------------------------------------------------------------------
    # Detection callback
    # ------------------------------------------------------------------
    def _det_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"無法解析 JSON: {e}")
            return

        with self._lock:
            depth_img = self._depth_image
            cam_info = self._camera_info

        if depth_img is None:
            self.get_logger().warn("尚未收到深度影像", once=True)
            return
        if cam_info is None:
            self.get_logger().warn("尚未收到 CameraInfo", once=True)
            return

        k = cam_info.k
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        stamp = self.get_clock().now().to_msg()

        results = []
        for det in data.get('detections', []):
            score = float(det.get('score', 0.0))
            if score < self.min_confidence:
                continue

            center = det.get('center', {})
            u = float(center.get('x', 0.0))
            v = float(center.get('y', 0.0))

            d = self._get_depth_at(u, v, depth_img)
            if d <= 0.0:
                continue

            cam_x = (u - cx) * d / fx
            cam_y = (v - cy) * d / fy
            cam_z = d

            try:
                pt_in = PointStamped()
                pt_in.header.frame_id = self.camera_frame
                pt_in.header.stamp = stamp
                pt_in.point.x, pt_in.point.y, pt_in.point.z = cam_x, cam_y, cam_z

                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.camera_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
                )
                pt_out = do_transform_point(pt_in, tf)

            except tf2_ros.LookupException as e:
                self.get_logger().warn(
                    f"TF LookupException: {self.camera_frame} → {self.map_frame} | {e}",
                    throttle_duration_sec=3.0,
                )
                return
            except tf2_ros.ConnectivityException as e:
                self.get_logger().warn(
                    f"TF ConnectivityException: {self.camera_frame} → {self.map_frame} | {e}",
                    throttle_duration_sec=3.0,
                )
                return
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(
                    f"TF ExtrapolationException: {self.camera_frame} → {self.map_frame} "
                    f"(超出 TF buffer 範圍) | {e}",
                    throttle_duration_sec=3.0,
                )
                return

            results.append({
                'class': det.get('class', 'unknown'),
                'score': score,
                'position': {
                    'x': pt_out.point.x,
                    'y': pt_out.point.y,
                    'z': pt_out.point.z,
                },
            })

        if not results:
            return

        out = String()
        out.data = json.dumps(
            {'frame_id': self.map_frame, 'detections': results},
            ensure_ascii=False,
        )
        self.raw_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
