#!/usr/bin/env python3
"""
Map Integrator Node — 地圖整合層
----------------------------------
職責：接收已轉換至 map frame 的 3D 偵測結果，進行去重、EMA 位置融合，
      維護持久化語意地圖並發布 RViz 可視化。

訂閱:
  - /semantic_map/raw_detections (std_msgs/String, JSON)
發布:
  - /semantic_map     (visualization_msgs/MarkerArray)
  - /semantic_objects (std_msgs/String, JSON)
"""

import json
import math
import os
from threading import Lock

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray


class SemanticObject:
    def __init__(self, obj_id, class_name, x, y, z, confidence=1.0):
        self.obj_id = obj_id
        self.class_name = class_name
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence
        self.observe_count = 1
        self.last_seen = None

    def update(self, x, y, z, confidence, alpha=0.3):
        self.x = (1 - alpha) * self.x + alpha * x
        self.y = (1 - alpha) * self.y + alpha * y
        self.z = (1 - alpha) * self.z + alpha * z
        self.confidence = max(self.confidence, confidence)
        self.observe_count += 1

    def to_dict(self):
        return {
            'id': self.obj_id,
            'class': self.class_name,
            'position': {'x': self.x, 'y': self.y, 'z': self.z},
            'confidence': self.confidence,
            'observe_count': self.observe_count,
        }


class MapIntegratorNode(Node):
    def __init__(self):
        super().__init__('map_integrator_node')

        # ---------- 參數 ----------
        self.declare_parameter('raw_detections_topic', '/semantic_map/raw_detections')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('merge_distance', 0.5)
        self.declare_parameter('ema_alpha', 0.3)
        self.declare_parameter('map_save_path', '/home/hungyu/work_ws/semantic_map.json')
        self.declare_parameter('auto_save_interval', 30.0)
        self.declare_parameter('publish_rate', 2.0)

        self.raw_topic = self.get_parameter('raw_detections_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        self.merge_distance = self.get_parameter('merge_distance').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.map_save_path = self.get_parameter('map_save_path').value
        self.auto_save_interval = self.get_parameter('auto_save_interval').value
        self.publish_rate = self.get_parameter('publish_rate').value

        # ---------- 內部狀態 ----------
        self.objects: dict = {}
        self.lock = Lock()
        self._next_id = 0

        # ---------- Subscriber ----------
        self.create_subscription(String, self.raw_topic, self._raw_callback, 10)

        # ---------- Publishers ----------
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_map', 10)
        self.objects_pub = self.create_publisher(String, '/semantic_objects', 10)

        # ---------- Timers ----------
        self.create_timer(1.0 / self.publish_rate, self._publish)
        self.create_timer(self.auto_save_interval, self.save_map)

        self.load_map()

        self.get_logger().info(
            f"MapIntegratorNode 已啟動。\n"
            f"  raw_topic      = {self.raw_topic}\n"
            f"  merge_distance = {self.merge_distance} m\n"
            f"  map_save_path  = {self.map_save_path}"
        )

    # ------------------------------------------------------------------
    # Incoming detections
    # ------------------------------------------------------------------
    def _raw_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"無法解析 JSON: {e}")
            return

        with self.lock:
            for det in data.get('detections', []):
                self._integrate(det)

    def _integrate(self, det: dict):
        class_name = det.get('class', 'unknown')
        score = float(det.get('score', 1.0))
        pos = det.get('position', {})
        x = float(pos.get('x', 0.0))
        y = float(pos.get('y', 0.0))
        z = float(pos.get('z', 0.0))

        best = self._find_nearby(class_name, x, y, z)
        if best is not None:
            best.update(x, y, z, score, self.ema_alpha)
            return

        new_id = f"{class_name}_{self._next_id}"
        self._next_id += 1
        self.objects[new_id] = SemanticObject(new_id, class_name, x, y, z, score)
        self.get_logger().info(f"新物件: {new_id} @ ({x:.2f}, {y:.2f}, {z:.2f})")

    def _find_nearby(self, class_name, x, y, z):
        best, best_dist = None, self.merge_distance
        for obj in self.objects.values():
            if obj.class_name != class_name:
                continue
            dist = math.sqrt((obj.x - x)**2 + (obj.y - y)**2 + (obj.z - z)**2)
            if dist < best_dist:
                best_dist = dist
                best = obj
        return best

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def _publish(self):
        with self.lock:
            objs = list(self.objects.values())

        marker_array = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        now = self.get_clock().now().to_msg()
        for idx, obj in enumerate(objs):
            m = Marker()
            m.header.frame_id = self.map_frame
            m.header.stamp = now
            m.ns = 'semantic_objects'
            m.id = idx * 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=obj.x, y=obj.y, z=obj.z)
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=0.3, y=0.3, z=0.3)
            m.color = self._class_color(obj.class_name)
            marker_array.markers.append(m)

            t = Marker()
            t.header.frame_id = self.map_frame
            t.header.stamp = now
            t.ns = 'semantic_labels'
            t.id = idx * 2 + 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(x=obj.x, y=obj.y, z=obj.z + 0.4)
            t.pose.orientation.w = 1.0
            t.scale.z = 0.25
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            t.text = f"{obj.class_name} ({obj.observe_count})"
            marker_array.markers.append(t)

        self.marker_pub.publish(marker_array)

        out = String()
        out.data = json.dumps(
            {'frame_id': self.map_frame, 'count': len(objs),
             'objects': [o.to_dict() for o in objs]},
            ensure_ascii=False,
        )
        self.objects_pub.publish(out)

    def _class_color(self, class_name) -> ColorRGBA:
        h = hash(class_name)
        return ColorRGBA(
            r=((h >> 16) & 0xFF) / 255.0,
            g=((h >> 8) & 0xFF) / 255.0,
            b=(h & 0xFF) / 255.0,
            a=0.9,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_map(self):
        with self.lock:
            data = {
                'frame_id': self.map_frame,
                'next_id': self._next_id,
                'objects': [o.to_dict() for o in self.objects.values()],
            }
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.map_save_path)), exist_ok=True)
            with open(self.map_save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.get_logger().debug(f"已存檔: {self.map_save_path}")
        except OSError as e:
            self.get_logger().warn(f"存檔失敗: {e}")

    def load_map(self):
        if not os.path.exists(self.map_save_path):
            return
        try:
            with open(self.map_save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._next_id = data.get('next_id', 0)
            for o in data.get('objects', []):
                obj = SemanticObject(
                    o['id'], o['class'],
                    o['position']['x'], o['position']['y'], o['position']['z'],
                    o.get('confidence', 1.0),
                )
                obj.observe_count = o.get('observe_count', 1)
                self.objects[obj.obj_id] = obj
            self.get_logger().info(
                f"載入語意地圖: {len(self.objects)} 個物件 ({self.map_save_path})"
            )
        except (OSError, json.JSONDecodeError, KeyError) as e:
            self.get_logger().warn(f"載入失敗: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MapIntegratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_map()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
