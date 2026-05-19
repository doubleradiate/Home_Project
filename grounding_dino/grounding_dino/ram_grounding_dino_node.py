#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Point


def normalize_tag(tag: str) -> str:
    tag = str(tag).lower().strip()
    tag = tag.replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = tag.strip(" .,:;[](){}\"'")
    tag = re.sub(r"^(a|an|the)\s+", "", tag)
    return tag.strip()


def load_stop_tags_from_yaml(yaml_path: str) -> set:
    if not yaml_path:
        raise ValueError("stop_tags_yaml is empty")

    path = Path(yaml_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"stop_tags yaml not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"YAML is empty: {path}")

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dictionary")

    tags = data.get("stop_tags", None)

    if tags is None:
        raise ValueError("YAML must contain key: stop_tags")

    if not isinstance(tags, list):
        raise ValueError("stop_tags must be a list")

    stop_tags = set()

    for tag in tags:
        tag = normalize_tag(str(tag))
        if tag:
            stop_tags.add(tag)

    if not stop_tags:
        raise ValueError("stop_tags is empty after normalize")

    return stop_tags


def clean_prompt(tag: str, stop_tags: set) -> str:
    tag = normalize_tag(tag)

    if not tag:
        return ""

    if tag in stop_tags:
        return ""

    words = tag.split()
    words = [w for w in words if w not in stop_tags]

    cleaned = " ".join(words).strip()

    if not cleaned:
        return ""

    if cleaned in stop_tags:
        return ""

    if len(cleaned) < 2:
        return ""

    return cleaned


def parse_tag_string(text: str) -> List[str]:
    text = str(text).strip()

    if not text:
        return []

    # 支援 JSON list，例如:
    # ["chair", "table", "white"]
    try:
        data = json.loads(text)

        if isinstance(data, list):
            return [str(x) for x in data]

        if isinstance(data, dict):
            for key in ["tags", "labels", "classes", "objects"]:
                if key in data and isinstance(data[key], list):
                    return [str(x) for x in data[key]]

    except Exception:
        pass

    # 支援一般字串:
    # chair, table, white, indoor
    text = text.replace("[", "").replace("]", "")
    text = text.replace("(", "").replace(")", "")
    text = text.replace("{", "").replace("}", "")
    text = text.replace('"', "").replace("'", "")

    parts = re.split(r"[,;\n|]+", text)

    return [p.strip() for p in parts if p.strip()]


def limit_and_filter_prompts(
    raw_tags: List[str],
    max_prompt_count: int,
    stop_tags: set,
) -> List[str]:
    prompts = []
    used = set()

    for tag in raw_tags:
        prompt = clean_prompt(tag, stop_tags)

        if not prompt:
            continue

        if prompt in used:
            continue

        used.add(prompt)
        prompts.append(prompt)

        if len(prompts) >= max_prompt_count:
            break

    return prompts


def box_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def nms_by_class(
    detections: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    if iou_threshold <= 0:
        return detections

    kept = []
    class_names = sorted(set(d["class"] for d in detections))

    for cls in class_names:
        cls_dets = [d for d in detections if d["class"] == cls]
        cls_dets.sort(key=lambda d: d["score"], reverse=True)

        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)

            remain = []

            for d in cls_dets:
                if box_iou(best["bbox"], d["bbox"]) < iou_threshold:
                    remain.append(d)

            cls_dets = remain

    return kept


class RamGroundingDinoNode(Node):
    def __init__(self):
        super().__init__("ram_grounding_dino_node")

        self.get_logger().info("Initializing RamGroundingDinoNode...")

        self.declare_parameter("image_topic", "/realsense/rgb")
        self.declare_parameter("tags_topic", "/ram/tags")

        self.declare_parameter(
            "stop_tags_yaml",
            "/home/hungyu/work_ws/src/grounding_dino/grounding_dino/config/linit_prompt.yaml",
        )

        self.declare_parameter("model_id", "IDEA-Research/grounding-dino-tiny")

        self.declare_parameter("max_prompt_count", 20)
        self.declare_parameter("max_detections", 30)

        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)
        self.declare_parameter("nms_iou_threshold", 0.50)

        # score / area / left_to_right / top_to_bottom / bottom_right
        self.declare_parameter("sort_mode", "score")

        # 避免每張 frame 都跑 GroundingDINO
        self.declare_parameter("min_period_s", 0.50)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.tags_topic = str(self.get_parameter("tags_topic").value)
        self.stop_tags_yaml = str(self.get_parameter("stop_tags_yaml").value)

        self.model_id = str(self.get_parameter("model_id").value)

        self.max_prompt_count = int(self.get_parameter("max_prompt_count").value)
        self.max_detections = int(self.get_parameter("max_detections").value)

        self.box_threshold = float(self.get_parameter("box_threshold").value)
        self.text_threshold = float(self.get_parameter("text_threshold").value)
        self.nms_iou_threshold = float(self.get_parameter("nms_iou_threshold").value)

        self.sort_mode = str(self.get_parameter("sort_mode").value)
        self.min_period_s = float(self.get_parameter("min_period_s").value)

        self.stop_tags = load_stop_tags_from_yaml(self.stop_tags_yaml)

        self.get_logger().info(f"Loaded stop_tags YAML: {self.stop_tags_yaml}")
        self.get_logger().info(f"STOP_TAGS count: {len(self.stop_tags)}")

        self.current_prompts = []

        self.last_process_time = 0.0
        self.is_processing = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.get_logger().info(f"Loading GroundingDINO model: {self.model_id}")
        self.get_logger().info(f"Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id)

        self.model.to(self.device)
        self.model.eval()

        self.image_subscriber = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.tags_subscriber = self.create_subscription(
            String,
            self.tags_topic,
            self.tags_callback,
            10,
        )

        self.detections_publisher = self.create_publisher(
            String,
            "/grounding_dino/detections",
            10,
        )

        self.detections_point_publisher = self.create_publisher(
            Point,
            "/grounding_dino/detections_point",
            10,
        )

        self.get_logger().info(f"Subscribed image topic: {self.image_topic}")
        self.get_logger().info(f"Subscribed tags topic: {self.tags_topic}")
        self.get_logger().warn("No prompt yet. Waiting for /ram/tags...")

    def tags_callback(self, msg: String):
        raw_tags = parse_tag_string(msg.data)

        prompts = limit_and_filter_prompts(
            raw_tags,
            self.max_prompt_count,
            self.stop_tags,
        )

        if not prompts:
            self.get_logger().warn(
                f"No valid prompts after stop_tags filter. Raw tags: {raw_tags}"
            )
            return

        self.current_prompts = prompts

        self.get_logger().info(
            f"Updated prompts after stop_tags filter "
            f"({len(self.current_prompts)}): {self.current_prompts}"
        )

    def image_callback(self, msg: Image):
        now = time.time()

        if self.is_processing:
            return

        if now - self.last_process_time < self.min_period_s:
            return

        if not self.current_prompts:
            self.get_logger().warn("No prompt available. Skip GroundingDINO.")
            return

        self.last_process_time = now
        self.is_processing = True

        try:
            cv_image = self.ros_image_to_cv2(msg)

            self.get_logger().info(
                f"Processing image with prompts: {self.current_prompts}"
            )

            detections = self.detect_objects(cv_image)
            detections = nms_by_class(detections, self.nms_iou_threshold)
            detections = self.sort_detections(detections)

            if self.max_detections > 0:
                detections = detections[:self.max_detections]

            self.publish_detections(detections)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

        finally:
            self.is_processing = False

    def ros_image_to_cv2(self, ros_image: Image) -> np.ndarray:
        encoding = ros_image.encoding.lower()
        height = ros_image.height
        width = ros_image.width
        step = ros_image.step

        data = np.frombuffer(ros_image.data, dtype=np.uint8)

        if encoding in ["rgb8", "bgr8"]:
            channels = 3
            expected_step = width * channels

            if step == expected_step:
                img = data.reshape((height, width, channels))
            else:
                rows = data.reshape((height, step))
                img = rows[:, :expected_step].reshape((height, width, channels))

            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return img.copy()

        if encoding in ["rgba8", "bgra8"]:
            channels = 4
            expected_step = width * channels

            if step == expected_step:
                img = data.reshape((height, width, channels))
            else:
                rows = data.reshape((height, step))
                img = rows[:, :expected_step].reshape((height, width, channels))

            if encoding == "rgba8":
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return img.copy()

        if encoding == "mono8":
            expected_step = width

            if step == expected_step:
                img = data.reshape((height, width))
            else:
                rows = data.reshape((height, step))
                img = rows[:, :expected_step].reshape((height, width))

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            return img.copy()

        raise ValueError(f"Unsupported image encoding: {ros_image.encoding}")

    def detect_objects(self, bgr_image: np.ndarray) -> List[Dict[str, Any]]:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)

        text_labels = [self.current_prompts]

        inputs = self.processor(
            images=pil_image,
            text=text_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        height = pil_image.height
        width = pil_image.width

        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )
        except TypeError:
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    input_ids=inputs.input_ids,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(height, width)],
                )
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    target_sizes=[(height, width)],
                )

        result = results[0]

        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])

        detections = []

        for box, score, label in zip(boxes, scores, labels):
            box_list = box.detach().cpu().tolist()
            score_value = float(score.detach().cpu().item())

            x1, y1, x2, y2 = [float(v) for v in box_list]

            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            area = float(max(0.0, bw) * max(0.0, bh))

            class_name = str(label)
            class_name = clean_prompt(class_name, self.stop_tags)

            if not class_name:
                continue

            det = {
                "class": class_name,
                "score": score_value,
                "bbox": [x1, y1, x2, y2],
                "center": {
                    "x": cx,
                    "y": cy,
                },
                "area": area,
                "width": bw,
                "height": bh,
            }

            detections.append(det)

        return detections

    def sort_detections(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mode = self.sort_mode

        if mode == "area":
            detections.sort(key=lambda d: d["area"], reverse=True)

        elif mode == "left_to_right":
            detections.sort(key=lambda d: d["center"]["x"])

        elif mode == "top_to_bottom":
            detections.sort(key=lambda d: d["center"]["y"])

        elif mode == "bottom_right":
            detections.sort(
                key=lambda d: d["center"]["x"] + d["center"]["y"],
                reverse=True,
            )

        else:
            detections.sort(key=lambda d: d["score"], reverse=True)

        return detections

    def publish_detections(self, detections: List[Dict[str, Any]]):
        payload = {
            "prompt_used": self.current_prompts,
            "sort_mode": self.sort_mode,
            "count": len(detections),
            "detections": detections,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)

        self.detections_publisher.publish(msg)

        if detections:
            selected = detections[0]

            point_msg = Point()
            point_msg.x = float(selected["center"]["x"])
            point_msg.y = float(selected["center"]["y"])
            point_msg.z = float(selected["score"])

            self.detections_point_publisher.publish(point_msg)

            self.get_logger().info(
                f"Selected: class={selected['class']} "
                f"score={selected['score']:.3f} "
                f"center=({selected['center']['x']:.1f}, {selected['center']['y']:.1f})"
            )

        else:
            self.get_logger().info("No detections.")


def main(args=None):
    rclpy.init(args=args)

    node = RamGroundingDinoNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
