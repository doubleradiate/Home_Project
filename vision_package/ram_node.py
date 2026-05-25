#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAM / RAM++ tag publisher node.

Flow:
  image topic -> RAM/RAM++ -> std_msgs/String /ram/tags

Notes:
- This node expects the Recognize-Anything repository/package to be installed or importable.
- If it is not installed as a Python package, set ram_repo_path to your local repo path,
  for example: /workspace/visual/src/recognize-anything
- Set pretrained_path to your RAM or RAM++ checkpoint.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def normalize_tag(tag: str) -> str:
    tag = str(tag).lower().strip()
    tag = tag.replace("_", " ")
    tag = re.sub(r"\s+", " ", tag)
    tag = tag.strip(" .,:;[](){}\"'")
    tag = re.sub(r"^(a|an|the)\s+", "", tag)
    return tag.strip()


def split_ram_result(result: Any) -> List[str]:
    """
    RAM/RAM++ inference functions may return:
      - "tag1 | tag2 | tag3"
      - ("tag1 | tag2", "中文標籤...")
      - ["tag1", "tag2"]
    This function normalizes them into a list of English tags.
    """
    if result is None:
        return []

    if isinstance(result, tuple):
        # Usually first field is English tags.
        result = result[0]

    if isinstance(result, list):
        raw_items = result
    else:
        text = str(result)
        text = text.replace(";", ",")
        text = text.replace("|", ",")
        raw_items = text.split(",")

    tags = []
    used = set()
    for item in raw_items:
        tag = normalize_tag(item)
        if not tag:
            continue
        if tag in used:
            continue
        used.add(tag)
        tags.append(tag)
    return tags


class RamTagNode(Node):
    def __init__(self):
        super().__init__("ram_tag_node")

        self.declare_parameter("image_topic", "/realsense/rgb")
        self.declare_parameter("tags_topic", "/ram/tags")
        self.declare_parameter("debug_topic", "/ram/debug_image")

        # RAM repo path. Use this when recognize-anything is cloned but not pip-installed.
        self.declare_parameter("ram_repo_path", "/workspace/visual/src/recognize-anything")

        # RAM model settings.
        # model_type: ram_plus or ram
        self.declare_parameter("model_type", "ram_plus")
        self.declare_parameter("pretrained_path", "/workspace/visual/src/recognize-anything/pretrained/ram_plus_swin_large_14m.pth")
        self.declare_parameter("vit", "swin_l")
        self.declare_parameter("image_size", 384)

        # Runtime settings.
        self.declare_parameter("min_period_s", 1.0)
        self.declare_parameter("max_tags", 30)
        self.declare_parameter("publish_json", False)
        self.declare_parameter("publish_empty", False)

        # Optional threshold override. RAM official inference usually handles threshold internally.
        # This node tries to override common threshold attributes if available.
        self.declare_parameter("tag_threshold", -1.0)

        # Fallback tags only used when RAM returns no tag and fallback_when_empty is true.
        self.declare_parameter("fallback_when_empty", False)
        self.declare_parameter(
            "fallback_tags",
            "sofa,couch,chair,table,coffee table,television,tv,window,lamp,door,cabinet,bottle,person",
        )

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.tags_topic = str(self.get_parameter("tags_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.ram_repo_path = str(self.get_parameter("ram_repo_path").value)
        self.model_type = str(self.get_parameter("model_type").value).lower().strip()
        self.pretrained_path = str(self.get_parameter("pretrained_path").value)
        self.vit = str(self.get_parameter("vit").value)
        self.image_size = int(self.get_parameter("image_size").value)

        self.min_period_s = float(self.get_parameter("min_period_s").value)
        self.max_tags = int(self.get_parameter("max_tags").value)
        self.publish_json = bool(self.get_parameter("publish_json").value)
        self.publish_empty = bool(self.get_parameter("publish_empty").value)
        self.tag_threshold = float(self.get_parameter("tag_threshold").value)

        self.fallback_when_empty = bool(self.get_parameter("fallback_when_empty").value)
        self.fallback_tags = [
            normalize_tag(x)
            for x in str(self.get_parameter("fallback_tags").value).split(",")
            if normalize_tag(x)
        ]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.transform = None
        self.ram_inference = None

        self.last_process_time = 0.0
        self.is_processing = False

        self.get_logger().info("Initializing RAM tag node...")
        self.get_logger().info(f"image_topic     : {self.image_topic}")
        self.get_logger().info(f"tags_topic      : {self.tags_topic}")
        self.get_logger().info(f"model_type      : {self.model_type}")
        self.get_logger().info(f"pretrained_path : {self.pretrained_path}")
        self.get_logger().info(f"device          : {self.device}")

        self._load_ram_model()

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )
        self.tags_pub = self.create_publisher(String, self.tags_topic, 10)

        self.get_logger().info("RAM tag node ready.")

    def _load_ram_model(self):
        if self.ram_repo_path and Path(self.ram_repo_path).exists():
            sys.path.insert(0, self.ram_repo_path)
            self.get_logger().info(f"Added RAM repo path: {self.ram_repo_path}")

        if not Path(self.pretrained_path).exists():
            self.get_logger().warn(
                f"RAM checkpoint not found: {self.pretrained_path}. "
                "Set -p pretrained_path:=/path/to/ram_plus_swin_large_14m.pth"
            )

        try:
            from ram import get_transform
            from ram import inference_ram as ram_inference
            from ram.models import ram, ram_plus
        except Exception as e:
            self.get_logger().error(
                "Failed to import Recognize-Anything RAM package. "
                "Check ram_repo_path or install recognize-anything. "
                f"Import error: {repr(e)}"
            )
            raise

        self.transform = get_transform(image_size=self.image_size)
        self.ram_inference = ram_inference

        if self.model_type in ["ram_plus", "ram++", "ramplus"]:
            self.model = ram_plus(
                pretrained=self.pretrained_path,
                image_size=self.image_size,
                vit=self.vit,
            )
        elif self.model_type == "ram":
            self.model = ram(
                pretrained=self.pretrained_path,
                image_size=self.image_size,
                vit=self.vit,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Use ram_plus or ram.")

        self.model.eval()
        self.model.to(self.device)

        # Try to override common RAM thresholds if the model exposes them.
        if self.tag_threshold >= 0:
            self._try_set_threshold(self.tag_threshold)

        self.get_logger().info("RAM/RAM++ model loaded.")

    def _try_set_threshold(self, threshold: float):
        changed = False

        for attr in ["class_threshold", "threshold", "tag_threshold"]:
            if hasattr(self.model, attr):
                old_value = getattr(self.model, attr)
                try:
                    if torch.is_tensor(old_value):
                        setattr(self.model, attr, torch.ones_like(old_value) * float(threshold))
                    else:
                        setattr(self.model, attr, float(threshold))
                    changed = True
                    self.get_logger().info(f"Set RAM model.{attr} to {threshold}")
                except Exception as e:
                    self.get_logger().warn(f"Could not set model.{attr}: {repr(e)}")

        if not changed:
            self.get_logger().warn(
                "tag_threshold was provided, but this RAM model did not expose a known threshold attribute. "
                "The official RAM inference threshold may still be used internally."
            )

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
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).copy()

        raise ValueError(f"Unsupported image encoding: {ros_image.encoding}")

    def image_callback(self, msg: Image):
        now = time.time()
        if self.is_processing:
            return
        if now - self.last_process_time < self.min_period_s:
            return

        self.last_process_time = now
        self.is_processing = True

        try:
            bgr = self.ros_image_to_cv2(msg)
            tags = self.run_ram(bgr)

            if not tags and self.fallback_when_empty:
                tags = self.fallback_tags
                self.get_logger().warn(f"RAM returned no tag. Use fallback tags: {tags}")

            if self.max_tags > 0:
                tags = tags[: self.max_tags]

            if tags or self.publish_empty:
                out = String()
                if self.publish_json:
                    out.data = json.dumps({"tags": tags}, ensure_ascii=False)
                else:
                    out.data = ", ".join(tags)
                self.tags_pub.publish(out)

            self.get_logger().info(f"RAM tags ({len(tags)}): {tags}")

        except Exception as e:
            self.get_logger().error(f"RAM inference error: {repr(e)}")

        finally:
            self.is_processing = False

    def run_ram(self, bgr_image: np.ndarray) -> List[str]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        pil = PILImage.fromarray(rgb)

        image_tensor = self.transform(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.ram_inference(image_tensor, self.model)

        return split_ram_result(result)


def main(args=None):
    rclpy.init(args=args)
    node = RamTagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()