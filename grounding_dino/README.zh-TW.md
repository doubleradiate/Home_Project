# grounding_dino

ROS 2 套件，使用 [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) 進行零樣本物件偵測，偵測提示詞由上游 RAM（Recognize Anything Model）節點動態提供。

## 系統架構

```
/realsense/rgb  ──┐
                  ├──► ram_grounding_dino_node ──► /grounding_dino/detections       (JSON String)
/ram/tags       ──┘                           └──► /grounding_dino/detections_point (geometry_msgs/Point)
```

節點啟動後等待 `/ram/tags` 傳入標籤字串，經 stop-tags 過濾後以剩餘標籤作為提示詞，對每幀影像執行 GroundingDINO 推論。

## 依賴套件

| 套件 | 用途 |
|------|------|
| `transformers` | GroundingDINO 模型與前後處理 |
| `torch` | 推論後端（支援 CUDA / CPU） |
| `opencv-python` | ROS Image 轉 numpy |
| `Pillow` | numpy 轉 PIL 供模型輸入 |
| `rclpy` | ROS 2 Python 客戶端 |
| `sensor_msgs`, `std_msgs`, `geometry_msgs` | ROS 訊息型別 |

## 安裝與執行

```bash
colcon build --packages-select grounding_dino
source install/setup.bash

ros2 run grounding_dino ram_grounding_dino_node
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `image_topic` | `/realsense/rgb` | 輸入影像 topic |
| `tags_topic` | `/ram/tags` | 來自 RAM 的標籤 topic |
| `stop_tags_yaml` | `…/config/linit_prompt.yaml` | stop-tags YAML 路徑（必填） |
| `model_id` | `IDEA-Research/grounding-dino-tiny` | HuggingFace 模型 ID |
| `max_prompt_count` | `20` | 每幀最多傳入模型的提示詞數量 |
| `max_detections` | `30` | NMS 後保留的最大偵測數量 |
| `box_threshold` | `0.35` | 邊界框信心度門檻 |
| `text_threshold` | `0.25` | 文字標籤信心度門檻 |
| `nms_iou_threshold` | `0.50` | 各類別 NMS 的 IoU 門檻（0 表示停用） |
| `sort_mode` | `score` | 偵測結果排序方式（見下表） |
| `min_period_s` | `0.50` | 兩次推論之間的最短間隔秒數 |

### sort_mode 選項

| 值 | 排序行為 |
|----|---------|
| `score` | 信心度由高到低（預設） |
| `area` | 邊界框面積由大到小 |
| `left_to_right` | 依中心點 x 座標由左到右 |
| `top_to_bottom` | 依中心點 y 座標由上到下 |
| `bottom_right` | 依 `cx + cy` 由大到小（偏右下角優先） |

## Topics

### 訂閱

| Topic | 型別 | 說明 |
|-------|------|------|
| `/realsense/rgb` | `sensor_msgs/Image` | 相機影像（支援 rgb8 / bgr8 / rgba8 / bgra8 / mono8） |
| `/ram/tags` | `std_msgs/String` | 來自 RAM 的標籤，支援逗號分隔或 JSON 格式 |

### 發布

| Topic | 型別 | 說明 |
|-------|------|------|
| `/grounding_dino/detections` | `std_msgs/String` | 所有偵測結果的 JSON |
| `/grounding_dino/detections_point` | `geometry_msgs/Point` | 排序後第一個目標的中心點（`x`、`y` 為像素座標，`z` 為信心度） |

### 偵測結果 JSON 格式

```json
{
  "prompt_used": ["chair", "table"],
  "sort_mode": "score",
  "count": 2,
  "detections": [
    {
      "class": "chair",
      "score": 0.82,
      "bbox": [x1, y1, x2, y2],
      "center": {"x": 320.0, "y": 240.0},
      "area": 14400.0,
      "width": 120.0,
      "height": 120.0
    }
  ]
}
```

## Stop-tags 設定

不應作為偵測提示詞的標籤（例如顏色、泛用場景詞）列於 `grounding_dino/config/linit_prompt.yaml`：

```yaml
stop_tags:
  - indoor
  - white
  - furniture
  # …
```

此檔案**必須存在**且 `stop_tags` 列表**不可為空**，否則節點啟動時直接報錯。

執行時可覆蓋路徑：

```bash
ros2 run grounding_dino ram_grounding_dino_node \
  --ros-args -p stop_tags_yaml:=/path/to/my_stop_tags.yaml
```
