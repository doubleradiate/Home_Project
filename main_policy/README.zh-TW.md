# main_policy

ROS 2 套件，負責語意地圖的 **地圖整合層**：接收已投影至 map frame 的 3D 偵測結果，進行去重、EMA 位置融合，維護持久化語意地圖並發布 RViz 視覺化。

## 系統架構

```
/semantic_map/raw_detections ──► map_integrator_node ──► /semantic_map     (MarkerArray)
                                                     └──► /semantic_objects (JSON String)
```

TF 投影由上游的 `Semanti_Map` 套件負責，本套件只處理地圖邏輯，不碰深度影像或 TF。

## 物件整合機制

每筆偵測進來時：

1. 在已知物件中尋找距離小於 `merge_distance` 的**同類別**物件
2. 找到 → 以 EMA（`ema_alpha`）更新位置與信心度，`observe_count` +1
3. 找不到 → 新增物件，分配格式為 `{class}_{自增數字}` 的 ID

由於 GroundingDINO 不提供跨幀穩定 ID，去重完全依賴空間距離判斷。

## 地圖持久化

- 每隔 `auto_save_interval` 秒自動存檔一次
- 節點關閉（Ctrl+C）時存檔一次
- 下次啟動自動載入，繼續累積

## 安裝與執行

```bash
colcon build --packages-select main_policy
source install/setup.bash

ros2 run main_policy map_integrator_node \
  --ros-args --params-file src/Semanti_Map/params.yaml
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `raw_detections_topic` | `/semantic_map/raw_detections` | 上游投影結果 topic |
| `map_frame` | `map` | 地圖座標系名稱 |
| `merge_distance` | `0.5` | 同類別物件合併距離門檻（公尺） |
| `ema_alpha` | `0.3` | 位置融合的指數移動平均係數（0～1，越大越偏向新觀測） |
| `map_save_path` | `/home/hungyu/work_ws/semantic_map.json` | 存檔路徑 |
| `auto_save_interval` | `30.0` | 自動存檔週期（秒） |
| `publish_rate` | `2.0` | MarkerArray 與 JSON 發布頻率（Hz） |

## Topics

### 訂閱

| Topic | 型別 | 說明 |
|-------|------|------|
| `/semantic_map/raw_detections` | `std_msgs/String` | 來自 Semanti_Map 的 map frame 3D 偵測 |

### 發布

| Topic | 型別 | 說明 |
|-------|------|------|
| `/semantic_map` | `visualization_msgs/MarkerArray` | RViz 球體 + 文字標籤 |
| `/semantic_objects` | `std_msgs/String` | 目前所有物件的 JSON 清單 |

### `/semantic_objects` JSON 格式

```json
{
  "frame_id": "map",
  "count": 3,
  "objects": [
    {
      "id": "chair_0",
      "class": "chair",
      "position": {"x": 1.23, "y": -0.45, "z": 0.80},
      "confidence": 0.82,
      "observe_count": 7
    }
  ]
}
```

## 完整系統 Topic 流向

```
RAM 節點
  └─► /ram/tags
        └─► grounding_dino (ram_grounding_dino_node)
              ├─► /grounding_dino/detections
              └─► /grounding_dino/detections_point

相機
  ├─► /realsense/rgb         → grounding_dino
  ├─► /realsense/depth       ┐
  └─► /realsense/camera_info ┼─► Semanti_Map (semantic_map_node)
                              ┘        └─► /semantic_map/raw_detections
                                               └─► main_policy (map_integrator_node)
                                                     ├─► /semantic_map      (RViz)
                                                     └─► /semantic_objects  (JSON)
```
