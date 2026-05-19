# Semanti_Map

ROS 2 套件，負責語意地圖的 **TF 投影層**：接收 GroundingDINO 偵測結果，結合深度影像與相機內參將 2D pixel 座標投影為 map frame 3D 座標，轉發給下游地圖整合節點處理。

## 系統架構

```
/grounding_dino/detections ──┐
/realsense/depth             ├──► semantic_map_node ──► /semantic_map/raw_detections
/realsense/camera_info       ┘
```

本套件**不做**去重、融合、存檔，只負責感測資料的座標轉換。地圖整合邏輯由 `main_policy` 的 `map_integrator_node` 處理。

## 2D → 3D 投影流程

```
DINO center (u, v)  +  深度影像取 depth_window 視窗中位數 d
          ↓  相機內參 (fx, fy, cx, cy)
  cam_x = (u - cx) × d / fx
  cam_y = (v - cy) × d / fy
  cam_z = d
          ↓  TF2  (camera_frame → map_frame)
     (map_x, map_y, map_z)  →  發布至 raw_detections
```

深度取樣使用 `depth_window` 半徑的視窗對有效值取**中位數**，避免單一無效像素造成誤差。

## 依賴套件

| 套件 | 用途 |
|------|------|
| `numpy` | 深度影像解碼與視窗取樣 |
| `tf2_ros` / `tf2_geometry_msgs` | 座標系轉換 |
| `rclpy` | ROS 2 Python 客戶端 |
| `sensor_msgs` | Image、CameraInfo |
| `geometry_msgs` | PointStamped |
| `std_msgs` | String（JSON 發布） |

## 安裝與執行

```bash
colcon build --packages-select Semanti_Map
source install/setup.bash

ros2 run Semanti_Map semantic_map_node \
  --ros-args --params-file src/Semanti_Map/params.yaml
```

## 參數說明

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `use_sim_time` | `false` | 模擬環境請改 `true` |
| `detection_topic` | `/grounding_dino/detections` | DINO 偵測結果 topic |
| `depth_topic` | `/realsense/depth` | 深度影像 topic（16UC1 或 32FC1） |
| `camera_info_topic` | `/realsense/camera_info` | 相機內參 topic |
| `raw_detections_topic` | `/semantic_map/raw_detections` | 投影結果發布 topic |
| `camera_frame` | `Camera_OmniVision_OV9782_Color` | 相機 TF frame |
| `map_frame` | `map` | 目標座標系 |
| `min_confidence` | `0.35` | 過濾低信心度偵測 |
| `depth_scale` | `0.001` | 深度縮放係數（16UC1 mm→m 用 `0.001`；32FC1 已是公尺用 `1.0`） |
| `depth_window` | `2` | 深度取樣視窗半徑（pixels） |

## Topics

### 訂閱

| Topic | 型別 | 說明 |
|-------|------|------|
| `/grounding_dino/detections` | `std_msgs/String` | DINO JSON 偵測結果 |
| `/realsense/depth` | `sensor_msgs/Image` | 深度影像 |
| `/realsense/camera_info` | `sensor_msgs/CameraInfo` | 相機內參 |

### 發布

| Topic | 型別 | 說明 |
|-------|------|------|
| `/semantic_map/raw_detections` | `std_msgs/String` | map frame 3D 偵測（未去重） |

### `/semantic_map/raw_detections` JSON 格式

```json
{
  "frame_id": "map",
  "detections": [
    {
      "class": "chair",
      "score": 0.82,
      "position": {"x": 1.23, "y": -0.45, "z": 0.80}
    }
  ]
}
```

## TF 錯誤說明

| 例外 | 原因 | 處理 |
|------|------|------|
| `LookupException` | frame 不存在（AMCL 未初始化） | 略過，每 3 秒警告一次 |
| `ConnectivityException` | frame tree 中無連結路徑 | 略過，每 3 秒警告一次 |
| `ExtrapolationException` | TF buffer 尚未填滿（啟動競爭） | 略過，每 3 秒警告一次，數秒後自動恢復 |
