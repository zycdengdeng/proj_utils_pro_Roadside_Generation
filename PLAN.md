# Segment Pipeline 实现计划

## 新增模块：`segment_pipeline/`

### 文件结构
```
segment_pipeline/
├── segment_pipeline.py          # 主入口，读取 filtered_segments.json，调度全流程
├── ego_transform.py             # 从标注提取 ego 变换 (get_vehicle_transform / getWorld2Carlidar)
├── pose_generator.py            # 生成 pose.csv (世界坐标 + yaw→quaternion)
├── annotation_converter.py      # 世界坐标标注 → ego LiDAR 坐标系标注
├── direction_detector.py        # 从轨迹推算行驶方向
└── run_segment_pipeline.py      # CLI 入口 (支持 --scene / --vehicle-id 过滤)
```

### 需要修改的现有文件
```
transfer_video_maker/
├── caption一键修理/update_captions.py   # 改为从 direction.json 自动读取方向
└── generate_transfer2_videos.py         # 适配 segment 结构（可选）
```

---

## 实现步骤

### Step 1: ego_transform.py
- 移植用户提供的 `get_vehicle_transform()` 和 `getWorld2Carlidar()` 逻辑
- LIDAR_ADJUST 设为 (0, 0, 0)
- 输入: annotation JSON + vehicle_id → 输出: world2lidar (rotate, trans)

### Step 2: pose_generator.py
- 从标注中提取 ego 车辆的 (x, y, z, roll, pitch, yaw)
- roll/pitch/yaw → quaternion (qx, qy, qz, qw)
- 输出: pose.csv (timestamp,x,y,z,qx,qy,qz,qw,frame_id)

### Step 3: annotation_converter.py
- 对每帧标注: 取出 ego 车的 world pose
- 把所有其他物体的 (x,y,z,yaw) 从世界坐标系转到 ego LiDAR 坐标系
- 排除 ego 自身
- 输出格式同用户示例 (id, label, x,y,z, l,w,h, roll,pitch,yaw, occlusion, num_points, vx,vy)

### Step 4: direction_detector.py
- 从 29 帧 ego 轨迹计算位移方向
- 映射到 "west to east" / "east to west" / "north to south" / "south to north"
- 输出: direction.json

### Step 5: segment_pipeline.py (主调度)
- 读取 filtered_segments.json
- 支持 --scene / --vehicle-id 过滤
- 对每个 segment:
  1. 生成 pose.csv
  2. 生成 per-frame annotation JSON
  3. 生成 direction.json
  4. 调用投影模块 (depth/blur/hdmap) 处理 29 帧
  5. 打包视频 + caption

### Step 6: update_captions.py 修改
- 新增从 direction.json 自动读取方向的逻辑
- 不再依赖硬编码的 SCENE_DIRECTION（保留作为 fallback）

---

## 输出目录结构
```
output/
└── scene{id}/
    └── vehicle{vid}_seg{nn}/
        ├── pose.csv
        ├── direction.json
        ├── annotations/
        │   ├── {timestamp_1}.json
        │   └── ...  (29个)
        ├── projections/
        │   └── {timestamp}/
        │       ├── depth/  {FN,FW,FL,FR,RL,RR,RN}.jpg
        │       ├── gt/     {FN,...}.jpg
        │       └── overlay/ ...
        ├── videos/
        │   └── {transfer_camera_name}/
        │       └── {scene}_{vid}_seg{nn}.mp4
        ├── control_input_{type}/
        │   └── {transfer_camera_name}/
        │       └── ...mp4
        └── captions/
            └── {transfer_camera_name}/
                └── ...json
```
