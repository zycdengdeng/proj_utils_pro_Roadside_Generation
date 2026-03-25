# 项目上下文文档

> 此文件供 Claude Code 在上下文压缩后恢复工作记忆。人类开发者也可参考。

---

## 项目简介

**路侧→车端多模态数据集生成工具链**。从路侧传感器（相机+LiDAR）的标注和点云数据，生成模拟车端视角的投影图像、3D标注、pose、视频，用于训练自动驾驶模型（视频生成 + 3D感知）。

---

## 核心流水线（2步）

```
原始数据 (/mnt/car_road_data_fix)
  ├── road_labels/interpolation_labels/*.json   (路侧标注)
  ├── road/cameras/                             (路侧图像)
  ├── road/lidar/merged_pcd/*.pcd               (路侧点云)
  ├── car/images/                               (车端GT图像)
  └── support_info/*.json                       (标定)

     ↓  intersection_filter (前置, 已跑完)
     ↓  输出: filtered_segments.json (29帧一段)

第1步: segment_pipeline
  → pose.csv + annotations/*.json + direction.json + 投影图像

第2步: transfer_video_maker
  → 训练视频 (.mp4) + caption (.json)
```

---

## 坐标变换链（关键！）

### world → car → lidar（当前实现）

```
文件: segment_pipeline/ego_transform.py

1. car2world: R = euler2rotmat(roll, pitch, yaw),  t = [x, y, z]  (来自标注)
2. world2car: R_world2car = R_car2world.T,  t = -R_world2car @ t_car2world
3. lidar2car: R = I (单位阵),  t = [0, 0, height/2 + 0.25]  (lidar在车顶)
4. car2lidar: R_car2lidar = I,  t = -t_lidar2car = [0, 0, -(height/2+0.25)]
5. world2lidar = R_car2lidar @ R_world2car,  t = R_car2lidar @ t_world2car + t_car2lidar
```

### lidar → camera（各投影模块内部）

```
文件: 各投影目录/undistort_projection_multithread_v2.py

cam2lidar 外参 (R_cam2lidar, t_cam2lidar) 从标定文件加载
R_lidar2cam = R_cam2lidar.T
t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar
points_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam
→ 然后 camera → image (内参 K + 畸变 D, 含鱼眼处理)
```

### 参考来源

坐标变换逻辑参照外部代码 `getRoad2lidar()` 中 `carid is None` 的情况：
- `euler2rotmat(roll, pitch, yaw)`: Rz @ Ry @ Rx 顺序
- lidar 偏移: `[0, 0, height/2 + 0.25]`（虚拟 lidar 安装在 bbox 顶部上方 0.25m）
- R_lidar2car = 单位阵（lidar 与车体坐标系轴向一致）

### ⚠️ 历史坑

- **旧版用 `cv2.Rodrigues([roll, pitch, yaw])`**，这是把欧拉角当作旋转向量（axis-angle），是错误的。已在 commit `195473d` 修复为 `euler2rotmat`。
- **旧版无 lidar 偏移**，bbox 中心直接当 lidar 原点。已修复。

---

## pose.csv 说明

**记录的是车辆在世界坐标系的位姿**（不含 lidar 偏移），供训练框架读取定位信息。

格式: `timestamp, x, y, z, qx, qy, qz, qw, frame_id`

欧拉角→四元数: `scipy.spatial.transform.Rotation.from_euler('xyz', [roll, pitch, yaw])`

---

## 目录结构

```
proj_utils_pro_Roadside_Generation/
├── common_utils.py                  # 公共工具: 场景路径、标定加载、world2lidar变换、时间戳匹配
├── run_interactive.sh               # 旧版交互式投影入口
├── batch_wrapper.py                 # 批量模式配置
│
├── 基本点云投影/                     # 6种投影模块, 每个含:
├── blur投影/                         #   undistort_projection_multithread_v2.py (投影器类)
├── blur稠密化投影/                   #   run_batch_v2.py (旧版批量入口)
├── depth投影/                        #
├── depth稠密化投影/                  # 投影链: world→lidar→camera→image
├── HDMap投影/                        # HDMap 特殊: 输入是标注JSON而非PCD
│
├── intersection_filter/             # 路口筛选 + 轨迹分段
│   ├── intersection_filter.py       # REFERENCE_VEHICLES 定义路口区域
│   └── output/filtered_segments.json
│
├── segment_pipeline/                # 核心流水线
│   ├── segment_pipeline.py          # 主调度 (--interactive)
│   ├── ego_transform.py             # 坐标变换 ★ (euler2rotmat + world→car→lidar)
│   ├── pose_generator.py            # pose.csv 生成
│   ├── annotation_converter.py      # 标注 world→lidar 转换
│   ├── direction_detector.py        # 行驶方向推算 (W2E/E2W/N2S/S2N)
│   ├── projection_runner.py         # 动态加载投影器, 运行 29 帧
│   └── output/{scene}_id{vid}_seg{NN}/
│       ├── pose.csv
│       ├── direction.json
│       ├── annotations/{ts}.json    # ego lidar 坐标系标注
│       └── {proj_type}/{ts}/{subdir}/{cam}.jpg
│
└── transfer_video_maker/            # 视频 + caption 生成
    ├── generate_transfer2_videos.py # 核心: 图像→视频, 含 SCENE_DESCRIPTION
    ├── generate_videos.sh           # 交互式入口
    ├── caption一键修理/update_captions.py
    └── output/
        ├── BlurProjection/
        ├── DepthSparse/
        └── HDMapBbox/
            ├── videos/{camera}/{seg}.mp4
            ├── control_input_{type}/{camera}/{seg}.mp4
            └── captions/{camera}/{seg}.json
```

---

## 变换一致性保证

所有路径最终都收敛到 `ego_transform.py:get_world2lidar_transform()`：

| 使用场景 | 调用入口 | 内部函数 |
|---------|---------|---------|
| 3D标注转换 | `annotation_converter` → `get_world2ego_transform` | → `get_world2lidar_transform` (别名) |
| 6种投影 | `projection_runner` → `get_world2ego_as_rodrigues` | → `get_world2lidar_transform` |
| 标注中物体朝向 | `transform_object_to_ego_frame(obj, R, t)` | 同一个 R_world2lidar |

投影模块内部消费方式:
```
common_utils.transform_points_to_lidar()
  → cv2.Rodrigues(rodrigues_vec) 还原旋转矩阵
  → p_lidar = R @ p_world + t
```

---

## 7 个车端相机

`FN`(前窄), `FW`(前宽), `FL`(前左), `FR`(前右), `RL`(后左), `RR`(后右), `RN`(后窄)

标定文件包含内参 K、畸变系数 D、cam2lidar 外参 (四元数→旋转矩阵)。

cam_id 2/3/4 为鱼眼相机，使用 `cv2.fisheye` 处理。

---

## 6 种投影类型

| 类型 | 输入 | 输出子目录 | 说明 |
|------|------|-----------|------|
| basic | PCD | proj/ | 基本彩色点云投影 |
| blur | PCD | proj/ | 路侧图像着色的模糊投影 |
| blur_dense | PCD | proj/ | blur 的 4 级稠密化版 |
| depth | PCD | depth/ | 深度图投影 |
| depth_dense | PCD | depth/ | depth 的稠密化版 |
| hdmap | annotation JSON | overlay/ | 3D bbox 框线投影 |

---

## 路口参考车辆（REFERENCE_VEHICLES）

当前 4 个方向各 1 辆参考车，定义路口矩形区域边界：

| 场景 | 车辆ID | 方向 | 进入时间戳 | 离开时间戳 |
|------|-------|------|-----------|-----------|
| 002 | 29 | W2E | 1742877436322 | 1742877441799 |
| 003 | 45 | E2W | 1742877823770 | 1742877829337 |
| 014 | 6 | N2S | 1742883999397 | 1742884002379 |
| 015 | 55 | S2N | 1742884382914 | 1742884385763 |

**后续会扩展更多场景**，需要保持 REFERENCE_VEHICLES 配置的灵活性。目前硬编码在 `intersection_filter.py` 和 `segment_pipeline.py` 两处，需统一。

---

## 开发注意事项

1. **euler2rotmat 是 ZYX 顺序** (Rz @ Ry @ Rx)，不要用 cv2.Rodrigues 处理欧拉角
2. **pose.csv 保持世界坐标**，不加 lidar 偏移
3. **标注和投影必须用同一套 world2lidar 变换**，改了 ego_transform 就全局生效
4. **中文目录名**：6个投影目录用中文命名（基本点云投影、blur投影 等），注意路径处理
5. **每 segment 固定 29 帧**，由 intersection_filter 决定
6. **PCD 时间戳匹配容差**: 500ms（projection_runner.py）
7. **方向推算容差**: 5000ms（direction_detector.py）
8. **投影器动态加载**: 通过 `importlib` 加载，脚本名/类名变化会导致加载失败
9. **Transfer2 输出格式**: videos/ + control_input_{type}/ + captions/，按相机分子目录

---

## 最近开发历史摘要

- 修复坐标变换: `cv2.Rodrigues(euler)` → `euler2rotmat` + lidar 高度偏移 (当前)
- README 2步化: 3步流程简化为2步
- caption 模板集成到视频生成中 (SCENE_DESCRIPTION)
- HDMap 投影升级: 实心 3D bbox 渲染
- 方向信息集成到 caption 生成
- 流水线重构: filter → segment → projection → video

---

## Git 分支

开发分支: `claude/explore-dataset-architecture-9qAZk`
