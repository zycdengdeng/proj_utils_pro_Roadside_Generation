# 使用指南

## 完整流程

```
intersection_filter  →  filtered_segments.json（筛选路口区域车辆，切29帧seg）
        ↓
segment_pipeline     →  每个seg: pose + annotation + direction + 投影（只投29帧）
        ↓
transfer_video_maker →  读seg逐帧图片，拼成训练视频 + caption
        ↓
update_captions.py   →  （可选）批量替换 caption 文本
```

---

## 一、路口筛选

```bash
cd intersection_filter
python intersection_filter.py --step all
```

输出 `intersection_filter/output/filtered_segments.json`，每个 segment 包含 29 帧的时间戳和标注文件路径。

---

## 二、Segment Pipeline（核心，一站式）

```bash
python -m segment_pipeline.segment_pipeline --interactive
```

### 交互流程

**第1步：选场景/车辆**
```
1) scene 002: 5 个 segments, 车辆 [29, 31]
2) scene 003: 3 个 segments, 车辆 [45]
0) 全部处理
```

**第2步：选投影类型**
```
1) basic        control_subdir=proj
2) blur         control_subdir=proj
3) blur_dense   control_subdir=proj
4) depth        control_subdir=depth
5) depth_dense  control_subdir=depth
6) hdmap        control_subdir=overlay
0) 全部
回车) 跳过投影
```

命令行模式：
```bash
# 只生成 pose/annotation/direction，不投影
python -m segment_pipeline.segment_pipeline --scene 002 --no-projection

# 指定投影类型
python -m segment_pipeline.segment_pipeline --scene 002 --projections basic depth hdmap

# 指定车辆
python -m segment_pipeline.segment_pipeline --scene 004 --vehicle-id 45 --projections depth
```

### 输出结构

命名格式：`{scene}_id{vehicle_id}_seg{NN}`（如 `004_id45_seg01`）

```
segment_pipeline/output/004_id45_seg01/
├── pose.csv                        # ego位姿
├── direction.json                  # 行驶方向 (W2E/E2W/N2S/S2N)
├── annotations/{ts}.json           # ego坐标系3D标注
├── basic/{ts}/                     # 基础点云投影
│   ├── proj/{cam}.jpg
│   └── gt/{cam}.jpg
├── depth/{ts}/                     # 深度投影
│   ├── depth/{cam}.jpg
│   └── gt/{cam}.jpg
└── hdmap/{ts}/                     # HDMap投影
    ├── overlay/{cam}.jpg
    └── gt/{cam}.jpg
```

### 坐标变换

```
world → ego (bbox中心取逆 = LiDAR位置) → camera
```
不需要额外 transform_json，直接从标注 bbox 位姿计算。

---

## 三、生成训练视频

```bash
python transfer_video_maker/generate_transfer2_videos.py \
  --segments-dir segment_pipeline/output \
  --project-type depth \
  --output-dir transfer_video_maker/output \
  --fps 10
```

- `--segments-dir`：segment_pipeline 的输出目录
- `--project-type`：用哪种投影做控制输入（depth/basic/blur/hdmap 等）
- 视频名 = seg 目录名（如 `004_id45_seg01.mp4`）
- 每个 seg 目录的所有帧 = 一个视频（29帧）

### Caption 生成逻辑

自动从每个 seg 的 `direction.json` 读取朝向，生成 caption：
```
"Front telephoto view. The ego vehicle is traveling from west to east."
```

caption JSON 字段：`segment_name`, `camera`, `direction`, `caption`

可用 `--caption-template` 自定义模板，支持变量：`{view_prefix}`, `{direction}`, `{camera}`

### 输出结构（兼容下游训练）

```
transfer_video_maker/output/
├── videos/{camera}/004_id45_seg01.mp4              # GT视频
├── control_input_depth/{camera}/004_id45_seg01.mp4 # 控制输入视频
└── captions/{camera}/004_id45_seg01.json           # caption
```

7个相机子目录：`ftheta_camera_front_tele_30fov` 等。

---

## 四、Caption 批量替换（可选）

生成视频后如果需要修改 caption 文本，用 `update_captions.py`。

```bash
python transfer_video_maker/caption一键修理/update_captions.py
```

### 交互流程

**第1步：选数据集**
```
1) DepthSparse (28 个caption文件)
2) HDMapBbox (20 个caption文件)
3) 全部数据集
0) 退出
```

**第2步：选模板**
- 选 `0`（自动匹配）→ 每个数据集自动用对应的预设模板
- 选 `1`（unified）→ 统一模板：`"{view_prefix}. The ego vehicle is traveling from {direction}. [场景描述]"`
- 选最后一个 → 自定义模板

**第3步：预览** → 确认 `y` 执行

### 模板变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `{camera}` | Transfer2 相机名 | `ftheta_camera_front_tele_30fov` |
| `{view_prefix}` | 相机视角描述 | `Front telephoto view` |
| `{direction}` | 行驶朝向 | `west to east` |
| `{scene}` | 场景ID | `004` |
| `{seg}` | segment ID | `seg01` |

朝向来源：文件名里的 scene_id → `SCENE_DIRECTION` 硬编码映射（覆盖 001~089 所有场景）。

### 命令行模式

```bash
# 预览（不修改）
python transfer_video_maker/caption一键修理/update_captions.py \
  --dataset DepthSparse --preset unified --dry-run

# 执行
python transfer_video_maker/caption一键修理/update_captions.py \
  --dataset DepthSparse --preset unified

# 自定义模板
python transfer_video_maker/caption一键修理/update_captions.py \
  --template "{view_prefix}. Driving from {direction}."
```
