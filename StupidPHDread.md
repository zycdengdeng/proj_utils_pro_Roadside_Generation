# 使用指南

## 完整流程

```
intersection_filter  →  filtered_segments.json（筛选路口区域车辆，切29帧seg）
        ↓
segment_pipeline     →  每个seg: pose + annotation + direction + 投影（只投29帧）
        ↓
transfer_video_maker →  读seg逐帧图片，拼成训练视频 + caption
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
├── direction.json                  # 行驶方向
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

GT 图片每种投影类型各存一份（projector 自带生成）。

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

### 输出结构（不变，兼容下游训练）

```
transfer_video_maker/output/
├── videos/{camera}/004_id45_seg01.mp4              # GT视频
├── control_input_depth/{camera}/004_id45_seg01.mp4 # 控制输入视频
└── captions/{camera}/004_id45_seg01.json           # caption
```

7个相机子目录：`ftheta_camera_front_tele_30fov` 等。

### Legacy 模式（兼容旧版）

```bash
python transfer_video_maker/generate_transfer2_videos.py \
  --project-type depth \
  --project-root /mnt/zihanw/proj_utils_pro \
  --scenes 002 004 \
  --frames-per-seg 21 --num-segs 4 \
  --output-dir transfer_video_maker/output
```
