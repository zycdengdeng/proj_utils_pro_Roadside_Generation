# 使用指南

## 一、投影生成（控制帧图像）

```bash
conda activate zihanw
bash run_interactive.sh
```

### 交互流程

**第1步：选项目**
```
1) basic    2) blur    3) blur_dense
4) depth    5) depth_dense    6) hdmap
7) batch（批量串行，一般选这个）
```
- 选 `7` → 再选要跑的项目编号，如 `2 4 6`（空格或逗号分隔）

**第2步：填场景ID**（仅第一个项目会问，后续自动复用）
```
请输入场景ID: 002 004 005
```
- 直接填 `001`~`089` 的编号，支持多个空格分隔

**第3步：world2lidar变换JSON**
```
请选择模式 [auto]:
```
- 直接回车选 `auto`，自动从 `transform_json/{场景ID}/` 找

**第4步：批次模式**
```
请输入批次模式 [all]:
```
- 直接回车 = 全部处理。其他选项：`10`(前10个), `middle_90`, `range_10_50`

**第5步：并行配置**
```
并行进程数 [默认16]:
每帧线程数 [默认7]:
```
- 直接回车用默认值

**第6步：（仅 hdmap）自车ID**
```
请选择模式 [auto]:
```
- 选 `auto`，自动从 `carid.json` 读取

> batch 模式下只有第一个项目需要交互，后续项目自动复用配置。
> 但 hdmap 排在最后时会额外问一次自车ID，选 auto。

### 输出

各子功能文件夹下按场景分类的逐帧图片：
- `基本点云投影/output/{场景}/proj/` — 基础点云投影图
- `blur投影/output/{场景}/proj/` — 路侧相机着色点云
- `depth投影/output/{场景}/proj/` — 深度图
- `HDMap投影/output/{场景}/proj/` — 3D bbox 投影到 2D

---

## 二、Segment Pipeline（轨迹分段 + ego标注）

```bash
python -m segment_pipeline.segment_pipeline --interactive
```

**前置条件**：先跑 `intersection_filter/intersection_filter.py` 生成 `filtered_segments.json`

### 交互流程

**第1步：选场景**
```
1) scene 002: 5 个 segments, 车辆 [29, 31]
2) scene 003: 3 个 segments, 车辆 [45]
0) 全部处理
q) 退出
```

**第2步：（场景有多辆车时）选车辆**
```
1) 车辆 29 (3 个 segments)
2) 车辆 31 (2 个 segments)
0) 全部
```

也可以命令行直接指定：
```bash
python -m segment_pipeline.segment_pipeline --scene 002 003
python -m segment_pipeline.segment_pipeline --scene 002 --vehicle-id 29
```

### 每个 segment 输出 3 个文件

```
segment_pipeline/output/scene{ID}/vehicle{VID}_seg{NN}/
├── pose.csv           # ego位姿 (timestamp, x,y,z, qx,qy,qz,qw)
├── annotations/       # ego坐标系下的3D标注 (每帧一个JSON)
│   ├── 174287743xxxx.json
│   └── ...
└── direction.json     # 行驶方向 (W2E/E2W/N2S/S2N + confidence)
```

### 坐标变换逻辑

```
标注转换:  world → ego (bbox中心 = 坐标原点)
投影:      world → ego → camera
```

bbox 的 (x,y,z,roll,pitch,yaw) 取逆就是 world2ego，不需要任何 LiDAR 偏移。

---

## 三、控制头视频生成

```bash
bash transfer_video_maker/generate_videos.sh
```
- 选 `7`(Batch) → 选控制头类型
- 默认 1280x720, 帧率符合 cosmos post training demo
- 输出在 `transfer_video_maker/output/`，按控制头类型分类（不是场景号）

## 四、Caption 替换

```bash
python transfer_video_maker/generate_transfer2_videos.py
```
- 替换单个头 → 选对应控制头的 caption 选项
- 替换多个头 → 选"替换全部" → 选"自适应匹配"
- 改预设 caption → 直接改 py 脚本里的 txt
