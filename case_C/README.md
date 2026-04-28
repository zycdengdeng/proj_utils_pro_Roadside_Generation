# Case C - 4-Vantage Counterfactual Pose Query

让 4 辆**虚拟观察车**以 ego (真采集车) 为圆心、跟随 ego 一起平移（与 ego
保持相对静止），在 N/E/S/W 4 个偏移位置各跑一次 segment_pipeline，得到 4
套独立的 7-相机 × 29-帧 控制条件 (blur / depth / hdmap)，最终训练 4 段
"反事实"假想视频。

> **关键点**：每辆观察车跑自己的一段 29 帧视频；4 段视频分开喂给 R2A
> 推理。每帧 `observer_world(t) = ego_world(t) + 恒定 offset`，所以观察
> 车与 ego 同步运动，"行驶"看起来正确。观察车朝向恒定指向 ego。

## 流程

```
008 路侧标注 (含真采集车 ego, 整段 163 帧轨迹)
        │
        ▼
[1] find_ego_pose.py
        │  →  case_C/output/ego_pose_t_star.json   (单帧 t*)
        │  →  case_C/output/ego_trajectory.json    (29 帧 ego 世界 pose)
        ▼
[2] build_virtual_observers.py     (R = 12m, 朝向 ego)
        │  →  case_C/output/virtual_observers.json
        ▼
[3] visualize_bev.py    (可选, 输出 PNG 看 BEV 布局)
        │  →  case_C/output/bev_observers.png
        ▼
[4] build_pseudo_segments.py
        │  → intersection_filter/output/filtered_segments_caseC.json
        │   (4 段 × 29 帧, 每帧含 virtual_poses[i] = ego(t_i) + offset)
        ▼
[5] python -m segment_pipeline.segment_pipeline \
        --segments-file intersection_filter/output/filtered_segments_caseC.json \
        --projections blur depth hdmap
        │  →  segment_pipeline/output/008_idvirt{N,E,S,W}_seg01/
        ▼
[6] bash transfer_video_maker/generate_videos.sh
        │  →  transfer_video_maker/output/.../008_idvirt*.mp4
        ▼
[7] OmniR2A 推理 → 抽帧 → 拼 5-panel compass figure
```

## 命令逐条

```bash
cd /home/user/proj_utils_pro_Roadside_Generation     # 你的实际路径

# Step 1: 选 t* + 取 ego 在世界系的单帧 pose + 29 帧轨迹
python case_C/find_ego_pose.py --scene 008
#   常用变体:
#     --fit-radius 12        自动挑 ego 离 R2A 红框四壁都 >= 12m 的 t*
#     --timestamp <ms>       手动指定 t*
#     --ego-id <int>         覆盖 carid.json
#     --num-frames 29        轨迹帧数

# Step 2: 在 ego 周围放 4 个观察车 (R = 12m, 朝向 ego)
python case_C/build_virtual_observers.py --radius 12

# Step 3: BEV 可视化 (输出 PNG)
python case_C/visualize_bev.py
# 含点云背景:
python case_C/visualize_bev.py --with-pcd

# Step 4: 把 4 观察车 × 29 帧 ego 轨迹 合成伪 segments
python case_C/build_pseudo_segments.py

# Step 5: 跑流水线生成 7 相机 × 29 帧 投影 (3 类型)
python -m segment_pipeline.segment_pipeline \
    --segments-file intersection_filter/output/filtered_segments_caseC.json \
    --projections blur depth hdmap

#   输出: segment_pipeline/output/008_idvirt{N,E,S,W}_seg01/{blur,depth,hdmap}/<ts>/<sub>/<cam>.jpg

# Step 6: 图 → 视频 + caption
bash transfer_video_maker/generate_videos.sh
# 选 blur+depth+hdmap，选 008_idvirt* 这 4 个 seg 全跑
```

## 数据安全

全程**只读** `/mnt/car_road_data_fix`。所有写入都落在仓库内：

- `case_C/output/` — ego pose / 轨迹 / observer 配置 / BEV 图
- `intersection_filter/output/filtered_segments_caseC.json` — pseudo segments
- `segment_pipeline/output/008_idvirt*/` — 投影 JPG
- `transfer_video_maker/output/...` — 视频 / caption

## Pipeline 改动点（最小侵入）

| 文件 | 改动 |
|------|------|
| `segment_pipeline/ego_transform.py` | 新增 `get_world2lidar_transform_from_pose` / `get_world2lidar_rodrigues_from_pose`，绕过标注查找 |
| `segment_pipeline/pose_generator.py` | `generate_pose_csv` 接 `virtual_pose=`(单帧广播) 或 `virtual_poses=`(每帧列表) |
| `segment_pipeline/annotation_converter.py` | `convert_segment_annotations` 同上；虚拟模式不排除真 ego |
| `segment_pipeline/projection_runner.py` | `build_transforms_from_annotations` 同上；每帧用各自的 world2lidar |
| `segment_pipeline/segment_pipeline.py` | `process_single_segment` 透传 `virtual_pose` / `virtual_poses` |

`vehicle_id` 在虚拟 segment 里是字符串 (`virtN/E/S/W`)。HDMap 投影做
`obj['id'] == ego_vehicle_id` 时 `str ≠ int` 永不匹配 → **保留所有真实
物体（含真采集车 ego）**，每个观察车的视图里都能看到真 ego。

## Segment JSON 格式（virtual_poses 模式）

```jsonc
{
  "scene": "008",
  "vehicle_id": "virtN",
  "segment_index": 0,
  "n_frames": 29,
  "timestamps":  [t_0, t_1, ..., t_28],          // 来自 ego 真实标注帧
  "label_files": ["…/t_0.json", …],              // 同上, 用于 hdmap 投影读真物体
  "virtual_poses": [                              // 每帧一个观察车 pose
    {"x": ego.x_0 + offset.x, "y": ego.y_0 + offset.y, "z": ego.z_0 + offset.z,
     "yaw": const, "roll": 0, "pitch": 0, "vehicle_height": 1.6},
    …
  ],
  "case_C": {
    "observer_name": "N",
    "ego_reference_id": 82,
    "t_star": 1742879650843,
    "offset_world": [0.0, 12.0, 0.0],
    "mode": "follow_ego_constant_offset"
  }
}
```

向后兼容：若 segment 只含 `virtual_pose`(单数 dict)，pipeline 自动广
播到所有帧（即原"4 静止观察车"模式）。

## 自检清单

跑完 Step 4 (build_pseudo_segments) 后：
- [ ] `intersection_filter/output/filtered_segments_caseC.json` 有 4 段
- [ ] 每段 `virtual_poses` 是长度 29 的列表，不是 dict

跑完 Step 5 (segment_pipeline) 后：
- [ ] `segment_pipeline/output/` 下出现 `008_idvirt{N,E,S,W}_seg01/` 4 个目录
- [ ] 每个目录含 `blur/`, `depth/`, `hdmap/` 子目录, 各 29 个 timestamp 子文件夹
- [ ] 每个 `<ts>/<sub>/` 下有 7 张相机 JPG (FN/FW/FL/FR/RL/RR/RN)
- [ ] `pose.csv` 里 29 行 (x, y, z) 与 ego 轨迹平行偏移, 不是恒定值
- [ ] FW 视图里 ego 居中可见, 4 个观察车看到的 ego 车型/颜色/yaw 一致
- [ ] HDMap 视图里 ego 的 3D bbox 闭合

跑完 Step 6 后：
- [ ] `transfer_video_maker/output/.../008_idvirt*.mp4` 4 套都存在
- [ ] 视频里 ego 不是静止的 (镜头前 ego 真在动 → "行驶正确")

## 注意

- 4 个观察车 (x, y) 必须在 R2A 红框 `x∈[-95.1,-30.7], y∈[-30.6,9.8]` 内，
  `build_virtual_observers.py` / `visualize_bev.py` 会标 ⚠️ 出框警告。
  若 ego 位置导致出框，用 `find_ego_pose.py --fit-radius 12` 自动挑帧。
- 跟随观察车的 `direction.json` 沿用 ego 的位移方向（W2E/E2W/N2S/S2N），
  与真采集车一致。
- 观察车朝向恒定指向 ego；因 offset 在世界系恒定，相对几何永不漂移。
