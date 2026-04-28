# Case C - 4-Vantage Counterfactual Pose Query

在不修改投影模块的前提下，让 segment_pipeline 接受人为给定的虚拟观察车 pose，
为同一时刻 t* 生成 N/E/S/W 4 个反事实视角的 7-相机 × 29 帧 控制条件
（blur / depth / hdmap），最终训练 4 段假想视频。

## 流程一图流

```
008 路侧标注 (含真采集车 ego)
        │
        ▼
[1] find_ego_pose.py
        │  →  case_C/output/ego_pose_t_star.json
        ▼
[2] build_virtual_observers.py     (R = 12m, 朝向 ego)
        │  →  case_C/output/virtual_observers.json
        ▼
[3] build_pseudo_segments.py
        │  →  intersection_filter/output/filtered_segments_caseC.json
        ▼
[4] python -m segment_pipeline.segment_pipeline \
        --segments-file intersection_filter/output/filtered_segments_caseC.json \
        --projections blur depth hdmap
        │  →  segment_pipeline/output/008_idvirt{N,E,S,W}_seg01/
        ▼
[5] bash transfer_video_maker/generate_videos.sh
        │  →  transfer_video_maker/output/{Blur,Depth,HDMap}*/.../008_idvirt*.mp4
        ▼
[6] OmniR2A 推理 → 抽帧 → 拼 5-panel compass figure
```

## 命令逐条

```bash
cd /home/user/proj_utils_pro_Roadside_Generation

# Step 1: 选 t* 并取 ego 在世界系的 pose
#   --timestamp 不传时自动取 008 标注时间戳的中间一帧
python case_C/find_ego_pose.py --scene 008
# (如需指定: python case_C/find_ego_pose.py --scene 008 --ego-id 12 --timestamp 1742XXXXXXXXX)

# Step 2: 在 ego 周围正交 4 方向放观察车
python case_C/build_virtual_observers.py --radius 12

# Step 3: 把 4 观察车包装成 segments
python case_C/build_pseudo_segments.py

# Step 4: 跑流水线生成 7 相机 × 29 帧 投影 (3 类型)
python -m segment_pipeline.segment_pipeline \
    --segments-file intersection_filter/output/filtered_segments_caseC.json \
    --projections blur depth hdmap

# 输出: segment_pipeline/output/008_idvirtN_seg01/{blur,depth,hdmap}/<ts>/<subdir>/<cam>.jpg
#       共 4 个 seg 目录 (virtN/E/S/W)

# Step 5: 图 → 视频 + caption (走原 transfer_video_maker)
bash transfer_video_maker/generate_videos.sh
# 选 blur+depth+hdmap，选 008_idvirt* 这 4 个 seg 全跑
```

## Pipeline 改动点（最小侵入）

| 文件 | 改动 |
|------|------|
| `segment_pipeline/ego_transform.py` | 新增 `get_world2lidar_transform_from_pose` 与 `get_world2lidar_rodrigues_from_pose`，复用同一套 world→car→lidar 数学 |
| `segment_pipeline/pose_generator.py` | `generate_pose_csv` 增 `virtual_pose=` 参数，绕过 ego 查找 |
| `segment_pipeline/annotation_converter.py` | `convert_segment_annotations` 增 `virtual_pose=`，虚拟模式不排除真 ego（因为 ego 现在是观察者，不再是采集车） |
| `segment_pipeline/projection_runner.py` | `build_transforms_from_annotations` / `run_projection_for_segment` 接受 `virtual_pose`；29 帧用同一个 world2lidar |
| `segment_pipeline/segment_pipeline.py` | `process_single_segment` 透传 `segment.get('virtual_pose')` |

`vehicle_id` 在虚拟 segment 里是字符串 (`virtN`, `virtE`, ...)；HDMap 投影做 `obj['id'] == ego_vehicle_id` 比较时 str≠int 自然不匹配，所以会**保留所有真实物体（含真采集车）**——这正是 Case C 想要的：每个虚拟观察车的视图里都能看到真 ego。

## 自检清单

跑完 Step 4 后核对：
- [ ] `segment_pipeline/output/` 下出现 4 个 `008_idvirt{N,E,S,W}_seg01/` 目录
- [ ] 每个目录含 `blur/`, `depth/`, `hdmap/` 子目录，每个子目录有 29 个 timestamp 子文件夹（虽然时间戳是同一个，但生成 29 份冗余文件，符合 transfer_video_maker 要求）
- [ ] 每个 `<ts>/<subdir>/` 下有 7 个相机的 JPG（FN/FW/FL/FR/RL/RR/RN）
- [ ] FW (前广角) 视图里 ego 居中可见，4 个方向看到的 ego 颜色 / 车型一致
- [ ] HDMap 视图里 ego 的 3D bbox 闭合

跑完 Step 5、6 后：
- [ ] `transfer_video_maker/output/.../008_idvirt*.mp4` 4 套都存在
- [ ] 推理产出 4 套 generated.mp4，抽 frame_014 拼图

## 注意

- 4 个观察车的 (x, y) 必须在 R2A 红框 `x∈[-95.1,-30.7], y∈[-30.6,9.8]` 内，
  `build_virtual_observers.py` 会标 ⚠️ 出框警告。若 ego 位置导致出框，需调小 `--radius` 或换 t*。
- 静止观察车的 `direction.json` 会落到 `unknown`（因为位移≈0），这是符合预期的。
- 29 帧标注是同一帧重复 → annotations/ 下的 29 个 JSON 内容一致；这只是为了让 transfer_video_maker 拼 29 帧视频。
