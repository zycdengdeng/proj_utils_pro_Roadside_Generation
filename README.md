## 完整流程

```
intersection_filter (第0步, 前置)
  → filtered_segments.json (29帧一段)

segment_pipeline (第1步)
  → pose.csv + annotations/ + direction.json + 投影图像

transfer_video_maker (第2步)
  → 训练视频 (.mp4) + caption (.json)
```

```
conda activate zihanw
```

### 第0步：路口筛选 + 轨迹分段（前置，通常只跑一次）

```bash
python intersection_filter/intersection_filter.py --step all
```

- 根据 4 个方向的参考车辆（REFERENCE_VEHICLES）定义路口矩形区域
- 扫描所有场景，找出穿越路口的车辆轨迹
- 切分为 29 帧一段的 segment
- 输出 `intersection_filter/output/filtered_segments.json`（后续步骤的输入）
- 同时输出 BEV 可视化 `bev_intersection_region.png`

查询指定车辆在路口内的帧数和时间戳范围：

```bash
python intersection_filter/query_vehicle_in_region.py
```

### 第1步：Segment Pipeline（生成 pose + 朝向 + 标注）

```bash
python -m segment_pipeline.segment_pipeline --interactive
```

- 选场景 → 选车辆 → 选投影类型
- 自动生成 `pose.csv`、`direction.json`（朝向）、ego坐标系标注
- 可选同时跑投影（basic/blur/depth/hdmap）

### 第2步：生成训练视频 + caption

```bash
bash transfer_video_maker/generate_videos.sh
```

- 选投影类型（多选）→ 选场景 → 确认 → 批量生成
- 默认投影：blur + depth + hdmap，默认场景：all
- 直接全部回车就开始跑，输出在 `transfer_video_maker/output/` 按投影类型分目录
- 也可以直接用 python 命令（跳过交互）：
  ```bash
  python transfer_video_maker/generate_transfer2_videos.py \
    --segments-dir segment_pipeline/output \
    --project-type depth \
    --output-dir transfer_video_maker/output/DepthSparse
  ```

### （可选）caption 批量修改

第2步已经生成完整 caption（含朝向 + 场景描述）。如需批量改模板：

```bash
python transfer_video_maker/caption一键修理/update_captions.py
```

---

## 单独跑投影（不走 segment_pipeline）

```bash
bash run_interactive.sh
```

- 选1-6单个投影，或选7批量（如输入 `2 4 6` 同时跑 blur + depth + hdmap）
- 输入场景号（如 `004`）→ 其余参数直接回车用默认
- hdmap 的自车ID选 `auto`

## 要改 caption 模板？

改 `transfer_video_maker/generate_transfer2_videos.py` 里的 `SCENE_DESCRIPTION`（生成时直接写入）。

如需事后批量替换已有 caption，用 `transfer_video_maker/caption一键修理/update_captions.py`。
