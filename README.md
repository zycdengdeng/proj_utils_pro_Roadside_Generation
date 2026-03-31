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

### 第0步：路口筛选 + 轨迹分段

通常只跑一次，后续步骤读取其输出。

```bash
python intersection_filter/intersection_filter.py --step all
```

- 根据 REFERENCE_VEHICLES 定义路口矩形区域
- 扫描所有场景，筛选穿越路口的车辆轨迹，切分为 29 帧一段
- 输出 `intersection_filter/output/filtered_segments.json`
- 同时输出 BEV 可视化 `bev_intersection_region.png`

查询指定车辆在路口内的帧数和时间戳范围：

```bash
python intersection_filter/query_vehicle_in_region.py
```

### 第1步：生成 pose + 标注 + 投影

```bash
python -m segment_pipeline.segment_pipeline --interactive
```

- 选场景 → 选车辆 → 选投影类型
- 生成 `pose.csv`、`direction.json`、ego 坐标系标注
- 可选同时跑投影（basic/blur/depth/hdmap）

### 第2步：生成训练视频 + caption

```bash
bash transfer_video_maker/generate_videos.sh
```

- 选投影类型（多选）→ 选场景 → 确认 → 批量生成
- 默认投影 blur + depth + hdmap，默认场景 all，直接回车即可
- 输出在 `transfer_video_maker/output/` 按投影类型分目录
- 跳过交互：
  ```bash
  python transfer_video_maker/generate_transfer2_videos.py \
    --segments-dir segment_pipeline/output \
    --project-type depth \
    --output-dir transfer_video_maker/output/DepthSparse
  ```

如需事后批量修改 caption 模板：

```bash
python transfer_video_maker/caption一键修理/update_captions.py
```

caption 模板定义在 `transfer_video_maker/generate_transfer2_videos.py` 的 `SCENE_DESCRIPTION` 中。
