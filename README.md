## 完整流程（3步）

```
conda activate zihanw
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
python transfer_video_maker/generate_transfer2_videos.py \
  --segments-dir segment_pipeline/output \
  --project-type depth \
  --output-dir transfer_video_maker/output/DepthSparse \
  --fps 10
```

- `--segments-dir`：第1步的输出目录，每个 seg 目录 = 一个视频（29帧，不需要手动指定帧数或seg数量）
- `--project-type`：选哪种投影做控制输入（depth / depth_dense / blur / blur_dense / basic / hdmap）
- 朝向自动从每个 seg 的 `direction.json` 读取，写入 caption
- 输出在 `--output-dir`，按控制头类型分目录

### 第3步：caption 批量替换（含朝向）

```bash
python transfer_video_maker/caption一键修理/update_captions.py
```

- 自动从第1步生成的 `direction.json` 读取朝向（west to east / east to west / ...）
- 选数据集 → 选模板（或选自动匹配）→ 预览 → 确认
- caption 格式：`"{视角}. The ego vehicle is traveling from {朝向}. {场景描述}"`

**朝向怎么对应上的**：caption文件名 `004_id45_seg01.json` → 自动找 `segment_pipeline/output/scene004/vehicle45_seg01/direction.json`

---

## 单独跑投影（不走 segment_pipeline）

```bash
bash run_interactive.sh
```

- 选1-6单个投影，或选7批量（如输入 `2 4 6` 同时跑 blur + depth + hdmap）
- 输入场景号（如 `004`）→ 其余参数直接回车用默认
- hdmap 的自车ID选 `auto`

## 要改 caption 模板？

直接改 `transfer_video_maker/caption一键修理/update_captions.py` 里的 `SCENE_DESCRIPTION` 和 `PRESET_TEMPLATES`。
