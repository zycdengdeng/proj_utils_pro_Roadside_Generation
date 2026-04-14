# 车端 LiDAR 投影

把车载 LiDAR 点云投影到车端 7 个相机的 GT 图像上，按"相机系深度 z_cam"做 JET 着色，效果近似经典 KITTI / RoadsideCamera 风格的 LiDAR-on-image 可视化。

## 数据要求

```
{scene_root}/
  ├── car/
  │   ├── pcds/main/{seconds}.{microseconds}.pcd     ← 车端 LiDAR (已在 lidar 系)
  │   └── images/{cam_name}/*_{sec}.{usec}.jpg       ← 车端 GT 图像
  └── ...
support_info/NoEER705_v3/camera/
  ├── camera_01_intrinsics.yaml / extrinsics.yaml    ← FN
  ├── camera_02_*.yaml                               ← FW (鱼眼)
  ├── ...
  └── camera_07_*.yaml                               ← RN
```

外参 yaml 里的 `transform.rotation/translation` 语义是 **`cam2lidar`**。脚本内部会取转置算 `lidar2cam`。

## 变换链

```
p_lidar  ── R_cam2lidar.T, -R_cam2lidar.T·t_cam2lidar ──>  p_cam
p_cam (z>0.1)  ── new_K (去畸变后内参) ──>  (u, v)
颜色 ← JET( clip(z_cam, near, far) )
```

注意：**不需要** `world2lidar` —— 因为 `car/pcds/main/*.pcd` 直接就是车载 lidar 系下的点云。这一点和 `depth投影/`、`基本点云投影/` 等模块不同（那些处理的是路侧 `merged_pcd`，需要先 world→lidar 才能投到车端相机）。

## 用法

```bash
# 1. 默认: 投到全部 7 个相机, 固定深度 [1m, 50m], 输出 ./output/{scene}/{cam}/{ts}.jpg
python project_vehicle_lidar.py --scene 004

# 2. 只投前视, 自适应深度范围 (远近场景都好看), 点稍大些
python project_vehicle_lidar.py \
    --scene 004 \
    --cameras FN FW \
    --auto-range \
    --point-size 3

# 3. 限制帧范围 (调试用)
python project_vehicle_lidar.py --scene 004 --max-frames 5

# 4. 半透明 (隐约能看到底图)
python project_vehicle_lidar.py --scene 004 --alpha 0.7

# 5. 多场景批量
python project_vehicle_lidar.py --scene 004 006 014 --output /data/lidar_overlay
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--scene` | (必填) | 场景 ID, 可多个 (`004 006`) |
| `--cameras` | 全部 7 个 | 相机名: `FN FW FL FR RL RR RN` |
| `--output` | `./output` | 输出根目录 |
| `--near` / `--far` | 1.0 / 50.0 | 固定深度范围 (米), JET 颜色映射区间 |
| `--auto-range` | off | 每帧用 2%~98% 分位自适应深度范围, 覆盖 `--near/--far` |
| `--point-size` | 2 | 投影点边长 (像素) |
| `--alpha` | 1.0 | 点对底图的不透明度, `1.0` 完全覆盖, `<1` 半透明混合 |
| `--start-frame` / `--max-frames` | 0 / 0 | 帧范围控制 |
| `--jpg-quality` | 92 | 输出 JPG 质量 |

## 颜色含义

- **蓝/青 (近)** ← `z_cam ≈ near`
- **绿/黄 (中)**
- **红/橙 (远)** ← `z_cam ≈ far`

如果想换成 TURBO 或 VIRIDIS 等其他配色，把 `colorize_by_depth(...)` 调用里的 `colormap=cv2.COLORMAP_JET` 改成 `cv2.COLORMAP_TURBO` / `cv2.COLORMAP_VIRIDIS` 等即可。

## 性能

- 每帧 7 个相机用 `ThreadPoolExecutor` 并行处理 (cv2 操作释放 GIL, 多线程有效)
- 4K 分辨率 + ~10 万点 / 帧, 单帧约 0.5~1.5 秒 (取决于点云密度和 CPU)
- 如需更快, 可在 `process_one_camera` 外层再套 `multiprocessing` (按 PCD 分进程)

## 已知限制

1. **GT 时间戳容差 100 ms** —— `find_gt_image` 的 `tol_us`。如果车端相机和 LiDAR 时间戳偏差较大, 改大 `GT_MATCH_TOL_US`。
2. **遮挡处理** —— 同一像素被多个深度的点击中时, 这里**只用绘制顺序解决**（先画远的、后画近的），不做严格 z-buffer。如果点云特别密 / 想要更干净的结果，可以加一个 `np.minimum.at(depth_buffer, ...)` 风格的 z-buffer 写入。
3. **去畸变后图像视角变化** —— 鱼眼相机去畸变后会丢掉边缘部分。如果需要保留全视场, 在 `undistort_image` 里把 `balance=0.0` 改成 `1.0`。
