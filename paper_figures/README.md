# 论文图：单 clip 交通流复杂度

`traffic_complexity_figure.py` —— 为数据集论文绘制"某一个 clip 的交通流复杂程度"图。

思路沿用 `intersection_filter.py`：在路口矩形区域内、在该 clip 的时间窗口
`[ts_start, ts_end]` 内，重建**所有**车辆（不仅是参考车）的 BEV 轨迹并量化复杂度。

## 图的四个部分

| 子图 | 内容 |
|------|------|
| (a) BEV traffic flow | 俯视图叠加 clip 内所有车辆轨迹，按车辆着色；起点圆点、终点箭头表方向；黑色 ✕ 标注轨迹交叉/冲突点；红色虚线框为路口区域；底图为 LiDAR 点云（或示意路口）。 |
| (b) Complexity metrics | 在场车辆数、峰值/平均同时在场数、方向熵、转向比例、冲突点数、平均车速，以及 0~100 综合复杂度评分（High/Medium/Low）。 |
| (c) Heading distribution | 车辆净行驶方向的极坐标玫瑰图（8 扇区）。 |
| (d) Temporal density | 每帧"在场车辆数"与"交互对数"随帧变化曲线。 |

## 综合复杂度评分

启发式加权（0~100）：

```
score = 100 * ( 0.30 * 平均同时在场数/8
              + 0.25 * 方向熵(归一化)
              + 0.25 * 冲突点数/车辆数
              + 0.15 * 转向车辆比例
              + 0.05 * 车辆数/15 )
```

`>=70 High，>=45 Medium，否则 Low`。权重可在 `compute_metrics()` 内调整。

## 用法

依赖：`matplotlib numpy scipy`（点云底图可选 `open3d`）。

真实数据（需挂载 `/mnt/car_road_data_fix`，并已跑过 `intersection_filter.py`）：

```bash
# 从 filtered_segments.json 取第 0 个片段
python traffic_complexity_figure.py --from-segments 0

# 显式指定 clip
python traffic_complexity_figure.py --scene 002 --vid 29 --seg 0

# 给定场景 + 时间窗口
python traffic_complexity_figure.py --scene 002 --ts-start 1742877436322 --ts-end 1742877441799
```

合成演示（无需真实数据，用于预览版式）：

```bash
python traffic_complexity_figure.py --demo
```

常用开关：`--no-region-clip`（统计窗口内所有车辆，不限路口框）、`--no-pcd`（不加载点云底图）。

输出：`paper_figures/output/traffic_complexity_<tag>.png`（300 dpi）和同名 `.pdf`（矢量，投稿用）。
