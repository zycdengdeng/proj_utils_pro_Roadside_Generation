# 论文图：单 clip 交通流复杂度

`traffic_complexity_figure.py` —— 为数据集论文绘制"某一个 clip 的交通流复杂程度"图。

## 概念

一个 **clip = 一整段路侧录制目录**，例如：

```
/mnt/car_road_data_TianJin/002_car0325_road0327_t2/
└── road_labels/interpolation_labels/*.json   # 文件名是毫秒时间戳
```

脚本读这个目录下**全时序**的所有标注帧，统计**通过该路口**的车辆，量化交通流复杂度。

> ⚠️ 与旧流水线的区别：`intersection_filter.py` 里每段固定 **29 帧**，那是"选某辆车做自车、
> 它经过路口时的那几帧"用于出 video；算复杂度**不需要**这个限制，这里用整段录制的全部帧。
> （`--vid 29` 那种是**车辆 id**，与 29 帧无关，别混淆。）

## 图的四个部分

| 子图 | 内容 |
|------|------|
| (a) BEV traffic flow | 俯视图叠加所有"通过路口"的车辆轨迹，**按行驶方向(航向)着色**；终点箭头表方向；黑色 ✕ 标注路口内轨迹交叉/冲突点；红色虚线框为路口区域；底图为 LiDAR 点云（或示意路口）。 |
| (b) Complexity metrics | **通过路口车辆数**、**吞吐量 (veh/min)**、峰值/平均同时在路口数、方向熵、转向比例、冲突点数、中位车速，以及 0~100 综合复杂度评分（High/Medium/Low）。 |
| (c) Heading rose | 通过车辆净行驶方向的极坐标玫瑰图（12 扇区）。 |
| (d) Temporal density | 每帧"路口内车辆数"（左轴）与"累计通过数"（右轴）随时间(秒)变化。 |

## 综合复杂度评分

启发式加权（0~100），权重在 `compute_metrics()` 内可调：

```
score = 100 * ( 0.25 * 平均同时在路口数/5
              + 0.20 * 方向熵(归一化)
              + 0.25 * 路口内冲突点数/通过车辆数
              + 0.10 * 转向车辆比例
              + 0.20 * 吞吐量/30(veh/min) )
```

`>=65 High，>=40 Medium，否则 Low`。

## 用法

依赖：`matplotlib numpy scipy`（点云底图可选 `open3d`）。

真实数据：

```bash
# 直接给 clip 目录
python traffic_complexity_figure.py --clip-dir /mnt/car_road_data_TianJin/002_car0325_road0327_t2

# 或给数据集根 + clip 名/前缀
python traffic_complexity_figure.py --dataset-root /mnt/car_road_data_TianJin --clip 002
```

合成演示（无需真实数据，预览版式）：

```bash
python traffic_complexity_figure.py --demo
```

## 批量扫描：找出最复杂的 clip（论文展示用）

并行算完数据集里所有 clip 的复杂度，排名，并自动渲染最复杂那个的论文图：

```bash
python traffic_complexity_figure.py --scan \
  --dataset-root /mnt/car_road_data_TianJin \
  --reference-region \        # 同一路口建议统一区域，保证跨 clip 可比
  --workers 64 \
  --plot-top 1                # 自动渲染 top-1 最复杂 clip
```

- **并行**：CPU 多进程（`ProcessPoolExecutor`），`--workers` 默认 `min(64, CPU核数)`。
  此任务是几何/统计计算，CPU 多进程即最优，无需 GPU。
- **区域统一**：若 89 个 clip 是同一路口，强烈建议加 `--reference-region`（或 `--region`），
  让所有 clip 用同一路口框，评分才可比；否则各 clip 用自身中位数估区域，可比性稍弱。
- **输出**：
  - 终端打印排名表（按评分降序）；
  - `paper_figures/output/complexity_ranking.csv`（**全量**排名，含各项指标，可自己再排序/作图）；
  - `--plot-top N`：渲染前 N 个最复杂 clip 的论文图（默认 1，`0` 则只排名不画）。
- `--pattern`：限定扫描范围，如 `--pattern '0??_*'`。`--top`：终端显示条数（默认 15）。

典型流程：先 `--scan` 拿到排名 → 看 CSV/终端选定要展示的 clip → 若想换一个，单独对它跑
`--clip-dir <那个clip> --reference-region --dataset-root ...` 出最终图。

常用开关：

| 参数 | 说明 |
|------|------|
| `--ts-start / --ts-end` | 只统计某个时间窗（毫秒），默认整段录制 |
| `--region XMIN XMAX YMIN YMAX` | 手动指定路口矩形 |
| `--region-json PATH` | 读路口区域 json（默认找 `intersection_filter/output/intersection_region.json`） |
| `--region-half H` | 自动估计路口区域时的半边长（米，默认 50），以全部车辆位置中位数为中心 |
| `--labels-subdir SUB` | 覆盖标注子目录（默认 `road_labels/interpolation_labels`，找不到会自动搜索） |
| `--no-pcd` | 不加载点云底图 |

**路口区域优先级**：`--region` > `--region-json` > `intersection_filter` 输出 > 自动估计（中位数中心 ±`--region-half`）。
建议优先用 `--region` 或 `--region-json` 给定真实路口范围，使"通过路口计数"准确。

输出：`paper_figures/output/traffic_complexity_<clip名>.png`（300 dpi）+ 同名 `.pdf`（矢量，投稿用）。
