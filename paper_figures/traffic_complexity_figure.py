#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic-flow complexity figure for one clip (publication-quality).

为数据集论文绘制"某一个 clip 的交通流复杂程度"图。

概念（重要）
-----------
一个 clip = 一整段路侧录制目录，例如::

    /mnt/car_road_data_TianJin/002_car0325_road0327_t2/

其路侧标注在 ``road_labels/interpolation_labels/*.json``（文件名是毫秒时间戳）。
我们用**全时序**的所有标注帧，统计**通过该路口**的车辆，量化交通流复杂度。

  ⚠ 注意：旧流水线里每段固定 "29 帧"，那是为某辆自车选出它过路口的那几帧用来
  出 video 的；算复杂度**不需要**这个限制 —— 这里用整段录制的全部帧。
  （`--vid 29` 那种是"车辆 id"，与帧数无关，别混淆。）

图由四部分组成
-------------
  (a) BEV traffic flow   俯视图叠加所有"通过路口"的车辆轨迹，按行驶方向(航向)着色，
                         终点箭头表方向；黑色 ✕ 标注路口内轨迹交叉/冲突点；
                         红色虚线框为路口区域；底图为 LiDAR 点云（或示意路口）。
  (b) Complexity metrics 通过路口车辆数、吞吐量(veh/min)、峰值/平均同时在路口数、
                         方向熵、转向比例、冲突点数、平均车速，及 0~100 综合复杂度评分。
  (c) Heading rose       通过车辆净行驶方向的极坐标玫瑰图。
  (d) Temporal density   每帧"路口内车辆数"与"累计通过数"随时间变化。

用法
----
真实数据::

    python traffic_complexity_figure.py --clip-dir /mnt/car_road_data_TianJin/002_car0325_road0327_t2
    # 或：--dataset-root /mnt/car_road_data_TianJin --clip 002
    # 限定时间窗：--ts-start 1742877424148 --ts-end 1742877460000
    # 自定义路口区域：--region xmin xmax ymin ymax （否则自动估计或读 intersection_filter）

合成演示（无需真实数据）::

    python traffic_complexity_figure.py --demo

输出：paper_figures/output/traffic_complexity_<tag>.png（300 dpi）+ .pdf（矢量）
"""

import os
import sys
import json
import glob
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from common_utils import DATASET_ROOT as _DEFAULT_ROOT  # noqa: E402
except Exception:
    _DEFAULT_ROOT = "/mnt/car_road_data_fix"

VEHICLE_LABELS = {"Car", "Suv", "Truck", "Bus", "Van"}
INTERSECTION_FILTER_DIR = REPO_ROOT / "intersection_filter" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 路口内"同时在场"判定的距离阈值（用于交互对计数）
INTERACT_DIST = 15.0

# 图中红色虚线框的名称（数据生成区域，名字未定，改这一行即可）
REGION_LABEL = "Generation region"
# 图标题
FIGURE_TITLE = "Traffic flow complexity of THICV-R2V"


# ==================================================================
# 数据读取
# ==================================================================
def find_label_dir(clip_dir, override=None):
    """在 clip 目录下定位路侧标注目录（含时间戳 *.json）。"""
    clip_dir = Path(clip_dir)
    if override:
        cand = clip_dir / override if not os.path.isabs(override) else Path(override)
        if cand.is_dir():
            return str(cand)
    # 默认与仓库 common_utils 一致
    default = clip_dir / "road_labels" / "interpolation_labels"
    if default.is_dir():
        return str(default)
    # 兜底：递归查找含数字时间戳 json 的目录
    for sub in clip_dir.rglob("interpolation_labels"):
        if sub.is_dir():
            return str(sub)
    for sub in clip_dir.rglob("*.json"):
        if sub.stem.isdigit():
            return str(sub.parent)
    raise FileNotFoundError(
        f"在 {clip_dir} 下找不到路侧标注目录（期望 road_labels/interpolation_labels/*.json）"
    )


def _heading_of(obj, prev_xy=None):
    """目标航向（弧度，x 向右 y 向上）。优先 yaw，其次速度，最后相邻位移。"""
    if obj.get("yaw") is not None:
        return float(obj["yaw"])
    vx, vy = obj.get("vx"), obj.get("vy")
    if vx is not None and vy is not None and (abs(vx) + abs(vy)) > 1e-3:
        return math.atan2(vy, vx)
    if prev_xy is not None:
        dx, dy = obj["x"] - prev_xy[0], obj["y"] - prev_xy[1]
        if abs(dx) + abs(dy) > 1e-3:
            return math.atan2(dy, dx)
    return None


def read_all_labels(label_dir, ts_start=None, ts_end=None):
    """读取整段录制（或时间窗）内所有车辆轨迹。

    Returns
    -------
    tracks : dict[int, list[dict{ts,x,y,yaw,label}]]  （按时间排序的完整轨迹）
    frame_ts : list[int]   所有标注帧时间戳（升序）
    """
    files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    tracks = defaultdict(list)
    frame_ts = []
    prev_xy = {}
    for lf in files:
        try:
            ts = int(Path(lf).stem)
        except ValueError:
            continue
        if ts_start is not None and ts < ts_start:
            continue
        if ts_end is not None and ts > ts_end:
            continue
        with open(lf, "r") as f:
            data = json.load(f)
        frame_ts.append(ts)
        for obj in data.get("object", []):
            if obj.get("label") not in VEHICLE_LABELS:
                continue
            x, y = float(obj["x"]), float(obj["y"])
            vid = obj["id"]
            yaw = _heading_of(obj, prev_xy.get(vid))
            tracks[vid].append({"ts": ts, "x": x, "y": y, "yaw": yaw,
                                "label": obj.get("label", "Car")})
            prev_xy[vid] = (x, y)
    for vid in tracks:
        tracks[vid].sort(key=lambda r: r["ts"])
    return dict(tracks), sorted(set(frame_ts))


def _in_region(x, y, region):
    return (region["x_min"] <= x <= region["x_max"] and
            region["y_min"] <= y <= region["y_max"])


def reference_region(dataset_root):
    """复用 intersection_filter.py 的 REFERENCE_VEHICLES 逻辑算出"官方"路口矩形。

    取 4 个方向参考车进/出路口共 8 个 (x,y) 点的 min/max（与原版完全一致），
    用给定的 dataset_root 解析场景，绕开 common_utils 里写死的 DATASET_ROOT。
    """
    import importlib.util
    import common_utils
    common_utils.DATASET_ROOT = dataset_root  # find_scene_path 用它定位场景

    ifp = REPO_ROOT / "intersection_filter" / "intersection_filter.py"
    spec = importlib.util.spec_from_file_location("_ifilt", str(ifp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DATASET_ROOT = dataset_root

    xs, ys, used = [], [], 0
    for ref in mod.REFERENCE_VEHICLES:
        files = mod.get_label_files(ref["scene_prefix"])
        if not files:
            continue
        for ts in (ref["entry_ts"], ref["exit_ts"]):
            f, _ = mod.find_closest_label(files, ts)
            if not f:
                continue
            pos = mod.find_vehicle_in_label(mod.load_label_file(f), ref["vehicle_id"])
            if pos:
                xs.append(pos[0]); ys.append(pos[1]); used += 1
    if used < 4:
        print(f"[WARN] reference-region 仅取到 {used} 个参考点，无法复算官方路口框")
        return None
    region = {"x_min": min(xs), "x_max": max(xs),
              "y_min": min(ys), "y_max": max(ys)}
    print(f"[INFO] reference-region 由 {used} 个参考车进出点算出（与 intersection_filter 一致）")
    return region


def resolve_region(args, tracks):
    """确定路口矩形区域。

    优先级：--region > --reference-region > --region-json/intersection_filter > 自动估计。
    """
    if args.region:
        x0, x1, y0, y1 = args.region
        return ({"x_min": min(x0, x1), "x_max": max(x0, x1),
                 "y_min": min(y0, y1), "y_max": max(y0, y1)}, "manual")
    if args.reference_region:
        reg = reference_region(args.dataset_root)
        if reg:
            return reg, "reference-vehicles"
        print("[WARN] 退回到下一优先级的区域来源")
    rf = Path(args.region_json) if args.region_json else \
        (INTERSECTION_FILTER_DIR / "intersection_region.json")
    if rf.exists():
        with open(rf, "r") as f:
            reg = json.load(f).get("region")
        if reg:
            return reg, f"file:{rf.name}"
    # 自动估计：以所有车辆位置中位数为中心，± half
    xs, ys = [], []
    for fr in tracks.values():
        xs += [r["x"] for r in fr]
        ys += [r["y"] for r in fr]
    if not xs:
        return ({"x_min": -50, "x_max": 50, "y_min": -50, "y_max": 50}, "default")
    cx, cy = float(np.median(xs)), float(np.median(ys))
    h = args.region_half
    return ({"x_min": cx - h, "x_max": cx + h,
             "y_min": cy - h, "y_max": cy + h}, "auto(median)")


# ==================================================================
# 合成演示数据（一整段录制：车辆陆续进入路口）
# ==================================================================
def make_demo(seed=7):
    rng = np.random.default_rng(seed)
    region = {"x_min": -45, "x_max": 45, "y_min": -45, "y_max": 45}
    dt_ms = 400               # 标注帧间隔
    total_s = 60.0
    n_frames = int(total_s * 1000 / dt_ms)
    base_ts = 1742877424148
    frame_ts = [base_ts + i * dt_ms for i in range(n_frames)]

    tracks = {}
    vid = 100

    def spawn(kind, lane, speed, enter_frame):
        nonlocal vid
        fr = []
        for k in range(n_frames - enter_frame):
            fi = enter_frame + k
            t = k * dt_ms / 1000.0
            if kind == "W2E":
                x, y, yaw = -45 + speed * t, lane, 0.0
            elif kind == "E2W":
                x, y, yaw = 45 - speed * t, lane, math.pi
            elif kind == "N2S":
                x, y, yaw = lane, 45 - speed * t, -math.pi / 2
            elif kind == "S2N":
                x, y, yaw = lane, -45 + speed * t, math.pi / 2
            elif kind == "LEFT":     # 南进 -> 东出 左转：先 1/4 圆弧，再沿东直行
                R, cx0, cy0 = 12.0, 12.0 - 3.5, -12.0
                if t <= 2.0:
                    ang = -math.pi / 2 + (math.pi / 2) * (t / 2.0)
                    x, y, yaw = cx0 + R * math.cos(ang), cy0 + R * math.sin(ang), ang + math.pi / 2
                else:
                    x, y, yaw = cx0 + R + speed * (t - 2.0), cy0, 0.0
            else:
                return
            if abs(x) > 60 or abs(y) > 60:
                break
            fr.append({"ts": frame_ts[fi], "x": float(x), "y": float(y),
                       "yaw": float(yaw), "label": kind_label(kind, rng)})
        if len(fr) >= 4:
            tracks[vid] = fr
            vid += 1

    def kind_label(kind, rng):
        return rng.choice(["Car", "Car", "Suv", "Truck", "Van"])

    # 四个方向陆续放车
    for ef in range(0, n_frames - 8, 9):
        spawn("W2E", rng.choice([-3.5, -7.0]), rng.uniform(16, 24), ef)
    for ef in range(4, n_frames - 8, 10):
        spawn("E2W", rng.choice([3.5, 7.0]), rng.uniform(16, 24), ef)
    for ef in range(2, n_frames - 8, 11):
        spawn("N2S", rng.choice([3.5, 7.0]), rng.uniform(14, 22), ef)
    for ef in range(6, n_frames - 8, 12):
        spawn("S2N", rng.choice([-3.5, -7.0]), rng.uniform(14, 22), ef)
    for ef in range(8, n_frames - 8, 18):
        spawn("LEFT", 0, rng.uniform(10, 14), ef)

    return "DEMO_intersection", tracks, frame_ts, region


# ==================================================================
# 复杂度指标
# ==================================================================
def _seg_intersect(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < 1e-9:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def _ang_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def net_heading(frames):
    if len(frames) < 2:
        return None
    dx = frames[-1]["x"] - frames[0]["x"]
    dy = frames[-1]["y"] - frames[0]["y"]
    if abs(dx) + abs(dy) < 1e-3:
        return None
    return math.atan2(dy, dx)


def turning_amount(frames):
    yaws = [r["yaw"] for r in frames if r["yaw"] is not None]
    if len(yaws) >= 2:
        return sum(abs(math.degrees(_ang_diff(b, a)))
                   for a, b in zip(yaws[:-1], yaws[1:]))
    h0 = net_heading(frames[:max(2, len(frames) // 2)])
    h1 = net_heading(frames[max(2, len(frames) // 2):])
    if h0 is None or h1 is None:
        return 0.0
    return abs(math.degrees(_ang_diff(h1, h0)))


def find_crossings(passing_polylines):
    """路口内轨迹两两空间交叉点（potential conflicts）。输入为已裁剪到路口的折线。"""
    vids = list(passing_polylines.keys())
    crossings = []
    for i in range(len(vids)):
        a = passing_polylines[vids[i]]
        if len(a) < 2:
            continue
        for j in range(i + 1, len(vids)):
            b = passing_polylines[vids[j]]
            if len(b) < 2:
                continue
            hit = None
            for k in range(len(a) - 1):
                for m in range(len(b) - 1):
                    pt = _seg_intersect(a[k], a[k + 1], b[m], b[m + 1])
                    if pt is not None:
                        hit = pt
                        break
                if hit:
                    break
            if hit:
                crossings.append(hit)
    return crossings


def compute_metrics(tracks, frame_ts, region):
    # 通过路口的车辆 = 轨迹至少有一点落在路口区域
    passing = {}
    region_poly = {}
    for vid, fr in tracks.items():
        in_pts = [(r["x"], r["y"]) for r in fr if _in_region(r["x"], r["y"], region)]
        if in_pts:
            passing[vid] = fr
            region_poly[vid] = in_pts
    n_pass = len(passing)

    # 每帧路口内车辆数 + 交互对数
    pos_by_ts = defaultdict(list)
    for fr in passing.values():
        for r in fr:
            if _in_region(r["x"], r["y"], region):
                pos_by_ts[r["ts"]].append((r["x"], r["y"]))
    per_frame_counts, per_frame_inter = [], []
    for ts in frame_ts:
        pts = pos_by_ts.get(ts, [])
        per_frame_counts.append(len(pts))
        inter = sum(1 for i in range(len(pts)) for j in range(i + 1, len(pts))
                    if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < INTERACT_DIST)
        per_frame_inter.append(inter)
    peak = max(per_frame_counts) if per_frame_counts else 0
    mean_conc = float(np.mean(per_frame_counts)) if per_frame_counts else 0.0

    # 累计通过数（按首次进入路口时间）
    entry_ts = {}
    for vid, pts in region_poly.items():
        for r in passing[vid]:
            if _in_region(r["x"], r["y"], region):
                entry_ts[vid] = r["ts"]
                break
    cumulative = [sum(1 for t in entry_ts.values() if t <= ts) for ts in frame_ts]

    # 时长 & 吞吐量
    duration = (frame_ts[-1] - frame_ts[0]) / 1000.0 if len(frame_ts) >= 2 else 0.0
    throughput = n_pass / (duration / 60.0) if duration > 0 else 0.0

    # 方向熵（8 扇区）
    headings = [h for h in (net_heading(fr) for fr in passing.values()) if h is not None]
    bins = np.zeros(8)
    for h in headings:
        bins[int(((math.degrees(h) % 360) + 22.5) // 45) % 8] += 1
    if bins.sum() > 0:
        p = bins / bins.sum()
        nz = p[p > 0]
        dir_entropy = float(-(nz * np.log(nz)).sum() / math.log(8))
    else:
        dir_entropy = 0.0

    # 转向比例
    turners = sum(1 for fr in passing.values() if turning_amount(fr) > 30.0)
    turn_ratio = turners / n_pass if n_pass else 0.0

    # 平均车速
    speeds = []
    for fr in passing.values():
        for a, b in zip(fr[:-1], fr[1:]):
            dt = (b["ts"] - a["ts"]) / 1000.0
            if dt > 1e-3:
                speeds.append(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt)
    mean_speed = float(np.median(speeds)) if speeds else 0.0

    crossings = find_crossings(region_poly)
    n_cross = len(crossings)

    # 综合复杂度评分（0~100，启发式加权）
    dens = min(mean_conc / 5.0, 1.0)
    crossn = min(n_cross / max(n_pass, 1), 1.0)
    thru = min(throughput / 30.0, 1.0)
    score = 100.0 * (0.25 * dens + 0.20 * dir_entropy + 0.25 * crossn +
                     0.10 * turn_ratio + 0.20 * thru)
    level = "High" if score >= 65 else ("Medium" if score >= 40 else "Low")

    return {
        "passing": passing, "region_poly": region_poly,
        "n_total": len(tracks),  # 输入里去重后的全部车辆数（仅车辆类）
        "n_pass": n_pass, "duration": duration, "throughput": throughput,
        "peak_concurrent": peak, "mean_concurrent": mean_conc,
        "dir_entropy": dir_entropy, "turn_ratio": turn_ratio, "n_turners": turners,
        "n_crossings": n_cross, "crossings": crossings, "mean_speed": mean_speed,
        "per_frame_counts": per_frame_counts, "per_frame_interactions": per_frame_inter,
        "cumulative": cumulative, "frame_ts": frame_ts,
        "score": score, "level": level,
    }


# ==================================================================
# 点云底图
# ==================================================================
def load_pcd_points(pcd_path, max_pts=120000):
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(pcd_path).points)
    except Exception:
        pts, started = [], False
        with open(pcd_path, "r", errors="ignore") as f:
            for line in f:
                if started:
                    p = line.split()
                    if len(p) >= 3:
                        try:
                            pts.append([float(p[0]), float(p[1]), float(p[2])])
                        except ValueError:
                            pass
                elif line.startswith("DATA"):
                    started = True
        pts = np.array(pts) if pts else np.zeros((0, 3))
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts


def get_base_pcd(clip_dir, mid_ts):
    pcd_dir = Path(clip_dir) / "road" / "lidar" / "merged_pcd"
    pcd_files = sorted(glob.glob(os.path.join(str(pcd_dir), "*.pcd")))
    if not pcd_files:
        return None
    best, bd = None, float("inf")
    for pf in pcd_files:
        try:
            ts = int(Path(pf).stem)
        except ValueError:
            continue
        if abs(ts - mid_ts) < bd:
            bd, best = abs(ts - mid_ts), pf
    return load_pcd_points(best or pcd_files[len(pcd_files) // 2])


# ==================================================================
# 绘图
# ==================================================================
def _heading_color(h):
    from matplotlib.colors import hsv_to_rgb
    if h is None:
        return (0.5, 0.5, 0.5)
    return tuple(hsv_to_rgb([(h % (2 * math.pi)) / (2 * math.pi), 0.72, 0.88]))


def _setup_serif_font(plt):
    """优先 Times New Roman，环境没装则回退到同族衬线字体。"""
    import matplotlib.font_manager as fm
    avail = {f.name for f in fm.fontManager.ttflist}
    serif = [n for n in ("Times New Roman", "Nimbus Roman", "Liberation Serif",
                         "DejaVu Serif") if n in avail]
    serif = serif or ["serif"]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": serif,
        "mathtext.fontset": "stix",
        "font.size": 12,
        "axes.linewidth": 0.8,
    })
    return serif[0]


def draw_figure(scene_name, region, region_src, metrics, tag,
                base_pts=None, demo=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch

    _setup_serif_font(plt)

    fig = plt.figure(figsize=(13.5, 7.9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.9, 1.0],
                           height_ratios=[1.62, 1.0], wspace=0.06, hspace=0.4)
    ax = fig.add_subplot(gs[:, 0])
    ax_card = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, 1])

    passing = metrics["passing"]

    # 底图
    if base_pts is not None and len(base_pts) > 0:
        ax.scatter(base_pts[:, 0], base_pts[:, 1], s=0.15, c="#c8c8c8",
                   alpha=0.5, rasterized=True, zorder=0)
    elif demo:
        cx = 0.5 * (region["x_min"] + region["x_max"])
        cy = 0.5 * (region["y_min"] + region["y_max"])
        rw = 18
        ax.add_patch(Rectangle((region["x_min"], cy - rw / 2),
                               region["x_max"] - region["x_min"], rw,
                               color="#ededed", zorder=0))
        ax.add_patch(Rectangle((cx - rw / 2, region["y_min"]),
                               rw, region["y_max"] - region["y_min"],
                               color="#ededed", zorder=0))

    # 注：分析区域仅用于统计/取景，最终图不再绘制区域框

    # 轨迹（按航向着色）
    many = len(passing) > 45
    for vid, fr in passing.items():
        xs = [r["x"] for r in fr]
        ys = [r["y"] for r in fr]
        col = _heading_color(net_heading(fr))
        ax.plot(xs, ys, "-", color=col, lw=1.3 if many else 1.8,
                alpha=0.78, zorder=4, solid_capstyle="round")
        if not many and len(xs) >= 2:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            nrm = math.hypot(dx, dy) or 1.0
            ax.add_patch(FancyArrow(xs[-1], ys[-1], dx / nrm * 2.0, dy / nrm * 2.0,
                                    width=0.4, head_width=2.0, head_length=2.2,
                                    length_includes_head=True, color=col, zorder=6))

    # 冲突点
    for (px, py) in metrics["crossings"]:
        ax.scatter(px, py, marker="x", s=55, c="#111111", linewidths=1.6, zorder=7)
    if metrics["crossings"]:
        ax.scatter([], [], marker="x", c="#111111", linewidths=1.6,
                   label=f"Conflict point (×{metrics['n_crossings']})")
    ax.plot([], [], "-", color="0.4", label="Track (hue = heading)")

    ext = max(region["x_max"] - region["x_min"], region["y_max"] - region["y_min"]) * 0.32
    ext = max(ext, 12)
    ax.set_xlim(region["x_min"] - ext, region["x_max"] + ext)
    ax.set_ylim(region["y_min"] - ext, region["y_max"] + ext)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("(a) Bird's-eye view of traffic flow", fontsize=13, loc="left",
                 fontweight="bold")
    ax.grid(True, alpha=0.22, lw=0.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    # 指标面板
    ax_card.axis("off")
    ax_card.set_xlim(0, 1)
    ax_card.set_ylim(0, 1)
    ax_card.set_title("(b) Complexity metrics", fontsize=13, loc="left",
                      fontweight="bold")
    lvl_color = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}[metrics["level"]]
    rows = [
        ("Duration (s)", f"{metrics['duration']:.0f}"),
        ("Vehicles annotated", f"{metrics['n_total']}"),
        ("Max. vehicles in intersection", f"{metrics['peak_concurrent']}"),
        ("Average vehicles in intersection", f"{metrics['mean_concurrent']:.1f}"),
        ("Directional diversity", f"{metrics['dir_entropy']:.2f}"),
        ("Turning vehicles (%)", f"{metrics['turn_ratio']*100:.0f}"),
        ("Conflict points", f"{metrics['n_crossings']}"),
        ("Median speed (m/s)", f"{metrics['mean_speed']:.1f}"),
    ]
    # 指标表（带边框；底部单独一行放复杂度评级，红字，无数字）
    box_top, box_bot, sep_y = 0.965, 0.045, 0.165
    ax_card.add_patch(Rectangle((0.0, box_bot), 1.0, box_top - box_bot, fill=False,
                                edgecolor="black", lw=1.3, zorder=2))
    ax_card.plot([0.0, 1.0], [sep_y, sep_y], color="black", lw=1.0, zorder=2)

    y_top, y_bot = 0.90, 0.24
    step = (y_top - y_bot) / (len(rows) - 1)
    y = y_top
    for k, v in rows:
        ax_card.text(0.03, y, k, fontsize=11.5, va="center")
        ax_card.text(0.97, y, v, fontsize=11.5, va="center", ha="right", fontweight="bold")
        y -= step

    cell_y = (sep_y + box_bot) / 2
    ax_card.text(0.03, cell_y, "Scene complexity", fontsize=12.5, va="center",
                 fontweight="bold")
    ax_card.text(0.97, cell_y, metrics["level"].upper(), fontsize=15, va="center",
                 ha="right", color=lvl_color, fontweight="bold")

    # 时序密度：瞬时在场数（左轴）+ 累计通过数（右轴）
    ax_time.set_title("(c) Temporal density", fontsize=13, loc="left",
                      fontweight="bold")
    t0 = metrics["frame_ts"][0]
    tsec = [(ts - t0) / 1000.0 for ts in metrics["frame_ts"]]
    ax_time.plot(tsec, metrics["per_frame_counts"], "-", lw=1.7, color="#1f77b4",
                 label="In intersection")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Vehicles", color="#1f77b4")
    ax_time.tick_params(axis="y", labelcolor="#1f77b4")
    ax_time.grid(True, alpha=0.3, lw=0.5)
    ax2 = ax_time.twinx()
    ax2.plot(tsec, metrics["cumulative"], "-", lw=1.9, color="#d62728",
             label="Cumulative passed")
    ax2.set_ylabel("Cumulative", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    l1, lb1 = ax_time.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax_time.legend(l1 + l2, lb1 + lb2, fontsize=9, loc="upper left", framealpha=0.9)

    fig.suptitle(FIGURE_TITLE, fontsize=16, fontweight="bold", y=0.99)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"traffic_complexity_{tag}.png"
    pdf = OUTPUT_DIR / f"traffic_complexity_{tag}.pdf"
    fig.savefig(str(png), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf), bbox_inches="tight")
    plt.close(fig)
    return str(png), str(pdf)


# ==================================================================
# 批量扫描 + 排名（找出最复杂的 clip）
# ==================================================================
def analyze_clip(clip_dir, fixed_region, region_half, ts_start, ts_end,
                 labels_subdir, region_json):
    """计算单个 clip 的复杂度摘要（供多进程并行调用，不绘图）。"""
    name = os.path.basename(os.path.normpath(clip_dir))
    try:
        label_dir = find_label_dir(clip_dir, labels_subdir)
        tracks, frame_ts = read_all_labels(label_dir, ts_start, ts_end)
        if not tracks:
            return {"clip": name, "ok": False, "error": "no vehicle tracks"}

        if fixed_region is not None:
            region = fixed_region
        elif region_json and Path(region_json).exists():
            with open(region_json, "r") as f:
                region = json.load(f).get("region")
        else:
            xs, ys = [], []
            for fr in tracks.values():
                xs += [r["x"] for r in fr]; ys += [r["y"] for r in fr]
            cx, cy = float(np.median(xs)), float(np.median(ys))
            region = {"x_min": cx - region_half, "x_max": cx + region_half,
                      "y_min": cy - region_half, "y_max": cy + region_half}

        m = compute_metrics(tracks, frame_ts, region)
        return {
            "clip": name, "ok": True,
            "score": round(m["score"], 2), "level": m["level"],
            "n_pass": m["n_pass"], "throughput": round(m["throughput"], 2),
            "peak_concurrent": m["peak_concurrent"],
            "mean_concurrent": round(m["mean_concurrent"], 2),
            "dir_entropy": round(m["dir_entropy"], 3),
            "turn_ratio": round(m["turn_ratio"], 3),
            "n_crossings": m["n_crossings"],
            "duration": round(m["duration"], 1),
            "n_total_vehicles": len(tracks),
        }
    except Exception as e:  # noqa: BLE001 - 单个 clip 失败不应中断整体
        return {"clip": name, "ok": False, "error": f"{type(e).__name__}: {e}"}


def discover_clips(root, pattern, labels_subdir=None):
    """在数据根下发现 clip 目录（含路侧标注目录的子目录）。"""
    sub = labels_subdir or os.path.join("road_labels", "interpolation_labels")
    out = []
    for p in sorted(glob.glob(os.path.join(root, pattern))):
        if not os.path.isdir(p):
            continue
        if (Path(p) / sub).is_dir() or list(Path(p).rglob("interpolation_labels"))[:1]:
            out.append(os.path.abspath(p))
    return out


def run_scan(args):
    import concurrent.futures as cf
    import csv

    clip_dirs = discover_clips(args.dataset_root, args.pattern, args.labels_subdir)
    if not clip_dirs:
        print(f"[ERROR] 在 {args.dataset_root} 下用 pattern '{args.pattern}' 没发现 clip")
        sys.exit(1)
    print(f"[INFO] 发现 {len(clip_dirs)} 个 clip")

    # 统一路口区域（保证跨 clip 可比）：--region > --reference-region > 各自自动估计
    fixed_region, region_src = None, "per-clip auto(median)"
    if args.region:
        x0, x1, y0, y1 = args.region
        fixed_region = {"x_min": min(x0, x1), "x_max": max(x0, x1),
                        "y_min": min(y0, y1), "y_max": max(y0, y1)}
        region_src = "manual"
    elif args.reference_region:
        fixed_region = reference_region(args.dataset_root)
        region_src = "reference-vehicles" if fixed_region else "per-clip auto(median)"
    print(f"[INFO] 路口区域来源: {region_src}")
    if fixed_region is None:
        print("[WARN] 各 clip 用自身中位数估区域，跨 clip 评分可比性稍弱；"
              "若是同一路口建议加 --reference-region 或 --region 以统一区域")

    workers = args.workers or min(64, os.cpu_count() or 8)
    print(f"[INFO] 并行 workers = {workers}\n")

    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(analyze_clip, cd, fixed_region, args.region_half,
                          args.ts_start, args.ts_end, args.labels_subdir,
                          args.region_json): cd for cd in clip_dirs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            tag = (f"score={r['score']:5.1f} ({r['level']:<6}) "
                   f"pass={r['n_pass']:>3} thru={r['throughput']:.1f}"
                   if r.get("ok") else f"[FAIL] {r.get('error')}")
            print(f"[{i:>3}/{len(clip_dirs)}] {r['clip']:<42} {tag}")

    ok = sorted([r for r in results if r.get("ok")],
                key=lambda r: r["score"], reverse=True)
    bad = [r for r in results if not r.get("ok")]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "complexity_ranking.csv"
    cols = ["rank", "clip", "score", "level", "n_pass", "throughput",
            "peak_concurrent", "mean_concurrent", "dir_entropy", "turn_ratio",
            "n_crossings", "duration", "n_total_vehicles"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rank, r in enumerate(ok, 1):
            w.writerow({"rank": rank, **{c: r.get(c, "") for c in cols if c != "rank"}})

    print(f"\n{'='*72}\n排名（按复杂度评分降序）  区域来源: {region_src}\n{'='*72}")
    print(f"{'#':>3}  {'clip':<42}{'score':>7} {'lvl':<7}{'pass':>5}{'thru':>7}"
          f"{'peak':>5}{'cross':>6}")
    for rank, r in enumerate(ok[:args.top], 1):
        print(f"{rank:>3}  {r['clip']:<42}{r['score']:>7.1f} {r['level']:<7}"
              f"{r['n_pass']:>5}{r['throughput']:>7.1f}{r['peak_concurrent']:>5}"
              f"{r['n_crossings']:>6}")
    if bad:
        print(f"\n[WARN] {len(bad)} 个 clip 失败：")
        for r in bad[:10]:
            print(f"    {r['clip']:<42} {r.get('error')}")
    print(f"\n[OK] 完整排名已存: {csv_path}")

    # 渲染 top-N 论文图
    if ok and args.plot_top > 0:
        region_pass = (fixed_region, region_src) if fixed_region else None
        print(f"\n[INFO] 渲染 top-{min(args.plot_top, len(ok))} 复杂 clip 的论文图...")
        for r in ok[:args.plot_top]:
            cd = next(c for c in clip_dirs
                      if os.path.basename(os.path.normpath(c)) == r["clip"])
            try:
                png, pdf, _ = render_clip(cd, args, fixed_region=region_pass)
                print(f"  [{r['clip']}] score={r['score']:.1f} -> {png}")
            except Exception as e:  # noqa: BLE001
                print(f"  [{r['clip']}] 渲染失败: {e}")


def resolve_clip_dir(token, dataset_root):
    """把 clip 名/前缀/路径解析成绝对目录。"""
    if os.path.isdir(token):
        return os.path.abspath(token)
    matches = [m for m in sorted(glob.glob(os.path.join(dataset_root, token + "*")))
               if os.path.isdir(m)]
    if not matches:
        raise FileNotFoundError(f"在 {dataset_root} 下找不到匹配 {token}* 的 clip 目录")
    return os.path.abspath(matches[0])


def build_overlay(tokens, args):
    """叠加多个（不同朝向的）clip：时间重基对齐 + 合并到同一路口坐标系。

    Returns: (merged_tracks, frame_ts, region, region_src, base_pts, tag, dirs)
    """
    dirs = [resolve_clip_dir(t, args.dataset_root) for t in tokens]
    print(f"[INFO] 叠加 {len(dirs)} 个 clip:")
    for d in dirs:
        print(f"    {d}")

    # 路口区域（叠加必须共用同一框）
    region, region_src = None, None
    if args.region:
        x0, x1, y0, y1 = args.region
        region = {"x_min": min(x0, x1), "x_max": max(x0, x1),
                  "y_min": min(y0, y1), "y_max": max(y0, y1)}
        region_src = "manual"
    elif args.reference_region:
        region = reference_region(args.dataset_root)
        region_src = "reference-vehicles" if region else None

    # 读各 clip，估计统一时间网格
    per_clip, intervals = [], []
    for d in dirs:
        ld = find_label_dir(d, args.labels_subdir)
        tr, fts = read_all_labels(ld, args.ts_start, args.ts_end)
        if not tr:
            print(f"[WARN] {os.path.basename(d)} 无轨迹，跳过")
            continue
        per_clip.append((d, tr, fts))
        intervals += [b - a for a, b in zip(fts[:-1], fts[1:]) if b > a]
    if not per_clip:
        raise RuntimeError("叠加的 clip 都没有可用轨迹")
    bin_ms = max(int(np.median(intervals)) if intervals else 100, 1)

    # 时间重基（各自从 0 开始，snap 到统一网格）+ 车辆 id 去重
    merged, merged_ts, short = {}, set(), []
    for i, (d, tr, fts) in enumerate(per_clip):
        t0 = fts[0]
        for vid, fr in tr.items():
            nfr = []
            for r in fr:
                nts = int(round((r["ts"] - t0) / bin_ms)) * bin_ms
                nr = dict(r); nr["ts"] = nts
                nfr.append(nr); merged_ts.add(nts)
            merged[i * 1_000_000 + vid] = nfr
        short.append(os.path.basename(d).split("_")[0])
    frame_ts = sorted(merged_ts)

    if region is None:
        region, region_src = resolve_region(args, merged)

    tag = "overlay_" + "+".join(short)
    mid = (per_clip[0][2][0] + per_clip[0][2][-1]) // 2
    base_pts = None if args.no_pcd else get_base_pcd(per_clip[0][0], mid)
    return merged, frame_ts, region, region_src, base_pts, tag, dirs


def render_clip(clip_dir, args, fixed_region=None):
    """读取 + 计算 + 绘图，返回 (png, pdf, metrics)。fixed_region 为 (region, src) 或 None。"""
    clip_dir = os.path.abspath(clip_dir)
    label_dir = find_label_dir(clip_dir, args.labels_subdir)
    tracks, frame_ts = read_all_labels(label_dir, args.ts_start, args.ts_end)
    if not tracks:
        raise RuntimeError("未读到任何车辆轨迹")
    if fixed_region is not None:
        region, region_src = fixed_region
    else:
        region, region_src = resolve_region(args, tracks)
    name = os.path.basename(clip_dir)
    mid_ts = (frame_ts[0] + frame_ts[-1]) // 2
    base_pts = None if args.no_pcd else get_base_pcd(clip_dir, mid_ts)
    metrics = compute_metrics(tracks, frame_ts, region)
    png, pdf = draw_figure(name, region, region_src, metrics, name,
                           base_pts=base_pts, demo=False)
    return png, pdf, metrics


# ==================================================================
# 主流程
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="单 clip 交通流复杂度论文图（全时序/路口吞吐）")
    ap.add_argument("--demo", action="store_true", help="合成数据演示，无需真实数据集")
    ap.add_argument("--clip-dir", type=str, default=None,
                    help="clip 目录绝对路径，如 /mnt/car_road_data_TianJin/002_car0325_road0327_t2")
    ap.add_argument("--dataset-root", type=str, default=_DEFAULT_ROOT,
                    help="数据集根目录（配合 --clip 名使用）")
    ap.add_argument("--clip", type=str, default=None,
                    help="clip 目录名或前缀（配合 --dataset-root），如 002 或完整名")
    ap.add_argument("--labels-subdir", type=str, default=None,
                    help="覆盖标注子目录（默认 road_labels/interpolation_labels）")
    ap.add_argument("--ts-start", type=int, default=None, help="时间窗起（毫秒，可选）")
    ap.add_argument("--ts-end", type=int, default=None, help="时间窗止（毫秒，可选）")
    ap.add_argument("--region", type=float, nargs=4, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="手动指定路口矩形")
    ap.add_argument("--reference-region", action="store_true",
                    help="复用 intersection_filter 的 REFERENCE_VEHICLES 复算官方路口框"
                         "（需要 --dataset-root 指向含参考场景的数据根）")
    ap.add_argument("--region-json", type=str, default=None,
                    help="路口区域 json（默认读 intersection_filter/output/intersection_region.json）")
    ap.add_argument("--region-half", type=float, default=50.0,
                    help="自动估计路口区域时的半边长（米，默认50）")
    ap.add_argument("--no-pcd", action="store_true", help="不加载点云底图")
    # 批量扫描 / 排名
    ap.add_argument("--scan", action="store_true",
                    help="扫描 --dataset-root 下所有 clip，并行算复杂度并排名")
    ap.add_argument("--pattern", type=str, default="*",
                    help="--scan 时的 clip 目录通配（默认 '*'，可如 '0??_*'）")
    ap.add_argument("--workers", type=int, default=None,
                    help="--scan 并行进程数（默认 min(64, CPU核数)）")
    ap.add_argument("--top", type=int, default=15,
                    help="--scan 终端打印的排名条数（默认15，CSV 始终全量）")
    ap.add_argument("--plot-top", type=int, default=1,
                    help="--scan 后渲染前 N 个最复杂 clip 的论文图（默认1，0=不渲染）")
    # 叠加不同朝向的 clip
    ap.add_argument("--overlay-clips", type=str, nargs="+", default=None,
                    metavar="CLIP",
                    help="叠加多个不同朝向的 clip（名/前缀/路径），如 "
                         "--overlay-clips 010 051；轨迹时间重基对齐后合并到同一路口")
    args = ap.parse_args()

    if args.scan:
        run_scan(args)
        return

    if args.overlay_clips:
        tracks, frame_ts, region, region_src, base_pts, tag, _ = build_overlay(
            args.overlay_clips, args)
        metrics = compute_metrics(tracks, frame_ts, region)
        print("\n=== Overlay traffic-flow complexity ===")
        print(f"  region              : x[{region['x_min']:.1f},{region['x_max']:.1f}] "
              f"y[{region['y_min']:.1f},{region['y_max']:.1f}]  ({region_src})")
        for k in ("n_pass", "duration", "throughput", "peak_concurrent",
                  "mean_concurrent", "dir_entropy", "turn_ratio", "n_crossings",
                  "mean_speed", "score", "level"):
            print(f"  {k:18s}: {metrics[k]}")
        png, pdf = draw_figure(tag, region, region_src, metrics, tag,
                               base_pts=base_pts, demo=False)
        print(f"\n[OK] 叠加图已保存:\n  {png}\n  {pdf}")
        return

    if args.demo:
        scene_name, tracks, frame_ts, region = make_demo()
        region_src, base_pts, demo, tag = "demo-fixed", None, True, "demo"
    else:
        # 解析 clip 目录
        clip_dir = args.clip_dir
        if clip_dir is None:
            if args.clip is None:
                ap.error("请指定 --clip-dir <目录> 或 --dataset-root + --clip <名>，或 --demo")
            matches = sorted(glob.glob(os.path.join(args.dataset_root, args.clip + "*")))
            if not matches:
                ap.error(f"在 {args.dataset_root} 下找不到匹配 {args.clip}* 的 clip 目录")
            clip_dir = matches[0]
            if len(matches) > 1:
                print(f"[WARN] 匹配到多个，使用: {clip_dir}")
        clip_dir = os.path.abspath(clip_dir)
        print(f"[INFO] clip 目录: {clip_dir}")
        label_dir = find_label_dir(clip_dir, args.labels_subdir)
        print(f"[INFO] 标注目录: {label_dir}")
        tracks, frame_ts = read_all_labels(label_dir, args.ts_start, args.ts_end)
        if not tracks:
            print("[ERROR] 未读到任何车辆轨迹"); sys.exit(1)
        region, region_src = resolve_region(args, tracks)
        scene_name = os.path.basename(clip_dir)
        tag = scene_name
        mid_ts = (frame_ts[0] + frame_ts[-1]) // 2
        base_pts = None if args.no_pcd else get_base_pcd(clip_dir, mid_ts)
        demo = False

    metrics = compute_metrics(tracks, frame_ts, region)
    print("\n=== Clip traffic-flow complexity ===")
    print(f"  region              : x[{region['x_min']:.1f},{region['x_max']:.1f}] "
          f"y[{region['y_min']:.1f},{region['y_max']:.1f}]  ({region_src})")
    for k in ("n_pass", "duration", "throughput", "peak_concurrent", "mean_concurrent",
              "dir_entropy", "turn_ratio", "n_crossings", "mean_speed", "score", "level"):
        print(f"  {k:18s}: {metrics[k]}")

    png, pdf = draw_figure(scene_name, region, region_src, metrics, tag,
                           base_pts=base_pts, demo=demo)
    print(f"\n[OK] 图已保存:\n  {png}\n  {pdf}")


if __name__ == "__main__":
    main()
