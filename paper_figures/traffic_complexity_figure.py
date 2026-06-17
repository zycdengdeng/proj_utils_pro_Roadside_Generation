#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic-flow complexity figure for one clip (publication-quality).

为数据集论文绘制"单个 clip 交通流复杂程度"图。思路沿用 intersection_filter.py：
在路口矩形区域内、在该 clip 的时间窗口 [ts_start, ts_end] 内，重建 **所有** 车辆
（不仅是参考车）的 BEV 轨迹，并量化交通流复杂度。

图由四部分组成（可单独开关）：
  1. BEV 多车轨迹       —— 俯视图叠加该 clip 内所有车辆轨迹，按车辆着色，
                            起点圆点、终点箭头表示行驶方向；红色 ✕ 标注轨迹交叉/冲突点。
  2. 复杂度指标面板     —— 在场车辆数、峰值/平均同时在场数、方向熵、转向比例、
                            交叉冲突点数、综合复杂度评分等。
  3. 时序密度曲线       —— 每帧"在场车辆数"与"交互对数"随帧变化。
  4. 点云/路口底图       —— 真实 LiDAR 点云作底图（无数据时用示意路口）。

用法
----
真实数据（在挂载了 /mnt/car_road_data_fix 的机器上运行）::

    # 直接从 intersection_filter 的 filtered_segments.json 选第 0 个片段
    python traffic_complexity_figure.py --from-segments 0

    # 或显式指定一个 clip
    python traffic_complexity_figure.py --scene 002 --vid 29 --seg 0

    # 或给定场景 + 时间窗口
    python traffic_complexity_figure.py --scene 002 --ts-start 1742877436322 --ts-end 1742877441799

合成演示（无需真实数据，用于验证脚本/预览版式）::

    python traffic_complexity_figure.py --demo

输出：paper_figures/output/traffic_complexity_<tag>.png / .pdf
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

# ------------------------------------------------------------------
# 复用仓库公共配置（可选；真实数据模式需要）
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from common_utils import DATASET_ROOT, get_scene_paths  # noqa: E402
except Exception:  # pragma: no cover - 仅在仓库结构变动时触发
    DATASET_ROOT = "/mnt/car_road_data_fix"
    get_scene_paths = None

# 与 intersection_filter.py 保持一致
VEHICLE_LABELS = {"Car", "Suv", "Truck", "Bus", "Van"}
SEGMENT_LENGTH = 29

INTERSECTION_FILTER_DIR = REPO_ROOT / "intersection_filter" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ==================================================================
# 数据读取（真实数据）
# ==================================================================
def _label_dir_for_scene(scene_prefix):
    if get_scene_paths is None:
        return None
    paths = get_scene_paths(scene_prefix)
    if not paths:
        return None
    return paths.get("roadside_labels")


def _heading_of(obj, prev_xy=None):
    """获取目标朝向（弧度，atan2 约定，x 向右、y 向上）。

    优先 yaw，其次速度 (vx, vy)，最后用相邻位移。返回 None 表示未知。
    """
    if "yaw" in obj and obj["yaw"] is not None:
        return float(obj["yaw"])
    vx, vy = obj.get("vx"), obj.get("vy")
    if vx is not None and vy is not None and (abs(vx) + abs(vy)) > 1e-3:
        return math.atan2(vy, vx)
    if prev_xy is not None:
        dx = obj["x"] - prev_xy[0]
        dy = obj["y"] - prev_xy[1]
        if abs(dx) + abs(dy) > 1e-3:
            return math.atan2(dy, dx)
    return None


def collect_tracks(scene_prefix, ts_start, ts_end, region=None):
    """读取时间窗口 [ts_start, ts_end] 内的所有车辆轨迹。

    Returns
    -------
    tracks : dict[int, list[dict]]   vid -> [{ts,x,y,yaw,label}, ...]（按时间排序）
    frame_ts : list[int]             落入窗口的去重时间戳（升序）
    """
    label_dir = _label_dir_for_scene(scene_prefix)
    if not label_dir or not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f"找不到场景 {scene_prefix} 的标注目录: {label_dir}\n"
            f"（真实数据需要挂载 {DATASET_ROOT}；无数据时请用 --demo）"
        )

    files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    tracks = defaultdict(list)
    frame_ts = []
    prev_xy = {}

    for lf in files:
        try:
            ts = int(Path(lf).stem)
        except ValueError:
            continue
        if ts < ts_start or ts > ts_end:
            continue
        with open(lf, "r") as f:
            data = json.load(f)

        kept = False
        for obj in data.get("object", []):
            if obj.get("label") not in VEHICLE_LABELS:
                continue
            x, y = float(obj["x"]), float(obj["y"])
            if region is not None and not _in_region(x, y, region):
                continue
            vid = obj["id"]
            yaw = _heading_of(obj, prev_xy.get(vid))
            tracks[vid].append({"ts": ts, "x": x, "y": y, "yaw": yaw,
                                 "label": obj.get("label", "Car")})
            prev_xy[vid] = (x, y)
            kept = True
        if kept:
            frame_ts.append(ts)

    for vid in tracks:
        tracks[vid].sort(key=lambda r: r["ts"])
    frame_ts = sorted(set(frame_ts))
    return dict(tracks), frame_ts


def _in_region(x, y, region):
    return (region["x_min"] <= x <= region["x_max"] and
            region["y_min"] <= y <= region["y_max"])


def load_region():
    """从 intersection_filter 输出加载路口矩形区域（若存在）。"""
    rf = INTERSECTION_FILTER_DIR / "intersection_region.json"
    if rf.exists():
        with open(rf, "r") as f:
            return json.load(f).get("region")
    return None


def region_from_tracks(tracks, pad=8.0):
    xs, ys = [], []
    for frames in tracks.values():
        xs += [r["x"] for r in frames]
        ys += [r["y"] for r in frames]
    if not xs:
        return None
    return {"x_min": min(xs) - pad, "x_max": max(xs) + pad,
            "y_min": min(ys) - pad, "y_max": max(ys) + pad}


def resolve_clip(args):
    """根据命令行参数确定一个 clip：返回 (scene, ts_start, ts_end, tag, ego_vid)。"""
    if args.from_segments is not None:
        sf = INTERSECTION_FILTER_DIR / "filtered_segments.json"
        if not sf.exists():
            raise FileNotFoundError(f"未找到 {sf}，请先运行 intersection_filter.py")
        with open(sf, "r") as f:
            segs = json.load(f)
        if not segs:
            raise ValueError("filtered_segments.json 为空")
        seg = segs[args.from_segments]
        return (seg["scene"], seg["ts_start"], seg["ts_end"],
                f"{seg['scene']}_id{seg['vehicle_id']}_seg{seg['segment_index']:02d}",
                seg["vehicle_id"])

    if args.scene and args.ts_start and args.ts_end:
        return (args.scene, args.ts_start, args.ts_end,
                f"{args.scene}_{args.ts_start}_{args.ts_end}", args.vid)

    if args.scene and args.vid is not None:
        # 从 filtered_segments.json 里找匹配的 scene/vid/seg
        sf = INTERSECTION_FILTER_DIR / "filtered_segments.json"
        if sf.exists():
            with open(sf, "r") as f:
                segs = json.load(f)
            for seg in segs:
                if (seg["scene"] == args.scene and seg["vehicle_id"] == args.vid
                        and seg["segment_index"] == args.seg):
                    return (seg["scene"], seg["ts_start"], seg["ts_end"],
                            f"{seg['scene']}_id{args.vid}_seg{args.seg:02d}", args.vid)
        raise ValueError("未找到匹配的片段，请改用 --ts-start/--ts-end 或 --from-segments")

    raise ValueError("请指定 clip：--demo / --from-segments N / --scene+--vid / "
                     "--scene+--ts-start+--ts-end")


# ==================================================================
# 合成演示数据
# ==================================================================
def make_demo(seed=7):
    """生成一个四岔路口、若干车辆（含直行/左转/右转）的合成 clip。"""
    rng = np.random.default_rng(seed)
    n_frames = SEGMENT_LENGTH
    dt = 0.1  # 10 Hz
    t = np.arange(n_frames) * dt
    region = {"x_min": -40, "x_max": 40, "y_min": -40, "y_max": 40}

    tracks = {}
    vid = 0

    def add(path_xy, yaws, label="Car", start_frame=0):
        nonlocal vid
        frames = []
        for k, (x, y) in enumerate(path_xy):
            fi = start_frame + k
            if fi >= n_frames:
                break
            frames.append({"ts": int(1000 * t[fi]), "x": float(x), "y": float(y),
                           "yaw": float(yaws[k]), "label": label})
        if len(frames) >= 5:
            tracks[vid] = frames
            vid += 1

    # 直行车（4 个方向各几辆，速度/相位不同；速度足够穿过路口中心以产生交叉）
    for lane, base in [("W2E", -3.5), ("W2E", -7.0)]:
        speed = rng.uniform(20, 26)
        sf = int(rng.integers(0, 5))
        xs = -38 + speed * t
        ys = np.full_like(xs, base)
        add(list(zip(xs, ys)), np.zeros(n_frames), start_frame=sf)
    for lane, base in [("E2W", 3.5), ("E2W", 7.0)]:
        speed = rng.uniform(20, 26)
        sf = int(rng.integers(0, 5))
        xs = 38 - speed * t
        ys = np.full_like(xs, base)
        add(list(zip(xs, ys)), np.full(n_frames, math.pi), start_frame=sf)
    for base in [3.5, 7.0]:
        speed = rng.uniform(18, 24)
        sf = int(rng.integers(0, 6))
        ys = -38 + speed * t
        xs = np.full_like(ys, base)
        add(list(zip(xs, ys)), np.full(n_frames, math.pi / 2), label="Suv", start_frame=sf)
    for base in [-3.5, -7.0]:
        speed = rng.uniform(18, 24)
        sf = int(rng.integers(0, 6))
        ys = 38 - speed * t
        xs = np.full_like(ys, base)
        add(list(zip(xs, ys)), np.full(n_frames, -math.pi / 2), label="Truck", start_frame=sf)

    # 左转车（南进 -> 东出，沿 1/4 圆弧）
    R = 12.0
    cx, cy = R - 3.5, -R
    ang = np.linspace(-math.pi / 2, 0, n_frames) + 0.0
    xs = cx + R * np.cos(ang)
    ys = cy + R * np.sin(ang)
    yaws = ang + math.pi / 2
    add(list(zip(xs, ys)), yaws, label="Car", start_frame=2)

    # 右转车（西进 -> 南出）
    R2 = 7.0
    cx2, cy2 = -R2, R2 - 7.0
    ang2 = np.linspace(0, -math.pi / 2, n_frames)
    xs2 = cx2 + R2 * np.cos(ang2)
    ys2 = cy2 + R2 * np.sin(ang2)
    yaws2 = ang2 - math.pi / 2
    add(list(zip(xs2, ys2)), yaws2, label="Van", start_frame=4)

    # 一辆缓慢通过、轻微抖动的车（增加交互）
    speed = 5.0
    xs = -20 + speed * t
    ys = 0.5 * np.sin(t * 2.0) - 3.5
    yaws = np.gradient(ys, xs)
    add(list(zip(xs, ys)), np.arctan(yaws), label="Bus", start_frame=0)

    frame_ts = sorted({int(1000 * x) for x in t})
    return "DEMO_intersection", tracks, frame_ts, region, None


# ==================================================================
# 复杂度指标
# ==================================================================
def _seg_intersect(p1, p2, p3, p4):
    """判断线段 p1p2 与 p3p4 是否相交，返回交点或 None。"""
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < 1e-9:
        return None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def find_crossings(tracks):
    """找出不同车辆轨迹之间的空间交叉点（潜在冲突点）。"""
    polylines = {vid: [(r["x"], r["y"]) for r in fr] for vid, fr in tracks.items()}
    vids = list(polylines.keys())
    crossings = []
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            a = polylines[vids[i]]
            b = polylines[vids[j]]
            found = None
            for k in range(len(a) - 1):
                for m in range(len(b) - 1):
                    pt = _seg_intersect(a[k], a[k + 1], b[m], b[m + 1])
                    if pt is not None:
                        found = pt
                        break
                if found:
                    break
            if found:
                crossings.append(found)
    return crossings


def net_heading(frames):
    """整段轨迹的净行驶方向（弧度）。"""
    if len(frames) < 2:
        return None
    dx = frames[-1]["x"] - frames[0]["x"]
    dy = frames[-1]["y"] - frames[0]["y"]
    if abs(dx) + abs(dy) < 1e-3:
        return None
    return math.atan2(dy, dx)


def turning_amount(frames):
    """整段累计航向变化（度），用 yaw 或位移方向估计。"""
    yaws = [r["yaw"] for r in frames if r["yaw"] is not None]
    if len(yaws) < 2:
        h0 = net_heading(frames[:max(2, len(frames) // 2)])
        h1 = net_heading(frames[max(2, len(frames) // 2):])
        if h0 is None or h1 is None:
            return 0.0
        return abs(math.degrees(_ang_diff(h1, h0)))
    total = 0.0
    for a, b in zip(yaws[:-1], yaws[1:]):
        total += abs(math.degrees(_ang_diff(b, a)))
    return total


def _ang_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def compute_metrics(tracks, frame_ts):
    n_agents = len(tracks)

    # 每帧在场车辆数 & 交互对数（中心距 < THRESH）
    THRESH = 12.0
    per_frame_counts = []
    per_frame_interactions = []
    pos_by_ts = defaultdict(list)
    for fr in tracks.values():
        for r in fr:
            pos_by_ts[r["ts"]].append((r["x"], r["y"]))
    for ts in frame_ts:
        pts = pos_by_ts.get(ts, [])
        per_frame_counts.append(len(pts))
        inter = 0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < THRESH:
                    inter += 1
        per_frame_interactions.append(inter)

    peak = max(per_frame_counts) if per_frame_counts else 0
    mean_conc = float(np.mean(per_frame_counts)) if per_frame_counts else 0.0

    # 方向熵（8 个罗盘扇区）
    headings = [net_heading(fr) for fr in tracks.values()]
    headings = [h for h in headings if h is not None]
    bins = np.zeros(8)
    for h in headings:
        idx = int(((math.degrees(h) % 360) + 22.5) // 45) % 8
        bins[idx] += 1
    if bins.sum() > 0:
        p = bins / bins.sum()
        nz = p[p > 0]
        dir_entropy = float(-(nz * np.log(nz)).sum() / math.log(8))
    else:
        dir_entropy = 0.0

    # 转向比例（累计航向变化 > 30°）
    turners = sum(1 for fr in tracks.values() if turning_amount(fr) > 30.0)
    turn_ratio = turners / n_agents if n_agents else 0.0

    # 路径长度 & 速度
    speeds = []
    for fr in tracks.values():
        for a, b in zip(fr[:-1], fr[1:]):
            dt = (b["ts"] - a["ts"]) / 1000.0
            if dt > 1e-3:
                speeds.append(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dt)
    mean_speed = float(np.mean(speeds)) if speeds else 0.0

    crossings = find_crossings(tracks)
    n_cross = len(crossings)

    # 综合复杂度评分（0~100，启发式加权）
    dens = min(mean_conc / 8.0, 1.0)
    crossn = min(n_cross / max(n_agents, 1), 1.0)
    score = 100.0 * (0.30 * dens + 0.25 * dir_entropy +
                     0.25 * crossn + 0.15 * turn_ratio +
                     0.05 * min(n_agents / 15.0, 1.0))
    if score >= 70:
        level = "High"
    elif score >= 45:
        level = "Medium"
    else:
        level = "Low"

    return {
        "n_agents": n_agents,
        "n_frames": len(frame_ts),
        "peak_concurrent": peak,
        "mean_concurrent": mean_conc,
        "dir_entropy": dir_entropy,
        "turn_ratio": turn_ratio,
        "n_turners": turners,
        "n_crossings": n_cross,
        "crossings": crossings,
        "mean_speed": mean_speed,
        "per_frame_counts": per_frame_counts,
        "per_frame_interactions": per_frame_interactions,
        "score": score,
        "level": level,
    }


# ==================================================================
# 点云底图
# ==================================================================
def load_pcd_points(pcd_path, max_pts=120000):
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        pts = np.asarray(pcd.points)
    except Exception:
        pts = []
        started = False
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


def get_base_pcd(scene_prefix, mid_ts):
    if get_scene_paths is None:
        return None
    paths = get_scene_paths(scene_prefix)
    if not paths:
        return None
    pcd_files = sorted(glob.glob(os.path.join(paths.get("pcd", ""), "*.pcd")))
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
    if best is None:
        best = pcd_files[len(pcd_files) // 2]
    return load_pcd_points(best)


# ==================================================================
# 绘图
# ==================================================================
def draw_figure(scene_name, tracks, frame_ts, region, metrics, tag,
                base_pts=None, demo=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle, FancyArrow

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "mathtext.default": "regular",
    })

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = gridspec.GridSpec(3, 3, width_ratios=[2.0, 0.04, 1.0],
                           height_ratios=[1, 1, 1], wspace=0.18, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])          # 主 BEV
    ax_card = fig.add_subplot(gs[0, 2])     # 指标面板
    ax_rose = fig.add_subplot(gs[1, 2], projection="polar")  # 方向玫瑰
    ax_time = fig.add_subplot(gs[2, 2])     # 时序曲线

    cx = 0.5 * (region["x_min"] + region["x_max"])
    cy = 0.5 * (region["y_min"] + region["y_max"])

    # --- 底图 ---
    if base_pts is not None and len(base_pts) > 0:
        ax.scatter(base_pts[:, 0], base_pts[:, 1], s=0.15, c="#c8c8c8",
                   alpha=0.5, rasterized=True, zorder=0)
    elif demo:
        # 示意四岔路口
        road_w = 16
        ax.add_patch(Rectangle((region["x_min"], cy - road_w / 2),
                               region["x_max"] - region["x_min"], road_w,
                               color="#ececec", zorder=0))
        ax.add_patch(Rectangle((cx - road_w / 2, region["y_min"]),
                               road_w, region["y_max"] - region["y_min"],
                               color="#ececec", zorder=0))
        # 车道虚线
        for yy in (cy,):
            ax.plot([region["x_min"], region["x_max"]], [yy, yy], "--",
                    color="#f4c430", lw=1.2, dashes=(6, 6), zorder=0.5, alpha=0.8)
        for xx in (cx,):
            ax.plot([xx, xx], [region["y_min"], region["y_max"]], "--",
                    color="#f4c430", lw=1.2, dashes=(6, 6), zorder=0.5, alpha=0.8)

    # --- 路口区域框 ---
    ax.add_patch(Rectangle((region["x_min"], region["y_min"]),
                           region["x_max"] - region["x_min"],
                           region["y_max"] - region["y_min"],
                           fill=False, edgecolor="#d62728", lw=1.8,
                           linestyle=(0, (6, 4)), zorder=2,
                           label="Intersection region"))

    # --- 轨迹 ---
    cmap = plt.get_cmap("turbo")
    vids = list(tracks.keys())
    n = max(len(vids), 1)
    for i, vid in enumerate(vids):
        fr = tracks[vid]
        xs = [r["x"] for r in fr]
        ys = [r["y"] for r in fr]
        col = cmap((i + 0.5) / n)
        ax.plot(xs, ys, "-", color=col, lw=2.0, alpha=0.9, zorder=4,
                solid_capstyle="round")
        ax.scatter(xs[0], ys[0], s=34, color=col, edgecolors="white",
                   linewidths=0.7, zorder=5)  # 起点
        # 终点方向箭头
        if len(xs) >= 2:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            norm = math.hypot(dx, dy) or 1.0
            ax.add_patch(FancyArrow(xs[-1], ys[-1], dx / norm * 2.2, dy / norm * 2.2,
                                    width=0.5, head_width=2.2, head_length=2.4,
                                    length_includes_head=True, color=col, zorder=6))

    # --- 冲突点 ---
    for (px, py) in metrics["crossings"]:
        ax.scatter(px, py, marker="x", s=70, c="#111111", linewidths=2.0, zorder=7)
    if metrics["crossings"]:
        ax.scatter([], [], marker="x", c="#111111", linewidths=2.0,
                   label=f"Trajectory conflict (×{metrics['n_crossings']})")

    ax.scatter([], [], marker="o", c="gray", edgecolors="white",
               label="Track start")
    ax.set_aspect("equal")
    ax.set_xlim(region["x_min"], region["x_max"])
    ax.set_ylim(region["y_min"], region["y_max"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"(a) BEV traffic flow  —  clip {tag}", fontsize=12, loc="left",
                 fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # --- 指标面板 ---
    ax_card.axis("off")
    ax_card.set_title("(b) Complexity metrics", fontsize=12, loc="left",
                      fontweight="bold")
    level_color = {"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"}[
        metrics["level"]]
    rows = [
        ("Agents in clip", f"{metrics['n_agents']}"),
        ("Frames", f"{metrics['n_frames']}"),
        ("Peak concurrent", f"{metrics['peak_concurrent']}"),
        ("Mean concurrent", f"{metrics['mean_concurrent']:.1f}"),
        ("Direction entropy", f"{metrics['dir_entropy']:.2f}"),
        ("Turning agents", f"{metrics['n_turners']} ({metrics['turn_ratio']*100:.0f}%)"),
        ("Conflict points", f"{metrics['n_crossings']}"),
        ("Mean speed", f"{metrics['mean_speed']:.1f} m/s"),
    ]
    y = 0.95
    for k, v in rows:
        ax_card.text(0.02, y, k, fontsize=10, va="center")
        ax_card.text(0.98, y, v, fontsize=10, va="center", ha="right",
                     fontweight="bold")
        y -= 0.092
    # 评分条
    bar_y = y - 0.07
    ax_card.text(0.02, y, "Complexity score", fontsize=10.5,
                 va="center", fontweight="bold")
    ax_card.text(0.98, y, f"{metrics['score']:.0f}/100  ({metrics['level']})",
                 fontsize=10.5, va="center", ha="right", fontweight="bold",
                 color=level_color)
    ax_card.add_patch(plt.Rectangle((0.02, bar_y - 0.04), 0.96, 0.055,
                                    color="#e8e8e8"))
    ax_card.add_patch(plt.Rectangle((0.02, bar_y - 0.04),
                                    0.96 * metrics["score"] / 100.0, 0.055,
                                    color=level_color))
    ax_card.set_xlim(0, 1)
    ax_card.set_ylim(0, 1)

    # --- 方向玫瑰 ---
    ax_rose.set_title("(c) Heading distribution", fontsize=11, loc="left",
                      fontweight="bold", pad=12)
    headings = [net_heading(fr) for fr in tracks.values()]
    headings = [h for h in headings if h is not None]
    nb = 8
    edges = np.linspace(-math.pi, math.pi, nb + 1)
    counts, _ = np.histogram(headings, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    ax_rose.bar(centers, counts, width=2 * math.pi / nb, bottom=0.0,
                color=cmap(np.linspace(0.15, 0.9, nb)), edgecolor="white",
                alpha=0.9, align="center")
    ax_rose.set_theta_zero_location("E")
    ax_rose.set_theta_direction(1)
    ax_rose.set_yticklabels([])
    ax_rose.set_xticks(np.linspace(0, 2 * math.pi, 4, endpoint=False))
    ax_rose.set_xticklabels(["E", "N", "W", "S"], fontsize=9)
    ax_rose.tick_params(pad=-2)

    # --- 时序曲线 ---
    ax_time.set_title("(d) Temporal density", fontsize=11, loc="left",
                      fontweight="bold")
    f = np.arange(metrics["n_frames"])
    ax_time.plot(f, metrics["per_frame_counts"], "-o", ms=3, lw=1.6,
                 color="#1f77b4", label="Vehicles in scene")
    ax_time.plot(f, metrics["per_frame_interactions"], "-s", ms=3, lw=1.6,
                 color="#d62728", label="Interaction pairs")
    ax_time.set_xlabel("Frame index")
    ax_time.set_ylabel("Count")
    ax_time.grid(True, alpha=0.3, lw=0.5)
    ax_time.legend(fontsize=8, loc="upper left")

    src = "synthetic demo data" if demo else f"scene {scene_name}"
    fig.suptitle(f"Traffic-flow complexity of a single clip  ({src})",
                 fontsize=14, fontweight="bold", y=0.995)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"traffic_complexity_{tag}.png"
    pdf = OUTPUT_DIR / f"traffic_complexity_{tag}.pdf"
    fig.savefig(str(png), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf), bbox_inches="tight")
    plt.close(fig)
    return str(png), str(pdf)


# ==================================================================
# 主流程
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="单 clip 交通流复杂度论文图")
    ap.add_argument("--demo", action="store_true", help="用合成数据演示（无需真实数据集）")
    ap.add_argument("--from-segments", type=int, default=None,
                    help="从 intersection_filter/output/filtered_segments.json 取第 N 个片段")
    ap.add_argument("--scene", type=str, default=None, help="场景前缀，如 002")
    ap.add_argument("--vid", type=int, default=None, help="（可选）clip 对应的参考车辆 id")
    ap.add_argument("--seg", type=int, default=0, help="片段序号（配合 --scene/--vid）")
    ap.add_argument("--ts-start", type=int, default=None, help="时间窗口起（毫秒）")
    ap.add_argument("--ts-end", type=int, default=None, help="时间窗口止（毫秒）")
    ap.add_argument("--no-region-clip", action="store_true",
                    help="不限制在路口矩形内（统计窗口内所有车辆）")
    ap.add_argument("--no-pcd", action="store_true", help="不加载点云底图")
    args = ap.parse_args()

    if args.demo:
        scene_name, tracks, frame_ts, region, ego_vid = make_demo()
        base_pts = None
        demo = True
        tag = "demo"
    else:
        scene, ts_start, ts_end, tag, ego_vid = resolve_clip(args)
        region = None if args.no_region_clip else load_region()
        tracks, frame_ts = collect_tracks(scene, ts_start, ts_end, region=region)
        if not tracks:
            print("[ERROR] 窗口内没有车辆轨迹"); sys.exit(1)
        if region is None:
            region = region_from_tracks(tracks)
        scene_name = scene
        mid_ts = (ts_start + ts_end) // 2
        base_pts = None if args.no_pcd else get_base_pcd(scene, mid_ts)
        demo = False

    metrics = compute_metrics(tracks, frame_ts)
    print("\n=== Clip complexity ===")
    for k in ("n_agents", "n_frames", "peak_concurrent", "mean_concurrent",
              "dir_entropy", "turn_ratio", "n_crossings", "mean_speed",
              "score", "level"):
        print(f"  {k:18s}: {metrics[k]}")

    png, pdf = draw_figure(scene_name, tracks, frame_ts, region, metrics, tag,
                           base_pts=base_pts, demo=demo)
    print(f"\n[OK] 图已保存:\n  {png}\n  {pdf}")


if __name__ == "__main__":
    main()
