#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-platform BEV: roadside merged LiDAR + roadside 3D labels + vehicle LiDAR.

车路协同空间对齐图：把路侧 merged 点云、路侧 3D 标注框、以及变换到路侧坐标系的
车端点云叠加到同一张 BEV 上，展示跨平台（车端/路侧）的时空对齐。

数据来源（/mnt/car_road_data_TianJin/<clip>/）：
  road/lidar/merged_pcd/<road_ts_ms>.pcd            路侧合并点云（底图）
  road_labels/interpolation_labels/<road_ts_ms>.json 路侧 3D 标注框
  car/pcds/main/<car_ts_s>.pcd                       车端点云
  support_info/carid.json                            自车在路侧标注里的 id

坐标变换（与 segment_pipeline/ego_transform.py 一致）：
  自车在路侧标注里有个 3D 框 (x,y,z,roll,pitch,yaw)。车端 LiDAR 点 → 路侧：
    p_road = euler2rotmat(roll,pitch,yaw) @ (p_carlidar + [0,0,h/2+0.25]) + [x,y,z]

帧选取：默认用 carid.json 的 visualize_roadtime 作为路侧帧（自车在此帧可见），
再在 car/pcds/main 里找时间戳最近的车端点云（路侧 ms ↔ 车端 s）。

用法::
    python cross_platform_bev.py --clip 002 \
        --dataset-root /mnt/car_road_data_TianJin
    # 或：--clip-dir /mnt/car_road_data_TianJin/002_car0325_road0327_t2

输出：paper_figures/output/cross_platform_bev_<clip>.png / .pdf
"""

import os
import sys
import json
import glob
import argparse
import math
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_ROOT = "/mnt/car_road_data_TianJin"
DEFAULT_CARID = "/mnt/car_road_data_TianJin/support_info/carid.json"

# 展示范围（世界坐标）：在 x宽70/y宽150 基础上各放大 1.5×、中心不变
# x: 中心 -55, 宽 105; y: 中心 -12.5, 宽 225
DEFAULT_XLIM = (-107.5, -2.5)
DEFAULT_YLIM = (-125.0, 100.0)
LIDAR_Z_EXTRA = 0.25  # 虚拟 LiDAR 在 bbox 顶部之上的偏移（ego_transform 约定）

FIGURE_TITLE = "Cross-platform LiDAR and annotation alignment (THICV-R2V)"


# ==================================================================
# 几何
# ==================================================================
def euler2rotmat(roll, pitch, yaw):
    """欧拉角 → 旋转矩阵 (Rz @ Ry @ Rx)，与 ego_transform.py 一致。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def car_lidar_to_road(points, ego, lidar_z_extra=LIDAR_Z_EXTRA, yaw_offset_deg=0.0):
    """车端 LiDAR 点 (N,3) → 路侧坐标系。ego 为自车路侧标注 dict。

    yaw_offset_deg: 车端 LiDAR 相对车体的安装朝向修正（若点云整体转了 90°/180° 用它纠正）。
    """
    if len(points) == 0:
        return points
    if abs(yaw_offset_deg) > 1e-6:
        a = math.radians(yaw_offset_deg)
        Rz = np.array([[math.cos(a), -math.sin(a), 0],
                       [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        points = (Rz @ points.T).T
    R = euler2rotmat(ego.get("roll", 0.0), ego.get("pitch", 0.0), ego["yaw"])
    offset = np.array([0.0, 0.0, ego.get("height", 0.0) / 2.0 + lidar_z_extra])
    t = np.array([ego["x"], ego["y"], ego["z"]])
    return (R @ (points + offset).T).T + t


def box_corners_bev(obj):
    """3D 框俯视投影的 4 角点 (顺时针)。"""
    l, w = obj.get("length", 4.0), obj.get("width", 1.8)
    yaw = obj.get("yaw", 0.0)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = l / 2.0, w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    R = np.array([[c, -s], [s, c]])
    return (R @ local.T).T + np.array([obj["x"], obj["y"]])


# ==================================================================
# 点云读取
# ==================================================================
def _ascii_pcd(path):
    pts, started = [], False
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if started:
                p = line.split()
                if len(p) >= 3:
                    try:
                        pts.append([float(p[0]), float(p[1]), float(p[2])])
                    except ValueError:
                        pass
            elif line.startswith("DATA"):
                if "ascii" not in line:
                    raise ValueError("二进制 PCD，需要 open3d")
                started = True
    return np.array(pts) if pts else np.zeros((0, 3))


def load_pcd(path, max_pts=None):
    try:
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(path).points)
    except Exception:
        pts = _ascii_pcd(path)
    pts = pts[np.isfinite(pts).all(axis=1)] if len(pts) else pts
    if max_pts and len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
    return pts


# ==================================================================
# 帧/路径解析
# ==================================================================
def resolve_clip_dir(args):
    if args.clip_dir:
        return os.path.abspath(args.clip_dir)
    if not args.clip:
        raise SystemExit("请指定 --clip-dir 或 --dataset-root + --clip")
    matches = [m for m in sorted(glob.glob(os.path.join(args.dataset_root, args.clip + "*")))
               if os.path.isdir(m)]
    if not matches:
        raise SystemExit(f"在 {args.dataset_root} 找不到匹配 {args.clip}* 的 clip")
    return os.path.abspath(matches[0])


def load_carid_entry(carid_json, clip_name):
    if not os.path.exists(carid_json):
        return None
    with open(carid_json, "r") as f:
        data = json.load(f)
    prefix = clip_name.split("_")[0]
    for r in data.get("results", []):
        cn = r.get("clip_name", "")
        if cn == clip_name or cn.split("_")[0] == prefix:
            return r
    return None


def _ts_list(dirpath, unit):
    """返回 [(ts_ms, filepath)]；unit='ms' 整数毫秒，'s' 浮点秒。"""
    out = []
    for p in glob.glob(os.path.join(dirpath, "*.pcd")):
        stem = Path(p).stem
        try:
            ts_ms = int(stem) if unit == "ms" else int(round(float(stem) * 1000))
        except ValueError:
            continue
        out.append((ts_ms, p))
    return sorted(out)


def nearest(ts_list, target_ms):
    return min(ts_list, key=lambda kv: abs(kv[0] - target_ms)) if ts_list else None


def _ego_in_range(ego, ego_x_range, ego_y_range):
    if ego_x_range and not (ego_x_range[0] <= ego["x"] <= ego_x_range[1]):
        return False
    if ego_y_range and not (ego_y_range[0] <= ego["y"] <= ego_y_range[1]):
        return False
    return True


def _try_road_ts(road_ts, road_pcd_dir, road_lab_dir, car_list, ego_id,
                 ego_x_range=None, ego_y_range=None, target_gap=0):
    """若该路侧帧有 pcd+label、含自车 id（且自车 x/y 在指定范围内），返回配对，否则 None。

    gap = car_ts - road_ts（有符号，正=车端晚于路侧）；cost = |gap - target_gap|，用于选帧。
    """
    road_pcd = os.path.join(road_pcd_dir, f"{road_ts}.pcd")
    lab = os.path.join(road_lab_dir, f"{road_ts}.json")
    if not (os.path.exists(road_pcd) and os.path.exists(lab)):
        return None
    objs = json.load(open(lab)).get("object", [])
    ego = next((o for o in objs if o.get("id") == ego_id), None)
    if ego is None:
        return None
    if not _ego_in_range(ego, ego_x_range, ego_y_range):
        return None
    car_ts, car_pcd = nearest(car_list, road_ts + target_gap)
    signed = car_ts - road_ts
    return {"road_ts": road_ts, "road_pcd": road_pcd, "labels": objs, "ego": ego,
            "car_ts": car_ts, "car_pcd": car_pcd, "gap": signed,
            "cost": abs(signed - target_gap)}


def choose_frame(args, road_list, road_pcd_dir, road_lab_dir, car_list, ego_id,
                 visualize_ts, ego_x_range=None, ego_y_range=None, target_gap=0):
    """选路侧/车端配对帧。

    --road-time: 用指定帧；--anchor visualize: 锚定 visualize_roadtime；
    默认 (best): 遍历所有含自车 id（且自车 x/y 在范围内）的路侧帧，选 gap 最接近 target_gap 的配对。
    """
    if args.road_time:
        f = _try_road_ts(int(args.road_time), road_pcd_dir, road_lab_dir,
                         car_list, ego_id, target_gap=target_gap)
        if f:
            return f
        print("[WARN] 指定的 --road-time 无效（缺 pcd/label 或无自车），改用最优搜索")

    if args.anchor == "visualize" and visualize_ts:
        for road_ts, _ in sorted(road_list, key=lambda kv: abs(kv[0] - visualize_ts)):
            f = _try_road_ts(road_ts, road_pcd_dir, road_lab_dir, car_list, ego_id,
                             ego_x_range, ego_y_range, target_gap)
            if f:
                return f

    best = None
    for road_ts, _ in road_list:
        f = _try_road_ts(road_ts, road_pcd_dir, road_lab_dir, car_list, ego_id,
                         ego_x_range, ego_y_range, target_gap)
        if f and (best is None or f["cost"] < best["cost"]):
            best = f
            if best["cost"] == 0:
                break
    return best


# ==================================================================
# 主流程
# ==================================================================
# ==================================================================
# 跨 clip 搜索：找全局时间差最小的配对
# ==================================================================
def clip_best_gap(clip_dir, carid_json, ego_x_range=None, ego_y_range=None,
                  target_gap=0):
    """单 clip 中 gap 最接近 target_gap 的配对（自车 x/y 须在范围内）。返回 dict 或 None。"""
    import bisect
    name = os.path.basename(os.path.normpath(clip_dir))
    road_pcd_dir = os.path.join(clip_dir, "road", "lidar", "merged_pcd")
    road_lab_dir = os.path.join(clip_dir, "road_labels", "interpolation_labels")
    car_pcd_dir = os.path.join(clip_dir, "car", "pcds", "main")
    if not all(os.path.isdir(d) for d in (road_pcd_dir, road_lab_dir, car_pcd_dir)):
        return None
    entry = load_carid_entry(carid_json, name)
    if entry is None:
        return None
    ego_id = entry["nearest_carid"]
    road_list = _ts_list(road_pcd_dir, "ms")
    car_list = _ts_list(car_pcd_dir, "s")
    if not road_list or not car_list:
        return None
    car_ts = sorted(t for t, _ in car_list)

    def _best_car(rts):
        """返回离 (rts+target_gap) 最近的车端帧 -> (cost, signed_gap, car_ts)。"""
        t = rts + target_gap
        i = bisect.bisect_left(car_ts, t)
        best = None
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(car_ts):
                c = abs(car_ts[j] - t)
                if best is None or c < best[0]:
                    best = (c, car_ts[j] - rts, car_ts[j])
        return best

    cand = sorted(((bc[0], bc[1], bc[2], rts) for rts, _ in road_list
                   if (bc := _best_car(rts)) is not None), key=lambda kv: kv[0])
    # 按 cost(到 target_gap 的距离) 从小到大，取首个"含自车 id 且自车 x/y 在范围内"的帧
    for cost, signed, cts, rts in cand:
        lab = os.path.join(road_lab_dir, f"{rts}.json")
        if not os.path.exists(lab):
            continue
        try:
            objs = json.load(open(lab)).get("object", [])
        except Exception:
            continue
        ego = next((o for o in objs if o.get("id") == ego_id), None)
        if ego is None:
            continue
        if not _ego_in_range(ego, ego_x_range, ego_y_range):
            continue
        return {"clip": name, "clip_dir": clip_dir, "gap": signed, "cost": cost,
                "road_ts": rts, "car_ts": cts, "ego_id": ego_id,
                "ego_x": round(ego["x"], 1), "ego_y": round(ego["y"], 1)}
    return None


def load_complexity_top(csv_path, n):
    """读复杂度排名 CSV，返回 (top-N clip 名集合, {clip: score})。CSV 缺失返回 (None, {})。"""
    import csv
    if not csv_path or not os.path.exists(csv_path):
        return None, {}
    names, scores = [], {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            names.append(row["clip"])
            try:
                scores[row["clip"]] = float(row.get("score", 0) or 0)
            except ValueError:
                scores[row["clip"]] = 0.0
    top = set(names[:n]) if n and n > 0 else set(names)
    return top, scores


def scan_all_clips(args):
    """优先在高复杂度 clip 中、自车 x 在范围内、挑时间差最小的配对。"""
    import concurrent.futures as cf
    ego_x_range = (args.ego_x_min, args.ego_x_max)
    ego_y_range = (args.ego_y_min, args.ego_y_max)
    clip_dirs = [os.path.abspath(p) for p in sorted(glob.glob(os.path.join(args.dataset_root, "*")))
                 if os.path.isdir(os.path.join(p, "road", "lidar", "merged_pcd"))]
    if not clip_dirs:
        raise SystemExit(f"{args.dataset_root} 下没发现含 road/lidar/merged_pcd 的 clip")

    top, scores = load_complexity_top(args.complexity_csv, args.top_complex)
    if top is not None:
        kept = [d for d in clip_dirs if os.path.basename(d) in top]
        if kept:
            clip_dirs = kept
            print(f"[INFO] 限定在复杂度 Top-{args.top_complex} 的 {len(clip_dirs)} 个 clip 中搜索")
        else:
            print(f"[WARN] 复杂度 CSV 的 clip 名与数据集对不上，改在全部 clip 中搜索")
    else:
        print(f"[WARN] 未找到复杂度 CSV（{args.complexity_csv}），在全部 clip 中搜索")
    print(f"[INFO] 约束: 自车 x ∈ [{args.ego_x_min}, {args.ego_x_max}], "
          f"y ∈ [{args.ego_y_min}, {args.ego_y_max}]；目标时间差 {args.target_gap} ms；"
          f"共 {len(clip_dirs)} 个候选")

    workers = args.workers or min(64, os.cpu_count() or 8)
    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(clip_best_gap, d, args.carid_json, ego_x_range,
                          ego_y_range, args.target_gap): d for d in clip_dirs}
        for fut in cf.as_completed(futs):
            r = fut.result()
            if r:
                r["score"] = scores.get(r["clip"], 0.0)
                results.append(r)
    if not results:
        raise SystemExit("候选 clip 里没有满足(含自车 id + 自车 x/y 在范围内)的帧；"
                         "可放宽 --ego-x-min/max 或 --ego-y-min/max")
    results.sort(key=lambda r: r["cost"])
    print(f"\n{'='*76}\n候选 clip：自车 x/y 在范围内的最小时间差（按 gap 升序）\n{'='*76}")
    print(f"{'#':>3}  {'clip':<34}{'score':>6}{'gap(ms)':>8}{'ego_x':>8}{'ego_y':>8}")
    for i, r in enumerate(results[:15], 1):
        print(f"{i:>3}  {r['clip']:<34}{r['score']:>6.1f}{r['gap']:>8}"
              f"{r['ego_x']:>8}{r['ego_y']:>8}")
    best = results[0]
    print(f"\n[INFO] 选中: {best['clip']} (score={best['score']:.1f}) "
          f"gap={best['gap']} ms  road_ts={best['road_ts']} "
          f"ego=({best['ego_x']},{best['ego_y']})")
    return best


def build_frame(args):
    clip_dir = resolve_clip_dir(args)
    clip_name = os.path.basename(clip_dir)
    print(f"[INFO] clip: {clip_dir}")

    road_pcd_dir = os.path.join(clip_dir, "road", "lidar", "merged_pcd")
    road_lab_dir = os.path.join(clip_dir, "road_labels", "interpolation_labels")
    car_pcd_dir = os.path.join(clip_dir, "car", "pcds", "main")
    for d in (road_pcd_dir, road_lab_dir, car_pcd_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"缺少目录: {d}")

    entry = load_carid_entry(args.carid_json, clip_name)
    if entry is None:
        raise SystemExit(f"carid.json 里找不到 {clip_name} 的记录")
    ego_id = entry["nearest_carid"]
    print(f"[INFO] 自车路侧 id = {ego_id} ({entry.get('nearest_label')})")

    road_list = _ts_list(road_pcd_dir, "ms")
    car_list = _ts_list(car_pcd_dir, "s")
    if not road_list or not car_list:
        raise SystemExit("路侧或车端点云为空")

    visualize_ts = int(entry.get("visualize_roadtime", 0) or 0)
    ego_x_range = (args.ego_x_min, args.ego_x_max)
    ego_y_range = (args.ego_y_min, args.ego_y_max)
    frame = choose_frame(args, road_list, road_pcd_dir, road_lab_dir, car_list,
                         ego_id, visualize_ts, ego_x_range, ego_y_range,
                         args.target_gap)
    if frame is None:
        raise SystemExit(f"找不到含自车 id={ego_id} 且有点云的路侧帧")
    road_ts, car_ts = frame["road_ts"], frame["car_ts"]
    labels, ego = frame["labels"], frame["ego"]
    print(f"[INFO] 路侧帧 ts = {road_ts} | 车端帧 ts = {car_ts} "
          f"(Δ={frame['gap']} ms, anchor={args.anchor})")

    road_pts = load_pcd(frame["road_pcd"], args.max_points)
    car_pts_raw = load_pcd(frame["car_pcd"], args.max_points)
    car_pts = car_lidar_to_road(car_pts_raw, ego, args.lidar_z_extra,
                                args.car_yaw_offset)
    print(f"[INFO] 路侧点 {len(road_pts)} | 车端点 {len(car_pts)} | 标注 {len(labels)} 个")

    return clip_name, road_pts, car_pts, labels, ego


def _render(clip_name, road_pts, car_pts, labels, ego, args,
            show_road, show_car, show_annot, suffix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, FancyArrow

    try:
        import matplotlib.font_manager as fm
        avail = {f.name for f in fm.fontManager.ttflist}
        serif = [n for n in ("Times New Roman", "Nimbus Roman", "Liberation Serif",
                             "DejaVu Serif") if n in avail] or ["serif"]
        plt.rcParams.update({"font.family": "serif", "font.serif": serif,
                             "mathtext.fontset": "stix"})
    except Exception:
        pass

    swap = args.swap_xy
    sc = args.point_scale
    clean = getattr(args, "points_only", False)
    xlim, ylim = tuple(args.xlim), tuple(args.ylim)  # 始终是世界坐标

    def _crop(p):
        if len(p) == 0:
            return p
        m = ((p[:, 0] >= xlim[0]) & (p[:, 0] <= xlim[1]) &
             (p[:, 1] >= ylim[0]) & (p[:, 1] <= ylim[1]))
        return p[m]

    def P(wx, wy):  # 世界 -> 画布；swap 时世界 Y 画到水平、世界 X 画到竖直
        return (wy, wx) if swap else (wx, wy)

    road_pts, car_pts = _crop(road_pts), _crop(car_pts)
    disp_xlim, disp_ylim = (ylim, xlim) if swap else (xlim, ylim)
    xlab, ylab = ("Y (m)", "X (m)") if swap else ("X (m)", "Y (m)")
    w, h = disp_xlim[1] - disp_xlim[0], disp_ylim[1] - disp_ylim[0]
    fig, ax = plt.subplots(figsize=(13.5, 13.5 * h / w + 0.6))

    if show_road:
        rx, ry = P(road_pts[:, 0], road_pts[:, 1])
        ax.scatter(rx, ry, s=0.25 * sc, c="#1f77b4", alpha=0.55,
                   linewidths=0, rasterized=True, label="Roadside LiDAR (merged)")
    if show_car:
        cx, cy = P(car_pts[:, 0], car_pts[:, 1])
        ax.scatter(cx, cy, s=0.5 * sc, c="#d62728", alpha=0.85,
                   linewidths=0, rasterized=True, label="Vehicle LiDAR (ego, projected)")

    def _poly_disp(corners):
        return [P(px, py) for px, py in corners]

    if show_annot and not clean:
        for o in labels:
            if o.get("id") == ego.get("id"):
                continue
            ax.add_patch(Polygon(_poly_disp(box_corners_bev(o)), closed=True,
                                 fill=False, edgecolor="#111111", lw=1.2, zorder=5))
        ax.plot([], [], "-", color="#111111", lw=1.2, label="Roadside 3D annotations")

    # 自车框：绿色半透明填充 + 粗边 + 朝向箭头 + 标注（points-only 时不画）
    if not clean:
        ax.add_patch(Polygon(_poly_disp(box_corners_bev(ego)), closed=True,
                             facecolor="#00d050", alpha=0.45, edgecolor="#007a30",
                             lw=2.8, zorder=10))
        exw, eyw = ego["x"], ego["y"]
        epx, epy = P(exw, eyw)
        yaw = ego.get("yaw", 0.0)
        al = max(ego.get("length", 4.0), 4.0)
        adx, ady = P(exw + al * math.cos(yaw), eyw + al * math.sin(yaw))
        ax.add_patch(FancyArrow(epx, epy, adx - epx, ady - epy, width=0.5,
                                head_width=3.0, head_length=3.0,
                                length_includes_head=True, color="#007a30", zorder=11))
        ax.annotate("Ego", (epx, epy), textcoords="offset points", xytext=(6, 6),
                    fontsize=12, fontweight="bold", color="#007a30", zorder=12)
        ax.plot([], [], "-", color="#00b050", lw=2.8, label="Ego vehicle")

    ax.set_xlim(*disp_xlim)
    ax.set_ylim(*disp_ylim)
    ax.set_aspect("equal")
    if clean:
        ax.axis("off")
    else:
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(FIGURE_TITLE, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.legend(loc="upper right", fontsize=11, framealpha=0.95, markerscale=12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = clip_name.split("_")[0]
    png = OUTPUT_DIR / f"cross_platform_bev_{tag}{suffix}.png"
    pdf = OUTPUT_DIR / f"cross_platform_bev_{tag}{suffix}.pdf"
    pad = 0 if clean else 0.1
    fig.savefig(str(png), dpi=300, bbox_inches="tight", pad_inches=pad)
    fig.savefig(str(pdf), bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return str(png)


def draw(clip_name, road_pts, car_pts, labels, ego, args):
    outs = [_render(clip_name, road_pts, car_pts, labels, ego, args,
                    show_road=True, show_car=True, show_annot=True, suffix="")]
    if args.separate:
        outs.append(_render(clip_name, road_pts, car_pts, labels, ego, args,
                            show_road=True, show_car=False, show_annot=True,
                            suffix="_road"))
        outs.append(_render(clip_name, road_pts, car_pts, labels, ego, args,
                            show_road=False, show_car=True, show_annot=False,
                            suffix="_car"))
    print("[OK] 已保存:")
    for p in outs:
        print(f"  {p}")
    return outs


def main():
    ap = argparse.ArgumentParser(description="车路协同 BEV 对齐图")
    ap.add_argument("--clip-dir", type=str, default=None)
    ap.add_argument("--dataset-root", type=str, default=DEFAULT_ROOT)
    ap.add_argument("--clip", type=str, default=None, help="clip 名/前缀，如 002")
    ap.add_argument("--carid-json", type=str, default=DEFAULT_CARID)
    ap.add_argument("--road-time", type=str, default=None, help="指定路侧帧 ts(ms)")
    ap.add_argument("--anchor", choices=["best", "visualize"], default="best",
                    help="选帧策略：best=gap 最接近 target(默认)；visualize=锚定 carid 的 visualize_roadtime")
    ap.add_argument("--target-gap", type=float, default=0.0,
                    help="期望的车端-路侧时间差 ms(有符号,正=车端晚于路侧)；"
                         "选帧时挑 gap 最接近此值的(如 5 表示车比路晚约5ms)")
    ap.add_argument("--xlim", type=float, nargs=2, default=list(DEFAULT_XLIM))
    ap.add_argument("--ylim", type=float, nargs=2, default=list(DEFAULT_YLIM))
    ap.add_argument("--lidar-z-extra", type=float, default=LIDAR_Z_EXTRA,
                    help="车端 LiDAR 相对 bbox 顶部的额外高度偏移（默认0.25）")
    ap.add_argument("--car-yaw-offset", type=float, default=0.0,
                    help="车端 LiDAR 安装朝向修正(度)；若车端点云整体转了角度用它纠正")
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--scan-all", action="store_true",
                    help="优先在高复杂度 clip 中、自车 x 在范围内、挑时间差最小的渲染")
    ap.add_argument("--workers", type=int, default=None,
                    help="--scan-all 并行进程数（默认 min(64, CPU核数)）")
    ap.add_argument("--complexity-csv", type=str,
                    default=str(OUTPUT_DIR / "complexity_ranking.csv"),
                    help="复杂度排名 CSV（traffic_complexity_figure.py --scan 产出）")
    ap.add_argument("--top-complex", type=int, default=10,
                    help="只在复杂度前 N 的 clip 中搜索（默认10，<=0 表示不限）")
    ap.add_argument("--ego-x-min", type=float, default=-100.0,
                    help="自车中心 x 下界（默认 -100）")
    ap.add_argument("--ego-x-max", type=float, default=-20.0,
                    help="自车中心 x 上界（默认 -20）")
    ap.add_argument("--ego-y-min", type=float, default=-30.0,
                    help="自车中心 y 下界（默认 -30）")
    ap.add_argument("--ego-y-max", type=float, default=10.0,
                    help="自车中心 y 上界（默认 10）")
    ap.add_argument("--swap-xy", action="store_true",
                    help="转置显示：世界 Y 画到水平轴、世界 X 画到竖直轴"
                         "（让自车沿水平方向行驶）；范围/区域随之对调")
    ap.add_argument("--separate", action="store_true",
                    help="除合并图外，再分别输出只含路侧(_road)和只含车端(_car)的图")
    ap.add_argument("--points-only", action="store_true",
                    help="只保留雷达点：不画坐标轴/标题/图例/标注框，输出纯净点云图")
    ap.add_argument("--point-scale", type=float, default=1.5,
                    help="雷达点大小倍数（默认 1.5）")
    args = ap.parse_args()

    if args.scan_all:
        best = scan_all_clips(args)
        args.clip_dir = best["clip_dir"]
        args.road_time = str(best["road_ts"])

    clip_name, road_pts, car_pts, labels, ego = build_frame(args)
    draw(clip_name, road_pts, car_pts, labels, ego, args)


if __name__ == "__main__":
    main()
