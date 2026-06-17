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

# 展示范围
DEFAULT_XLIM = (-140.0, 0.0)
DEFAULT_YLIM = (-50.0, 25.0)
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


def _try_road_ts(road_ts, road_pcd_dir, road_lab_dir, car_list, ego_id):
    """若该路侧帧有 pcd+label 且含自车 id，返回配对信息，否则 None。"""
    road_pcd = os.path.join(road_pcd_dir, f"{road_ts}.pcd")
    lab = os.path.join(road_lab_dir, f"{road_ts}.json")
    if not (os.path.exists(road_pcd) and os.path.exists(lab)):
        return None
    objs = json.load(open(lab)).get("object", [])
    ego = next((o for o in objs if o.get("id") == ego_id), None)
    if ego is None:
        return None
    car_ts, car_pcd = nearest(car_list, road_ts)
    return {"road_ts": road_ts, "road_pcd": road_pcd, "labels": objs, "ego": ego,
            "car_ts": car_ts, "car_pcd": car_pcd, "gap": abs(car_ts - road_ts)}


def choose_frame(args, road_list, road_pcd_dir, road_lab_dir, car_list, ego_id,
                 visualize_ts):
    """选路侧/车端配对帧。

    --road-time: 用指定帧；--anchor visualize: 锚定 visualize_roadtime；
    默认 (best): 遍历所有含自车 id 的路侧帧，选时间差最小的配对。
    """
    if args.road_time:
        f = _try_road_ts(int(args.road_time), road_pcd_dir, road_lab_dir,
                         car_list, ego_id)
        if f:
            return f
        print("[WARN] 指定的 --road-time 无效（缺 pcd/label 或无自车），改用最优搜索")

    if args.anchor == "visualize" and visualize_ts:
        # 按到 visualize_roadtime 的距离从近到远，取第一个有效帧
        for road_ts, _ in sorted(road_list, key=lambda kv: abs(kv[0] - visualize_ts)):
            f = _try_road_ts(road_ts, road_pcd_dir, road_lab_dir, car_list, ego_id)
            if f:
                return f

    # best：最小时间差
    best = None
    for road_ts, _ in road_list:
        f = _try_road_ts(road_ts, road_pcd_dir, road_lab_dir, car_list, ego_id)
        if f and (best is None or f["gap"] < best["gap"]):
            best = f
            if best["gap"] == 0:
                break
    return best


# ==================================================================
# 主流程
# ==================================================================
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
    frame = choose_frame(args, road_list, road_pcd_dir, road_lab_dir, car_list,
                         ego_id, visualize_ts)
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


def draw(clip_name, road_pts, car_pts, labels, ego, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    try:
        import matplotlib.font_manager as fm
        avail = {f.name for f in fm.fontManager.ttflist}
        serif = [n for n in ("Times New Roman", "Nimbus Roman", "Liberation Serif",
                             "DejaVu Serif") if n in avail] or ["serif"]
        plt.rcParams.update({"font.family": "serif", "font.serif": serif,
                             "mathtext.fontset": "stix"})
    except Exception:
        pass

    xlim, ylim = tuple(args.xlim), tuple(args.ylim)
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    fig, ax = plt.subplots(figsize=(13.5, 13.5 * h / w + 0.6))

    def _crop(p):
        if len(p) == 0:
            return p
        m = ((p[:, 0] >= xlim[0]) & (p[:, 0] <= xlim[1]) &
             (p[:, 1] >= ylim[0]) & (p[:, 1] <= ylim[1]))
        return p[m]

    road_pts, car_pts = _crop(road_pts), _crop(car_pts)

    ax.scatter(road_pts[:, 0], road_pts[:, 1], s=0.25, c="#1f77b4", alpha=0.55,
               linewidths=0, rasterized=True, label="Roadside LiDAR (merged)")
    ax.scatter(car_pts[:, 0], car_pts[:, 1], s=0.5, c="#d62728", alpha=0.85,
               linewidths=0, rasterized=True, label="Vehicle LiDAR (ego, projected)")

    # 3D 标注框
    for o in labels:
        if o.get("id") == ego.get("id"):
            continue
        poly = box_corners_bev(o)
        ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="#111111",
                             lw=1.2, zorder=5))
    ax.plot([], [], "-", color="#111111", lw=1.2, label="Roadside 3D annotations")

    # 自车框（绿色，避开与车端红点撞色）
    ego_poly = box_corners_bev(ego)
    ax.add_patch(Polygon(ego_poly, closed=True, fill=False, edgecolor="#00b050",
                         lw=2.4, zorder=6))
    ax.plot([], [], "-", color="#00b050", lw=2.4, label="Ego vehicle")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(FIGURE_TITLE, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, lw=0.5)
    leg = ax.legend(loc="upper right", fontsize=11, framealpha=0.95, markerscale=12)
    for h_ in leg.legend_handles:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = clip_name.split("_")[0]
    png = OUTPUT_DIR / f"cross_platform_bev_{tag}.png"
    pdf = OUTPUT_DIR / f"cross_platform_bev_{tag}.pdf"
    fig.savefig(str(png), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已保存:\n  {png}\n  {pdf}")
    return str(png), str(pdf)


def main():
    ap = argparse.ArgumentParser(description="车路协同 BEV 对齐图")
    ap.add_argument("--clip-dir", type=str, default=None)
    ap.add_argument("--dataset-root", type=str, default=DEFAULT_ROOT)
    ap.add_argument("--clip", type=str, default=None, help="clip 名/前缀，如 002")
    ap.add_argument("--carid-json", type=str, default=DEFAULT_CARID)
    ap.add_argument("--road-time", type=str, default=None, help="指定路侧帧 ts(ms)")
    ap.add_argument("--anchor", choices=["best", "visualize"], default="best",
                    help="选帧策略：best=时间差最小(默认)；visualize=锚定 carid 的 visualize_roadtime")
    ap.add_argument("--xlim", type=float, nargs=2, default=list(DEFAULT_XLIM))
    ap.add_argument("--ylim", type=float, nargs=2, default=list(DEFAULT_YLIM))
    ap.add_argument("--lidar-z-extra", type=float, default=LIDAR_Z_EXTRA,
                    help="车端 LiDAR 相对 bbox 顶部的额外高度偏移（默认0.25）")
    ap.add_argument("--car-yaw-offset", type=float, default=0.0,
                    help="车端 LiDAR 安装朝向修正(度)；若车端点云整体转了角度用它纠正")
    ap.add_argument("--max-points", type=int, default=400000)
    args = ap.parse_args()

    clip_name, road_pts, car_pts, labels, ego = build_frame(args)
    draw(clip_name, road_pts, car_pts, labels, ego, args)


if __name__ == "__main__":
    main()
