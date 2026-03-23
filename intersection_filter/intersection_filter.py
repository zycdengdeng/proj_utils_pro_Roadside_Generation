#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路口区域筛选工具
根据采集车进出路口的位置定义矩形区域，筛选区域内车辆轨迹，生成29帧片段

Usage:
    python intersection_filter.py --step define    # Step 1: 定义区域并BEV可视化
    python intersection_filter.py --step filter    # Step 2: 筛选车辆并分段
    python intersection_filter.py --step all       # 一步完成
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common_utils import DATASET_ROOT, find_scene_path, get_scene_paths

# ============================================================
# 配置参数
# ============================================================

SEGMENT_LENGTH = 29  # 每个视频片段的帧数

# 需要追踪的车辆类型（排除静态物体）
VEHICLE_LABELS = {"Car", "Suv", "Truck", "Bus", "Van"}

# 四个方向的参考车辆信息，用于定义路口区域
# scene_prefix: 场景ID前缀
# vehicle_id: 标注中的车辆id
# entry_ts: 进入路口时间戳（毫秒）
# exit_ts: 离开路口时间戳（毫秒）
REFERENCE_VEHICLES = [
    {
        "scene_prefix": "002",
        "vehicle_id": 29,
        "entry_ts": 1742877436322,
        "exit_ts": 1742877441522,
        "direction": "W2E",
        "camera": "pinhole2",
        "desc": "西向东",
    },
    {
        "scene_prefix": "003",
        "vehicle_id": 45,
        "entry_ts": 1742877823367,
        "exit_ts": 1742877828935,
        "direction": "E2W",
        "camera": "pinhole0",
        "desc": "东向西",
    },
    {
        "scene_prefix": "014",
        "vehicle_id": 6,
        "entry_ts": 1742883999258,
        "exit_ts": 1742884002516,
        "direction": "N2S",
        "camera": "pinhole3",
        "desc": "北向南",
    },
    {
        "scene_prefix": "015",
        "vehicle_id": 55,
        "entry_ts": 1742884382914,
        "exit_ts": 1742884385763,
        "direction": "S2N",
        "camera": "pinhole1",
        "desc": "南向北",
    },
]

# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ============================================================
# 标注文件读取
# ============================================================

def load_label_file(label_path):
    """读取单个标注文件"""
    with open(label_path, 'r') as f:
        data = json.load(f)
    return data


def get_label_files(scene_prefix):
    """获取某个场景的所有标注文件路径，按时间戳排序"""
    paths = get_scene_paths(scene_prefix)
    if not paths:
        print(f"[ERROR] 找不到场景 {scene_prefix}")
        return []

    label_dir = paths['roadside_labels']
    if not os.path.isdir(label_dir):
        print(f"[ERROR] 标注目录不存在: {label_dir}")
        return []

    files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    return files


def find_closest_label(label_files, target_ts):
    """找到最接近目标时间戳的标注文件"""
    best_file = None
    best_diff = float('inf')

    for f in label_files:
        # 文件名格式: 1742877031036.json
        ts_str = Path(f).stem
        try:
            ts = int(ts_str)
        except ValueError:
            continue

        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_file = f

    return best_file, best_diff


def find_vehicle_in_label(label_data, vehicle_id):
    """在标注数据中查找指定ID的车辆，返回其(x, y)位置"""
    for obj in label_data.get("object", []):
        if obj.get("id") == vehicle_id:
            return obj["x"], obj["y"]
    return None


# ============================================================
# Step 1: 定义路口矩形区域
# ============================================================

def define_intersection_region():
    """
    从四个方向的参考车辆位置，定义路口矩形区域

    Returns:
        region: dict with keys x_min, x_max, y_min, y_max
        ref_positions: list of (x, y, direction, type) for visualization
    """
    print("\n" + "=" * 60)
    print("Step 1: 定义路口矩形区域")
    print("=" * 60)

    ref_positions = []  # (x, y, direction, entry/exit)
    all_x = []
    all_y = []

    for ref in REFERENCE_VEHICLES:
        scene_prefix = ref["scene_prefix"]
        vid = ref["vehicle_id"]
        entry_ts = ref["entry_ts"]
        exit_ts = ref["exit_ts"]
        direction = ref["direction"]
        desc = ref["desc"]

        print(f"\n--- {desc} ({direction}): 场景{scene_prefix}, 车辆ID={vid} ---")

        label_files = get_label_files(scene_prefix)
        if not label_files:
            print(f"  [SKIP] 无法获取标注文件")
            continue

        # 查找进路口时刻的位置
        entry_file, entry_diff = find_closest_label(label_files, entry_ts)
        if entry_file:
            entry_data = load_label_file(entry_file)
            entry_pos = find_vehicle_in_label(entry_data, vid)
            if entry_pos:
                print(f"  进路口: ts={Path(entry_file).stem} (diff={entry_diff}ms), "
                      f"pos=({entry_pos[0]:.2f}, {entry_pos[1]:.2f})")
                ref_positions.append((entry_pos[0], entry_pos[1], direction, "entry"))
                all_x.append(entry_pos[0])
                all_y.append(entry_pos[1])
            else:
                print(f"  [WARN] 进路口标注中未找到车辆ID={vid}")
        else:
            print(f"  [WARN] 未找到进路口时间戳对应的标注文件")

        # 查找出路口时刻的位置
        exit_file, exit_diff = find_closest_label(label_files, exit_ts)
        if exit_file:
            exit_data = load_label_file(exit_file)
            exit_pos = find_vehicle_in_label(exit_data, vid)
            if exit_pos:
                print(f"  出路口: ts={Path(exit_file).stem} (diff={exit_diff}ms), "
                      f"pos=({exit_pos[0]:.2f}, {exit_pos[1]:.2f})")
                ref_positions.append((exit_pos[0], exit_pos[1], direction, "exit"))
                all_x.append(exit_pos[0])
                all_y.append(exit_pos[1])
            else:
                print(f"  [WARN] 出路口标注中未找到车辆ID={vid}")
        else:
            print(f"  [WARN] 未找到出路口时间戳对应的标注文件")

    if len(all_x) < 4:
        print(f"\n[ERROR] 参考点不足（仅{len(all_x)}个），无法定义区域")
        return None, ref_positions

    # 从所有进出路口位置计算矩形区域
    region = {
        "x_min": min(all_x),
        "x_max": max(all_x),
        "y_min": min(all_y),
        "y_max": max(all_y),
    }

    print(f"\n{'='*60}")
    print(f"路口矩形区域:")
    print(f"  X: [{region['x_min']:.2f}, {region['x_max']:.2f}]  "
          f"宽度: {region['x_max'] - region['x_min']:.2f}m")
    print(f"  Y: [{region['y_min']:.2f}, {region['y_max']:.2f}]  "
          f"高度: {region['y_max'] - region['y_min']:.2f}m")
    print(f"{'='*60}")

    # 保存区域定义
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    region_file = OUTPUT_DIR / "intersection_region.json"
    save_data = {
        "region": region,
        "reference_positions": [
            {"x": p[0], "y": p[1], "direction": p[2], "type": p[3]}
            for p in ref_positions
        ],
        "reference_vehicles": REFERENCE_VEHICLES,
    }
    with open(region_file, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n区域定义已保存: {region_file}")

    return region, ref_positions


# ============================================================
# Step 2: 筛选区域内车辆并生成29帧片段
# ============================================================

def is_in_region(x, y, region):
    """判断点是否在矩形区域内（BEV俯视，不考虑z轴）"""
    return (region["x_min"] <= x <= region["x_max"] and
            region["y_min"] <= y <= region["y_max"])


def track_vehicles_in_scene(scene_prefix, region):
    """
    在一个场景中追踪所有车辆，找出在区域内的帧

    Returns:
        tracks: dict of {vehicle_id: [(timestamp, x, y, label_file), ...]}
                只包含在区域内的帧
    """
    label_files = get_label_files(scene_prefix)
    if not label_files:
        return {}

    # vehicle_id -> [(timestamp_ms, x, y, label_file), ...]
    tracks = defaultdict(list)

    for lf in label_files:
        ts_str = Path(lf).stem
        try:
            ts = int(ts_str)
        except ValueError:
            continue

        data = load_label_file(lf)
        for obj in data.get("object", []):
            label = obj.get("label", "")
            if label not in VEHICLE_LABELS:
                continue

            vid = obj["id"]
            x, y = obj["x"], obj["y"]

            if is_in_region(x, y, region):
                tracks[vid].append((ts, x, y, lf))

    # 按时间戳排序
    for vid in tracks:
        tracks[vid].sort(key=lambda t: t[0])

    return tracks


def segment_track(frames, segment_length=SEGMENT_LENGTH):
    """
    将一条轨迹按segment_length分段

    规则：
    - 帧数 < segment_length: 丢弃
    - segment_length <= 帧数 < 2*segment_length: 取前segment_length帧
    - 2*segment_length <= 帧数 < 3*segment_length: 取前2*segment_length帧，分成2段
    - 以此类推：n = 帧数 // segment_length 段，每段segment_length帧

    Returns:
        list of lists, each inner list is a segment of frames
    """
    n_frames = len(frames)
    if n_frames < segment_length:
        return []

    n_segments = n_frames // segment_length
    segments = []
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(frames[start:end])

    return segments


def filter_vehicles_in_region(region, scene_prefixes=None):
    """
    筛选所有场景中区域内的车辆，并生成29帧片段

    Args:
        region: 矩形区域 dict
        scene_prefixes: 要处理的场景列表，None则处理所有

    Returns:
        all_segments: list of dicts, each containing segment info
    """
    print("\n" + "=" * 60)
    print("Step 2: 筛选区域内车辆并生成29帧片段")
    print("=" * 60)

    if scene_prefixes is None:
        # 自动发现所有场景
        scene_dirs = sorted(glob.glob(os.path.join(DATASET_ROOT, "[0-9][0-9][0-9]_*")))
        scene_prefixes = [os.path.basename(d).split("_")[0] for d in scene_dirs]
        print(f"发现 {len(scene_prefixes)} 个场景")

    all_segments = []
    all_tracks_for_viz = {}  # scene -> {vid: [(ts, x, y), ...]}

    for idx, sp in enumerate(scene_prefixes):
        print(f"\n[{idx+1}/{len(scene_prefixes)}] 处理场景 {sp}...")

        tracks = track_vehicles_in_scene(sp, region)

        if not tracks:
            print(f"  无区域内车辆")
            continue

        scene_tracks_viz = {}
        for vid, frames in tracks.items():
            n_frames = len(frames)
            segments = segment_track(frames, SEGMENT_LENGTH)

            if segments:
                print(f"  车辆 {vid}: {n_frames} 帧在区域内 -> {len(segments)} 个{SEGMENT_LENGTH}帧片段")
                for seg_idx, seg in enumerate(segments):
                    ts_start = seg[0][0]
                    ts_end = seg[-1][0]
                    all_segments.append({
                        "scene": sp,
                        "vehicle_id": vid,
                        "segment_index": seg_idx,
                        "n_frames": len(seg),
                        "ts_start": ts_start,
                        "ts_end": ts_end,
                        "timestamps": [f[0] for f in seg],
                        "label_files": [f[3] for f in seg],
                    })
            else:
                print(f"  车辆 {vid}: {n_frames} 帧在区域内 (不足{SEGMENT_LENGTH}帧，丢弃)")

            # 保存完整轨迹用于可视化（包括不足29帧的）
            scene_tracks_viz[vid] = [(f[0], f[1], f[2]) for f in frames]

        if scene_tracks_viz:
            all_tracks_for_viz[sp] = scene_tracks_viz

    # 汇总
    print(f"\n{'='*60}")
    print(f"筛选结果汇总:")
    print(f"  处理场景数: {len(scene_prefixes)}")
    print(f"  有效片段数: {len(all_segments)}")
    if all_segments:
        scenes_with_segments = set(s["scene"] for s in all_segments)
        print(f"  涉及场景数: {len(scenes_with_segments)}")
        for scene in sorted(scenes_with_segments):
            scene_segs = [s for s in all_segments if s["scene"] == scene]
            vids = set(s["vehicle_id"] for s in scene_segs)
            print(f"    场景 {scene}: {len(scene_segs)} 个片段, {len(vids)} 辆车")
    print(f"{'='*60}")

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    segments_file = OUTPUT_DIR / "filtered_segments.json"
    with open(segments_file, 'w') as f:
        json.dump(all_segments, f, indent=2, ensure_ascii=False)
    print(f"\n片段列表已保存: {segments_file}")

    # 保存轨迹用于可视化
    tracks_file = OUTPUT_DIR / "tracks_in_region.json"
    # 转换为可序列化格式
    tracks_serializable = {}
    for scene, vids in all_tracks_for_viz.items():
        tracks_serializable[scene] = {}
        for vid, points in vids.items():
            tracks_serializable[scene][str(vid)] = [
                {"ts": p[0], "x": p[1], "y": p[2]} for p in points
            ]
    with open(tracks_file, 'w') as f:
        json.dump(tracks_serializable, f, indent=2)
    print(f"轨迹数据已保存: {tracks_file}")

    return all_segments, all_tracks_for_viz


# ============================================================
# Step 3: BEV可视化
# ============================================================

def load_pcd_points(pcd_path):
    """
    简单PCD文件读取（ASCII/binary_compressed），返回xyz坐标
    优先使用open3d，回退到手动解析
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        return points
    except ImportError:
        pass

    # 手动解析ASCII格式PCD
    points = []
    data_start = False
    with open(pcd_path, 'r', errors='ignore') as f:
        for line in f:
            if data_start:
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
            elif line.strip().startswith("DATA"):
                data_start = True

    return np.array(points) if points else np.zeros((0, 3))


def visualize_bev(region, ref_positions, tracks_in_region=None,
                  pcd_scene_prefix=None, pcd_timestamp=None):
    """
    BEV俯视图可视化：点云背景 + 矩形区域 + 车辆轨迹

    Args:
        region: 矩形区域 dict
        ref_positions: 参考车辆位置列表
        tracks_in_region: {scene: {vid: [(ts, x, y), ...]}}
        pcd_scene_prefix: 用哪个场景的点云做背景
        pcd_timestamp: 用哪个时间戳的点云（如果None则自动选中间帧）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    print("\n" + "=" * 60)
    print("Step 3: BEV可视化")
    print("=" * 60)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))

    # --- 加载点云背景 ---
    if pcd_scene_prefix:
        paths = get_scene_paths(pcd_scene_prefix)
        if paths:
            pcd_dir = paths['pcd']
            pcd_files = sorted(glob.glob(os.path.join(pcd_dir, "*.pcd")))
            if pcd_files:
                if pcd_timestamp:
                    # 找最接近的
                    best_pcd = None
                    best_diff = float('inf')
                    for pf in pcd_files:
                        ts_str = Path(pf).stem
                        try:
                            ts = int(ts_str)
                        except ValueError:
                            continue
                        diff = abs(ts - pcd_timestamp)
                        if diff < best_diff:
                            best_diff = diff
                            best_pcd = pf
                else:
                    # 选中间帧
                    best_pcd = pcd_files[len(pcd_files) // 2]

                if best_pcd:
                    print(f"  加载点云: {best_pcd}")
                    points = load_pcd_points(best_pcd)
                    if len(points) > 0:
                        # BEV: 只取x, y，用灰色散点表示
                        ax.scatter(points[:, 0], points[:, 1],
                                   c='gray', s=0.1, alpha=0.3, rasterized=True)
                        print(f"  点云点数: {len(points)}")

    # --- 画矩形区域 ---
    rx = region["x_min"]
    ry = region["y_min"]
    rw = region["x_max"] - region["x_min"]
    rh = region["y_max"] - region["y_min"]
    rect = patches.Rectangle((rx, ry), rw, rh,
                              linewidth=3, edgecolor='red',
                              facecolor='red', alpha=0.1,
                              label='Intersection Region')
    ax.add_patch(rect)

    # --- 画参考车辆位置 ---
    direction_colors = {
        "W2E": "blue",
        "E2W": "green",
        "N2S": "orange",
        "S2N": "purple",
    }
    direction_descs = {
        "W2E": "西→东",
        "E2W": "东→西",
        "N2S": "北→南",
        "S2N": "南→北",
    }

    for x, y, direction, entry_exit in ref_positions:
        color = direction_colors.get(direction, "black")
        marker = "^" if entry_exit == "entry" else "v"
        size = 200
        label_text = f"{direction_descs.get(direction, direction)} {'进' if entry_exit == 'entry' else '出'}"
        ax.scatter(x, y, c=color, s=size, marker=marker, zorder=10,
                   edgecolors='black', linewidths=1)
        ax.annotate(label_text, (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=8,
                    fontproperties=None,  # 如果需要中文显示，设置字体
                    color=color, fontweight='bold')

    # --- 画车辆轨迹 ---
    if tracks_in_region:
        # 使用不同颜色区分不同场景/车辆
        color_map = plt.cm.get_cmap('tab20')
        track_idx = 0

        for scene, vids in tracks_in_region.items():
            for vid, points in vids.items():
                if not points:
                    continue
                xs = [p[1] for p in points]
                ys = [p[2] for p in points]
                color = color_map(track_idx % 20)
                ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.7)
                ax.scatter(xs[0], ys[0], c=[color], s=80, marker='o',
                           edgecolors='black', linewidths=0.5, zorder=5)
                ax.scatter(xs[-1], ys[-1], c=[color], s=80, marker='s',
                           edgecolors='black', linewidths=0.5, zorder=5)
                # 标注
                mid_idx = len(xs) // 2
                ax.annotate(f"S{scene}_V{vid}({len(points)}f)",
                            (xs[mid_idx], ys[mid_idx]),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=6, color=color, alpha=0.8)
                track_idx += 1

    # --- 图例和标注 ---
    # 手动添加图例
    legend_elements = [
        patches.Patch(facecolor='red', alpha=0.2, edgecolor='red', linewidth=2,
                      label=f'Region: x[{region["x_min"]:.1f},{region["x_max"]:.1f}] '
                            f'y[{region["y_min"]:.1f},{region["y_max"]:.1f}]'),
    ]
    for direction, color in direction_colors.items():
        desc = direction_descs.get(direction, direction)
        legend_elements.append(
            plt.Line2D([0], [0], marker='^', color=color, markersize=10,
                       linestyle='None', label=f'{desc} entry/exit')
        )

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # 设置坐标轴
    # 自动范围：以区域为中心，扩展一定比例
    margin_x = rw * 0.5
    margin_y = rh * 0.5
    margin = max(margin_x, margin_y, 30)  # 至少30m margin

    ax.set_xlim(region["x_min"] - margin, region["x_max"] + margin)
    ax.set_ylim(region["y_min"] - margin, region["y_max"] + margin)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('BEV Intersection Region Visualization')
    ax.grid(True, alpha=0.3)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "bev_intersection_region.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nBEV可视化已保存: {out_path}")

    return str(out_path)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="路口区域筛选工具")
    parser.add_argument("--step", type=str, default="all",
                        choices=["define", "filter", "all"],
                        help="执行步骤: define=定义区域, filter=筛选车辆, all=全部")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="要处理的场景列表（默认所有场景）")
    parser.add_argument("--pcd-scene", type=str, default=None,
                        help="BEV可视化使用的点云场景（默认使用第一个参考场景）")
    parser.add_argument("--segment-length", type=int, default=SEGMENT_LENGTH,
                        help=f"每个片段的帧数（默认{SEGMENT_LENGTH}）")
    args = parser.parse_args()

    global SEGMENT_LENGTH
    SEGMENT_LENGTH = args.segment_length

    region = None
    ref_positions = None
    tracks_in_region = None

    # Step 1: 定义区域
    if args.step in ("define", "all"):
        region, ref_positions = define_intersection_region()
        if region is None:
            print("[ERROR] 区域定义失败，退出")
            sys.exit(1)

    # Step 2: 筛选车辆
    if args.step in ("filter", "all"):
        # 如果只运行filter步骤，从文件加载区域定义
        if region is None:
            region_file = OUTPUT_DIR / "intersection_region.json"
            if not region_file.exists():
                print(f"[ERROR] 未找到区域定义文件: {region_file}")
                print("请先运行 --step define")
                sys.exit(1)
            with open(region_file, 'r') as f:
                saved = json.load(f)
            region = saved["region"]
            ref_positions = [
                (p["x"], p["y"], p["direction"], p["type"])
                for p in saved["reference_positions"]
            ]
            print(f"从文件加载区域定义: {region_file}")

        all_segments, tracks_in_region = filter_vehicles_in_region(
            region, scene_prefixes=args.scenes
        )

    # Step 3: BEV可视化
    if args.step in ("define", "all"):
        pcd_scene = args.pcd_scene or REFERENCE_VEHICLES[0]["scene_prefix"]
        visualize_bev(
            region, ref_positions,
            tracks_in_region=tracks_in_region,
            pcd_scene_prefix=pcd_scene,
        )

    print("\n[DONE] 完成")


if __name__ == "__main__":
    main()
