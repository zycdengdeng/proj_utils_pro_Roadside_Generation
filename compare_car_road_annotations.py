#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车端 vs 路端标注偏差分析

将路端标注（世界坐标系）通过 world2lidar 变换转到车端 LiDAR 坐标系，
与车端标注按位置就近匹配同一物体，计算中心点偏差。

用于为 BEVFormer 评测设计匹配阈值。

Usage:
    python compare_car_road_annotations.py
    python compare_car_road_annotations.py --clips 008 031 088
    python compare_car_road_annotations.py --max-match-dist 5.0
"""

import os
import sys
import json
import glob
import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2

# ============================================================
# 配置
# ============================================================

DATASET_ROOT = "/mnt/car_road_data_fix"
TRANSFORM_ROOT = "/mnt/zihanw/proj_utils_pro/transform_json"

# 时间戳匹配容差（秒）：路端和车端帧之间的最大时间差
TS_MATCH_TOLERANCE_SEC = 0.1

# 物体位置匹配距离上限（米）：超过此距离不认为是同一物体
DEFAULT_MAX_MATCH_DIST = 10.0

# 只比较这些类别
VEHICLE_LABELS = {"Car", "Suv", "Truck", "Bus", "Van",
                  "Pedestrian", "Bicycle", "Motorcycle",
                  "Non_motor_rider", "Motor_rider", "Tricycle"}


# ============================================================
# 数据加载
# ============================================================

def find_clip_dir(clip_num):
    """根据 clip 编号找到完整路径"""
    pattern = os.path.join(DATASET_ROOT, f"{clip_num}_*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_annotations(label_dir):
    """加载标注目录下所有 JSON，返回 {timestamp_sec: annotation_dict}"""
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return {}

    annotations = {}
    for f in sorted(label_dir.glob("*.json")):
        stem = f.stem
        try:
            # 车端格式: "1742879639.800415" (秒)
            # 路端格式: "1742879639783" (毫秒)
            if '.' in stem:
                ts_sec = float(stem)
            else:
                ts_sec = int(stem) / 1000.0
        except ValueError:
            continue

        with open(f, 'r') as fh:
            annotations[ts_sec] = json.load(fh)

    return annotations


def load_transforms(transform_path):
    """加载 world2lidar_transforms.json，返回 [(ts_sec, rotation, translation)]"""
    with open(transform_path, 'r') as f:
        data = json.load(f)

    transforms = []
    for item in data:
        ts = item['timestamp']
        rot = np.array(item['world2lidar']['rotation'])
        trans = np.array(item['world2lidar']['translation'])
        transforms.append((ts, rot, trans))

    return transforms


def find_nearest_timestamp(target_ts, ts_list, tolerance_sec):
    """在时间戳列表中找最近的，返回 (best_ts, diff) 或 (None, inf)"""
    best_ts = None
    best_diff = float('inf')
    for ts in ts_list:
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_ts = ts
    if best_ts is not None and best_diff <= tolerance_sec:
        return best_ts, best_diff
    return None, float('inf')


def find_nearest_transform(target_ts, transforms):
    """找最近的 world2lidar 变换"""
    best = None
    best_diff = float('inf')
    for ts, rot, trans in transforms:
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = (rot, trans)
    return best


# ============================================================
# 坐标变换
# ============================================================

def transform_point_world2lidar(point_world, rotation_rodrigues, translation):
    """将世界坐标系的点转到 LiDAR 坐标系"""
    R, _ = cv2.Rodrigues(rotation_rodrigues)
    point_lidar = R @ point_world + translation
    return point_lidar


# ============================================================
# 物体匹配
# ============================================================

def match_objects(road_objects, car_objects, max_dist):
    """
    按位置就近匹配路端和车端物体（贪心，距离最小优先）

    Returns:
        matches: [(road_obj, car_obj, distance, dx, dy, dz)]
        unmatched_road: [road_obj]
        unmatched_car: [car_obj]
    """
    if not road_objects or not car_objects:
        return [], road_objects, car_objects

    # 计算距离矩阵
    pairs = []
    for i, r_obj in enumerate(road_objects):
        for j, c_obj in enumerate(car_objects):
            r_pos = np.array([r_obj['x'], r_obj['y'], r_obj['z']])
            c_pos = np.array([c_obj['x'], c_obj['y'], c_obj['z']])
            dist = np.linalg.norm(r_pos - c_pos)
            if dist <= max_dist:
                pairs.append((dist, i, j))

    # 贪心匹配：距离小的优先
    pairs.sort(key=lambda x: x[0])
    matched_road = set()
    matched_car = set()
    matches = []

    for dist, i, j in pairs:
        if i in matched_road or j in matched_car:
            continue
        r_obj = road_objects[i]
        c_obj = car_objects[j]
        dx = r_obj['x'] - c_obj['x']
        dy = r_obj['y'] - c_obj['y']
        dz = r_obj['z'] - c_obj['z']
        matches.append((r_obj, c_obj, dist, dx, dy, dz))
        matched_road.add(i)
        matched_car.add(j)

    unmatched_road = [road_objects[i] for i in range(len(road_objects)) if i not in matched_road]
    unmatched_car = [car_objects[j] for j in range(len(car_objects)) if j not in matched_car]

    return matches, unmatched_road, unmatched_car


# ============================================================
# 主流程
# ============================================================

def process_clip(clip_num, max_match_dist):
    """处理单个 clip，返回匹配结果列表"""
    clip_dir = find_clip_dir(clip_num)
    if not clip_dir:
        print(f"  [ERROR] 找不到 clip {clip_num}")
        return []

    transform_path = os.path.join(TRANSFORM_ROOT, clip_num, "world2lidar_transforms.json")
    if not os.path.exists(transform_path):
        print(f"  [SKIP] 无 transform: {transform_path}")
        return []

    road_label_dir = os.path.join(clip_dir, "road_labels", "interpolation_labels")
    car_label_dir = os.path.join(clip_dir, "car_labels", "interpolation_labels")

    # 加载数据
    road_anns = load_annotations(road_label_dir)
    car_anns = load_annotations(car_label_dir)
    transforms = load_transforms(transform_path)

    if not road_anns:
        print(f"  [SKIP] 无路端标注")
        return []
    if not car_anns:
        print(f"  [SKIP] 无车端标注")
        return []

    road_ts_list = sorted(road_anns.keys())
    car_ts_list = sorted(car_anns.keys())

    print(f"  路端: {len(road_anns)}帧, 车端: {len(car_anns)}帧, transforms: {len(transforms)}")

    all_matches = []
    matched_frames = 0

    # 以车端帧为基准，找最近路端帧
    for car_ts in car_ts_list:
        road_ts, ts_diff = find_nearest_timestamp(car_ts, road_ts_list, TS_MATCH_TOLERANCE_SEC)
        if road_ts is None:
            continue

        matched_frames += 1
        road_ann = road_anns[road_ts]
        car_ann = car_anns[car_ts]

        # 找最近的 world2lidar 变换
        transform = find_nearest_transform(car_ts, transforms)
        if transform is None:
            continue
        rot, trans = transform

        # 路端物体转到 LiDAR 坐标系
        road_objects_lidar = []
        for obj in road_ann.get('object', []):
            if obj.get('label', '') not in VEHICLE_LABELS:
                continue
            pos_world = np.array([obj['x'], obj['y'], obj['z']])
            pos_lidar = transform_point_world2lidar(pos_world, rot, trans)
            road_obj = {
                'x': float(pos_lidar[0]),
                'y': float(pos_lidar[1]),
                'z': float(pos_lidar[2]),
                'label': obj['label'],
                'id': obj.get('id', -1),
                'length': obj.get('length', 0),
                'width': obj.get('width', 0),
                'height': obj.get('height', 0),
            }
            road_objects_lidar.append(road_obj)

        # 车端物体（已在 LiDAR 坐标系）
        car_objects = []
        for obj in car_ann.get('object', []):
            if obj.get('label', '') not in VEHICLE_LABELS:
                continue
            car_objects.append({
                'x': obj['x'],
                'y': obj['y'],
                'z': obj['z'],
                'label': obj['label'],
                'id': obj.get('id', -1),
                'length': obj.get('length', 0),
                'width': obj.get('width', 0),
                'height': obj.get('height', 0),
            })

        # 匹配
        matches, _, _ = match_objects(road_objects_lidar, car_objects, max_match_dist)

        for r_obj, c_obj, dist, dx, dy, dz in matches:
            all_matches.append({
                'clip': clip_num,
                'car_ts': car_ts,
                'road_label': r_obj['label'],
                'car_label': c_obj['label'],
                'dist_3d': dist,
                'dx': dx, 'dy': dy, 'dz': dz,
                'dist_bev': np.sqrt(dx**2 + dy**2),
            })

    print(f"  匹配帧数: {matched_frames}, 匹配物体对数: {len(all_matches)}")
    return all_matches


def main():
    parser = argparse.ArgumentParser(description='车端 vs 路端标注偏差分析')
    parser.add_argument('--clips', nargs='+', default=None,
                        help='指定 clip 编号（如 008 031），默认所有')
    parser.add_argument('--max-match-dist', type=float, default=DEFAULT_MAX_MATCH_DIST,
                        help=f'物体匹配距离上限（米），默认 {DEFAULT_MAX_MATCH_DIST}')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).resolve().parent / 'output'),
                        help='输出目录')
    args = parser.parse_args()

    # 确定要处理的 clip 列表
    if args.clips:
        clip_list = args.clips
    else:
        # 自动发现所有有 transform 的 clip
        clip_list = sorted([
            d for d in os.listdir(TRANSFORM_ROOT)
            if os.path.isdir(os.path.join(TRANSFORM_ROOT, d))
        ])

    print(f"处理 {len(clip_list)} 个 clip, 匹配距离上限: {args.max_match_dist}m")
    print("=" * 80)

    all_matches = []
    for clip_num in clip_list:
        print(f"\n--- Clip {clip_num} ---")
        matches = process_clip(clip_num, args.max_match_dist)
        all_matches.extend(matches)

    if not all_matches:
        print("\n没有匹配到任何物体对")
        return

    # ---- 统计分析 ----
    dists_3d = np.array([m['dist_3d'] for m in all_matches])
    dists_bev = np.array([m['dist_bev'] for m in all_matches])
    dxs = np.array([m['dx'] for m in all_matches])
    dys = np.array([m['dy'] for m in all_matches])
    dzs = np.array([m['dz'] for m in all_matches])

    print("\n" + "=" * 80)
    print(f"总匹配物体对数: {len(all_matches)}")
    print(f"涉及 clip 数:   {len(set(m['clip'] for m in all_matches))}")
    print()
    print("3D 距离偏差 (米):")
    print(f"  mean: {np.mean(dists_3d):.4f}")
    print(f"  std:  {np.std(dists_3d):.4f}")
    print(f"  median: {np.median(dists_3d):.4f}")
    print(f"  p50/p75/p90/p95/p99: "
          f"{np.percentile(dists_3d, 50):.4f} / "
          f"{np.percentile(dists_3d, 75):.4f} / "
          f"{np.percentile(dists_3d, 90):.4f} / "
          f"{np.percentile(dists_3d, 95):.4f} / "
          f"{np.percentile(dists_3d, 99):.4f}")
    print()
    print("BEV 距离偏差 (米, 只看 xy):")
    print(f"  mean: {np.mean(dists_bev):.4f}")
    print(f"  std:  {np.std(dists_bev):.4f}")
    print(f"  median: {np.median(dists_bev):.4f}")
    print(f"  p50/p75/p90/p95/p99: "
          f"{np.percentile(dists_bev, 50):.4f} / "
          f"{np.percentile(dists_bev, 75):.4f} / "
          f"{np.percentile(dists_bev, 90):.4f} / "
          f"{np.percentile(dists_bev, 95):.4f} / "
          f"{np.percentile(dists_bev, 99):.4f}")
    print()
    print("各轴偏差 (米):")
    print(f"  dx: mean={np.mean(dxs):.4f}, std={np.std(dxs):.4f}")
    print(f"  dy: mean={np.mean(dys):.4f}, std={np.std(dys):.4f}")
    print(f"  dz: mean={np.mean(dzs):.4f}, std={np.std(dzs):.4f}")

    # 按类别统计
    print()
    print("按类别 BEV 偏差:")
    label_dists = defaultdict(list)
    for m in all_matches:
        label_dists[m['road_label']].append(m['dist_bev'])

    print(f"  {'类别':<20} {'数量':>6} {'mean':>8} {'median':>8} {'p90':>8}")
    for label in sorted(label_dists.keys()):
        d = np.array(label_dists[label])
        print(f"  {label:<20} {len(d):>6} {np.mean(d):>8.4f} {np.median(d):>8.4f} "
              f"{np.percentile(d, 90):>8.4f}")

    print("=" * 80)

    # ---- 保存结果 ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存详细匹配结果
    detail_path = output_dir / "car_road_match_details.csv"
    with open(detail_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'clip', 'car_ts', 'road_label', 'car_label',
            'dist_3d', 'dist_bev', 'dx', 'dy', 'dz'
        ])
        writer.writeheader()
        for m in all_matches:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in m.items()})
    print(f"\n详细结果: {detail_path}")

    # 保存统计摘要
    summary = {
        "total_matches": len(all_matches),
        "total_clips": len(set(m['clip'] for m in all_matches)),
        "max_match_dist": args.max_match_dist,
        "ts_tolerance_sec": TS_MATCH_TOLERANCE_SEC,
        "dist_3d": {
            "mean": float(np.mean(dists_3d)),
            "std": float(np.std(dists_3d)),
            "median": float(np.median(dists_3d)),
            "p50": float(np.percentile(dists_3d, 50)),
            "p75": float(np.percentile(dists_3d, 75)),
            "p90": float(np.percentile(dists_3d, 90)),
            "p95": float(np.percentile(dists_3d, 95)),
            "p99": float(np.percentile(dists_3d, 99)),
        },
        "dist_bev": {
            "mean": float(np.mean(dists_bev)),
            "std": float(np.std(dists_bev)),
            "median": float(np.median(dists_bev)),
            "p50": float(np.percentile(dists_bev, 50)),
            "p75": float(np.percentile(dists_bev, 75)),
            "p90": float(np.percentile(dists_bev, 90)),
            "p95": float(np.percentile(dists_bev, 95)),
            "p99": float(np.percentile(dists_bev, 99)),
        },
        "per_axis": {
            "dx_mean": float(np.mean(dxs)), "dx_std": float(np.std(dxs)),
            "dy_mean": float(np.mean(dys)), "dy_std": float(np.std(dys)),
            "dz_mean": float(np.mean(dzs)), "dz_std": float(np.std(dzs)),
        },
    }
    summary_path = output_dir / "car_road_match_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"统计摘要: {summary_path}")


if __name__ == "__main__":
    main()
