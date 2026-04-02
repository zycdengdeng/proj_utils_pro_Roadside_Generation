#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车端 vs 路端标注偏差分析（ego_transform 虚拟LiDAR方式）

用 ego_transform 的变换链（euler2rotmat + 虚拟LiDAR偏移）将路端标注转到
虚拟LiDAR坐标系，与车端标注（真实LiDAR坐标系）按位置就近匹配，计算偏差。

与 compare_car_road_annotations.py 的区别：
  - 那个用 transform_json 的真实外参
  - 这个用 ego_transform 的虚拟LiDAR (R=I, t=[0,0,h/2+0.25])

Usage:
    python compare_car_road_ego_transform.py
    python compare_car_road_ego_transform.py --clips 008 031
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from segment_pipeline.ego_transform import get_world2lidar_transform

# ============================================================
# 配置
# ============================================================

DATASET_ROOT = "/mnt/car_road_data_fix"

# 时间戳匹配容差（秒）
TS_MATCH_TOLERANCE_SEC = 0.1

# 物体位置匹配距离上限（米）
DEFAULT_MAX_MATCH_DIST = 10.0

VEHICLE_LABELS = {"Car", "Suv", "Truck", "Bus", "Van",
                  "Pedestrian", "Bicycle", "Motorcycle",
                  "Non_motor_rider", "Motor_rider", "Tricycle"}

# 所有 clip 的采集车 ID（来自汇总表 nearsetCarID）
EGO_VEHICLE_IDS = {
    "001": 45, "002": 29, "003": 45, "004": 17, "005": 48,
    "006": 66, "007": 7,  "008": 82, "009": 35, "010": 101,
    "011": 59, "012": 66, "013": 21, "014": 6,  "015": 55,
    "016": 28, "017": 106,"018": 56, "019": 81, "020": 77,
    "021": 70, "022": 90, "023": 29, "024": 2,  "025": 22,
    "026": 52, "027": 20, "028": 70, "029": 28, "030": 23,
    "031": 41, "032": 68, "033": 22, "034": 25, "035": 86,
    "036": 13, "037": 10, "038": 21, "039": 81, "040": 14,
    "041": 54, "042": 64, "043": 8,  "044": 82, "045": 95,
    "046": 50, "047": 102,"048": 27, "049": 106,"050": 78,
    "051": 99, "052": 31, "053": 52, "054": 32, "055": 13,
    "056": 25, "057": 7,  "058": 63, "059": 29, "060": 84,
    "061": 12, "062": 43, "063": 75, "064": 57, "065": 67,
    "066": 67, "067": 58, "068": 67, "069": 11, "070": 36,
    "071": 22, "072": 29, "073": 3,  "074": 20, "075": 111,
    "076": 12, "077": 7,  "078": 28, "079": 35, "080": 55,
    "081": 29, "082": 31, "083": 22, "084": 37, "085": 47,
    "086": 23, "087": 17, "088": 19, "089": 34,
}


# ============================================================
# 数据加载
# ============================================================

def find_clip_dir(clip_num):
    pattern = os.path.join(DATASET_ROOT, f"{clip_num}_*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def load_annotations(label_dir):
    """加载标注目录，返回 {timestamp_sec: annotation_dict}"""
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return {}

    annotations = {}
    for f in sorted(label_dir.glob("*.json")):
        stem = f.stem
        try:
            if '.' in stem:
                ts_sec = float(stem)
            else:
                ts_sec = int(stem) / 1000.0
        except ValueError:
            continue

        with open(f, 'r') as fh:
            annotations[ts_sec] = json.load(fh)

    return annotations


def find_nearest_timestamp(target_ts, ts_list, tolerance_sec):
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


# ============================================================
# 物体匹配（和 compare_car_road_annotations.py 完全相同）
# ============================================================

def match_objects(road_objects, car_objects, max_dist):
    if not road_objects or not car_objects:
        return [], road_objects, car_objects

    pairs = []
    for i, r_obj in enumerate(road_objects):
        for j, c_obj in enumerate(car_objects):
            r_pos = np.array([r_obj['x'], r_obj['y'], r_obj['z']])
            c_pos = np.array([c_obj['x'], c_obj['y'], c_obj['z']])
            dist = np.linalg.norm(r_pos - c_pos)
            if dist <= max_dist:
                pairs.append((dist, i, j))

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
    clip_dir = find_clip_dir(clip_num)
    if not clip_dir:
        print(f"  [ERROR] 找不到 clip {clip_num}")
        return []

    ego_id = EGO_VEHICLE_IDS.get(clip_num)
    if ego_id is None:
        print(f"  [SKIP] 无采集车 ID")
        return []

    road_label_dir = os.path.join(clip_dir, "road_labels", "interpolation_labels")
    car_label_dir = os.path.join(clip_dir, "car_labels", "interpolation_labels")

    road_anns = load_annotations(road_label_dir)
    car_anns = load_annotations(car_label_dir)

    if not road_anns:
        print(f"  [SKIP] 无路端标注")
        return []
    if not car_anns:
        print(f"  [SKIP] 无车端标注")
        return []

    road_ts_list = sorted(road_anns.keys())
    car_ts_list = sorted(car_anns.keys())

    print(f"  采集车ID: {ego_id}, 路端: {len(road_anns)}帧, 车端: {len(car_anns)}帧")

    all_matches = []
    matched_frames = 0
    transform_fail = 0

    for car_ts in car_ts_list:
        road_ts, ts_diff = find_nearest_timestamp(car_ts, road_ts_list, TS_MATCH_TOLERANCE_SEC)
        if road_ts is None:
            continue

        road_ann = road_anns[road_ts]
        car_ann = car_anns[car_ts]

        # 用 ego_transform 计算 world2lidar
        try:
            R_w2l, t_w2l, _, _, _ = get_world2lidar_transform(road_ann, ego_id)
        except ValueError:
            transform_fail += 1
            continue

        matched_frames += 1

        # 路端物体转到虚拟 LiDAR 坐标系
        road_objects_lidar = []
        for obj in road_ann.get('object', []):
            if obj.get('label', '') not in VEHICLE_LABELS:
                continue
            if obj['id'] == ego_id:
                continue  # 排除采集车自身

            pos_world = np.array([obj['x'], obj['y'], obj['z']]).reshape(3, 1)
            pos_lidar = R_w2l @ pos_world + t_w2l
            road_objects_lidar.append({
                'x': float(pos_lidar[0, 0]),
                'y': float(pos_lidar[1, 0]),
                'z': float(pos_lidar[2, 0]),
                'label': obj['label'],
                'id': obj.get('id', -1),
            })

        # 车端物体
        car_objects = []
        for obj in car_ann.get('object', []):
            if obj.get('label', '') not in VEHICLE_LABELS:
                continue
            car_objects.append({
                'x': obj['x'], 'y': obj['y'], 'z': obj['z'],
                'label': obj['label'],
                'id': obj.get('id', -1),
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

    if transform_fail:
        print(f"  变换失败帧数: {transform_fail} (路端标注中未找到采集车)")
    print(f"  匹配帧数: {matched_frames}, 匹配物体对数: {len(all_matches)}")
    return all_matches


def print_stats(all_matches):
    dists_3d = np.array([m['dist_3d'] for m in all_matches])
    dists_bev = np.array([m['dist_bev'] for m in all_matches])
    dxs = np.array([m['dx'] for m in all_matches])
    dys = np.array([m['dy'] for m in all_matches])
    dzs = np.array([m['dz'] for m in all_matches])

    print(f"\n总匹配物体对数: {len(all_matches)}")
    print(f"涉及 clip 数:   {len(set(m['clip'] for m in all_matches))}")

    for name, vals in [("3D 距离", dists_3d), ("BEV 距离", dists_bev)]:
        print(f"\n{name}偏差 (米):")
        print(f"  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  median={np.median(vals):.4f}")
        print(f"  p50={np.percentile(vals,50):.4f}  p75={np.percentile(vals,75):.4f}  "
              f"p90={np.percentile(vals,90):.4f}  p95={np.percentile(vals,95):.4f}  "
              f"p99={np.percentile(vals,99):.4f}")

    print(f"\n各轴偏差 (米):")
    print(f"  dx: mean={np.mean(dxs):.4f}, std={np.std(dxs):.4f}")
    print(f"  dy: mean={np.mean(dys):.4f}, std={np.std(dys):.4f}")
    print(f"  dz: mean={np.mean(dzs):.4f}, std={np.std(dzs):.4f}")

    print(f"\n按类别 BEV 偏差:")
    label_dists = defaultdict(list)
    for m in all_matches:
        label_dists[m['road_label']].append(m['dist_bev'])
    print(f"  {'类别':<20} {'数量':>6} {'mean':>8} {'median':>8} {'p90':>8}")
    for label in sorted(label_dists.keys()):
        d = np.array(label_dists[label])
        print(f"  {label:<20} {len(d):>6} {np.mean(d):>8.4f} {np.median(d):>8.4f} "
              f"{np.percentile(d, 90):>8.4f}")

    return dists_3d, dists_bev, dxs, dys, dzs


def main():
    parser = argparse.ArgumentParser(description='车端 vs 路端标注偏差 (ego_transform 方式)')
    parser.add_argument('--clips', nargs='+', default=None,
                        help='指定 clip 编号，默认所有')
    parser.add_argument('--max-match-dist', type=float, default=DEFAULT_MAX_MATCH_DIST,
                        help=f'物体匹配距离上限（米），默认 {DEFAULT_MAX_MATCH_DIST}')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).resolve().parent / 'output'),
                        help='输出目录')
    args = parser.parse_args()

    if args.clips:
        clip_list = args.clips
    else:
        clip_list = sorted(EGO_VEHICLE_IDS.keys())

    print(f"处理 {len(clip_list)} 个 clip (ego_transform 虚拟LiDAR方式)")
    print(f"匹配距离上限: {args.max_match_dist}m")
    print("=" * 80)

    all_matches = []
    for clip_num in clip_list:
        print(f"\n--- Clip {clip_num} ---")
        matches = process_clip(clip_num, args.max_match_dist)
        all_matches.extend(matches)

    if not all_matches:
        print("\n没有匹配到任何物体对")
        return

    print("\n" + "=" * 80)
    dists_3d, dists_bev, dxs, dys, dzs = print_stats(all_matches)
    print("=" * 80)

    # 保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "car_road_ego_transform_details.csv"
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

    summary = {
        "method": "ego_transform (virtual lidar: R=I, t=[0,0,h/2+0.25])",
        "total_matches": len(all_matches),
        "total_clips": len(set(m['clip'] for m in all_matches)),
        "max_match_dist": args.max_match_dist,
        "dist_3d": {
            "mean": float(np.mean(dists_3d)), "std": float(np.std(dists_3d)),
            "median": float(np.median(dists_3d)),
            "p90": float(np.percentile(dists_3d, 90)),
            "p95": float(np.percentile(dists_3d, 95)),
        },
        "dist_bev": {
            "mean": float(np.mean(dists_bev)), "std": float(np.std(dists_bev)),
            "median": float(np.median(dists_bev)),
            "p90": float(np.percentile(dists_bev, 90)),
            "p95": float(np.percentile(dists_bev, 95)),
        },
    }
    summary_path = output_dir / "car_road_ego_transform_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"统计摘要: {summary_path}")


if __name__ == "__main__":
    main()
