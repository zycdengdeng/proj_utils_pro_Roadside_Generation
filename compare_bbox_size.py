#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比车端标注 vs 路转车标注中 Bus/Pedestrian 的框尺寸分布

车端标注: /mnt/car_road_data_fix/{clip}/car_labels/interpolation_labels/
路转车标注: segment_pipeline/output/{seg_name}/annotations/

按类别统计 length/width/height 的 mean/std/median/min/max，
找出框大小不一致的原因。

Usage:
    python compare_bbox_size.py
    python compare_bbox_size.py --end 025_id22_seg01
    python compare_bbox_size.py --labels Bus Pedestrian Truck
"""

import json
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = "/mnt/car_road_data_fix"
DEFAULT_SEGMENTS_DIR = str(Path(__file__).resolve().parent / "segment_pipeline" / "output")
DEFAULT_LABELS = ["Bus", "Pedestrian"]


def collect_car_side_sizes(clip_nums, target_labels):
    """收集车端标注的框尺寸"""
    sizes = defaultdict(list)  # label -> [(l, w, h), ...]

    for clip_num in clip_nums:
        pattern = os.path.join(DATASET_ROOT, f"{clip_num}_*")
        matches = glob.glob(pattern)
        if not matches:
            continue
        clip_dir = matches[0]
        label_dir = os.path.join(clip_dir, "car_labels", "interpolation_labels")
        if not os.path.isdir(label_dir):
            continue

        for f in glob.glob(os.path.join(label_dir, "*.json")):
            with open(f, 'r') as fh:
                data = json.load(fh)
            for obj in data.get('object', []):
                label = obj.get('label', '')
                if label in target_labels:
                    sizes[label].append((obj['length'], obj['width'], obj['height']))

    return sizes


def collect_road2car_sizes(segments_dir, end_seg, target_labels):
    """收集路转车标注（segment_pipeline 输出）的框尺寸"""
    sizes = defaultdict(list)

    seg_dir = Path(segments_dir)
    all_segs = sorted([d.name for d in seg_dir.iterdir() if d.is_dir()])
    selected = [s for s in all_segs if s <= end_seg]

    for seg_name in selected:
        ann_dir = seg_dir / seg_name / "annotations"
        if not ann_dir.exists():
            continue

        for f in sorted(ann_dir.glob("*.json")):
            with open(f, 'r') as fh:
                data = json.load(fh)
            for obj in data.get('object', []):
                label = obj.get('label', '')
                if label in target_labels:
                    sizes[label].append((obj['length'], obj['width'], obj['height']))

    return sizes


def print_size_stats(sizes, source_name):
    """打印尺寸统计"""
    for label in sorted(sizes.keys()):
        data = np.array(sizes[label])
        n = len(data)
        if n == 0:
            continue

        lengths = data[:, 0]
        widths = data[:, 1]
        heights = data[:, 2]

        print(f"\n  [{source_name}] {label} ({n} 个框)")
        print(f"    {'':12s} {'mean':>8} {'std':>8} {'median':>8} {'min':>8} {'max':>8}")
        for name, vals in [("length", lengths), ("width", widths), ("height", heights)]:
            print(f"    {name:<12s} {np.mean(vals):>8.3f} {np.std(vals):>8.3f} "
                  f"{np.median(vals):>8.3f} {np.min(vals):>8.3f} {np.max(vals):>8.3f}")


def main():
    parser = argparse.ArgumentParser(description='对比车端 vs 路转车标注的框尺寸')
    parser.add_argument('--segments-dir', type=str, default=DEFAULT_SEGMENTS_DIR)
    parser.add_argument('--end', type=str, default='025_id22_seg01',
                        help='路转车标注统计到哪个 seg')
    parser.add_argument('--labels', nargs='+', default=DEFAULT_LABELS,
                        help='要统计的类别')
    args = parser.parse_args()

    target_labels = set(args.labels)
    print(f"统计类别: {', '.join(sorted(target_labels))}")
    print(f"路转车范围: 到 {args.end}")

    # 从路转车 seg 名提取 clip 编号
    seg_dir = Path(args.segments_dir)
    all_segs = sorted([d.name for d in seg_dir.iterdir() if d.is_dir()])
    selected_segs = [s for s in all_segs if s <= args.end]
    clip_nums = sorted(set(s.split('_id')[0] for s in selected_segs))
    print(f"涉及 clip: {len(clip_nums)} 个, seg: {len(selected_segs)} 个")

    # 收集
    print("\n收集车端标注...")
    car_sizes = collect_car_side_sizes(clip_nums, target_labels)
    print(f"  " + ", ".join(f"{k}: {len(v)}" for k, v in car_sizes.items()))

    print("收集路转车标注...")
    r2c_sizes = collect_road2car_sizes(args.segments_dir, args.end, target_labels)
    print(f"  " + ", ".join(f"{k}: {len(v)}" for k, v in r2c_sizes.items()))

    # 输出对比
    print("\n" + "=" * 70)
    for label in sorted(target_labels):
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        if label in car_sizes:
            print_size_stats({label: car_sizes[label]}, "车端")
        else:
            print(f"\n  [车端] {label}: 无数据")

        if label in r2c_sizes:
            print_size_stats({label: r2c_sizes[label]}, "路转车")
        else:
            print(f"\n  [路转车] {label}: 无数据")

        # 差异
        if label in car_sizes and label in r2c_sizes:
            car_data = np.array(car_sizes[label])
            r2c_data = np.array(r2c_sizes[label])
            print(f"\n  差异 (路转车 - 车端):")
            for i, dim in enumerate(["length", "width", "height"]):
                diff = np.mean(r2c_data[:, i]) - np.mean(car_data[:, i])
                ratio = np.mean(r2c_data[:, i]) / np.mean(car_data[:, i]) if np.mean(car_data[:, i]) > 0 else float('inf')
                print(f"    {dim:<12s} mean差: {diff:>+8.3f}m  比值: {ratio:.3f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
