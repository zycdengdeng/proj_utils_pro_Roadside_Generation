#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 segment_pipeline 输出标注中的物体类别分布

按 nuScenes 映射关系统计，范围：从第一个 seg 到指定 seg。

Usage:
    python count_labels.py
    python count_labels.py --end 025_id22_seg01
    python count_labels.py --segments-dir /path/to/segment_pipeline/output
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

# 映射关系
CATES_OUR_TO_NUSC = {
    "Car": "car",
    "Suv": "car",
    "Vehicle_else": "car",
    "Truck": "truck",
    "Bus": "bus",
    "Bicycle": "bicycle",
    "Tricycle": "bicycle",
    "Non_motor_rider": "bicycle",
    "Motorcycle": "motorcycle",
    "Motor_rider": "motorcycle",
    "Pedestrian": "pedestrian",
    "Pedestrian_else": "pedestrian",
    "Bollards": "barrier",
    "Crash_bucket": "barrier",
}

DEFAULT_SEGMENTS_DIR = str(Path(__file__).resolve().parent / "segment_pipeline" / "output")


def main():
    parser = argparse.ArgumentParser(description='统计标注物体类别分布')
    parser.add_argument('--segments-dir', type=str, default=DEFAULT_SEGMENTS_DIR,
                        help='segment_pipeline 输出目录')
    parser.add_argument('--end', type=str, default='025_id22_seg01',
                        help='统计到哪个 seg（含），按字典序排序')
    args = parser.parse_args()

    seg_dir = Path(args.segments_dir)
    all_segs = sorted([d.name for d in seg_dir.iterdir() if d.is_dir()])

    # 筛选范围
    selected = [s for s in all_segs if s <= args.end]
    print(f"总 seg 数: {len(all_segs)}, 统计范围: 前 {len(selected)} 个 (到 {args.end})")
    print()

    # 统计
    our_counts = defaultdict(int)      # 原始标签计数
    nusc_counts = defaultdict(int)     # nuScenes 映射后计数
    unmapped = defaultdict(int)        # 未映射的标签
    total_objects = 0
    total_frames = 0

    for seg_name in selected:
        ann_dir = seg_dir / seg_name / "annotations"
        if not ann_dir.exists():
            continue

        for ann_file in sorted(ann_dir.glob("*.json")):
            with open(ann_file, 'r') as f:
                data = json.load(f)

            total_frames += 1
            for obj in data.get('object', []):
                label = obj.get('label', '')
                our_counts[label] += 1
                total_objects += 1

                nusc = CATES_OUR_TO_NUSC.get(label)
                if nusc:
                    nusc_counts[nusc] += 1
                else:
                    unmapped[label] += 1

    # 输出
    print(f"总帧数: {total_frames}, 总物体数: {total_objects}")
    print()

    print("=== 原始标签统计 ===")
    for label in sorted(our_counts.keys(), key=lambda x: -our_counts[x]):
        nusc = CATES_OUR_TO_NUSC.get(label, "(未映射)")
        print(f"  {label:<20} {our_counts[label]:>8}  → {nusc}")

    print()
    print("=== nuScenes 映射后统计 ===")
    for label in sorted(nusc_counts.keys(), key=lambda x: -nusc_counts[x]):
        print(f"  {label:<20} {nusc_counts[label]:>8}")

    if unmapped:
        print()
        print("=== 未映射标签 ===")
        for label in sorted(unmapped.keys(), key=lambda x: -unmapped[x]):
            print(f"  {label:<20} {unmapped[label]:>8}")

    mapped_total = sum(nusc_counts.values())
    unmapped_total = sum(unmapped.values())
    print()
    print(f"已映射: {mapped_total}, 未映射: {unmapped_total}, 总计: {total_objects}")


if __name__ == "__main__":
    main()
