#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询指定车辆在路口区域内的时间戳范围和帧数

Usage:
    python query_vehicle_in_region.py
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common_utils import DATASET_ROOT, get_scene_paths

# ============================================================
# 要查询的车辆列表
# ============================================================
QUERY_VEHICLES = [
    {"clip": "031", "label": "Suv", "vehicle_id": 41},
    {"clip": "088", "label": "Suv", "vehicle_id": 19},
    {"clip": "053", "label": "Suv", "vehicle_id": 52},
    {"clip": "077", "label": "Suv", "vehicle_id": 7},
    {"clip": "056", "label": "Suv", "vehicle_id": 25},
    {"clip": "076", "label": "Suv", "vehicle_id": 12},
    {"clip": "033", "label": "Suv", "vehicle_id": 22},
    {"clip": "089", "label": "Suv", "vehicle_id": 34},
]

SEGMENT_LENGTH = 29


def load_region():
    """加载已有的路口区域定义"""
    region_file = Path(__file__).resolve().parent / "output" / "intersection_region.json"
    if not region_file.exists():
        print(f"[ERROR] 未找到区域定义文件: {region_file}")
        print("请先运行: python intersection_filter.py --step define")
        sys.exit(1)
    with open(region_file, 'r') as f:
        data = json.load(f)
    return data["region"]


def is_in_region(x, y, region):
    return (region["x_min"] <= x <= region["x_max"] and
            region["y_min"] <= y <= region["y_max"])


def query_vehicle(clip, vehicle_id, region):
    """
    查询某个clip中某个vehicle_id在路口区域内的所有帧

    Returns:
        frames_in_region: [(timestamp, x, y), ...]  按时间排序
        total_frames_in_clip: 该车辆在clip中出现的总帧数
    """
    paths = get_scene_paths(clip)
    if not paths:
        return None, 0

    label_dir = paths['roadside_labels']
    if not os.path.isdir(label_dir):
        return None, 0

    label_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))

    frames_in_region = []
    total_appearances = 0

    for lf in label_files:
        ts_str = Path(lf).stem
        try:
            ts = int(ts_str)
        except ValueError:
            continue

        with open(lf, 'r') as f:
            data = json.load(f)

        for obj in data.get("object", []):
            if obj["id"] == vehicle_id:
                total_appearances += 1
                x, y = obj["x"], obj["y"]
                if is_in_region(x, y, region):
                    frames_in_region.append((ts, x, y))
                break

    frames_in_region.sort(key=lambda t: t[0])
    return frames_in_region, total_appearances


def find_continuous_segments(frames_in_region, label_files_ts):
    """
    找出连续的在区域内的片段（考虑中间可能短暂离开区域又回来的情况）

    Returns:
        segments: [(start_ts, end_ts, frame_count, all_timestamps), ...]
    """
    if not frames_in_region:
        return []

    # 按时间顺序排列所有在区域内的时间戳
    all_ts = sorted(set(f[0] for f in frames_in_region))

    # 按间隔 >500ms 切分为连续片段
    segments = []
    current_ts_list = [all_ts[0]]

    for i in range(1, len(all_ts)):
        if all_ts[i] - all_ts[i - 1] > 500:
            segments.append((current_ts_list[0], current_ts_list[-1],
                             len(current_ts_list), current_ts_list))
            current_ts_list = [all_ts[i]]
        else:
            current_ts_list.append(all_ts[i])

    segments.append((current_ts_list[0], current_ts_list[-1],
                     len(current_ts_list), current_ts_list))

    return segments


def main():
    region = load_region()

    print("=" * 80)
    print(f"路口区域: X[{region['x_min']:.2f}, {region['x_max']:.2f}]  "
          f"Y[{region['y_min']:.2f}, {region['y_max']:.2f}]")
    print("=" * 80)

    results = []

    for q in QUERY_VEHICLES:
        clip = q["clip"]
        vid = q["vehicle_id"]
        label = q["label"]

        print(f"\n--- Clip {clip}, {label}{vid} ---")

        frames_in_region, total_appearances = query_vehicle(clip, vid, region)

        if frames_in_region is None:
            print(f"  [ERROR] 找不到场景 {clip}")
            results.append({"clip": clip, "vid": vid, "label": label, "error": "场景不存在"})
            continue

        if total_appearances == 0:
            print(f"  [WARN] 车辆 {label}{vid} 在 clip {clip} 中未出现")
            results.append({"clip": clip, "vid": vid, "label": label,
                            "total_appearances": 0, "in_region": 0})
            continue

        n_in = len(frames_in_region)
        print(f"  clip内总出现帧数: {total_appearances}")
        print(f"  路口区域内帧数:   {n_in}")

        if n_in == 0:
            print(f"  该车辆未进入路口区域")
            results.append({"clip": clip, "vid": vid, "label": label,
                            "total_appearances": total_appearances, "in_region": 0})
            continue

        # 找连续片段
        segments = find_continuous_segments(frames_in_region, None)

        print(f"  连续片段数: {len(segments)}")
        for i, (start_ts, end_ts, count, ts_list) in enumerate(segments):
            duration_ms = end_ts - start_ts
            enough = "YES" if count >= SEGMENT_LENGTH else "NO"
            n_segs = count // SEGMENT_LENGTH
            print(f"    片段{i+1}: {start_ts} ~ {end_ts}  "
                  f"({count}帧, {duration_ms}ms)  "
                  f"够{SEGMENT_LENGTH}帧: {enough}"
                  + (f"  可分{n_segs}段" if n_segs > 0 else ""))

            # 如果够29帧，打印每个29帧segment的时间戳范围
            if n_segs > 0:
                for s_idx in range(n_segs):
                    seg_start = s_idx * SEGMENT_LENGTH
                    seg_end = seg_start + SEGMENT_LENGTH
                    seg_ts = ts_list[seg_start:seg_end]
                    print(f"      seg{s_idx}: ts {seg_ts[0]} ~ {seg_ts[-1]}  "
                          f"(间隔 {seg_ts[-1] - seg_ts[0]}ms)")

        # 总计
        total_possible_segs = n_in // SEGMENT_LENGTH
        enough_total = "YES" if n_in >= SEGMENT_LENGTH else "NO"
        print(f"  >> 总计: {n_in}帧在区域内, 够{SEGMENT_LENGTH}帧: {enough_total}, "
              f"可生成{total_possible_segs}个segment")

        # 构建每个29帧segment的详细信息
        seg29_details = []
        for _start, _end, _count, _ts_list in segments:
            n_s = _count // SEGMENT_LENGTH
            for s_idx in range(n_s):
                s_begin = s_idx * SEGMENT_LENGTH
                s_slice = _ts_list[s_begin:s_begin + SEGMENT_LENGTH]
                seg29_details.append({
                    "start_ts": s_slice[0],
                    "end_ts": s_slice[-1],
                    "duration_ms": s_slice[-1] - s_slice[0],
                })

        results.append({
            "clip": clip, "vid": vid, "label": label,
            "total_appearances": total_appearances,
            "in_region": n_in,
            "enough_29": n_in >= SEGMENT_LENGTH,
            "possible_segments": total_possible_segs,
            "continuous_segments": [
                {"start_ts": s[0], "end_ts": s[1], "frames": s[2]}
                for s in segments
            ],
            "seg29": seg29_details,
        })

    # 汇总表格
    print("\n" + "=" * 80)
    print(f"{'Clip':>5} | {'车辆':>8} | {'clip总帧':>8} | {'区域内帧':>8} | {'够29帧':>6} | {'可生成段数':>10} | {'时间戳范围'}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['clip']:>5} | {r['label']}{r['vid']:>5} | {'ERROR':>8} |")
            continue
        ts_range = ""
        if r.get("seg29"):
            ranges = [f"seg{i}: {s['start_ts']}~{s['end_ts']}({s['duration_ms']}ms)"
                      for i, s in enumerate(r["seg29"])]
            ts_range = " | ".join(ranges)
        elif r["in_region"] > 0 and "continuous_segments" in r:
            ranges = [f"{s['start_ts']}~{s['end_ts']}({s['frames']}f)"
                      for s in r["continuous_segments"]]
            ts_range = " | ".join(ranges)

        print(f"{r['clip']:>5} | {r['label']}{r['vid']:>5} | "
              f"{r['total_appearances']:>8} | {r['in_region']:>8} | "
              f"{'YES' if r.get('enough_29') else 'NO':>6} | "
              f"{r.get('possible_segments', 0):>10} | {ts_range}")
    print("=" * 80)

    # 保存结果
    output_file = Path(__file__).resolve().parent / "output" / "query_vehicle_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
