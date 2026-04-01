#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询采集车在路口区域内的时间戳范围

根据汇总表中的 nearsetCarID，追踪采集车在各 clip 中穿越路口的时间段。

Usage:
    python query_ego_in_region.py
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
# 要查询的 clip + 采集车 ID（来自汇总表 nearsetCarID）
# ============================================================
QUERY_LIST = [
    {"clip": "003", "label": "Suv", "vehicle_id": 45},
    {"clip": "004", "label": "Suv", "vehicle_id": 17},
    {"clip": "009", "label": "Suv", "vehicle_id": 35},
    {"clip": "015", "label": "Suv", "vehicle_id": 55},
    {"clip": "020", "label": "Suv", "vehicle_id": 77},
    {"clip": "031", "label": "Suv", "vehicle_id": 41},
    {"clip": "035", "label": "Suv", "vehicle_id": 86},
    {"clip": "039", "label": "Suv", "vehicle_id": 81},
    {"clip": "050", "label": "Suv", "vehicle_id": 78},
    {"clip": "055", "label": "Suv", "vehicle_id": 13},
    {"clip": "056", "label": "Suv", "vehicle_id": 25},
    {"clip": "059", "label": "Suv", "vehicle_id": 29},
    {"clip": "063", "label": "Suv", "vehicle_id": 75},
    {"clip": "076", "label": "Suv", "vehicle_id": 12},
    {"clip": "082", "label": "Suv", "vehicle_id": 31},
    {"clip": "085", "label": "Suv", "vehicle_id": 47},
    {"clip": "086", "label": "Suv", "vehicle_id": 23},
    {"clip": "088", "label": "Suv", "vehicle_id": 19},
]

def load_region():
    region_file = Path(__file__).resolve().parent / "output" / "intersection_region.json"
    if not region_file.exists():
        print(f"[ERROR] 未找到区域定义: {region_file}")
        sys.exit(1)
    with open(region_file, 'r') as f:
        data = json.load(f)
    return data["region"]


def is_in_region(x, y, region):
    return (region["x_min"] <= x <= region["x_max"] and
            region["y_min"] <= y <= region["y_max"])


def query_ego_in_clip(clip, vehicle_id, region):
    """追踪采集车在 clip 中穿越路口区域的帧"""
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


def find_continuous_segments(frames):
    """按 >500ms 间隔切分连续片段"""
    if not frames:
        return []

    all_ts = sorted(set(f[0] for f in frames))
    segments = []
    current = [all_ts[0]]

    for i in range(1, len(all_ts)):
        if all_ts[i] - all_ts[i - 1] > 500:
            segments.append(current)
            current = [all_ts[i]]
        else:
            current.append(all_ts[i])
    segments.append(current)

    return segments


def main():
    region = load_region()
    print("=" * 100)
    print(f"路口区域: X[{region['x_min']:.2f}, {region['x_max']:.2f}]  "
          f"Y[{region['y_min']:.2f}, {region['y_max']:.2f}]")
    print("=" * 100)

    results = []

    for q in QUERY_LIST:
        clip = q["clip"]
        vid = q["vehicle_id"]
        label = q["label"]

        frames_in_region, total = query_ego_in_clip(clip, vid, region)

        if frames_in_region is None:
            print(f"\n  Clip {clip}: [ERROR] 场景不存在")
            results.append({"clip": clip, "ego": f"{label}{vid}", "error": "场景不存在"})
            continue

        n_in = len(frames_in_region)
        print(f"\n--- Clip {clip}, 采集车 {label}{vid} ---")
        print(f"  clip内出现: {total}帧, 路口内: {n_in}帧")

        if n_in == 0:
            print(f"  采集车未进入路口区域")
            results.append({"clip": clip, "ego": f"{label}{vid}",
                            "total": total, "in_region": 0, "segments": []})
            continue

        segs = find_continuous_segments(frames_in_region)
        seg_details = []

        for i, ts_list in enumerate(segs):
            count = len(ts_list)
            print(f"  连续段{i}: {ts_list[0]} ~ {ts_list[-1]}  "
                  f"({count}帧, {ts_list[-1]-ts_list[0]}ms)")

            seg_details.append({
                "start_ts": ts_list[0], "end_ts": ts_list[-1],
                "frames": count,
            })

        results.append({
            "clip": clip, "ego": f"{label}{vid}",
            "total": total, "in_region": n_in,
            "segments": seg_details,
        })

    # 汇总表
    print("\n" + "=" * 100)
    print(f"{'Clip':>5} | {'采集车':>8} | {'clip帧':>6} | {'区域内':>6} | {'时间戳范围 (帧数)'}")
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['clip']:>5} | {r['ego']:>8} | {'ERR':>6} |")
            continue

        seg_info = ""
        for seg in r.get("segments", []):
            if seg_info:
                seg_info += " | "
            seg_info += f"{seg['start_ts']}~{seg['end_ts']}({seg['frames']}f)"

        print(f"{r['clip']:>5} | {r['ego']:>8} | {r['total']:>6} | "
              f"{r['in_region']:>6} | {seg_info}")
    print("=" * 100)

    # 保存
    out_file = Path(__file__).resolve().parent / "output" / "query_ego_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_file}")


if __name__ == "__main__":
    main()
