#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment Pipeline 主调度模块
读取 filtered_segments.json，为每个 segment 生成:
  - pose.csv (ego 位姿)
  - annotations/ (ego 坐标系下的 3D 标注)
  - direction.json (行驶方向)

Usage:
    python -m segment_pipeline.segment_pipeline                    # 处理全部
    python -m segment_pipeline.segment_pipeline --scene 002        # 指定场景
    python -m segment_pipeline.segment_pipeline --interactive      # 交互式选择
"""

import json
import sys
import argparse
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment_pipeline.pose_generator import generate_pose_csv
from segment_pipeline.annotation_converter import convert_segment_annotations
from segment_pipeline.direction_detector import (
    build_reference_vectors, detect_direction, save_direction
)
from common_utils import find_scene_path, get_scene_paths


# ============================================================
# 配置
# ============================================================

# filtered_segments.json 默认路径
DEFAULT_SEGMENTS_FILE = (
    Path(__file__).resolve().parent.parent / "intersection_filter" / "output" / "filtered_segments.json"
)

# 输出根目录
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# intersection_filter 参考车辆 (导入或内联)
REFERENCE_VEHICLES = [
    {
        "scene_prefix": "002", "vehicle_id": 29,
        "entry_ts": 1742877436322, "exit_ts": 1742877441799,
        "direction": "W2E", "desc": "西向东",
    },
    {
        "scene_prefix": "003", "vehicle_id": 45,
        "entry_ts": 1742877823770, "exit_ts": 1742877829337,
        "direction": "E2W", "desc": "东向西",
    },
    {
        "scene_prefix": "014", "vehicle_id": 6,
        "entry_ts": 1742883999397, "exit_ts": 1742884002379,
        "direction": "N2S", "desc": "北向南",
    },
    {
        "scene_prefix": "015", "vehicle_id": 55,
        "entry_ts": 1742884382914, "exit_ts": 1742884385763,
        "direction": "S2N", "desc": "南向北",
    },
]


# ============================================================
# 核心处理
# ============================================================

def get_label_dir(scene_prefix):
    """获取场景的标注目录路径"""
    paths = get_scene_paths(scene_prefix)
    if paths and 'roadside_labels' in paths:
        label_dir = Path(paths['roadside_labels'])
        if label_dir.exists():
            return str(label_dir)
    return None


def process_single_segment(segment, ref_vectors, output_dir):
    """
    处理单个 segment

    Args:
        segment: segment dict from filtered_segments.json
        ref_vectors: 方向参考向量
        output_dir: 输出根目录

    Returns:
        success: bool
    """
    scene_id = segment['scene']
    vehicle_id = segment['vehicle_id']
    seg_idx = segment.get('segment_index', 0)
    timestamps = segment['timestamps']
    label_files = segment['label_files']

    seg_name = f"vehicle{vehicle_id}_seg{seg_idx:02d}"
    seg_output = Path(output_dir) / f"scene{scene_id}" / seg_name

    print(f"\n{'─'*60}")
    print(f"处理: scene={scene_id}, vehicle={vehicle_id}, seg={seg_idx}")
    print(f"帧数: {len(timestamps)}, 输出: {seg_output}")
    print(f"{'─'*60}")

    # Step 1: 生成 pose.csv
    print("\n[1/3] 生成 pose.csv ...")
    pose_path = seg_output / "pose.csv"
    poses, missing = generate_pose_csv(
        label_files, timestamps, vehicle_id, pose_path
    )

    if not poses:
        print(f"  错误: 无法提取任何 pose, 跳过此 segment")
        return False

    # Step 2: 生成 annotations
    print("\n[2/3] 转换标注到 ego 坐标系 ...")
    annotations_dir = seg_output / "annotations"
    convert_segment_annotations(
        label_files, timestamps, vehicle_id, annotations_dir
    )

    # Step 3: 检测方向
    print("\n[3/3] 检测行驶方向 ...")
    direction_key, direction_text, confidence = detect_direction(poses, ref_vectors)
    save_direction(
        direction_key, direction_text, confidence,
        seg_output / "direction.json"
    )

    return True


# ============================================================
# 交互式选择
# ============================================================

def interactive_select(segments):
    """
    交互式选择要处理的 segments

    Args:
        segments: 全部 segments 列表

    Returns:
        selected: 选中的 segments 列表
    """
    print("\n" + "="*60)
    print("Segment Pipeline - 交互式选择")
    print("="*60)

    # 按场景分组
    scenes = {}
    for seg in segments:
        scene_id = seg['scene']
        if scene_id not in scenes:
            scenes[scene_id] = []
        scenes[scene_id].append(seg)

    # 显示场景列表
    scene_ids = sorted(scenes.keys())
    print(f"\n共 {len(segments)} 个 segments, {len(scene_ids)} 个场景:")
    for i, sid in enumerate(scene_ids, 1):
        segs = scenes[sid]
        vehicle_ids = sorted(set(s['vehicle_id'] for s in segs))
        print(f"  {i}) scene {sid}: {len(segs)} 个 segments, "
              f"车辆 {vehicle_ids}")

    print(f"\n  0) 全部处理 ({len(segments)} 个)")
    print(f"  q) 退出")

    choice = input("\n请选择场景 [0]: ").strip()

    if choice.lower() == 'q':
        return []

    if choice == '' or choice == '0':
        return segments

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(scene_ids):
            selected_scene = scene_ids[idx]
            scene_segs = scenes[selected_scene]

            # 如果场景有多个车辆，允许进一步选择
            vehicle_ids = sorted(set(s['vehicle_id'] for s in scene_segs))
            if len(vehicle_ids) > 1:
                print(f"\n场景 {selected_scene} 中的车辆:")
                for i, vid in enumerate(vehicle_ids, 1):
                    count = sum(1 for s in scene_segs if s['vehicle_id'] == vid)
                    print(f"  {i}) 车辆 {vid} ({count} 个 segments)")
                print(f"  0) 全部")

                vid_choice = input("\n请选择车辆 [0]: ").strip()
                if vid_choice == '' or vid_choice == '0':
                    return scene_segs
                try:
                    vid_idx = int(vid_choice) - 1
                    if 0 <= vid_idx < len(vehicle_ids):
                        target_vid = vehicle_ids[vid_idx]
                        return [s for s in scene_segs if s['vehicle_id'] == target_vid]
                except ValueError:
                    pass

            return scene_segs
    except ValueError:
        pass

    print("无效输入，处理全部")
    return segments


# ============================================================
# 主函数
# ============================================================

def load_segments(segments_file):
    """加载 filtered_segments.json"""
    with open(segments_file, 'r') as f:
        data = json.load(f)

    segments = data if isinstance(data, list) else data.get('segments', [])
    print(f"加载了 {len(segments)} 个 segments")
    return segments


def filter_segments(segments, scene_filter=None, vehicle_filter=None):
    """按场景和车辆 ID 过滤"""
    if scene_filter:
        segments = [s for s in segments if s['scene'] in scene_filter]
    if vehicle_filter:
        segments = [s for s in segments if s['vehicle_id'] in vehicle_filter]
    return segments


def main():
    parser = argparse.ArgumentParser(description="Segment Pipeline - 生成 pose/annotation/direction")
    parser.add_argument('--segments-file', type=str, default=str(DEFAULT_SEGMENTS_FILE),
                        help='filtered_segments.json 路径')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='输出目录')
    parser.add_argument('--scene', type=str, nargs='*',
                        help='指定场景ID (如 --scene 001 002)')
    parser.add_argument('--vehicle-id', type=int, nargs='*',
                        help='指定车辆ID (如 --vehicle-id 1 5)')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式选择模式')
    args = parser.parse_args()

    print("="*60)
    print("Segment Pipeline")
    print("="*60)

    # 加载 segments
    segments_file = Path(args.segments_file)
    if not segments_file.exists():
        print(f"错误: {segments_file} 不存在")
        print(f"请先运行 intersection_filter.py 生成 filtered_segments.json")
        sys.exit(1)

    segments = load_segments(segments_file)
    if not segments:
        print("错误: 无有效 segments")
        sys.exit(1)

    # 过滤
    if args.interactive:
        segments = interactive_select(segments)
        if not segments:
            print("未选择任何 segment, 退出")
            sys.exit(0)
    else:
        segments = filter_segments(segments, args.scene, args.vehicle_id)
        if not segments:
            print("过滤后无匹配的 segment")
            sys.exit(1)

    print(f"\n将处理 {len(segments)} 个 segments")

    # 构建方向参考向量
    print("\n构建方向参考向量 ...")
    ref_vectors = build_reference_vectors(REFERENCE_VEHICLES, get_label_dir)
    if not ref_vectors:
        print("警告: 无法构建方向参考向量，方向将标记为 unknown")

    # 处理每个 segment
    success_count = 0
    for seg in segments:
        if process_single_segment(seg, ref_vectors, args.output_dir):
            success_count += 1

    # 总结
    print(f"\n{'='*60}")
    print(f"处理完成: {success_count}/{len(segments)} 个 segments")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
