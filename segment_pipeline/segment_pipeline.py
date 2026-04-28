#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment Pipeline 主调度模块
读取 filtered_segments.json，为每个 segment 生成:
  - pose.csv (ego 位姿)
  - annotations/ (ego 坐标系下的 3D 标注)
  - direction.json (行驶方向)
  - 投影 (basic/blur/depth/hdmap，可选)

命名格式: {scene}_id{vehicle_id}_seg{NN}  (如 004_id45_seg01)

Usage:
    python -m segment_pipeline.segment_pipeline                    # 处理全部
    python -m segment_pipeline.segment_pipeline --scene 002        # 指定场景
    python -m segment_pipeline.segment_pipeline --interactive      # 交互式选择
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment_pipeline.pose_generator import generate_pose_csv
from segment_pipeline.annotation_converter import convert_segment_annotations
from segment_pipeline.direction_detector import (
    build_reference_vectors, detect_direction, save_direction
)
from segment_pipeline.projection_runner import (
    run_projection_for_segment, interactive_select_projections
)
from common_utils import find_scene_path, get_scene_paths


# ============================================================
# 配置
# ============================================================

DEFAULT_SEGMENTS_FILE = (
    Path(__file__).resolve().parent.parent / "intersection_filter" / "output" / "filtered_segments.json"
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 方向参考车辆
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
# 命名
# ============================================================

def make_seg_name(scene_id, vehicle_id, segment_index):
    """
    生成 segment 目录名

    格式: {scene}_id{vid}_seg{NN}  (NN 从 01 开始)
    例: 004_id45_seg01
    """
    seg_num = segment_index + 1
    return f"{scene_id}_id{vehicle_id}_seg{seg_num:02d}"


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


def process_single_segment(segment, ref_vectors, output_dir, projection_types=None,
                           num_threads=7):
    """
    处理单个 segment

    Args:
        segment: segment dict from filtered_segments.json
        ref_vectors: 方向参考向量
        output_dir: 输出根目录
        projection_types: 要运行的投影类型列表，None 或 [] 表示跳过
        num_threads: 投影每帧线程数

    Returns:
        success: bool
    """
    scene_id = segment['scene']
    vehicle_id = segment['vehicle_id']
    seg_idx = segment.get('segment_index', 0)
    timestamps = segment['timestamps']
    label_files = segment['label_files']
    virtual_pose = segment.get('virtual_pose')      # Case C 静止观察车 (可选)
    virtual_poses = segment.get('virtual_poses')    # Case C 跟随观察车 (可选, 与 timestamps 等长)

    seg_name = make_seg_name(scene_id, vehicle_id, seg_idx)
    seg_output = Path(output_dir) / seg_name

    print(f"\n{'─'*60}")
    print(f"处理: {seg_name}")
    print(f"  scene={scene_id}, vehicle={vehicle_id}, seg_idx={seg_idx}")
    print(f"  帧数: {len(timestamps)}, 输出: {seg_output}")
    if virtual_poses is not None:
        vp0, vpN = virtual_poses[0], virtual_poses[-1]
        print(f"  virtual_poses (per-frame, n={len(virtual_poses)}):")
        print(f"    [0]    x={vp0['x']:.2f}, y={vp0['y']:.2f}, z={vp0['z']:.2f}, yaw={vp0['yaw']:.3f}")
        print(f"    [-1]   x={vpN['x']:.2f}, y={vpN['y']:.2f}, z={vpN['z']:.2f}, yaw={vpN['yaw']:.3f}")
    elif virtual_pose is not None:
        print(f"  virtual_pose: x={virtual_pose['x']:.2f}, y={virtual_pose['y']:.2f}, "
              f"z={virtual_pose['z']:.2f}, yaw={virtual_pose['yaw']:.3f}")
    print(f"{'─'*60}")

    # Step 1: 生成 pose.csv
    print("\n[1/4] 生成 pose.csv ...")
    pose_path = seg_output / "pose.csv"
    poses, missing = generate_pose_csv(
        label_files, timestamps, vehicle_id, pose_path,
        virtual_pose=virtual_pose, virtual_poses=virtual_poses,
    )

    if not poses:
        print(f"  错误: 无法提取任何 pose, 跳过此 segment")
        return False

    # Step 2: 生成 annotations
    print("\n[2/4] 转换标注到 ego 坐标系 ...")
    annotations_dir = seg_output / "annotations"
    convert_segment_annotations(
        label_files, timestamps, vehicle_id, annotations_dir,
        virtual_pose=virtual_pose, virtual_poses=virtual_poses,
    )

    # Step 3: 检测方向
    print("\n[3/4] 检测行驶方向 ...")
    direction_key, direction_text, confidence = detect_direction(poses, ref_vectors)
    save_direction(
        direction_key, direction_text, confidence,
        seg_output / "direction.json"
    )

    # Step 4: 投影（可选）
    if projection_types:
        print(f"\n[4/4] 投影 ({', '.join(projection_types)}) ...")
        run_projection_for_segment(
            segment, projection_types, seg_output, num_threads
        )
    else:
        print("\n[4/4] 跳过投影")

    return True


# ============================================================
# 交互式选择
# ============================================================

def _select_vehicles_for_scene(scene_id, scene_segs):
    """
    为单个场景选择车辆

    Args:
        scene_id: 场景ID
        scene_segs: 该场景下的所有 segments

    Returns:
        选中的 segments 列表
    """
    vehicle_ids = sorted(set(s['vehicle_id'] for s in scene_segs))

    if len(vehicle_ids) == 1:
        print(f"  场景 {scene_id}: 仅有车辆 {vehicle_ids[0]}, 自动选中")
        return scene_segs

    print(f"\n  场景 {scene_id} 中的车辆:")
    for i, vid in enumerate(vehicle_ids, 1):
        count = sum(1 for s in scene_segs if s['vehicle_id'] == vid)
        print(f"    {i}) 车辆 {vid} ({count} 个 segments)")
    print(f"    0) 全部车辆")

    vid_choice = input(f"  选择车辆 (可多选, 如 1 3 或 1,3) [0=全部]: ").strip()

    if vid_choice == '' or vid_choice == '0':
        return scene_segs

    # 解析多选
    vid_choice = vid_choice.replace(',', ' ')
    selected_segs = []
    for token in vid_choice.split():
        try:
            vid_idx = int(token) - 1
            if 0 <= vid_idx < len(vehicle_ids):
                target_vid = vehicle_ids[vid_idx]
                selected_segs.extend(s for s in scene_segs if s['vehicle_id'] == target_vid)
        except ValueError:
            pass

    if selected_segs:
        return selected_segs

    print("  无效输入, 选中该场景全部车辆")
    return scene_segs


def interactive_select(segments):
    """交互式选择要处理的 segments（支持多场景 + 每场景选车辆）"""
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

    scene_ids = sorted(scenes.keys())
    print(f"\n共 {len(segments)} 个 segments, {len(scene_ids)} 个场景:\n")
    for i, sid in enumerate(scene_ids, 1):
        segs = scenes[sid]
        vehicle_ids = sorted(set(s['vehicle_id'] for s in segs))
        print(f"  {i}) scene {sid}: {len(segs)} 个 segments, "
              f"车辆 {vehicle_ids}")

    print(f"\n  0) 全部处理 ({len(segments)} 个)")
    print(f"  q) 退出")
    print(f"\n  提示: 可多选场景, 用空格或逗号分隔, 如 \"1 3 5\" 或 \"1,3,5\"")

    choice = input("\n请选择场景: ").strip()

    if choice.lower() == 'q':
        return []

    if choice == '' or choice == '0':
        return segments

    # 解析多选场景
    choice = choice.replace(',', ' ')
    selected_scene_ids = []
    for token in choice.split():
        try:
            idx = int(token) - 1
            if 0 <= idx < len(scene_ids):
                selected_scene_ids.append(scene_ids[idx])
            else:
                print(f"  跳过无效编号: {token}")
        except ValueError:
            print(f"  跳过无效输入: {token}")

    if not selected_scene_ids:
        print("无效输入, 处理全部")
        return segments

    # 去重并保持顺序
    seen = set()
    unique_scene_ids = []
    for sid in selected_scene_ids:
        if sid not in seen:
            seen.add(sid)
            unique_scene_ids.append(sid)
    selected_scene_ids = unique_scene_ids

    print(f"\n已选场景: {', '.join(selected_scene_ids)}")

    # 对每个场景选择车辆
    all_selected = []
    for sid in selected_scene_ids:
        scene_segs = scenes[sid]
        selected = _select_vehicles_for_scene(sid, scene_segs)
        all_selected.extend(selected)

    return all_selected


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
    parser = argparse.ArgumentParser(description="Segment Pipeline - 生成 pose/annotation/direction/projection")
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
    parser.add_argument('--projections', type=str, nargs='*',
                        help='投影类型 (如 --projections basic depth hdmap)，不指定则交互选择')
    parser.add_argument('--no-projection', action='store_true',
                        help='跳过投影步骤')
    parser.add_argument('--num-threads', type=int, default=7,
                        help='投影每帧线程数 (默认 7)')
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

    # 预览 seg 命名
    for seg in segments:
        name = make_seg_name(seg['scene'], seg['vehicle_id'], seg.get('segment_index', 0))
        print(f"  {name}")

    # 投影类型选择
    if args.no_projection:
        projection_types = []
    elif args.projections is not None:
        projection_types = args.projections
    elif args.interactive:
        projection_types = interactive_select_projections()
    else:
        projection_types = []

    if projection_types:
        print(f"\n投影类型: {', '.join(projection_types)}")
    else:
        print(f"\n跳过投影")

    # 构建方向参考向量
    print("\n构建方向参考向量 ...")
    ref_vectors = build_reference_vectors(REFERENCE_VEHICLES, get_label_dir)
    if not ref_vectors:
        print("警告: 无法构建方向参考向量，方向将标记为 unknown")

    # 处理每个 segment
    success_count = 0
    for seg in segments:
        if process_single_segment(seg, ref_vectors, args.output_dir,
                                  projection_types, args.num_threads):
            success_count += 1

    # 总结
    print(f"\n{'='*60}")
    print(f"处理完成: {success_count}/{len(segments)} 个 segments")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
