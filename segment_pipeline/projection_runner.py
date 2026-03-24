#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影运行模块
对 segment 的 29 帧运行指定投影类型，复用现有 projector 类

每种投影类型输出到独立子目录:
  {seg_output}/{proj_type}/{timestamp}/{proj|gt|overlay|...}/{cam}.jpg
"""

import json
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils
from segment_pipeline.ego_transform import get_world2ego_as_rodrigues


# 投影类型配置
PROJECTION_TYPES = {
    'basic': {
        'script': '基本点云投影/undistort_projection_multithread_v2.py',
        'class_name': 'UndistortProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': False,
        'control_subdir': 'proj',
        'control_input_type': 'basic',
    },
    'blur': {
        'script': 'blur投影/undistort_projection_multithread_v2.py',
        'class_name': 'BlurProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': True,
        'control_subdir': 'proj',
        'control_input_type': 'blur',
    },
    'blur_dense': {
        'script': 'blur稠密化投影/undistort_projection_multithread_v2.py',
        'class_name': 'BlurDenseProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': True,
        'control_subdir': 'proj',
        'control_input_type': 'blur_dense',
    },
    'depth': {
        'script': 'depth投影/undistort_projection_multithread_v2.py',
        'class_name': 'DepthProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': False,
        'control_subdir': 'depth',
        'control_input_type': 'depth',
    },
    'depth_dense': {
        'script': 'depth稠密化投影/undistort_projection_multithread_v2.py',
        'class_name': 'DepthDenseProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': False,
        'control_subdir': 'depth',
        'control_input_type': 'depth_dense',
    },
    'hdmap': {
        'script': 'HDMap投影/undistort_projection_multithread_v2.py',
        'class_name': 'HDMapProjectorMultiThread',
        'input': 'annotation',
        'needs_roadside_images': False,
        'control_subdir': 'overlay',
        'control_input_type': 'hdmap_bbox',
    },
}

# 可选投影类型列表（用于交互显示）
PROJECTION_CHOICES = ['basic', 'blur', 'blur_dense', 'depth', 'depth_dense', 'hdmap']


def build_transforms_from_annotations(label_files, timestamps, vehicle_id):
    """
    从标注文件构建 projector 所需的 transforms 列表

    bbox 中心位姿取逆 = world2ego = world2lidar（bbox 中心即 LiDAR 位置）

    Returns:
        transforms: [{'timestamp': ms, 'world2lidar': {'rotation': [...], 'translation': [...]}}]
    """
    transforms = []
    for label_file, ts in zip(label_files, timestamps):
        with open(label_file, 'r') as f:
            annotation = json.load(f)
        try:
            rotate, trans, _, _, _ = get_world2ego_as_rodrigues(annotation, vehicle_id)
            transforms.append({
                'timestamp': ts,
                'world2lidar': {
                    'rotation': rotate.flatten().tolist(),
                    'translation': trans.flatten().tolist()
                }
            })
        except ValueError:
            continue
    return transforms


def build_pcd_timestamp_map(pcd_dir):
    """构建 PCD 目录的 timestamp → filepath 映射"""
    pcd_dir = Path(pcd_dir)
    ts_map = {}
    for pcd_file in pcd_dir.glob("*.pcd"):
        ts = common_utils.extract_timestamp_from_filename(str(pcd_file))
        if ts is not None:
            ts_map[int(ts)] = pcd_file
    return ts_map


def find_closest_pcd(pcd_ts_map, target_ts, tolerance_ms=500):
    """在 PCD 时间戳映射中找最接近的文件"""
    best_ts = None
    best_diff = float('inf')
    for ts in pcd_ts_map:
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_ts = ts
    if best_ts is not None and best_diff <= tolerance_ms:
        return pcd_ts_map[best_ts]
    return None


def _load_projector(proj_type, config, scene_paths, transforms):
    """动态加载并创建 projector 实例"""
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / config['script']

    if not script_path.exists():
        print(f"    错误: 脚本不存在 {script_path}")
        return None

    spec = importlib.util.spec_from_file_location(f"projector_{proj_type}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    projector_class = getattr(module, config['class_name'])

    # blur/blur_dense 需要额外的 roadside_images_folder 参数
    if config['needs_roadside_images']:
        return projector_class(
            scene_paths['roadside_calib'],
            scene_paths['roadside_images'],
            scene_paths['vehicle_calib'],
            scene_paths.get('vehicle_images', scene_paths['roadside_images']),
            transforms
        )
    else:
        return projector_class(
            scene_paths['roadside_calib'],
            scene_paths['vehicle_calib'],
            scene_paths.get('vehicle_images', scene_paths['roadside_images']),
            transforms
        )


def run_projection_for_segment(segment, projection_types, output_dir, num_threads=7):
    """
    对一个 segment 的所有帧运行指定的投影类型

    Args:
        segment: segment dict (from filtered_segments.json)
        projection_types: list of projection type names (e.g., ['basic', 'depth'])
        output_dir: segment output directory (e.g., output/004_id45_seg01/)
        num_threads: threads per frame for camera processing
    """
    scene_id = segment['scene']
    vehicle_id = segment['vehicle_id']
    timestamps = segment['timestamps']
    label_files = segment['label_files']

    # 获取场景路径
    scene_paths = common_utils.get_scene_paths(scene_id)
    if not scene_paths:
        print(f"  错误: 无法获取场景 {scene_id} 路径")
        return

    # 构建 transforms (world2ego = world2lidar)
    transforms = build_transforms_from_annotations(label_files, timestamps, vehicle_id)
    if not transforms:
        print(f"  错误: 无法构建变换矩阵")
        return

    # PCD 文件映射（仅在有 PCD 类投影时构建）
    pcd_ts_map = None
    if any(PROJECTION_TYPES[pt]['input'] == 'pcd' for pt in projection_types
           if pt in PROJECTION_TYPES):
        pcd_ts_map = build_pcd_timestamp_map(scene_paths['pcd'])
        if not pcd_ts_map:
            print(f"  警告: PCD 目录为空 {scene_paths['pcd']}")

    for proj_type in projection_types:
        config = PROJECTION_TYPES.get(proj_type)
        if not config:
            print(f"  跳过未知投影类型: {proj_type}")
            continue

        print(f"\n  [{proj_type}] 投影 ...")

        projector = _load_projector(proj_type, config, scene_paths, transforms)
        if projector is None:
            continue

        proj_output_dir = Path(output_dir) / proj_type
        success = 0

        for ts, label_file in zip(timestamps, label_files):
            frame_output = proj_output_dir / str(ts)

            try:
                if config['input'] == 'pcd':
                    if pcd_ts_map is None:
                        continue
                    pcd_file = find_closest_pcd(pcd_ts_map, ts)
                    if pcd_file is None:
                        print(f"    跳过帧 {ts}: 未找到PCD")
                        continue
                    ok = projector.process_single_frame(
                        str(pcd_file), str(frame_output), ts, num_threads
                    )
                elif config['input'] == 'annotation':
                    ok = projector.process_single_frame(
                        str(label_file), str(frame_output), ts,
                        ego_vehicle_id=vehicle_id, num_threads=num_threads
                    )
                else:
                    continue
            except Exception as e:
                print(f"    帧 {ts} 异常: {e}")
                ok = False

            if ok:
                success += 1

        print(f"    完成: {success}/{len(timestamps)} 帧")


def interactive_select_projections():
    """交互式选择投影类型"""
    print("\n选择投影类型（空格分隔，回车=跳过投影）:")
    for i, name in enumerate(PROJECTION_CHOICES, 1):
        config = PROJECTION_TYPES[name]
        print(f"  {i}) {name:<12s}  control_subdir={config['control_subdir']}")
    print(f"  0) 全部")
    print(f"  回车) 跳过投影")

    choice = input("\n请选择: ").strip()

    if not choice:
        return []

    if choice == '0':
        return list(PROJECTION_CHOICES)

    selected = []
    for c in choice.replace(',', ' ').split():
        try:
            idx = int(c) - 1
            if 0 <= idx < len(PROJECTION_CHOICES):
                selected.append(PROJECTION_CHOICES[idx])
        except ValueError:
            if c in PROJECTION_TYPES:
                selected.append(c)

    return selected
