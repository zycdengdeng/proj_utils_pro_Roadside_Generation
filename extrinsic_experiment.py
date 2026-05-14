#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机外参扰动实验

在不修改原始标定文件的前提下，对车端相机外参施加扰动（z 抬升 / yaw 偏转），
重跑 3 种路转车投影（blur + depth + hdmap），生成 FW 视角视频用于对比。

用法:
  # 实验1: z轴抬升1m
  python extrinsic_experiment.py --clip 067 --vehicle-id 49 --experiment zlift --z-offset 1.0

  # 实验2: yaw偏转5°
  python extrinsic_experiment.py --clip 067 --vehicle-id 49 --experiment yaw --yaw-degrees 5

  # 基线: 原始外参 (用于对比)
  python extrinsic_experiment.py --clip 067 --vehicle-id 49 --experiment original
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'segment_pipeline'))
import common_utils
from ego_transform import get_world2ego_as_rodrigues

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(common_utils.DATASET_ROOT)
VEHICLE_CALIB_DIR = Path(common_utils.VEHICLE_CALIB_DIR)

SEGMENT_LENGTH = 29
PROJECTION_CONFIGS = {
    'blur': {
        'script': 'blur投影/undistort_projection_multithread_v2.py',
        'class_name': 'BlurProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': True,
        'control_subdir': 'proj',
        'control_input_type': 'blur',
    },
    'depth': {
        'script': 'depth投影/undistort_projection_multithread_v2.py',
        'class_name': 'DepthProjectorMultiThread',
        'input': 'pcd',
        'needs_roadside_images': False,
        'control_subdir': 'depth',
        'control_input_type': 'depth',
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


def build_pcd_timestamp_map(pcd_dir):
    pcd_dir = Path(pcd_dir)
    ts_map = {}
    for pcd_file in pcd_dir.glob("*.pcd"):
        ts = common_utils.extract_timestamp_from_filename(str(pcd_file))
        if ts is not None:
            ts_map[int(ts)] = pcd_file
    return ts_map


def find_closest_pcd(pcd_ts_map, target_ts, tolerance_ms=500):
    best_ts, best_diff = None, float('inf')
    for ts in pcd_ts_map:
        diff = abs(ts - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_ts = ts
    if best_ts is not None and best_diff <= tolerance_ms:
        return pcd_ts_map[best_ts]
    return None

TRANSFER_CAM_NAME = 'ftheta_camera_front_wide_120fov'
FW_CAM_NAME = 'FW'


# ============================================================
# 外参修改
# ============================================================

def load_extrinsics(cam_id):
    path = VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_extrinsics.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def save_extrinsics(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def apply_zlift(extr, z_offset):
    """cam2lidar 的 translation.z += z_offset"""
    extr = json.loads(json.dumps(extr))  # deep copy
    extr['transform']['translation']['z'] += z_offset
    return extr


def apply_yaw_perturbation(extr, yaw_degrees):
    """在 lidar 系绕 z 轴旋转 cam2lidar"""
    extr = json.loads(json.dumps(extr))
    tr = extr['transform']

    # 当前 cam2lidar 旋转 (quaternion xyzw → scipy 格式 xyzw)
    q_old = [tr['rotation']['x'], tr['rotation']['y'],
             tr['rotation']['z'], tr['rotation']['w']]
    R_old = R.from_quat(q_old)  # scipy uses xyzw

    # 绕 lidar z 轴旋转
    R_yaw = R.from_euler('z', yaw_degrees, degrees=True)
    R_new = R_yaw * R_old  # R_new = R_z @ R_old

    q_new = R_new.as_quat()  # xyzw
    tr['rotation']['x'] = float(q_new[0])
    tr['rotation']['y'] = float(q_new[1])
    tr['rotation']['z'] = float(q_new[2])
    tr['rotation']['w'] = float(q_new[3])

    # 同时旋转 translation (相机位置也跟着转)
    t_old = np.array([tr['translation']['x'], tr['translation']['y'], tr['translation']['z']])
    t_new = R_yaw.apply(t_old)
    tr['translation']['x'] = float(t_new[0])
    tr['translation']['y'] = float(t_new[1])
    tr['translation']['z'] = float(t_new[2])

    return extr


def create_modified_calib_dir(experiment, z_offset=0.0, yaw_degrees=0.0):
    """创建临时标定目录，内含修改后的外参 + 原始内参"""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"calib_{experiment}_"))

    for cam_id in range(1, 8):
        # 复制内参 (不改)
        intr_src = VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"
        shutil.copy2(intr_src, tmp_dir / intr_src.name)

        # 修改外参
        extr = load_extrinsics(cam_id)
        if experiment == 'zlift':
            extr = apply_zlift(extr, z_offset)
        elif experiment == 'yaw':
            extr = apply_yaw_perturbation(extr, yaw_degrees)
        # 'original' → 不改

        save_extrinsics(extr, tmp_dir / f"camera_{cam_id:02d}_extrinsics.yaml")

    return tmp_dir


# ============================================================
# 帧选择
# ============================================================

def find_segment_frames(clip_id, vehicle_id):
    """从路侧标注中找 vehicle_id 存在的帧，取中间 29 帧"""
    paths = common_utils.get_scene_paths(clip_id)
    if not paths:
        print(f"[ERROR] 找不到场景 {clip_id}")
        sys.exit(1)

    import glob
    label_dir = paths['roadside_labels']
    label_files = sorted(glob.glob(str(Path(label_dir) / "*.json")))
    if not label_files:
        print(f"[ERROR] 无标注文件: {label_dir}")
        sys.exit(1)

    # 找包含该车辆的帧
    valid_frames = []
    for lf in label_files:
        ts_str = Path(lf).stem
        try:
            ts = int(ts_str)
        except ValueError:
            continue
        with open(lf) as f:
            ann = json.load(f)
        for obj in ann.get('object', []):
            if obj['id'] == vehicle_id:
                valid_frames.append((ts, lf))
                break

    if len(valid_frames) < SEGMENT_LENGTH:
        print(f"[ERROR] 车辆 {vehicle_id} 只出现在 {len(valid_frames)} 帧，不够 {SEGMENT_LENGTH}")
        sys.exit(1)

    # 取中间 29 帧
    start = (len(valid_frames) - SEGMENT_LENGTH) // 2
    segment = valid_frames[start:start + SEGMENT_LENGTH]

    timestamps = [ts for ts, _ in segment]
    label_files_sel = [lf for _, lf in segment]

    print(f"  选取 {len(segment)} 帧 (从 {len(valid_frames)} 帧中居中截取)")
    print(f"  时间戳: {timestamps[0]} ~ {timestamps[-1]}")

    return timestamps, label_files_sel, paths


# ============================================================
# 投影
# ============================================================

def build_transforms(label_files, timestamps, vehicle_id):
    transforms = []
    for lf, ts in zip(label_files, timestamps):
        with open(lf) as f:
            ann = json.load(f)
        try:
            rot, trans, _, _, _ = get_world2ego_as_rodrigues(ann, vehicle_id)
            transforms.append({
                'timestamp': ts,
                'world2lidar': {
                    'rotation': rot.flatten().tolist(),
                    'translation': trans.flatten().tolist(),
                }
            })
        except ValueError:
            continue
    return transforms


def load_projector(proj_type, scene_paths, calib_dir, transforms):
    """动态加载投影器，用修改后的 calib_dir"""
    config = PROJECTION_CONFIGS[proj_type]

    import importlib.util
    script_path = PROJECT_ROOT / config['script']
    spec = importlib.util.spec_from_file_location(f"proj_{proj_type}", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    projector_class = getattr(module, config['class_name'])

    if config['needs_roadside_images']:
        return projector_class(
            scene_paths['roadside_calib'],
            scene_paths['roadside_images'],
            str(calib_dir),
            scene_paths.get('vehicle_images', scene_paths['roadside_images']),
            transforms,
        )
    else:
        return projector_class(
            scene_paths['roadside_calib'],
            str(calib_dir),
            scene_paths.get('vehicle_images', scene_paths['roadside_images']),
            transforms,
        )


def run_projections(timestamps, label_files, scene_paths, calib_dir, transforms,
                    vehicle_id, output_dir):
    pcd_ts_map = build_pcd_timestamp_map(scene_paths['pcd'])
    if not pcd_ts_map:
        print(f"  [WARN] PCD 目录为空: {scene_paths['pcd']}")

    for proj_type in PROJECTION_CONFIGS:
        config = PROJECTION_CONFIGS[proj_type]
        print(f"\n  [{proj_type}] 投影 ...")

        projector = load_projector(proj_type, scene_paths, calib_dir, transforms)
        if projector is None:
            continue

        proj_dir = output_dir / proj_type
        success = 0

        for ts, lf in zip(timestamps, label_files):
            frame_dir = proj_dir / str(ts)
            try:
                if config['input'] == 'pcd':
                    pcd_file = find_closest_pcd(pcd_ts_map, ts)
                    if pcd_file is None:
                        continue
                    ok = projector.process_single_frame(
                        str(pcd_file), str(frame_dir), ts, 7)
                elif config['input'] == 'annotation':
                    ok = projector.process_single_frame(
                        str(lf), str(frame_dir), ts,
                        ego_vehicle_id=vehicle_id, num_threads=7)
                else:
                    continue
            except Exception as e:
                print(f"    帧 {ts} 异常: {e}")
                ok = False
            if ok:
                success += 1

        print(f"    完成: {success}/{len(timestamps)} 帧")


# ============================================================
# 视频生成 (FW only)
# ============================================================

CONTROL_SUBDIRS = {
    'blur': 'proj',
    'depth': 'depth',
    'hdmap': 'overlay',
}

CONTROL_INPUT_NAMES = {
    'blur': 'blur',
    'depth': 'depth',
    'hdmap': 'hdmap_bbox',
}


def generate_videos(output_dir, fps=10, resolution=(1280, 720)):
    print(f"\n  生成视频 (FW, {resolution[0]}x{resolution[1]}, {fps}fps) ...")

    for proj_type in PROJECTION_CONFIGS:
        ctrl_subdir = CONTROL_SUBDIRS[proj_type]
        ctrl_name = CONTROL_INPUT_NAMES[proj_type]

        proj_dir = output_dir / proj_type
        ts_dirs = sorted([d for d in proj_dir.iterdir() if d.is_dir()],
                         key=lambda d: int(d.name)) if proj_dir.exists() else []
        if not ts_dirs:
            continue

        # GT 视频
        gt_paths = [d / 'gt' / f'{FW_CAM_NAME}.jpg' for d in ts_dirs
                     if (d / 'gt' / f'{FW_CAM_NAME}.jpg').exists()]
        # Control 视频
        ctrl_paths = [d / ctrl_subdir / f'{FW_CAM_NAME}.jpg' for d in ts_dirs
                      if (d / ctrl_subdir / f'{FW_CAM_NAME}.jpg').exists()]

        seg_name = output_dir.name
        video_dir = output_dir / 'videos'

        if gt_paths:
            gt_video = video_dir / f'gt_{seg_name}.mp4'
            _make_video(gt_paths, gt_video, fps, resolution)

        if ctrl_paths:
            ctrl_video = video_dir / f'control_{ctrl_name}_{seg_name}.mp4'
            _make_video(ctrl_paths, ctrl_video, fps, resolution)

    print(f"    视频输出: {output_dir / 'videos'}/")


def _make_video(image_paths, output_path, fps, resolution):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[1] != resolution[0] or img.shape[0] != resolution[1]:
            img = cv2.resize(img, resolution, interpolation=cv2.INTER_LINEAR)
        writer.write(img)
    writer.release()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='相机外参扰动实验')
    parser.add_argument('--clip', required=True, help='clip ID (如 067)')
    parser.add_argument('--vehicle-id', type=int, required=True, help='ego 车辆 ID')
    parser.add_argument('--experiment', required=True,
                        choices=['original', 'zlift', 'yaw'],
                        help='实验类型: original / zlift / yaw')
    parser.add_argument('--z-offset', type=float, default=1.0,
                        help='z 轴抬升量 (米, 默认 1.0)')
    parser.add_argument('--yaw-degrees', type=float, default=5.0,
                        help='yaw 偏转角 (度, 默认 5.0)')
    parser.add_argument('--output', default='/mnt/zihanw/extrinsic_exp',
                        help='输出根目录')
    args = parser.parse_args()

    # 实验名
    if args.experiment == 'zlift':
        exp_name = f"zlift_{args.z_offset:.1f}m"
    elif args.experiment == 'yaw':
        exp_name = f"yaw_{args.yaw_degrees:.0f}deg"
    else:
        exp_name = "original"

    output_dir = Path(args.output) / f"{args.clip}_id{args.vehicle_id}_{exp_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"相机外参扰动实验")
    print(f"  Clip: {args.clip},  Vehicle ID: {args.vehicle_id}")
    print(f"  实验: {exp_name}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}")

    # 1. 选帧
    print(f"\n[1/4] 选取 {SEGMENT_LENGTH} 帧 ...")
    timestamps, label_files, scene_paths = find_segment_frames(args.clip, args.vehicle_id)

    # 2. 创建修改后的标定目录
    print(f"\n[2/4] 创建修改后的标定 ...")
    calib_dir = create_modified_calib_dir(
        args.experiment,
        z_offset=args.z_offset,
        yaw_degrees=args.yaw_degrees,
    )
    print(f"  临时标定目录: {calib_dir}")

    # 3. 构建 transforms + 跑投影
    print(f"\n[3/4] 投影 (blur + depth + hdmap) ...")
    transforms = build_transforms(label_files, timestamps, args.vehicle_id)
    run_projections(timestamps, label_files, scene_paths, calib_dir,
                    transforms, args.vehicle_id, output_dir)

    # 4. 生成视频
    print(f"\n[4/4] 生成视频 ...")
    generate_videos(output_dir)

    # 清理临时目录
    shutil.rmtree(calib_dir)
    print(f"\n{'='*60}")
    print(f"完成: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
