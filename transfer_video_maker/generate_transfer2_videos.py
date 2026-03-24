#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Transfer2 训练数据格式的视频

两种模式:
  1. segment 模式 (--segments-dir): 从 segment_pipeline 输出读取，每个 seg 目录 = 一个视频
  2. legacy 模式 (--project-root + --scenes): 从老的投影输出读取（兼容）

输出结构不变:
  {output_dir}/videos/{camera}/{seg_name}.mp4
  {output_dir}/control_input_{type}/{camera}/{seg_name}.mp4
  {output_dir}/captions/{camera}/{seg_name}.json
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

# 相机名称映射：我的相机名 → Transfer2 相机名
CAMERA_NAME_MAPPING = {
    'FN': 'ftheta_camera_front_tele_30fov',
    'FW': 'ftheta_camera_front_wide_120fov',
    'FL': 'ftheta_camera_cross_left_120fov',
    'FR': 'ftheta_camera_cross_right_120fov',
    'RL': 'ftheta_camera_rear_left_70fov',
    'RR': 'ftheta_camera_rear_right_70fov',
    'RN': 'ftheta_camera_rear_tele_30fov'
}

CAMERA_NAMES = ['FN', 'FW', 'FL', 'FR', 'RL', 'RR', 'RN']

# 相机视角描述（用于 caption）
CAMERA_VIEW_PREFIX = {
    'ftheta_camera_front_tele_30fov': 'Front telephoto view',
    'ftheta_camera_front_wide_120fov': 'Front wide view',
    'ftheta_camera_cross_left_120fov': 'Left cross view',
    'ftheta_camera_cross_right_120fov': 'Right cross view',
    'ftheta_camera_rear_left_70fov': 'Rear left view',
    'ftheta_camera_rear_right_70fov': 'Rear right view',
    'ftheta_camera_rear_tele_30fov': 'Rear telephoto view'
}

# direction_key → caption 文本
DIRECTION_TEXT = {
    'W2E': 'west to east',
    'E2W': 'east to west',
    'N2S': 'north to south',
    'S2N': 'south to north',
}


def parse_args():
    parser = argparse.ArgumentParser(description='生成 Transfer2 格式视频数据集')

    # ---- segment 模式参数 ----
    parser.add_argument('--segments-dir', type=str,
                        help='segment_pipeline 输出目录 (如 segment_pipeline/output/)')
    parser.add_argument('--project-type', type=str,
                        choices=['depth', 'depth_dense', 'hdmap', 'blur', 'blur_dense', 'basic'],
                        help='投影类型（用于确定控制输入子目录）')

    # ---- legacy 模式参数 ----
    parser.add_argument('--project-root', type=str,
                        default='/mnt/zihanw/proj_utils_pro',
                        help='[legacy] 项目根目录')
    parser.add_argument('--scenes', type=str, nargs='+',
                        help='[legacy] 场景ID列表')
    parser.add_argument('--frames-per-seg', type=int, default=29,
                        help='[legacy] 每个seg的帧数')
    parser.add_argument('--num-segs', type=int, default=1,
                        help='[legacy] 每个场景生成的seg数量')

    # ---- 共用参数 ----
    parser.add_argument('--fps', type=int, default=10,
                        help='视频帧率（默认 10 FPS）')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--control-subdir', type=str,
                        help='控制输入图像子目录名（如 depth, proj, overlay）')
    parser.add_argument('--control-input-type', type=str,
                        help='控制输入类型名（如 depth, hdmap_bbox, basic）')
    parser.add_argument('--caption-template', type=str, default='',
                        help='Caption 模板')

    return parser.parse_args()


# ============================================================
# 投影类型 → 默认子目录映射
# ============================================================

PROJECT_TYPE_DEFAULTS = {
    'basic':      {'control_subdir': 'proj',    'control_input_type': 'basic',
                   'caption': '{view_prefix}. The ego vehicle is traveling from {direction}.'},
    'blur':       {'control_subdir': 'proj',    'control_input_type': 'blur',
                   'caption': '{view_prefix}. The ego vehicle is traveling from {direction}.'},
    'blur_dense': {'control_subdir': 'proj',    'control_input_type': 'blur_dense',
                   'caption': '{view_prefix}. The ego vehicle is traveling from {direction}.'},
    'depth':      {'control_subdir': 'depth',   'control_input_type': 'depth',
                   'caption': '{view_prefix}. The ego vehicle is traveling from {direction}.'},
    'depth_dense':{'control_subdir': 'depth',   'control_input_type': 'depth_dense',
                   'caption': '{view_prefix}. The ego vehicle is traveling from {direction}.'},
    'hdmap':      {'control_subdir': 'overlay', 'control_input_type': 'hdmap_bbox',
                   'caption': '{view_prefix}. The ego vehicle is traveling from {direction}.'},
}


# ============================================================
# 工具函数
# ============================================================

def get_sorted_timestamp_folders(base_dir):
    """获取目录下所有时间戳文件夹，按时间戳排序"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    timestamp_folders = []
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            timestamp_folders.append(folder)

    timestamp_folders.sort(key=lambda x: int(x.name))
    return timestamp_folders


def create_video_from_images(image_paths, output_path, fps, target_resolution=(1280, 720)):
    """从图像列表创建视频（统一分辨率为1280x720）"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, target_resolution
    )

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue

        if img.shape[1] != target_resolution[0] or img.shape[0] != target_resolution[1]:
            img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)

        video_writer.write(img)

    video_writer.release()


def load_direction(seg_dir):
    """从 segment 目录读取 direction.json"""
    direction_file = Path(seg_dir) / 'direction.json'
    if direction_file.exists():
        with open(direction_file, 'r') as f:
            data = json.load(f)
        direction_key = data.get('direction_key', 'unknown')
        return DIRECTION_TEXT.get(direction_key, data.get('direction', 'unknown direction'))
    return 'unknown direction'


def create_caption_json(seg_name, camera_name, caption_template, direction='unknown direction'):
    """
    创建 caption JSON

    模板可用变量: {camera}, {view_prefix}, {direction}
    """
    view_prefix = CAMERA_VIEW_PREFIX.get(camera_name, camera_name)

    if caption_template:
        caption = caption_template.format(
            camera=camera_name,
            view_prefix=view_prefix,
            direction=direction
        )
    else:
        caption = f"{view_prefix}. The ego vehicle is traveling from {direction}."

    return {
        "segment_name": seg_name,
        "camera": camera_name,
        "direction": direction,
        "caption": caption
    }


# ============================================================
# Segment 模式
# ============================================================

def process_segment_mode(args):
    """
    从 segment_pipeline 输出生成视频

    目录结构:
      {segments_dir}/{seg_name}/{proj_type}/{timestamp}/gt/{cam}.jpg
      {segments_dir}/{seg_name}/{proj_type}/{timestamp}/{control_subdir}/{cam}.jpg
    """
    segments_dir = Path(args.segments_dir)
    project_type = args.project_type

    # 获取默认配置
    defaults = PROJECT_TYPE_DEFAULTS.get(project_type, {})
    control_subdir = args.control_subdir or defaults.get('control_subdir', 'proj')
    control_input_type = args.control_input_type or defaults.get('control_input_type', project_type)
    caption_template = args.caption_template or defaults.get('caption', '')

    print(f"模式: segment")
    print(f"投影类型: {project_type}")
    print(f"控制子目录: {control_subdir}")
    print(f"控制输入类型: {control_input_type}")

    # 扫描所有 segment 目录
    seg_dirs = sorted([
        d for d in segments_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    if not seg_dirs:
        print(f"错误: {segments_dir} 下未找到 segment 目录")
        return

    # 过滤出有指定投影类型输出的 segment
    valid_segs = []
    for seg_dir in seg_dirs:
        proj_dir = seg_dir / project_type
        if proj_dir.exists():
            valid_segs.append(seg_dir)

    print(f"找到 {len(valid_segs)}/{len(seg_dirs)} 个有 {project_type} 输出的 segment")

    if not valid_segs:
        print(f"错误: 没有可用的 segment")
        return

    success_count = 0
    for seg_dir in valid_segs:
        seg_name = seg_dir.name
        proj_dir = seg_dir / project_type

        # 获取时间戳目录
        ts_folders = get_sorted_timestamp_folders(proj_dir)
        if not ts_folders:
            print(f"跳过 {seg_name}: 无时间戳目录")
            continue

        # 读取朝向（从 segment_pipeline 生成的 direction.json）
        direction = load_direction(seg_dir)

        print(f"\n处理: {seg_name} ({len(ts_folders)} 帧, {direction})")

        for cam_name in CAMERA_NAMES:
            transfer_cam_name = CAMERA_NAME_MAPPING[cam_name]

            # 收集 GT 图像
            gt_paths = []
            for folder in ts_folders:
                img = folder / 'gt' / f"{cam_name}.jpg"
                if img.exists():
                    gt_paths.append(img)

            # 收集 control 图像
            control_paths = []
            for folder in ts_folders:
                img = folder / control_subdir / f"{cam_name}.jpg"
                if img.exists():
                    control_paths.append(img)

            if not gt_paths or not control_paths:
                continue

            video_filename = f"{seg_name}.mp4"

            # GT 视频
            gt_video = Path(args.output_dir) / 'videos' / transfer_cam_name / video_filename
            create_video_from_images(gt_paths, gt_video, args.fps)

            # Control 视频
            ctrl_video = Path(args.output_dir) / f'control_input_{control_input_type}' / transfer_cam_name / video_filename
            create_video_from_images(control_paths, ctrl_video, args.fps)

            # Caption JSON（包含朝向）
            caption_path = Path(args.output_dir) / 'captions' / transfer_cam_name / f"{seg_name}.json"
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption_data = create_caption_json(
                seg_name, transfer_cam_name, caption_template, direction
            )
            with open(caption_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, indent=2, ensure_ascii=False)

        print(f"  -> {seg_name}.mp4")
        success_count += 1

    print(f"\n完成: {success_count}/{len(valid_segs)} 个 segment")


# ============================================================
# Legacy 模式（兼容旧版投影输出）
# ============================================================

def select_middle_frames(timestamp_folders, total_frames):
    """从时间戳文件夹列表中选取中间的 N 帧"""
    total_available = len(timestamp_folders)
    if total_available < total_frames:
        print(f"警告: 可用帧数 {total_available} 少于需求 {total_frames}，将使用所有可用帧")
        return timestamp_folders

    start_idx = (total_available - total_frames) // 2
    selected = timestamp_folders[start_idx:start_idx + total_frames]
    print(f"  可用帧数: {total_available}")
    print(f"  选取范围: [{start_idx}, {start_idx + total_frames}) (中间 {total_frames} 帧)")
    return selected


def process_legacy_mode(args):
    """兼容旧版的投影目录结构"""
    project_dir_map = {
        'depth': 'depth投影', 'depth_dense': 'depth稠密化投影',
        'hdmap': 'HDMap投影', 'blur': 'blur投影',
        'blur_dense': 'blur稠密化投影', 'basic': '基本点云投影'
    }

    defaults = PROJECT_TYPE_DEFAULTS.get(args.project_type, {})
    control_subdir = args.control_subdir or defaults.get('control_subdir', 'proj')
    control_input_type = args.control_input_type or defaults.get('control_input_type', args.project_type)
    caption_template = args.caption_template or defaults.get('caption', '')

    print(f"模式: legacy")

    project_dir = project_dir_map.get(args.project_type, args.project_type)

    for scene_id in args.scenes:
        print(f"\n{'='*60}")
        print(f"处理场景: {scene_id}")
        scene_dir = Path(args.project_root) / project_dir / scene_id / scene_id

        ts_folders = get_sorted_timestamp_folders(scene_dir)
        if not ts_folders:
            print(f"跳过场景 {scene_id}：未找到时间戳文件夹")
            continue

        total_frames = args.frames_per_seg * args.num_segs
        selected = select_middle_frames(ts_folders, total_frames)

        for cam_name in CAMERA_NAMES:
            transfer_cam_name = CAMERA_NAME_MAPPING[cam_name]

            for seg_idx in range(args.num_segs):
                seg_id = seg_idx + 1
                start = seg_idx * args.frames_per_seg
                end = min(start + args.frames_per_seg, len(selected))
                seg_folders = selected[start:end]

                gt_paths = [f / 'gt' / f"{cam_name}.jpg" for f in seg_folders
                            if (f / 'gt' / f"{cam_name}.jpg").exists()]
                ctrl_paths = [f / control_subdir / f"{cam_name}.jpg" for f in seg_folders
                              if (f / control_subdir / f"{cam_name}.jpg").exists()]

                if not gt_paths or not ctrl_paths:
                    continue

                video_filename = f"{scene_id}_seg{seg_id:02d}.mp4"

                gt_video = Path(args.output_dir) / 'videos' / transfer_cam_name / video_filename
                create_video_from_images(gt_paths, gt_video, args.fps)

                ctrl_video = Path(args.output_dir) / f'control_input_{control_input_type}' / transfer_cam_name / video_filename
                create_video_from_images(ctrl_paths, ctrl_video, args.fps)

                caption_path = Path(args.output_dir) / 'captions' / transfer_cam_name / f"{scene_id}_seg{seg_id:02d}.json"
                caption_path.parent.mkdir(parents=True, exist_ok=True)
                caption_data = {
                    "scene_id": scene_id,
                    "segment_id": f"seg{seg_id:02d}",
                    "camera": transfer_cam_name,
                    "caption": caption_template.format(
                        scene=scene_id, seg=seg_id, camera=transfer_cam_name
                    ) if caption_template else f"Scene {scene_id} seg{seg_id:02d} from {transfer_cam_name}"
                }
                with open(caption_path, 'w', encoding='utf-8') as f:
                    json.dump(caption_data, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成")


# ============================================================
# 主函数
# ============================================================

def main():
    args = parse_args()

    print("="*60)
    print("Transfer2 视频数据集生成器")
    print("="*60)

    if args.segments_dir:
        # Segment 模式
        if not args.project_type:
            print("错误: segment 模式需要 --project-type 参数")
            sys.exit(1)
        process_segment_mode(args)
    elif args.scenes:
        # Legacy 模式
        if not args.project_type:
            print("错误: legacy 模式需要 --project-type 参数")
            sys.exit(1)
        process_legacy_mode(args)
    else:
        print("错误: 请指定 --segments-dir (segment模式) 或 --scenes (legacy模式)")
        sys.exit(1)

    # 显示输出结构
    print(f"\n输出目录: {args.output_dir}/")
    print(f"  videos/{{camera}}/{{seg_name}}.mp4")
    if args.project_type:
        cit = args.control_input_type or PROJECT_TYPE_DEFAULTS.get(
            args.project_type, {}).get('control_input_type', args.project_type)
        print(f"  control_input_{cit}/{{camera}}/{{seg_name}}.mp4")
    print(f"  captions/{{camera}}/{{seg_name}}.json")


if __name__ == "__main__":
    main()
