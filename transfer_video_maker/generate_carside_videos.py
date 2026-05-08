#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车端 HDMap bbox 视频生成器

从 generate_carside_hdmap.py 的输出生成训练用视频 + caption。
只处理 FW 相机。

输入结构:
  {input_root}/{clip_id}/seg{00-02}/{ts}/overlay/FW.jpg   黑底bbox
  {input_root}/{clip_id}/seg{00-02}/{ts}/gt/FW.jpg        去畸变GT

输出结构:
  {output}/videos/{camera}/{clip_id}_seg{NN}.mp4
  {output}/control_input_carside_hdmap/{camera}/{clip_id}_seg{NN}.mp4
  {output}/captions/{camera}/{clip_id}_seg{NN}.json

用法:
  python generate_carside_videos.py \
      --input /mnt/zihanw/car_side_bbox \
      --output /mnt/zihanw/car_side_bbox_videos
"""

import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

CAMERA_NAME = 'FW'
TRANSFER_CAM_NAME = 'ftheta_camera_front_wide_120fov'
VIEW_PREFIX = 'Front wide view'

SCENE_DESCRIPTION = (
    "Northern Chinese suburban intersection captured in early spring. "
    "Clear daytime conditions with bright blue sky and soft natural sunlight casting gentle shadows. "
    "Wide multi-lane asphalt road surface in good condition with crisp white lane markings, "
    "directional arrows, and crosswalk patterns. Beige and tan colored high-rise residential "
    "apartment buildings line both sides of the street, typical of Chinese suburban architecture. "
    "Rows of bare deciduous trees with leafless branches stand along the sidewalks, characteristic "
    "of late winter to early spring season. White painted metal safety railings separate the road "
    "from pedestrian areas. Green traffic signals mounted on overhead poles with directional signs. "
    "Street lamp posts visible along the road. Occasional mixed traffic including sedans, SUVs, "
    "buses, trucks, and non-motorized road users such as pedestrians, cyclists, and electric "
    "tricycles. Clean urban environment with well-maintained infrastructure."
)


def get_sorted_timestamp_folders(seg_dir):
    seg_path = Path(seg_dir)
    folders = []
    for f in seg_path.iterdir():
        if f.is_dir():
            try:
                float(f.name)
                folders.append(f)
            except ValueError:
                continue
    folders.sort(key=lambda x: float(x.name))
    return folders


def create_video_from_images(image_paths, output_path, fps, target_resolution=(1280, 720)):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, target_resolution)

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  警告: 无法读取 {img_path}")
            continue
        if img.shape[1] != target_resolution[0] or img.shape[0] != target_resolution[1]:
            img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
        writer.write(img)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description='车端 HDMap bbox 视频生成')
    parser.add_argument('--input', required=True,
                        help='generate_carside_hdmap.py 输出目录 (如 /mnt/zihanw/car_side_bbox)')
    parser.add_argument('--output', required=True,
                        help='视频输出目录')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--clips', nargs='*', default=None,
                        help='指定 clip ID (如 004 006)，不指定则处理全部')
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    # 扫描所有 clip
    clip_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    if args.clips:
        clip_dirs = [d for d in clip_dirs if d.name in args.clips]

    print(f"输入: {input_root}")
    print(f"输出: {output_root}")
    print(f"Clip 数: {len(clip_dirs)}")
    print()

    total_videos = 0
    for clip_dir in tqdm(clip_dirs, desc="Clips"):
        clip_id = clip_dir.name

        # 扫描 seg 目录
        seg_dirs = sorted([d for d in clip_dir.iterdir()
                           if d.is_dir() and d.name.startswith('seg')])

        for seg_dir in seg_dirs:
            seg_name = seg_dir.name  # seg00, seg01, seg02
            ts_folders = get_sorted_timestamp_folders(seg_dir)
            if not ts_folders:
                continue

            # 收集图像路径
            gt_paths = [f / 'gt' / f'{CAMERA_NAME}.jpg' for f in ts_folders
                        if (f / 'gt' / f'{CAMERA_NAME}.jpg').exists()]
            ctrl_paths = [f / 'overlay' / f'{CAMERA_NAME}.jpg' for f in ts_folders
                          if (f / 'overlay' / f'{CAMERA_NAME}.jpg').exists()]

            if not gt_paths or not ctrl_paths:
                continue

            video_name = f"{clip_id}_{seg_name}.mp4"

            # GT 视频
            gt_video = output_root / 'videos' / TRANSFER_CAM_NAME / video_name
            create_video_from_images(gt_paths, gt_video, args.fps)

            # Control 视频
            ctrl_video = output_root / f'control_input_carside_hdmap' / TRANSFER_CAM_NAME / video_name
            create_video_from_images(ctrl_paths, ctrl_video, args.fps)

            # Caption
            caption_path = output_root / 'captions' / TRANSFER_CAM_NAME / f"{clip_id}_{seg_name}.json"
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption = f"{VIEW_PREFIX}. {SCENE_DESCRIPTION}"
            caption_data = {
                "segment_name": f"{clip_id}_{seg_name}",
                "camera": TRANSFER_CAM_NAME,
                "caption": caption,
            }
            with open(caption_path, 'w', encoding='utf-8') as f:
                json.dump(caption_data, f, indent=2, ensure_ascii=False)

            total_videos += 1

    print(f"\n完成: 生成 {total_videos} 组视频")
    print(f"输出结构:")
    print(f"  {output_root}/videos/{TRANSFER_CAM_NAME}/{{clip}}_{{seg}}.mp4")
    print(f"  {output_root}/control_input_carside_hdmap/{TRANSFER_CAM_NAME}/{{clip}}_{{seg}}.mp4")
    print(f"  {output_root}/captions/{TRANSFER_CAM_NAME}/{{clip}}_{{seg}}.json")


if __name__ == '__main__':
    main()
