#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车端标注投影到车端GT图像（指定时间戳，全部相机，全部类别）

参照路侧投影的绘制风格：每个物体按ID着色，线宽1，标签显示。

Usage:
    python verify_carside_at_timestamp.py --clip 053 --timestamp 1743583130886
    python verify_carside_at_timestamp.py --clip 053 --timestamp 1743583130886 --output-dir verify_053
"""

import json
import sys
import re
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
import glob

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_utils

VEHICLE_CALIB_DIR = Path(common_utils.VEHICLE_CALIB_DIR)
DATASET_ROOT = common_utils.DATASET_ROOT

CAMS = {1: "FN", 2: "FW", 3: "FL", 4: "FR", 5: "RL", 6: "RR", 7: "RN"}
RES = {1: (3840, 2160), 2: (3840, 2160), 3: (3840, 2160), 4: (3840, 2160),
       5: (1920, 1080), 6: (1920, 1080), 7: (1920, 1080)}

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def get_color_by_id(obj_id):
    """基于ID生成HSV均匀分布的鲜艳颜色 (BGR)"""
    hue = (obj_id * 41) % 180
    color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color_bgr)


def quat2R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])


def load_cam_calib(cam_id):
    K = np.array(yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"))['K']).reshape(3, 3)
    D = np.array(yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"))['D'])
    ext = yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_extrinsics.yaml"))['transform']
    q = [ext['rotation'][k] for k in ['x', 'y', 'z', 'w']]
    t = np.array([ext['translation'][k] for k in ['x', 'y', 'z']])
    R = quat2R(q)
    w, h = RES[cam_id]
    # 所有相机都用 plumb_bob 去畸变
    nK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
    return {'K': K, 'D': D, 'R': R, 't': t, 'nK': nK, 'w': w, 'h': h}


def bbox3d_corners(obj):
    x, y, z = obj['x'], obj['y'], obj['z']
    l, w, h = obj['length'], obj['width'], obj['height']
    yaw = obj['yaw']
    corners = np.array([
        [l / 2, w / 2, -h / 2], [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2], [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, h / 2], [-l / 2, w / 2, h / 2],
        [-l / 2, -w / 2, h / 2], [l / 2, -w / 2, h / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ corners.T).T + np.array([x, y, z])


def find_nearest_image(img_dir, target_ts_sec):
    """找最近时间戳的图像"""
    best_file = None
    best_diff = float('inf')
    for f in Path(img_dir).glob("*.jpg"):
        match = re.search(r'_(\d+\.\d+)\.jpg$', f.name)
        if match:
            file_ts = float(match.group(1))
            diff = abs(file_ts - target_ts_sec)
            if diff < best_diff:
                best_diff = diff
                best_file = f
    return best_file, best_diff


def find_nearest_label(label_dir, target_ts_sec):
    """找最近时间戳的标注文件 (车端标注文件名是秒格式: 1743583130.886123.json)"""
    best_file = None
    best_diff = float('inf')
    for f in Path(label_dir).glob("*.json"):
        try:
            file_ts = float(f.stem)
        except ValueError:
            continue
        diff = abs(file_ts - target_ts_sec)
        if diff < best_diff:
            best_diff = diff
            best_file = f
    return best_file, best_diff


def undistort_image(img, cam):
    """去畸变图像 (plumb_bob 模型)"""
    return cv2.undistort(img, cam['K'], cam['D'], None, cam['nK'])


def project_and_draw(img, obj, cam):
    """投影3D框到去畸变后的图像上"""
    corners = bbox3d_corners(obj)

    # lidar → camera
    Rl2c = cam['R'].T
    tl2c = -cam['R'].T @ cam['t']
    pts_cam = (Rl2c @ corners.T).T + tl2c

    valid = pts_cam[:, 2] > 0.1
    if sum(valid) < 2:
        return False

    # 用去畸变后的内参 nK 做线性投影
    corners_2d = np.full((8, 2), -1.0)
    for i in range(8):
        if valid[i]:
            uv = cam['nK'] @ pts_cam[i]
            corners_2d[i] = uv[:2] / uv[2]

    in_image = ((corners_2d[:, 0] >= 0) & (corners_2d[:, 0] < cam['w']) &
                (corners_2d[:, 1] >= 0) & (corners_2d[:, 1] < cam['h']))
    drawable = valid & in_image

    if sum(drawable) < 2:
        return False

    color = get_color_by_id(obj.get('id', 0))
    label_text = f"{obj.get('label', '')}_{obj.get('id', '')}"

    for i, j in BOX_EDGES:
        if valid[i] and valid[j]:
            p1 = tuple(corners_2d[i].astype(int))
            p2 = tuple(corners_2d[j].astype(int))
            cv2.line(img, p1, p2, color, 1)

    # 标签
    for k in range(8):
        if drawable[k]:
            pos = tuple(corners_2d[k].astype(int))
            cv2.putText(img, label_text, (pos[0], pos[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            break

    return True


def main():
    parser = argparse.ArgumentParser(description='车端标注投影到车端GT图像（指定时间戳）')
    parser.add_argument('--clip', required=True, help='clip 编号 (如 053)')
    parser.add_argument('--timestamp', required=True, type=int,
                        help='目标时间戳（毫秒，如 1743583130886）')
    parser.add_argument('--output-dir', default='verify_carside_ts_output',
                        help='输出目录')
    args = parser.parse_args()

    clip_dir = common_utils.find_scene_path(args.clip)
    if not clip_dir:
        print(f"找不到 clip {args.clip}")
        return

    clip_dir = Path(clip_dir)
    car_label_dir = clip_dir / "car_labels" / "interpolation_labels"
    car_image_dir = clip_dir / "car" / "images"

    target_ts_sec = args.timestamp / 1000.0

    # 找最近标注
    label_file, label_diff = find_nearest_label(car_label_dir, target_ts_sec)
    if label_file is None:
        print(f"未找到车端标注")
        return
    print(f"标注: {label_file.name} (时间差: {label_diff * 1000:.1f}ms)")

    with open(label_file) as f:
        ann = json.load(f)

    objects = ann.get('object', [])
    print(f"物体数: {len(objects)}")

    # 加载相机标定
    cam_calibs = {}
    for cid in CAMS:
        cam_calibs[cid] = load_cam_calib(cid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cid, cam_name in CAMS.items():
        cam = cam_calibs[cid]
        img_dir = car_image_dir / cam_name

        if not img_dir.exists():
            continue

        img_file, img_diff = find_nearest_image(img_dir, target_ts_sec)
        if img_file is None or img_diff > 0.5:
            print(f"  {cam_name}: 未找到图像 (diff={img_diff:.3f}s)")
            continue

        print(f"  {cam_name}: 图像 {img_file.name} (时间差: {img_diff*1000:.1f}ms)")

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        # 去畸变
        img = undistort_image(img, cam)

        box_count = 0
        for obj in objects:
            if project_and_draw(img, obj, cam):
                box_count += 1

        # 信息
        info = f"Clip{args.clip} {cam_name} ts={args.timestamp} ({box_count} boxes)"
        cv2.putText(img, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out_path = output_dir / f"{args.clip}_{cam_name}_bbox.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"  {cam_name}: {box_count} 框 -> {out_path}")

    print(f"\n完成! 输出: {output_dir}")


if __name__ == "__main__":
    main()
