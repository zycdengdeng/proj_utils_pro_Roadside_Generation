#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将车端标注（车载LiDAR坐标系）投影到车端GT图像上验证

Usage:
    python verify_carside_labels.py --clip 003 --label Bus
    python verify_carside_labels.py --clip 003 --label Pedestrian
    python verify_carside_labels.py --clip 003 --label Bus Pedestrian --frame-idx 100
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
RES = {1:(3840,2160), 2:(3840,2160), 3:(3840,2160), 4:(3840,2160),
       5:(1920,1080), 6:(1920,1080), 7:(1920,1080)}

BBOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]

LABEL_COLORS = {
    "Bus": (0, 0, 255),        # 蓝
    "Pedestrian": (0, 255, 0), # 绿
    "Car": (255, 0, 0),        # 红
    "Suv": (255, 69, 0),
    "Truck": (255, 140, 0),
}


def quat2R(q):
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]
    ])


def load_cam_calib(cam_id):
    K = np.array(yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"))['K']).reshape(3,3)
    D = np.array(yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"))['D'])
    ext = yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_extrinsics.yaml"))['transform']
    q = [ext['rotation'][k] for k in ['x','y','z','w']]
    t = np.array([ext['translation'][k] for k in ['x','y','z']])
    R = quat2R(q)
    w, h = RES[cam_id]
    if cam_id in [2,3,4] and np.max(np.abs(D)) > 1:
        nK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D[:4], (w,h), np.eye(3), balance=0.0)
    else:
        nK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 0, (w,h))
    return {'K': K, 'D': D, 'R': R, 't': t, 'nK': nK, 'w': w, 'h': h}


def get_bbox_corners(obj):
    center = np.array([obj['x'], obj['y'], obj['z']])
    l, w, h = obj['length'], obj['width'], obj['height']
    yaw = obj['yaw']
    corners = np.array([
        [-l/2,-w/2,-h/2],[l/2,-w/2,-h/2],[l/2,w/2,-h/2],[-l/2,w/2,-h/2],
        [-l/2,-w/2,h/2],[l/2,-w/2,h/2],[l/2,w/2,h/2],[-l/2,w/2,h/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (R @ corners.T).T + center


def find_nearest_image(img_dir, target_ts):
    """在图像目录中找最接近目标时间戳的图像"""
    best_file = None
    best_diff = float('inf')

    for f in img_dir.glob("*.jpg"):
        # 格式: addc_xxx_{timestamp}.jpg
        match = re.search(r'_(\d+\.\d+)\.jpg$', f.name)
        if match:
            file_ts = float(match.group(1))
            diff = abs(file_ts - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_file = f

    return best_file, best_diff


def project_and_draw(img, obj, cam):
    """投影3D框到图像上，返回是否可见"""
    corners = get_bbox_corners(obj)
    Rl2c = cam['R'].T
    tl2c = -cam['R'].T @ cam['t']
    pts_cam = (Rl2c @ corners.T).T + tl2c

    valid = pts_cam[:, 2] > 0.1
    if sum(valid) < 5:
        return False

    corners_2d = []
    corners_ok = []
    for i in range(8):
        if valid[i]:
            uv = cam['nK'] @ pts_cam[i]
            u, v = uv[0]/uv[2], uv[1]/uv[2]
            if -500 <= u <= cam['w']+500 and -500 <= v <= cam['h']+500:
                corners_2d.append((u, v))
                corners_ok.append(True)
            else:
                corners_2d.append(None)
                corners_ok.append(False)
        else:
            corners_2d.append(None)
            corners_ok.append(False)

    if sum(corners_ok) < 5:
        return False

    # 检查是否在画面内
    valid_pts = [c for c, ok in zip(corners_2d, corners_ok) if ok]
    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    img_h, img_w = img.shape[:2]
    if max(xs) < 0 or min(xs) > img_w or max(ys) < 0 or min(ys) > img_h:
        return False

    color = LABEL_COLORS.get(obj['label'], (128, 128, 128))
    thickness = max(2, int(min(img_w, img_h) / 500))

    for i, j in BBOX_EDGES:
        if corners_ok[i] and corners_ok[j]:
            p1 = (int(corners_2d[i][0]), int(corners_2d[i][1]))
            p2 = (int(corners_2d[j][0]), int(corners_2d[j][1]))
            cv2.line(img, p1, p2, color, thickness)

    # 标签
    for i in range(8):
        if corners_ok[i]:
            pos = (int(corners_2d[i][0]), int(corners_2d[i][1]) - 5)
            cv2.putText(img, f"{obj['label']}_{obj['id']}", pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            break

    return True


def main():
    parser = argparse.ArgumentParser(description='车端标注投影到车端GT图像')
    parser.add_argument('--clip', required=True, help='clip 编号 (如 003)')
    parser.add_argument('--label', nargs='+', default=['Bus', 'Pedestrian'],
                        help='要验证的类别')
    parser.add_argument('--frame-idx', type=int, default=None,
                        help='指定第几帧标注 (默认取中间帧)')
    parser.add_argument('--output-dir', default='verify_carside_output',
                        help='输出目录')
    args = parser.parse_args()

    clip_dir = common_utils.find_scene_path(args.clip)
    if not clip_dir:
        print(f"找不到 clip {args.clip}")
        return

    clip_dir = Path(clip_dir)
    car_label_dir = clip_dir / "car_labels" / "interpolation_labels"
    car_image_dir = clip_dir / "car" / "images"

    if not car_label_dir.exists():
        print(f"车端标注不存在: {car_label_dir}")
        return

    # 加载标注
    label_files = sorted(car_label_dir.glob("*.json"))
    if not label_files:
        print("无车端标注文件")
        return

    # 选帧
    if args.frame_idx is not None:
        idx = min(args.frame_idx, len(label_files) - 1)
    else:
        idx = len(label_files) // 2

    label_file = label_files[idx]
    with open(label_file) as f:
        ann = json.load(f)

    # 从文件名提取时间戳 (秒)
    label_ts = float(label_file.stem)
    print(f"Clip {args.clip}, 帧 {idx}/{len(label_files)}, ts={label_ts}")

    target_labels = set(args.label)
    targets = [obj for obj in ann.get('object', []) if obj['label'] in target_labels]
    print(f"目标类别: {args.label}, 找到 {len(targets)} 个物体")

    if not targets:
        print("该帧没有目标类别的物体")
        return

    # 加载相机标定
    cam_calibs = {}
    for cid in CAMS:
        cam_calibs[cid] = load_cam_calib(cid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for cid, cam_name in CAMS.items():
        cam = cam_calibs[cid]
        img_dir = car_image_dir / cam_name

        if not img_dir.exists():
            continue

        # 找最近时间戳的图像
        img_file, ts_diff = find_nearest_image(img_dir, label_ts)
        if img_file is None or ts_diff > 0.5:
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            continue

        drawn = False
        for obj in targets:
            if project_and_draw(img, obj, cam):
                drawn = True

        if drawn:
            info = f"Clip{args.clip} {cam_name} frame{idx} ts={label_ts:.2f}"
            cv2.putText(img, info, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

            out_path = output_dir / f"{args.clip}_{cam_name}_frame{idx}.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"  保存: {out_path}")
            count += 1

    print(f"\n共保存 {count} 张图像到 {output_dir}/")


if __name__ == "__main__":
    main()
