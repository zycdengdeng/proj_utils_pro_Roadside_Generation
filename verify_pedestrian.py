#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证行人标注投影：将 Pedestrian 3D 框投影到生成视频帧上

Usage:
    python verify_pedestrian.py \
        --video-root /mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_new_trajectory_generation \
        --end 025_id22_seg01
"""

import json
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_utils

VEHICLE_CALIB_DIR = Path(common_utils.VEHICLE_CALIB_DIR)

CAMS = {1: "FN", 2: "FW", 3: "FL", 4: "FR", 5: "RL", 6: "RR", 7: "RN"}
RES = {1:(3840,2160), 2:(3840,2160), 3:(3840,2160), 4:(3840,2160),
       5:(1920,1080), 6:(1920,1080), 7:(1920,1080)}

VIDEO_TO_CAM = {
    "front_tele_30fov": 1, "front_wide_120fov": 2,
    "cross_left_120fov": 3, "cross_right_120fov": 4,
    "rear_left_70fov": 5, "rear_right_70fov": 6, "rear_tele_30fov": 7,
}

BBOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


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


def project_and_draw(img, obj, cam, scale_x, scale_y):
    """投影一个物体到图像上，返回是否可见"""
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
                corners_2d.append((u * scale_x, v * scale_y))
                corners_ok.append(True)
            else:
                corners_2d.append(None)
                corners_ok.append(False)
        else:
            corners_2d.append(None)
            corners_ok.append(False)

    if sum(corners_ok) < 5:
        return False

    # 检查投影后是否在画面内
    valid_pts = [c for c, ok in zip(corners_2d, corners_ok) if ok]
    xs = [p[0] for p in valid_pts]
    ys = [p[1] for p in valid_pts]
    img_h, img_w = img.shape[:2]

    # 如果所有有效点都在画面外，跳过
    if max(xs) < 0 or min(xs) > img_w or max(ys) < 0 or min(ys) > img_h:
        return False

    # 画线
    color = (0, 255, 0)  # 绿色
    thickness = max(2, int(min(img_w, img_h) / 500))  # 根据分辨率自适应线宽
    for i, j in BBOX_EDGES:
        if corners_ok[i] and corners_ok[j]:
            p1 = (int(corners_2d[i][0]), int(corners_2d[i][1]))
            p2 = (int(corners_2d[j][0]), int(corners_2d[j][1]))
            cv2.line(img, p1, p2, color, thickness)

    # 标签
    for i in range(8):
        if corners_ok[i]:
            pos = (int(corners_2d[i][0]), int(corners_2d[i][1]) - 5)
            cv2.putText(img, f"Ped_{obj['id']}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            break

    return True


def main():
    parser = argparse.ArgumentParser(description='验证行人标注投影到生成视频')
    parser.add_argument('--video-root', required=True,
                        help='生成视频根目录 (含 {seg_name}/ 子目录)')
    parser.add_argument('--segments-dir', default=str(Path(__file__).resolve().parent / 'segment_pipeline' / 'output'),
                        help='segment_pipeline 输出目录')
    parser.add_argument('--end', default='025_id22_seg01',
                        help='统计到哪个 seg')
    parser.add_argument('--output-dir', default='verify_pedestrian_output',
                        help='输出目录')
    parser.add_argument('--frame-idx', type=int, default=14,
                        help='取第几帧 (默认中间帧14)')
    args = parser.parse_args()

    video_root = Path(args.video_root)
    seg_root = Path(args.segments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有相机标定
    cam_calibs = {}
    for cid in CAMS:
        cam_calibs[cid] = load_cam_calib(cid)

    # 找有行人的 seg
    all_segs = sorted([d.name for d in seg_root.iterdir() if d.is_dir() and d.name <= args.end])
    count = 0

    for seg_name in all_segs:
        ann_dir = seg_root / seg_name / "annotations"
        if not ann_dir.exists():
            continue

        # 检查是否有行人
        ann_files = sorted(ann_dir.glob("*.json"))
        if not ann_files or args.frame_idx >= len(ann_files):
            continue

        ann_file = ann_files[args.frame_idx]
        with open(ann_file) as f:
            ann = json.load(f)

        peds = [obj for obj in ann.get('object', []) if obj['label'] == 'Pedestrian']
        if not peds:
            continue

        # 检查生成视频目录是否存在
        vid_dir = video_root / seg_name
        if not vid_dir.exists():
            continue

        # 对每个相机视角
        for vid_file in sorted(vid_dir.glob("*_generated.mp4")):
            cam_key = vid_file.stem.replace("_generated", "")
            if cam_key not in VIDEO_TO_CAM:
                continue

            cam_id = VIDEO_TO_CAM[cam_key]
            cam = cam_calibs[cam_id]

            # 提取指定帧
            cap = cv2.VideoCapture(str(vid_file))
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]
            scale_x = frame_w / cam['w']
            scale_y = frame_h / cam['h']

            # 投影行人
            drawn = False
            for ped in peds:
                if project_and_draw(frame, ped, cam, scale_x, scale_y):
                    drawn = True

            if drawn:
                # 写 seg 名和相机名
                cv2.putText(frame, f"{seg_name} | {CAMS[cam_id]} | frame{args.frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                out_path = output_dir / f"{seg_name}_{CAMS[cam_id]}.jpg"
                cv2.imwrite(str(out_path), frame)
                count += 1

    print(f"共保存 {count} 张含行人投影的图像到 {output_dir}/")


if __name__ == "__main__":
    main()
