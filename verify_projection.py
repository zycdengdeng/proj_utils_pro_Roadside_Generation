#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投影验证工具
将 ego lidar 坐标系下的 3D 标注投影到生成视频的帧上，验证坐标变换是否正确。

Usage:
    python verify_projection.py \
        --annotations-dir segment_pipeline/output/001_id5_seg01/annotations \
        --video-dir /mnt/zihanw/Output_R2V_world_foundation_model_v1/inference_new_trajectory_generation/001_id5_seg01 \
        --output-dir verify_output/001_id5_seg01
"""

import json
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_utils


# ============================================================
# 配置
# ============================================================

VEHICLE_CALIB_DIR = Path(common_utils.VEHICLE_CALIB_DIR)

VEHICLE_CAMERAS = {
    1: {"name": "FN", "resolution": (3840, 2160)},
    2: {"name": "FW", "resolution": (3840, 2160)},
    3: {"name": "FL", "resolution": (3840, 2160)},
    4: {"name": "FR", "resolution": (3840, 2160)},
    5: {"name": "RL", "resolution": (1920, 1080)},
    6: {"name": "RR", "resolution": (1920, 1080)},
    7: {"name": "RN", "resolution": (1920, 1080)},
}

# 视频文件名 → (cam_id, 内部名)
VIDEO_TO_CAM = {
    "front_tele_30fov":    (1, "FN"),
    "front_wide_120fov":   (2, "FW"),
    "cross_left_120fov":   (3, "FL"),
    "cross_right_120fov":  (4, "FR"),
    "rear_left_70fov":     (5, "RL"),
    "rear_right_70fov":    (6, "RR"),
    "rear_tele_30fov":     (7, "RN"),
}

LABEL_COLORS = {
    "Car": (255, 0, 0), "Suv": (255, 69, 0), "Truck": (255, 140, 0),
    "Bus": (0, 0, 255), "Pedestrian": (0, 255, 0), "Bicycle": (0, 255, 255),
    "Motorcycle": (255, 0, 255), "Non_motor_rider": (255, 215, 0),
    "Motor_rider": (255, 105, 180), "Tricycle": (135, 206, 250),
    "Huge_vehicle": (139, 0, 0), "Vehicle_else": (160, 82, 45),
}

BBOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),  # 底面
    (4,5),(5,6),(6,7),(7,4),  # 顶面
    (0,4),(1,5),(2,6),(3,7),  # 垂直
]


# ============================================================
# 工具函数
# ============================================================

def quaternion_to_rotation_matrix(q):
    """四元数 [x, y, z, w] → 旋转矩阵"""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


def load_camera_calibration(cam_id):
    """加载相机标定 (内参 K, D + 外参 R_cam2lidar, t_cam2lidar)"""
    intrinsics_path = VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"
    extrinsics_path = VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_extrinsics.yaml"

    with open(intrinsics_path, 'r') as f:
        intrinsics = yaml.safe_load(f)
    K = np.array(intrinsics['K']).reshape(3, 3)
    D = np.array(intrinsics['D'])

    with open(extrinsics_path, 'r') as f:
        extrinsics = yaml.safe_load(f)
    transform = extrinsics['transform']
    q = [transform['rotation']['x'], transform['rotation']['y'],
         transform['rotation']['z'], transform['rotation']['w']]
    t = np.array([transform['translation']['x'],
                  transform['translation']['y'],
                  transform['translation']['z']])
    R_cam2lidar = quaternion_to_rotation_matrix(q)

    return K, D, R_cam2lidar, t


def get_3d_bbox_corners(obj):
    """从标注对象计算 3D bbox 的 8 个角点（ego lidar 坐标系）"""
    center = np.array([obj['x'], obj['y'], obj['z']])
    l, w, h = obj['length'], obj['width'], obj['height']
    yaw = obj['yaw']

    corners = np.array([
        [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2],
        [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
        [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2],
        [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2],
    ])

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0, 0, 1]
    ])
    return (R @ corners.T).T + center


def project_corners_to_image(corners_lidar, K, D, R_cam2lidar, t_cam2lidar, cam_id, img_w, img_h):
    """
    将 lidar 坐标系的 3D 点投影到图像上

    Returns:
        corners_2d: list of (u, v) or None for invalid points
        corners_valid: list of bool
    """
    # lidar → camera
    R_lidar2cam = R_cam2lidar.T
    t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar
    points_cam = (R_lidar2cam @ corners_lidar.T).T + t_lidar2cam

    # 计算 new_K（去畸变后的内参）
    if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D[:4], (img_w, img_h), np.eye(3), balance=0.0
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 0, (img_w, img_h))

    corners_2d = []
    corners_valid = []
    margin = 500

    for i in range(len(points_cam)):
        pt = points_cam[i]
        if pt[2] > 0.1:
            uv_h = new_K @ pt
            uv = uv_h[:2] / uv_h[2]
            if -margin <= uv[0] <= img_w + margin and -margin <= uv[1] <= img_h + margin:
                corners_2d.append((float(uv[0]), float(uv[1])))
                corners_valid.append(True)
            else:
                corners_2d.append(None)
                corners_valid.append(False)
        else:
            corners_2d.append(None)
            corners_valid.append(False)

    return corners_2d, corners_valid


def draw_3d_bbox_on_image(img, corners_2d, corners_valid, color, label_text=None, thickness=2):
    """在图像上绘制 3D bbox 线框"""
    for i, j in BBOX_EDGES:
        if corners_valid[i] and corners_valid[j]:
            p1 = (int(corners_2d[i][0]), int(corners_2d[i][1]))
            p2 = (int(corners_2d[j][0]), int(corners_2d[j][1]))
            cv2.line(img, p1, p2, color, thickness)

    # 标签文字
    if label_text:
        for i in range(8):
            if corners_valid[i]:
                pos = (int(corners_2d[i][0]), int(corners_2d[i][1]) - 5)
                cv2.putText(img, label_text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
                break


def extract_video_frames(video_path):
    """提取视频的所有帧"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ============================================================
# 主流程
# ============================================================

def verify_projection(annotations_dir, video_dir, output_dir):
    annotations_dir = Path(annotations_dir)
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有标注文件（按时间戳排序）
    ann_files = sorted(annotations_dir.glob("*.json"))
    if not ann_files:
        print(f"错误: 未找到标注文件 in {annotations_dir}")
        return

    annotations = []
    for f in ann_files:
        with open(f, 'r') as fh:
            annotations.append(json.load(fh))
    print(f"加载 {len(annotations)} 帧标注")

    # 加载相机标定
    print("加载相机标定...")
    cam_calibs = {}
    for cam_id in VEHICLE_CAMERAS:
        K, D, R, t = load_camera_calibration(cam_id)
        cam_calibs[cam_id] = (K, D, R, t)
    print(f"加载 {len(cam_calibs)} 个相机标定")

    # 遍历每个视频
    video_files = sorted(video_dir.glob("*_generated.mp4"))
    if not video_files:
        print(f"错误: 未找到 generated 视频 in {video_dir}")
        return

    for video_path in video_files:
        # 解析相机名: e.g. "front_tele_30fov_generated.mp4"
        stem = video_path.stem  # "front_tele_30fov_generated"
        cam_key = stem.replace("_generated", "")  # "front_tele_30fov"

        if cam_key not in VIDEO_TO_CAM:
            print(f"跳过未知相机: {cam_key}")
            continue

        cam_id, cam_name = VIDEO_TO_CAM[cam_key]
        img_w, img_h = VEHICLE_CAMERAS[cam_id]["resolution"]
        K, D, R_cam2lidar, t_cam2lidar = cam_calibs[cam_id]

        print(f"\n处理 {cam_name} ({cam_key})...")

        # 提取视频帧
        frames = extract_video_frames(video_path)
        print(f"  视频帧数: {len(frames)}, 标注帧数: {len(annotations)}")

        num_frames = min(len(frames), len(annotations))
        annotated_frames = []

        for frame_idx in range(num_frames):
            frame = frames[frame_idx].copy()
            frame_h, frame_w = frame.shape[:2]
            ann = annotations[frame_idx]

            # 如果视频帧分辨率与标定分辨率不同，需要缩放投影
            scale_x = frame_w / img_w
            scale_y = frame_h / img_h

            obj_count = 0
            for obj in ann.get('object', []):
                color = LABEL_COLORS.get(obj['label'], (128, 128, 128))
                corners_lidar = get_3d_bbox_corners(obj)
                corners_2d, corners_valid = project_corners_to_image(
                    corners_lidar, K, D, R_cam2lidar, t_cam2lidar, cam_id, img_w, img_h
                )

                if not any(corners_valid):
                    continue

                # 至少需要两个面（5个以上角点）的投影才画，否则看起来像2D框
                if sum(corners_valid) < 5:
                    continue

                # 缩放到视频帧分辨率
                scaled_2d = []
                for c in corners_2d:
                    if c is not None:
                        scaled_2d.append((c[0] * scale_x, c[1] * scale_y))
                    else:
                        scaled_2d.append(None)

                label_text = f"{obj['label']}_{obj['id']}"
                draw_3d_bbox_on_image(frame, scaled_2d, corners_valid, color,
                                      label_text, thickness=2)
                obj_count += 1

            # 帧信息
            info_text = f"Frame {frame_idx}/{num_frames-1}  |  {cam_name}  |  {obj_count} objects"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            annotated_frames.append(frame)

        # 保存逐帧图像
        if annotated_frames:
            frames_dir = output_dir / f"{cam_key}_frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for idx, f in enumerate(annotated_frames):
                frame_path = frames_dir / f"{idx:03d}.jpg"
                cv2.imwrite(str(frame_path), f)
            print(f"  逐帧保存: {frames_dir}/ ({len(annotated_frames)} 帧)")

            # 保存为视频
            out_path = output_dir / f"{cam_key}_verify.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w = annotated_frames[0].shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, 10, (w, h))
            for f in annotated_frames:
                writer.write(f)
            writer.release()
            print(f"  视频保存: {out_path}")

    print(f"\n验证完成！输出目录: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='投影验证: 将3D标注投影到生成视频上')
    parser.add_argument('--annotations-dir', required=True,
                        help='标注目录 (ego lidar 坐标系)')
    parser.add_argument('--video-dir', required=True,
                        help='生成视频目录 (含 *_generated.mp4)')
    parser.add_argument('--output-dir', default='verify_output',
                        help='输出目录')
    args = parser.parse_args()

    verify_projection(args.annotations_dir, args.video_dir, args.output_dir)
