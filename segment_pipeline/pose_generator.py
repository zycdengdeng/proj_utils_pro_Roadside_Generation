#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pose 生成模块
从路侧标注中提取 ego 车辆位姿，输出 pose CSV
格式: timestamp,x,y,z,qx,qy,qz,qw,frame_id
"""

import json
import csv
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def euler_to_quaternion(roll, pitch, yaw):
    """
    欧拉角 (roll, pitch, yaw) → 四元数 (qx, qy, qz, qw)

    使用 scipy 的 ZYX 内旋顺序（即 extrinsic XYZ）
    """
    r = R.from_euler('xyz', [roll, pitch, yaw])
    quat = r.as_quat()  # [qx, qy, qz, qw]
    return quat


def extract_ego_pose_from_annotation(annotation_path, vehicle_id):
    """
    从单帧标注文件提取 ego 车辆位姿

    Args:
        annotation_path: 标注 JSON 文件路径
        vehicle_id: ego 车辆 ID

    Returns:
        pose_dict: {x, y, z, roll, pitch, yaw, qx, qy, qz, qw} or None
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    for obj in annotation.get('object', []):
        if obj['id'] == vehicle_id:
            roll = obj.get('roll', 0.0)
            pitch = obj.get('pitch', 0.0)
            yaw = obj['yaw']

            qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)

            return {
                'x': obj['x'],
                'y': obj['y'],
                'z': obj['z'],
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'qx': qx,
                'qy': qy,
                'qz': qz,
                'qw': qw,
            }

    return None


def generate_pose_csv(label_files, timestamps, vehicle_id, output_path):
    """
    从多帧标注生成 pose CSV

    Args:
        label_files: 标注文件路径列表 (按时间排序)
        timestamps: 对应的时间戳列表 (毫秒, int)
        vehicle_id: ego 车辆 ID
        output_path: 输出 CSV 文件路径

    Returns:
        poses: 提取的 pose 列表 (用于后续方向推算)
        missing_frames: 缺失帧的索引列表
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    poses = []
    missing_frames = []

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'frame_id'])

        for i, (label_file, ts) in enumerate(zip(label_files, timestamps)):
            pose = extract_ego_pose_from_annotation(label_file, vehicle_id)

            if pose is None:
                print(f"  警告: 帧 {i} ({ts}) 中未找到车辆 {vehicle_id}")
                missing_frames.append(i)
                continue

            # 时间戳转为秒 (float)
            ts_sec = ts / 1000.0

            writer.writerow([
                f"{ts_sec:.2f}",
                pose['x'],
                pose['y'],
                pose['z'],
                pose['qx'],
                pose['qy'],
                pose['qz'],
                pose['qw'],
                'base_link'
            ])

            poses.append({
                'timestamp': ts,
                'x': pose['x'],
                'y': pose['y'],
                'z': pose['z'],
                'yaw': pose['yaw'],
            })

    print(f"  pose.csv: {len(poses)}/{len(timestamps)} 帧")
    if missing_frames:
        print(f"  缺失帧: {missing_frames}")

    return poses, missing_frames
