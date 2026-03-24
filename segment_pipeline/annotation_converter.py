#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注坐标转换模块
将路侧标注从世界坐标系转换到 ego 车辆的 LiDAR 坐标系
"""

import json
import numpy as np
from pathlib import Path

from .ego_transform import get_vehicle_transform


def transform_object_to_ego_frame(obj, ego_rotate_mat, ego_trans, ego_yaw):
    """
    将单个物体从世界坐标系转换到 ego LiDAR 坐标系

    Args:
        obj: 标注对象 dict (世界坐标系)
        ego_rotate_mat: world2lidar 旋转矩阵 (3,3)
        ego_trans: world2lidar 平移向量 (3,1)
        ego_yaw: ego 车辆世界朝向 yaw

    Returns:
        obj_ego: 转换后的标注对象 dict (ego LiDAR 坐标系)
    """
    # 物体世界坐标
    pos_world = np.array([obj['x'], obj['y'], obj['z']]).reshape(3, 1)

    # 世界 → ego LiDAR
    pos_lidar = ego_rotate_mat @ pos_world + ego_trans

    # yaw 转换: 相对于 ego 的朝向
    obj_yaw = obj.get('yaw', 0.0)
    yaw_in_ego = obj_yaw - ego_yaw

    # 归一化 yaw 到 [-pi, pi]
    yaw_in_ego = (yaw_in_ego + np.pi) % (2 * np.pi) - np.pi

    return {
        'id': obj['id'],
        'label': obj['label'],
        'x': float(pos_lidar[0, 0]),
        'y': float(pos_lidar[1, 0]),
        'z': float(pos_lidar[2, 0]),
        'length': obj.get('length', 0.0),
        'width': obj.get('width', 0.0),
        'height': obj.get('height', 0.0),
        'roll': obj.get('roll', 0.0),
        'pitch': obj.get('pitch', 0.0),
        'yaw': float(yaw_in_ego),
        'occlusion': obj.get('occlusion', 0),
        'num_points': obj.get('num_points', 0),
        'vx': obj.get('vx', 0.0),
        'vy': obj.get('vy', 0.0),
    }


def convert_single_frame(annotation_path, vehicle_id, output_path):
    """
    转换单帧标注到 ego LiDAR 坐标系

    Args:
        annotation_path: 路侧标注 JSON 路径
        vehicle_id: ego 车辆 ID
        output_path: 输出 JSON 路径

    Returns:
        num_objects: 转换的物体数量, -1 表示失败
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    # 获取 ego 变换
    try:
        rotate, trans, world_pos, world_yaw, ego_obj = get_vehicle_transform(
            annotation, vehicle_id
        )
    except ValueError as e:
        print(f"  跳过: {e}")
        return -1

    # 旋转矩阵
    import cv2
    R_world2lidar = cv2.Rodrigues(rotate)[0]

    # 转换所有非 ego 物体
    objects_ego = []
    for obj in annotation.get('object', []):
        if obj['id'] == vehicle_id:
            continue  # 排除 ego 自身

        obj_ego = transform_object_to_ego_frame(
            obj, R_world2lidar, trans, world_yaw
        )
        objects_ego.append(obj_ego)

    # 构造输出
    # 保留原始标注的 metadata
    output = {
        'timestamp': annotation.get('timestamp', ''),
        'interpolated': annotation.get('interpolated', False),
        'pose_enhanced': annotation.get('pose_enhanced', False),
        'fixed_version': annotation.get('fixed_version', False),
        'object': objects_ego,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return len(objects_ego)


def convert_segment_annotations(label_files, timestamps, vehicle_id, output_dir):
    """
    转换一个 segment 的所有帧标注

    Args:
        label_files: 标注文件路径列表
        timestamps: 时间戳列表
        vehicle_id: ego 车辆 ID
        output_dir: 输出目录

    Returns:
        success_count: 成功转换的帧数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for label_file, ts in zip(label_files, timestamps):
        output_path = output_dir / f"{ts}.json"
        num_objects = convert_single_frame(label_file, vehicle_id, output_path)
        if num_objects >= 0:
            success_count += 1

    print(f"  annotations: {success_count}/{len(timestamps)} 帧, 输出到 {output_dir}")
    return success_count
