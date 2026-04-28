#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注坐标转换模块
将路侧标注从世界坐标系转换到 ego 车辆的 LiDAR 坐标系
"""

import json
import numpy as np
from pathlib import Path

from .ego_transform import (
    get_world2ego_transform,
    get_world2lidar_transform_from_pose,
)


def transform_object_to_ego_frame(obj, R_world2lidar, t_world2lidar):
    """
    将单个物体从世界坐标系转换到 ego LiDAR 坐标系

    位置: 通过完整的 world2lidar 旋转矩阵 + 平移
    朝向: 将物体朝向向量通过 R_world2lidar 旋转后提取 yaw/roll/pitch

    Args:
        obj: 标注对象 dict (世界坐标系)
        R_world2lidar: world2lidar 旋转矩阵 (3,3)
        t_world2lidar: world2lidar 平移向量 (3,1)

    Returns:
        obj_ego: 转换后的标注对象 dict (ego LiDAR 坐标系)
    """
    # 位置变换: world → ego LiDAR
    pos_world = np.array([obj['x'], obj['y'], obj['z']]).reshape(3, 1)
    pos_lidar = R_world2lidar @ pos_world + t_world2lidar

    # 朝向变换: 将物体的局部坐标轴通过 R_world2lidar 旋转
    obj_roll = obj.get('roll', 0.0)
    obj_pitch = obj.get('pitch', 0.0)
    obj_yaw = obj.get('yaw', 0.0)

    # 物体在世界坐标系中的旋转矩阵 (ZYX欧拉角)
    from scipy.spatial.transform import Rotation as Rot
    R_obj_world = Rot.from_euler('xyz', [obj_roll, obj_pitch, obj_yaw]).as_matrix()

    # 物体在 ego LiDAR 坐标系中的旋转矩阵
    R_obj_lidar = R_world2lidar @ R_obj_world

    # 从旋转矩阵提取欧拉角
    euler_lidar = Rot.from_matrix(R_obj_lidar).as_euler('xyz')
    roll_lidar, pitch_lidar, yaw_lidar = euler_lidar

    return {
        'id': obj['id'],
        'label': obj['label'],
        'x': float(pos_lidar[0, 0]),
        'y': float(pos_lidar[1, 0]),
        'z': float(pos_lidar[2, 0]),
        'length': obj.get('length', 0.0),
        'width': obj.get('width', 0.0),
        'height': obj.get('height', 0.0),
        'roll': float(roll_lidar),
        'pitch': float(pitch_lidar),
        'yaw': float(yaw_lidar),
        'occlusion': obj.get('occlusion', 0),
        'num_points': obj.get('num_points', 0),
        'vx': obj.get('vx', 0.0),
        'vy': obj.get('vy', 0.0),
    }


def convert_single_frame(annotation_path, vehicle_id, output_path,
                         virtual_pose=None):
    """
    转换单帧标注到 ego LiDAR 坐标系

    Args:
        annotation_path: 路侧标注 JSON 路径
        vehicle_id: ego 车辆 ID（仅在非虚拟模式下用于排除 ego 自身）
        output_path: 输出 JSON 路径
        virtual_pose: 可选 dict {x,y,z,roll,pitch,yaw}。若提供则使用虚拟观察车作为 ego，
                      此时所有标注里的物体（含真实采集车）都被转换到虚拟观察车 lidar 系。

    Returns:
        num_objects: 转换的物体数量, -1 表示失败
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)

    if virtual_pose is not None:
        R_world2ego, t_world2ego = get_world2lidar_transform_from_pose(
            world_pos=[virtual_pose['x'], virtual_pose['y'], virtual_pose['z']],
            world_yaw=float(virtual_pose['yaw']),
            roll=float(virtual_pose.get('roll', 0.0)),
            pitch=float(virtual_pose.get('pitch', 0.0)),
            vehicle_height=float(virtual_pose.get('vehicle_height', 1.6)),
        )
    else:
        try:
            R_world2ego, t_world2ego, world_pos, world_yaw, ego_obj = get_world2ego_transform(
                annotation, vehicle_id
            )
        except ValueError as e:
            print(f"  跳过: {e}")
            return -1

    # 转换所有物体；非虚拟模式下排除 ego 自身
    objects_ego = []
    for obj in annotation.get('object', []):
        if virtual_pose is None and obj['id'] == vehicle_id:
            continue  # 排除 ego 自身

        obj_ego = transform_object_to_ego_frame(
            obj, R_world2ego, t_world2ego
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


def convert_segment_annotations(label_files, timestamps, vehicle_id, output_dir,
                                virtual_pose=None):
    """
    转换一个 segment 的所有帧标注

    Args:
        label_files: 标注文件路径列表
        timestamps: 时间戳列表
        vehicle_id: ego 车辆 ID
        output_dir: 输出目录
        virtual_pose: 可选 dict，提供则所有帧都用虚拟观察车作为 ego

    Returns:
        success_count: 成功转换的帧数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for label_file, ts in zip(label_files, timestamps):
        output_path = output_dir / f"{ts}.json"
        num_objects = convert_single_frame(
            label_file, vehicle_id, output_path, virtual_pose=virtual_pose
        )
        if num_objects >= 0:
            success_count += 1

    print(f"  annotations: {success_count}/{len(timestamps)} 帧, 输出到 {output_dir}"
          + ("  (virtual_pose)" if virtual_pose is not None else ""))
    return success_count
