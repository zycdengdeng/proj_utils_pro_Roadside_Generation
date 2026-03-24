#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ego 车辆变换模块
从路侧标注中提取目标车辆的位姿，计算 world2ego 变换矩阵

核心逻辑：
  车体 bbox 中心 = ego 坐标原点（等同于 LiDAR 安装位置）
  world2ego = car2world 的逆

用于：
  - 标注转换: world → ego 坐标系
  - 点云投影: world → ego → camera
"""

import numpy as np
import cv2


def _find_vehicle_in_annotation(annotation, vehicle_id):
    """从标注中查找指定车辆"""
    for obj in annotation.get('object', []):
        if obj['id'] == vehicle_id:
            return obj
    return None


def get_world2ego_transform(annotation, vehicle_id):
    """
    从标注中获取指定车辆的 world2ego 变换

    直接将 bbox 中心位姿取逆，bbox 中心即为 ego 坐标原点。

    Args:
        annotation: 标注 JSON dict (含 'object' 列表)
        vehicle_id: 目标车辆 ID

    Returns:
        R_world2ego: world2ego 旋转矩阵, shape (3,3)
        t_world2ego: world2ego 平移向量, shape (3,1)
        world_pos: 车辆世界坐标 [x, y, z]
        world_yaw: 车辆世界朝向 yaw
        vehicle_obj: 原始标注对象 dict

    Raises:
        ValueError: 未找到指定车辆
    """
    vehicle = _find_vehicle_in_annotation(annotation, vehicle_id)
    if not vehicle:
        raise ValueError(f"未找到车辆 ID={vehicle_id}")

    # 车体在世界坐标系的位姿
    car2world_rotate = np.array([
        vehicle.get('roll', 0.0),
        vehicle.get('pitch', 0.0),
        vehicle['yaw']
    ]).reshape((3, 1))
    car2world_trans = np.array([
        vehicle['x'], vehicle['y'], vehicle['z']
    ]).reshape((3, 1))

    # car2world 的逆 = world2ego
    R_car2world = cv2.Rodrigues(car2world_rotate)[0]
    R_world2ego = R_car2world.T
    t_world2ego = -R_world2ego @ car2world_trans

    world_pos = car2world_trans.flatten()
    world_yaw = vehicle['yaw']

    return R_world2ego, t_world2ego, world_pos, world_yaw, vehicle


def get_world2ego_as_rodrigues(annotation, vehicle_id):
    """
    同 get_world2ego_transform，但返回罗德里格斯旋转向量（用于投影模块兼容）

    Returns:
        rotate: world2ego 旋转向量 (罗德里格斯), shape (3,1)
        trans: world2ego 平移向量, shape (3,1)
        world_pos: 车辆世界坐标 [x, y, z]
        world_yaw: 车辆世界朝向 yaw
        vehicle_obj: 原始标注对象 dict
    """
    R_world2ego, t_world2ego, world_pos, world_yaw, vehicle = \
        get_world2ego_transform(annotation, vehicle_id)

    rotate, _ = cv2.Rodrigues(R_world2ego)
    return rotate, t_world2ego, world_pos, world_yaw, vehicle


def points_world_to_ego(points, R_world2ego, t_world2ego):
    """
    将点云从世界坐标系转换到 ego 坐标系

    Args:
        points: 世界坐标系点云 (N, 3)
        R_world2ego: world2ego 旋转矩阵 (3,3)
        t_world2ego: world2ego 平移向量 (3,1)

    Returns:
        points_ego: ego 坐标系点云 (N, 3)
    """
    points_ego = (R_world2ego @ points.T).T + t_world2ego.T
    return points_ego
