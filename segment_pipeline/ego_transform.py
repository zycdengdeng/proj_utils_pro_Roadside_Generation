#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ego 车辆变换模块
从路侧标注中提取目标车辆的位姿，计算 world2lidar 变换矩阵

坐标变换链：
  world (路侧世界坐标系)
    → car (车体坐标系, bbox 中心为原点)
    → lidar (虚拟 LiDAR 坐标系, 车顶)

参照: getRoad2lidar() 逻辑
  - car2world: euler2rotmat(roll, pitch, yaw) + [x, y, z]
  - lidar2car: R=I, t=[0, 0, height/2+0.25]
"""

import numpy as np
import cv2


def euler2rotmat(roll, pitch, yaw):
    """
    欧拉角 → 旋转矩阵 (ZYX 顺序: Rz @ Ry @ Rx)

    Args:
        roll, pitch, yaw: 弧度

    Returns:
        R: (3,3) 旋转矩阵
    """
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def _find_vehicle_in_annotation(annotation, vehicle_id):
    """从标注中查找指定车辆"""
    for obj in annotation.get('object', []):
        if obj['id'] == vehicle_id:
            return obj
    return None


def get_world2lidar_transform(annotation, vehicle_id):
    """
    从标注中获取 world → 虚拟 LiDAR 坐标系的变换

    变换链 (参照 getRoad2lidar):
      1. world → car:  R_car2world 取逆
      2. car → lidar:  R_lidar2car 取逆 (R_lidar2car=I, t_lidar2car=[0,0,h/2+0.25])

    Args:
        annotation: 标注 JSON dict (含 'object' 列表)
        vehicle_id: 目标车辆 ID

    Returns:
        R_world2lidar: (3,3) 旋转矩阵
        t_world2lidar: (3,1) 平移向量
        world_pos: 车辆世界坐标 [x, y, z]
        world_yaw: 车辆世界朝向 yaw
        vehicle_obj: 原始标注对象 dict

    Raises:
        ValueError: 未找到指定车辆
    """
    vehicle = _find_vehicle_in_annotation(annotation, vehicle_id)
    if not vehicle:
        raise ValueError(f"未找到车辆 ID={vehicle_id}")

    roll = vehicle.get('roll', 0.0)
    pitch = vehicle.get('pitch', 0.0)
    yaw = vehicle['yaw']
    height = vehicle.get('height', 1.5)

    # ---- Step 1: world → car ----
    # car2world: R_car2world, t_car2world
    R_car2world = euler2rotmat(roll, pitch, yaw)
    t_car2world = np.array([vehicle['x'], vehicle['y'], vehicle['z']]).reshape(3, 1)

    R_world2car = R_car2world.T
    t_world2car = -R_world2car @ t_car2world

    # ---- Step 2: car → lidar ----
    # lidar2car: R=I, t=[0, 0, height/2+0.25] (lidar 安装在车顶)
    R_lidar2car = np.eye(3)
    t_lidar2car = np.array([0.0, 0.0, height / 2.0 + 0.25]).reshape(3, 1)

    R_car2lidar = R_lidar2car.T  # I.T = I
    t_car2lidar = -R_car2lidar @ t_lidar2car

    # ---- Step 3: 合并 world → lidar ----
    R_world2lidar = R_car2lidar @ R_world2car
    t_world2lidar = R_car2lidar @ t_world2car + t_car2lidar

    world_pos = t_car2world.flatten()
    world_yaw = yaw

    return R_world2lidar, t_world2lidar, world_pos, world_yaw, vehicle


# 向后兼容别名
get_world2ego_transform = get_world2lidar_transform


def get_world2ego_as_rodrigues(annotation, vehicle_id):
    """
    同 get_world2lidar_transform，但返回罗德里格斯旋转向量（用于投影模块兼容）

    Returns:
        rotate: world2lidar 旋转向量 (罗德里格斯), shape (3,1)
        trans: world2lidar 平移向量, shape (3,1)
        world_pos: 车辆世界坐标 [x, y, z]
        world_yaw: 车辆世界朝向 yaw
        vehicle_obj: 原始标注对象 dict
    """
    R_world2lidar, t_world2lidar, world_pos, world_yaw, vehicle = \
        get_world2lidar_transform(annotation, vehicle_id)

    rotate, _ = cv2.Rodrigues(R_world2lidar)
    return rotate, t_world2lidar, world_pos, world_yaw, vehicle


def points_world_to_ego(points, R_world2lidar, t_world2lidar):
    """
    将点云从世界坐标系转换到 LiDAR 坐标系

    Args:
        points: 世界坐标系点云 (N, 3)
        R_world2lidar: world2lidar 旋转矩阵 (3,3)
        t_world2lidar: world2lidar 平移向量 (3,1)

    Returns:
        points_lidar: LiDAR 坐标系点云 (N, 3)
    """
    points_lidar = (R_world2lidar @ points.T).T + t_world2lidar.T
    return points_lidar
