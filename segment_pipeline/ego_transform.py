#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ego 车辆变换模块
从路侧标注中提取目标车辆的位姿，计算变换矩阵

提供两种变换:
- world2lidar: 用于点云投影 (含 LiDAR 安装偏移)
- world2ego:   用于标注转换 (仅车体中心，不含 LiDAR 偏移)
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


# 车体几何参数（固定配置）
IMU_IN_CAR = {
    'x': 1.385,
    'y': 0.0,
    'z_func': lambda box_height: box_height / 2 - 1.12,  # 依赖 bbox 高度
}

LIDAR_TO_IMU = {
    'x': 0.003551,
    'y': 1.630648,
    'z': 1.262754,
}

# LiDAR 安装偏移微调（归零）
LIDAR_ADJUST = {
    'x_offset': 0.0,
    'y_offset': 0.0,
    'z_offset': 0.0,
}


def get_world2carlidar(rotate_car2world, trans_car2world, lidar2car_quat, lidar2car_trans):
    """
    计算从世界坐标系到车载LiDAR坐标系的变换

    Args:
        rotate_car2world: 车体→世界 旋转向量 (roll, pitch, yaw), shape (3,1)
        trans_car2world: 车体→世界 平移向量 (x, y, z), shape (3,1)
        lidar2car_quat: LiDAR→车体 四元数 [x, y, z, w]
        lidar2car_trans: LiDAR→车体 平移 [x, y, z]

    Returns:
        rotation_vector: world2lidar 旋转向量 (罗德里格斯), shape (3,1)
        translation: world2lidar 平移向量, shape (3,1)
    """
    # car2world → world2car
    R_car2world = cv2.Rodrigues(rotate_car2world)[0]
    R_world2car = R_car2world.T
    t_world2car = -R_world2car @ trans_car2world

    # lidar2car → car2lidar
    r_lidar2car = R.from_quat(lidar2car_quat)
    R_lidar2car = r_lidar2car.as_matrix()
    R_car2lidar = R_lidar2car.T
    lidar2car_trans_array = np.asarray(lidar2car_trans).reshape((3, 1))
    t_car2lidar = -R_car2lidar @ lidar2car_trans_array

    # world2lidar = car2lidar @ world2car
    R_world2lidar = R_car2lidar @ R_world2car
    t_world2lidar = R_car2lidar @ t_world2car + t_car2lidar

    rotation_vector, _ = cv2.Rodrigues(R_world2lidar)
    return rotation_vector, t_world2lidar


def _find_vehicle_in_annotation(annotation, vehicle_id):
    """从标注中查找指定车辆"""
    for obj in annotation.get('object', []):
        if obj['id'] == vehicle_id:
            return obj
    return None


def _get_car2world_params(vehicle):
    """从车辆标注对象中提取 car2world 参数"""
    car2world_rotate = np.array([
        vehicle.get('roll', 0.0),
        vehicle.get('pitch', 0.0),
        vehicle['yaw']
    ]).reshape((3, 1))
    car2world_trans = np.array([
        vehicle['x'], vehicle['y'], vehicle['z']
    ]).reshape((3, 1))
    return car2world_rotate, car2world_trans


def get_vehicle_transform(annotation, vehicle_id):
    """
    从标注中获取指定车辆的 world2lidar 变换（用于点云投影）

    包含 LiDAR 安装偏移，适用于点云投影到相机的坐标链。

    Args:
        annotation: 标注 JSON dict (含 'object' 列表)
        vehicle_id: 目标车辆 ID

    Returns:
        rotate: world2lidar 旋转向量, shape (3,1)
        trans: world2lidar 平移向量, shape (3,1)
        world_pos: 车辆世界坐标 [x, y, z]
        world_yaw: 车辆世界朝向 yaw
        vehicle_obj: 原始标注对象 dict

    Raises:
        ValueError: 未找到指定车辆
    """
    vehicle = _find_vehicle_in_annotation(annotation, vehicle_id)
    if not vehicle:
        raise ValueError(f"未找到车辆 ID={vehicle_id}")

    car2world_rotate, car2world_trans = _get_car2world_params(vehicle)

    # LiDAR 在车体中的安装位置
    box_height = vehicle.get('height', 1.72)
    lidar_quat = [0, 0, 0, 1]  # 无旋转
    lidar_trans = [
        IMU_IN_CAR['x'] + LIDAR_TO_IMU['x'] + LIDAR_ADJUST['x_offset'],
        IMU_IN_CAR['y'] + LIDAR_TO_IMU['y'] + LIDAR_ADJUST['y_offset'],
        IMU_IN_CAR['z_func'](box_height) + LIDAR_TO_IMU['z'] + LIDAR_ADJUST['z_offset'],
    ]

    # 计算 world2lidar
    rotate, trans = get_world2carlidar(
        car2world_rotate, car2world_trans,
        lidar_quat, lidar_trans
    )

    world_pos = car2world_trans.flatten()
    world_yaw = vehicle['yaw']

    return rotate, trans, world_pos, world_yaw, vehicle


def get_world2ego_transform(annotation, vehicle_id):
    """
    从标注中获取指定车辆的 world2ego 变换（用于标注坐标转换）

    仅从车体 bbox 中心位姿取逆，不含 LiDAR 安装偏移。
    适用于将其他物体的 3D 标注转换到 ego 车体坐标系。

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

    car2world_rotate, car2world_trans = _get_car2world_params(vehicle)

    # car2world 的逆 = world2ego
    R_car2world = cv2.Rodrigues(car2world_rotate)[0]
    R_world2ego = R_car2world.T
    t_world2ego = -R_world2ego @ car2world_trans

    world_pos = car2world_trans.flatten()
    world_yaw = vehicle['yaw']

    return R_world2ego, t_world2ego, world_pos, world_yaw, vehicle


def points_world_to_lidar(points, rotate, trans):
    """
    将点云从世界坐标系转换到车端 LiDAR 坐标系

    Args:
        points: 世界坐标系点云 (N, 3)
        rotate: world2lidar 旋转向量 (3,1)
        trans: world2lidar 平移向量 (3,1)

    Returns:
        points_lidar: LiDAR 坐标系点云 (N, 3)
    """
    R_mat = cv2.Rodrigues(rotate)[0]
    points_lidar = (R_mat @ points.T).T
    points_lidar[:, 0] += trans[0, 0]
    points_lidar[:, 1] += trans[1, 0]
    points_lidar[:, 2] += trans[2, 0]
    return points_lidar
