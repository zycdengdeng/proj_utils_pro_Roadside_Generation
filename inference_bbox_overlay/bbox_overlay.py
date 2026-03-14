#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理视频2D bbox叠加工具

将路侧标注的动态物体投影为2D bbox，叠加到推理生成的视频上
"""

import json
import yaml
import numpy as np
import cv2
from pathlib import Path
import argparse
import warnings
import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor

# 添加父目录到路径以导入 common_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)

# world2lidar变换JSON文件目录
TRANSFORM_JSON_DIR = "/mnt/zihanw/proj_utils_pro/transform_json"

# 相机名称映射（视频文件名 → 标准相机名）
CAMERA_NAME_MAPPING = {
    'front_tele_30fov': {'cam_id': 1, 'standard_name': 'FN'},
    'front_wide_120fov': {'cam_id': 2, 'standard_name': 'FW'},
    'cross_left_120fov': {'cam_id': 3, 'standard_name': 'FL'},
    'cross_right_120fov': {'cam_id': 4, 'standard_name': 'FR'},
    'rear_left_70fov': {'cam_id': 5, 'standard_name': 'RL'},
    'rear_right_70fov': {'cam_id': 6, 'standard_name': 'RR'},
    'rear_tele_30fov': {'cam_id': 7, 'standard_name': 'RN'},
}

# 车端相机配置
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "前视窄角30°", "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "前视广角120°", "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "左前视120°", "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "右前视120°", "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "左后视60°", "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "右后视60°", "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "后视60°", "resolution": (1920, 1080)}
}

# 自车ID配置文件路径
CARID_JSON_PATH = "/mnt/car_road_data_fix/support_info/carid.json"

# 颜色配置（20个类别 + unknown）
LABEL_COLORS = {
    "Car": [255, 0, 0],
    "Suv": [255, 69, 0],
    "Non_motor_rider": [255, 215, 0],
    "Bollards": [128, 128, 128],
    "Pedestrian": [0, 255, 0],
    "Crash_bucket": [192, 192, 192],
    "Tricycle": [135, 206, 250],
    "Truck": [255, 140, 0],
    "Motorcycle": [255, 0, 255],
    "Bus": [0, 0, 255],
    "Motor_rider": [255, 105, 180],
    "Pedestrian_else": [144, 238, 144],
    "Bicycle": [0, 255, 255],
    "Vehicle_else": [160, 82, 45],
    "Vehicle_door": [255, 192, 203],
    "Other_rider": [221, 160, 221],
    "Huge_vehicle": [139, 0, 0],
    "Unknown": [128, 128, 128],
    "Animal_small": [255, 228, 196],
    "Cone": [255, 255, 0],
    "unknown": [255, 0, 128]
}


def quaternion_to_rotation_matrix(q):
    """四元数转旋转矩阵"""
    x, y, z, w = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])
    return R


def get_3d_bbox_corners(center, size, yaw):
    """计算3D bbox的8个角点（世界坐标系）"""
    x, y, z = center
    l, w, h = size

    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
        [l/2, w/2, h/2], [-l/2, w/2, h/2]
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    corners_rotated = (R @ corners.T).T + center
    return corners_rotated


def load_carid_mapping(carid_json_path=CARID_JSON_PATH):
    """
    加载carid.json，建立scene_id到自车ID的映射

    Args:
        carid_json_path: carid.json文件路径

    Returns:
        dict: {scene_id: nearest_carid} 映射
    """
    mapping = {}
    try:
        with open(carid_json_path, 'r') as f:
            data = json.load(f)

        for result in data.get('results', []):
            clip_name = result.get('clip_name', '')
            # 从clip_name提取scene_id，如 "001_car0325_road0327_t1" → "001"
            match = re.match(r'^(\d+)_', clip_name)
            if match:
                scene_id = match.group(1)
                nearest_carid = result.get('nearest_carid')
                if nearest_carid is not None:
                    mapping[scene_id] = nearest_carid

        print(f"📋 已加载 {len(mapping)} 个场景的自车ID映射")
    except Exception as e:
        print(f"⚠️ 加载carid.json失败: {e}")

    return mapping


def load_world2lidar_transforms(scene_id):
    """
    从预计算的JSON文件加载world2lidar变换

    Args:
        scene_id: 场景ID（如 '031'）

    Returns:
        transforms: 变换列表，每个元素包含 timestamp_ms, rotation, translation
    """
    json_path = Path(TRANSFORM_JSON_DIR) / scene_id / "world2lidar_transforms.json"

    if not json_path.exists():
        print(f"  ⚠️ world2lidar变换文件不存在: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        transforms = []
        for item in data:
            # timestamp 是秒，转换为毫秒
            timestamp_sec = item['timestamp']
            timestamp_ms = int(timestamp_sec * 1000)

            rotation = np.array(item['world2lidar']['rotation'])
            translation = np.array(item['world2lidar']['translation'])

            transforms.append({
                'timestamp_ms': timestamp_ms,
                'rotation': rotation,
                'translation': translation
            })

        # 按时间戳排序
        transforms.sort(key=lambda x: x['timestamp_ms'])

        print(f"  📂 加载了 {len(transforms)} 个world2lidar变换")
        return transforms

    except Exception as e:
        print(f"  ⚠️ 加载world2lidar变换失败: {e}")
        return None


def find_closest_transform(timestamp_ms, transforms, tolerance_ms=5000):
    """
    根据时间戳找到最近的world2lidar变换

    Args:
        timestamp_ms: 目标时间戳（毫秒）
        transforms: 变换列表
        tolerance_ms: 容差（毫秒）

    Returns:
        (rotation, translation) 或 None
    """
    if not transforms:
        return None, None

    min_diff = float('inf')
    closest = None

    for t in transforms:
        diff = abs(t['timestamp_ms'] - timestamp_ms)
        if diff < min_diff:
            min_diff = diff
            closest = t

    if min_diff > tolerance_ms:
        return None, None

    return closest['rotation'], closest['translation']


def parse_folder_name(folder_name):
    """
    解析文件夹名称，提取 scene_id, seg_num

    格式: {scene_id}_seg{seg_num}
    例如: 031_seg01 → scene_id='031', seg_num=1
    """
    match = re.match(r'^(\d+)_seg(\d+)$', folder_name)
    if match:
        scene_id = match.group(1)
        seg_num = int(match.group(2))
        return scene_id, seg_num
    return None, None


def draw_2d_bbox(img, bbox_2d, color, thickness=2, label=None):
    """
    绘制2D bbox

    Args:
        img: 图像
        bbox_2d: [x1, y1, x2, y2]
        color: 颜色 (B, G, R)
        thickness: 线宽
        label: 可选的标签文本
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_2d]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        # 绘制标签背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), 1)


def draw_3d_bbox(img, corners_2d, color, thickness=1, label=None):
    """
    绘制3D bbox（12条边）

    Args:
        img: 图像
        corners_2d: 8个角点的2D坐标 [(x, y), ...] 或 None
                    顺序: 0-3底面, 4-7顶面
        color: 颜色 (B, G, R)
        thickness: 线宽
        label: 可选的标签文本
    """
    if corners_2d is None or len(corners_2d) < 8:
        return

    # 定义12条边的连接关系
    # 底面4条边
    edges_bottom = [(0, 1), (1, 2), (2, 3), (3, 0)]
    # 顶面4条边
    edges_top = [(4, 5), (5, 6), (6, 7), (7, 4)]
    # 垂直4条边
    edges_vertical = [(0, 4), (1, 5), (2, 6), (3, 7)]

    all_edges = edges_bottom + edges_top + edges_vertical

    # 绘制所有边（只绘制两端点都有效的边）
    valid_pts = []
    for i, j in all_edges:
        if corners_2d[i] is not None and corners_2d[j] is not None:
            pt1 = (int(corners_2d[i][0]), int(corners_2d[i][1]))
            pt2 = (int(corners_2d[j][0]), int(corners_2d[j][1]))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 收集有效点用于标签位置
    for pt in corners_2d:
        if pt is not None:
            valid_pts.append((int(pt[0]), int(pt[1])))

    # 绘制标签（在最高点附近）
    if label and valid_pts:
        # 找到最高的点（y最小）
        min_y_pt = min(valid_pts, key=lambda p: p[1])
        label_pos = (min_y_pt[0], max(0, min_y_pt[1] - 5))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        cv2.putText(img, label, label_pos, font, font_scale, color, 1, cv2.LINE_AA)


class BboxOverlayProcessor:
    def __init__(self, vehicle_calib_folder):
        """
        初始化bbox叠加处理器

        Args:
            vehicle_calib_folder: 车端标定文件夹路径
        """
        self.vehicle_calib_folder = Path(vehicle_calib_folder)
        self.camera_poses = {}
        self.camera_params = {}

        # 加载所有相机参数
        for cam_id in range(1, 8):
            self.load_camera_params(cam_id)

    def load_camera_params(self, cam_id):
        """加载车端相机参数"""
        with open(self.vehicle_calib_folder / f"camera_{cam_id:02d}_intrinsics.yaml", 'r') as f:
            intrinsics = yaml.safe_load(f)
        K = np.array(intrinsics['K']).reshape(3, 3)
        D = np.array(intrinsics['D'])

        with open(self.vehicle_calib_folder / f"camera_{cam_id:02d}_extrinsics.yaml", 'r') as f:
            extrinsics = yaml.safe_load(f)

        transform = extrinsics['transform']
        q = [transform['rotation']['x'], transform['rotation']['y'],
             transform['rotation']['z'], transform['rotation']['w']]
        t = np.array([transform['translation']['x'],
                     transform['translation']['y'],
                     transform['translation']['z']])
        R_cam = quaternion_to_rotation_matrix(q)

        self.camera_poses[cam_id] = {'R': R_cam, 't': t}
        self.camera_params[cam_id] = {
            'K': K,
            'D': D,
            'resolution': VEHICLE_CAMERAS[cam_id]["resolution"]
        }

    def project_bbox_to_2d(self, bbox_corners_world, rotate_world2lidar,
                           trans_world2lidar, cam_id, target_resolution=None):
        """
        将3D bbox投影到相机并计算2D bbox

        Args:
            bbox_corners_world: 3D bbox角点（世界坐标系）
            rotate_world2lidar: world2lidar旋转矩阵
            trans_world2lidar: world2lidar平移向量
            cam_id: 相机ID
            target_resolution: 目标分辨率 (width, height)，用于缩放bbox

        Returns:
            bbox_2d: [x1, y1, x2, y2] or None
        """
        cam_info = VEHICLE_CAMERAS[cam_id]
        orig_w, orig_h = cam_info["resolution"]  # 原始分辨率

        K = self.camera_params[cam_id]['K']
        D = self.camera_params[cam_id]['D']
        R_cam2lidar = self.camera_poses[cam_id]['R']
        t_cam2lidar = self.camera_poses[cam_id]['t']

        # 世界坐标系 → LiDAR坐标系
        points_lidar = common_utils.transform_points_to_lidar(
            bbox_corners_world,
            {'world2lidar': {
                'rotation': rotate_world2lidar.flatten().tolist(),
                'translation': trans_world2lidar.flatten().tolist()
            }}
        )

        # LiDAR坐标系 → 相机坐标系
        R_lidar2cam = R_cam2lidar.T
        t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar
        points_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam

        # 过滤相机前方的点
        valid_mask = points_cam[:, 2] > 0.1
        if not valid_mask.any():
            return None

        points_valid = points_cam[valid_mask]

        # 相机坐标系 → 图像坐标系（使用原始分辨率计算）
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (orig_w, orig_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (orig_w, orig_h), 0, (orig_w, orig_h))

        uv_homogeneous = (new_K @ points_valid.T).T
        z_proj = uv_homogeneous[:, 2]
        uv = uv_homogeneous[:, :2] / z_proj[:, np.newaxis]

        # 计算2D bbox（原始分辨率下）
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)

        # 检查是否在原始图像内
        if x2 < 0 or y2 < 0 or x1 > orig_w or y1 > orig_h:
            return None

        # 裁剪到原始图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(orig_w, x2)
        y2 = min(orig_h, y2)

        # 如果指定了目标分辨率，进行缩放
        if target_resolution is not None:
            target_w, target_h = target_resolution
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

        return [float(x1), float(y1), float(x2), float(y2)]

    def project_bbox_corners_to_2d(self, bbox_corners_world, rotate_world2lidar,
                                   trans_world2lidar, cam_id, target_resolution=None):
        """
        将3D bbox的8个角点投影到相机，返回2D坐标

        Args:
            bbox_corners_world: 3D bbox角点（世界坐标系），8个点
            rotate_world2lidar: world2lidar旋转矩阵
            trans_world2lidar: world2lidar平移向量
            cam_id: 相机ID
            target_resolution: 目标分辨率 (width, height)，用于缩放

        Returns:
            corners_2d: 8个角点的2D坐标列表 [(x, y), ...] or None
            valid_mask: 每个角点是否有效的mask
        """
        cam_info = VEHICLE_CAMERAS[cam_id]
        orig_w, orig_h = cam_info["resolution"]

        K = self.camera_params[cam_id]['K']
        D = self.camera_params[cam_id]['D']
        R_cam2lidar = self.camera_poses[cam_id]['R']
        t_cam2lidar = self.camera_poses[cam_id]['t']

        # 世界坐标系 → LiDAR坐标系
        points_lidar = common_utils.transform_points_to_lidar(
            bbox_corners_world,
            {'world2lidar': {
                'rotation': rotate_world2lidar.flatten().tolist(),
                'translation': trans_world2lidar.flatten().tolist()
            }}
        )

        # LiDAR坐标系 → 相机坐标系
        R_lidar2cam = R_cam2lidar.T
        t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar
        points_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam

        # 检查哪些点在相机前方
        valid_mask = points_cam[:, 2] > 0.1
        if not valid_mask.any():
            return None, None

        # 相机坐标系 → 图像坐标系
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (orig_w, orig_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (orig_w, orig_h), 0, (orig_w, orig_h))

        # 投影所有点（包括无效的，后面会用mask过滤）
        corners_2d = []
        for i, pt_cam in enumerate(points_cam):
            if pt_cam[2] > 0.1:
                uv_h = new_K @ pt_cam
                u = uv_h[0] / uv_h[2]
                v = uv_h[1] / uv_h[2]

                # 缩放到目标分辨率
                if target_resolution is not None:
                    target_w, target_h = target_resolution
                    u = u * target_w / orig_w
                    v = v * target_h / orig_h

                corners_2d.append((u, v))
            else:
                corners_2d.append(None)

        # 检查是否至少有部分点在图像内
        valid_corners = [c for c in corners_2d if c is not None]
        if not valid_corners:
            return None, None

        # 检查bbox中心是否大致在图像范围内
        xs = [c[0] for c in valid_corners]
        ys = [c[1] for c in valid_corners]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        if target_resolution:
            img_w, img_h = target_resolution
        else:
            img_w, img_h = orig_w, orig_h

        # 如果中心点完全在图像外，跳过
        margin = max(img_w, img_h) * 0.5
        if center_x < -margin or center_x > img_w + margin:
            return None, None
        if center_y < -margin or center_y > img_h + margin:
            return None, None

        return corners_2d, valid_mask

    def process_single_frame(self, frame, annotation, rotate_world2lidar,
                             trans_world2lidar, cam_id, ego_vehicle_id):
        """
        处理单帧：在帧上绘制所有物体的3D bbox

        Args:
            frame: 视频帧 (numpy array)
            annotation: 标注数据
            rotate_world2lidar: world2lidar旋转向量
            trans_world2lidar: world2lidar平移向量
            cam_id: 相机ID
            ego_vehicle_id: 自车ID（将被排除）

        Returns:
            annotated_frame: 带bbox的帧
        """
        annotated_frame = frame.copy()

        # 获取视频帧的实际分辨率
        frame_h, frame_w = frame.shape[:2]
        target_resolution = (frame_w, frame_h)

        for obj in annotation.get('object', []):
            if obj['id'] == ego_vehicle_id:
                continue

            # 计算3D bbox角点
            center = [obj['x'], obj['y'], obj['z']]
            size = [obj['length'], obj['width'], obj['height']]
            yaw = obj['yaw']
            bbox_corners = get_3d_bbox_corners(center, size, yaw)

            # 投影8个角点到2D
            corners_2d, valid_mask = self.project_bbox_corners_to_2d(
                bbox_corners, rotate_world2lidar, trans_world2lidar, cam_id,
                target_resolution=target_resolution
            )

            if corners_2d is not None:
                color = LABEL_COLORS.get(obj['label'], LABEL_COLORS['unknown'])
                draw_3d_bbox(annotated_frame, corners_2d, tuple(color),
                            thickness=1, label=obj['label'])

        return annotated_frame


def select_annotation_timestamps(annotation_timestamps, total_frames, num_segs,
                                  frames_per_seg, seg_num, frame_selection):
    """
    根据帧选择方式选取标注时间戳

    Args:
        annotation_timestamps: 所有标注时间戳列表
        total_frames: 需要的总帧数
        num_segs: seg总数
        frames_per_seg: 每个seg的帧数
        seg_num: 当前seg序号 (1-based)
        frame_selection: 帧选择方式 ('middle', 'start', 'end', 或 (start, end) 范围)

    Returns:
        selected_timestamps: 选取的时间戳列表
    """
    annotation_timestamps = sorted(annotation_timestamps)
    total_available = len(annotation_timestamps)

    # 计算总需要帧数
    total_needed = num_segs * frames_per_seg

    if frame_selection == 'middle':
        # 选取中间部分
        start_idx = (total_available - total_needed) // 2
        start_idx = max(0, start_idx)
    elif frame_selection == 'start':
        start_idx = 0
    elif frame_selection == 'end':
        start_idx = max(0, total_available - total_needed)
    elif isinstance(frame_selection, tuple):
        start_idx, _ = frame_selection
    else:
        start_idx = 0

    # 选取当前seg对应的帧
    seg_start = start_idx + (seg_num - 1) * frames_per_seg
    seg_end = seg_start + frames_per_seg

    selected = annotation_timestamps[seg_start:seg_end]

    # 如果不够，用最后一帧填充
    while len(selected) < frames_per_seg:
        selected.append(selected[-1] if selected else annotation_timestamps[-1])

    return selected[:frames_per_seg]


def process_single_video(video_path, output_path, processor, scene_id, vehicle_id,
                         seg_num, num_segs, frames_per_seg, frame_selection, fps=29):
    """
    处理单个视频文件

    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        processor: BboxOverlayProcessor实例
        scene_id: 场景ID
        vehicle_id: 车辆ID
        seg_num: seg序号
        num_segs: 总seg数
        frames_per_seg: 每seg帧数
        frame_selection: 帧选择方式
        fps: 输出视频帧率
    """
    # 解析相机信息
    video_name = video_path.stem  # e.g., "front_tele_30fov_generated"
    for cam_key, cam_info in CAMERA_NAME_MAPPING.items():
        if cam_key in video_name:
            cam_id = cam_info['cam_id']
            break
    else:
        print(f"  ⚠️ 无法识别相机: {video_name}")
        return False

    # 获取场景路径
    scene_paths = common_utils.get_scene_paths(scene_id)
    if not scene_paths:
        print(f"  ❌ 未找到场景 {scene_id}")
        return False

    labels_dir = scene_paths['roadside_labels']
    vehicle_images_dir = scene_paths['vehicle_images']

    # 使用和 video maker 相同的逻辑：从车端图像目录获取 timestamp
    # 这样确保 bbox_overlay 和 video maker 使用完全相同的帧
    vehicle_timestamps = []
    vehicle_images_path = Path(vehicle_images_dir)
    if vehicle_images_path.exists():
        for folder in vehicle_images_path.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                vehicle_timestamps.append(int(folder.name))
        vehicle_timestamps.sort()

    if not vehicle_timestamps:
        # 回退到路侧标注 timestamp
        print(f"  ⚠️ 车端图像目录为空，使用路侧标注timestamp")
        label_files = sorted(Path(labels_dir).glob("*.json"))
        for lf in label_files:
            match = re.search(r'(\d+)\.json$', lf.name)
            if match:
                vehicle_timestamps.append(int(match.group(1)))

    if not vehicle_timestamps:
        print(f"  ❌ 未找到有效timestamp")
        return False

    # 选取当前seg对应的时间戳（和 video maker 使用相同的中间选取逻辑）
    selected_timestamps = select_annotation_timestamps(
        vehicle_timestamps, frames_per_seg, num_segs,
        frames_per_seg, seg_num, frame_selection
    )

    # 加载预计算的 world2lidar 变换（仅对本模块生效）
    world2lidar_transforms = load_world2lidar_transforms(scene_id)
    if world2lidar_transforms is None:
        print(f"  ❌ 无法加载 world2lidar 变换")
        return False

    # 打开输入视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频: {video_path}")
        return False

    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(selected_timestamps):
            timestamp_ms = selected_timestamps[frame_idx]

            # 读取标注
            annotation_path = Path(labels_dir) / f"{timestamp_ms}.json"
            if annotation_path.exists():
                with open(annotation_path, 'r') as f:
                    annotation = json.load(f)

                # 从预计算的JSON获取world2lidar变换
                try:
                    rotate_world2lidar, trans_world2lidar = find_closest_transform(
                        timestamp_ms, world2lidar_transforms
                    )

                    if rotate_world2lidar is None:
                        if frame_idx == 0:
                            print(f"    ⚠️ 未找到匹配的world2lidar变换: timestamp={timestamp_ms}")
                        out.write(frame)
                        frame_idx += 1
                        continue

                    # 第一帧打印调试信息
                    if frame_idx == 0:
                        num_objects = len(annotation.get('object', []))
                        non_ego_objects = [o for o in annotation.get('object', []) if o['id'] != vehicle_id]
                        print(f"    调试: 标注中有 {num_objects} 个物体, 排除自车后 {len(non_ego_objects)} 个")
                        print(f"    视频分辨率: {width}x{height}")
                        print(f"    使用预计算的world2lidar变换")

                    # 处理帧
                    frame = processor.process_single_frame(
                        frame, annotation, rotate_world2lidar,
                        trans_world2lidar, cam_id, vehicle_id
                    )
                    processed_frames += 1
                except Exception as e:
                    if frame_idx == 0:
                        print(f"    ⚠️ 处理第一帧失败: {e}")

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"  ✓ {video_path.name} → {output_path.name} ({processed_frames}/{frame_idx} 帧带bbox)")
    return True


def process_folder(folder_path, output_root, processor, num_segs, frames_per_seg,
                   frame_selection, carid_mapping, fps=29):
    """
    处理单个文件夹（包含一个seg的所有相机视频）

    Args:
        folder_path: 输入文件夹路径
        output_root: 输出根目录
        processor: BboxOverlayProcessor实例
        num_segs: 总seg数
        frames_per_seg: 每seg帧数
        frame_selection: 帧选择方式
        carid_mapping: scene_id到自车ID的映射
        fps: 输出视频帧率
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    # 解析文件夹名
    scene_id, seg_num = parse_folder_name(folder_name)
    if scene_id is None:
        print(f"⚠️ 无法解析文件夹名: {folder_name}")
        return False

    # 从映射获取自车ID
    vehicle_id = carid_mapping.get(scene_id)
    if vehicle_id is None:
        print(f"⚠️ 未找到场景 {scene_id} 的自车ID，跳过: {folder_name}")
        return False

    print(f"\n📂 处理: {folder_name}")
    print(f"   场景: {scene_id}, 自车ID: {vehicle_id}, Seg: {seg_num}")

    # 创建输出目录
    output_folder = Path(output_root) / folder_name

    # 处理所有视频（generated和gt）
    video_files = list(folder_path.glob("*.mp4"))

    for video_path in video_files:
        output_path = output_folder / video_path.name

        try:
            process_single_video(
                video_path, output_path, processor, scene_id, vehicle_id,
                seg_num, num_segs, frames_per_seg, frame_selection, fps
            )
        except Exception as e:
            print(f"  ❌ 处理失败 {video_path.name}: {e}")

    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='推理视频2D bbox叠加工具')

    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入目录（包含推理视频的文件夹）')
    parser.add_argument('--output-dir', type=str,
                        help='输出目录（默认在输入目录下创建proj_hdmap）')
    parser.add_argument('--folders', type=str, nargs='*',
                        help='要处理的文件夹名列表（不指定则处理全部）')
    parser.add_argument('--num-segs', type=int, default=3,
                        help='seg总数 (默认3)')
    parser.add_argument('--frames-per-seg', type=int, default=29,
                        help='每seg帧数 (默认29)')
    parser.add_argument('--frame-selection', type=str, default='middle',
                        choices=['middle', 'start', 'end'],
                        help='帧选择方式 (默认middle)')
    parser.add_argument('--fps', type=int, default=29,
                        help='输出视频帧率 (默认29)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'proj_hdmap'

    print("="*60)
    print("🎯 推理视频2D bbox叠加工具")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"帧选择: {args.frame_selection}")
    print(f"Seg数: {args.num_segs}, 每Seg帧数: {args.frames_per_seg}")

    # 初始化处理器
    vehicle_calib = common_utils.VEHICLE_CALIB_DIR
    processor = BboxOverlayProcessor(vehicle_calib)

    # 获取要处理的文件夹
    if args.folders:
        folders = [input_dir / f for f in args.folders]
    else:
        # 自动发现所有符合格式的文件夹
        folders = []
        for item in input_dir.iterdir():
            if item.is_dir():
                scene_id, seg_num = parse_folder_name(item.name)
                if scene_id is not None:
                    folders.append(item)
        folders = sorted(folders)

    print(f"\n找到 {len(folders)} 个文件夹待处理")

    # 加载自车ID映射
    carid_mapping = load_carid_mapping()

    # 处理每个文件夹
    success_count = 0
    for folder in folders:
        if process_folder(folder, output_dir, processor, args.num_segs,
                         args.frames_per_seg, args.frame_selection,
                         carid_mapping, args.fps):
            success_count += 1

    print(f"\n✅ 完成! 成功处理 {success_count}/{len(folders)} 个文件夹")
    print(f"📂 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
