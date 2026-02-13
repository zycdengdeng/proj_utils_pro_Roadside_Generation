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


def parse_folder_name(folder_name):
    """
    解析文件夹名称，提取 scene_id, vehicle_id, seg_num

    格式: {scene_id}_id{vehicle_id}_seg{seg_num}
    例如: 031_id041_seg01 → scene_id='031', vehicle_id=41, seg_num=1
    """
    match = re.match(r'^(\d+)_id(\d+)_seg(\d+)$', folder_name)
    if match:
        scene_id = match.group(1)
        vehicle_id = int(match.group(2))
        seg_num = int(match.group(3))
        return scene_id, vehicle_id, seg_num
    return None, None, None


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
                           trans_world2lidar, cam_id):
        """
        将3D bbox投影到相机并计算2D bbox

        Returns:
            bbox_2d: [x1, y1, x2, y2] or None
        """
        cam_info = VEHICLE_CAMERAS[cam_id]
        img_w, img_h = cam_info["resolution"]

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

        # 相机坐标系 → 图像坐标系
        if cam_id in [2, 3, 4] and np.max(np.abs(D)) > 1:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (img_w, img_h), np.eye(3), balance=0.0
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 0, (img_w, img_h))

        uv_homogeneous = (new_K @ points_valid.T).T
        z_proj = uv_homogeneous[:, 2]
        uv = uv_homogeneous[:, :2] / z_proj[:, np.newaxis]

        # 计算2D bbox
        x1, y1 = uv.min(axis=0)
        x2, y2 = uv.max(axis=0)

        # 检查是否在图像内
        if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
            return None

        # 裁剪到图像范围
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        return [float(x1), float(y1), float(x2), float(y2)]

    def process_single_frame(self, frame, annotation, rotate_world2lidar,
                             trans_world2lidar, cam_id, ego_vehicle_id):
        """
        处理单帧：在帧上绘制所有物体的2D bbox

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

        for obj in annotation.get('object', []):
            if obj['id'] == ego_vehicle_id:
                continue

            # 计算3D bbox角点
            center = [obj['x'], obj['y'], obj['z']]
            size = [obj['length'], obj['width'], obj['height']]
            yaw = obj['yaw']
            bbox_corners = get_3d_bbox_corners(center, size, yaw)

            # 投影为2D bbox
            bbox_2d = self.project_bbox_to_2d(
                bbox_corners, rotate_world2lidar, trans_world2lidar, cam_id
            )

            if bbox_2d is not None:
                color = LABEL_COLORS.get(obj['label'], LABEL_COLORS['unknown'])
                draw_2d_bbox(annotated_frame, bbox_2d, tuple(color),
                            thickness=3, label=obj['label'])

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

    # 获取标注时间戳
    label_files = sorted(Path(labels_dir).glob("*.json"))
    annotation_timestamps = []
    for lf in label_files:
        match = re.search(r'(\d+)\.json$', lf.name)
        if match:
            annotation_timestamps.append(int(match.group(1)))

    if not annotation_timestamps:
        print(f"  ❌ 未找到标注文件")
        return False

    # 选取当前seg对应的时间戳
    selected_timestamps = select_annotation_timestamps(
        annotation_timestamps, frames_per_seg, num_segs,
        frames_per_seg, seg_num, frame_selection
    )

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

                # 计算world2lidar变换
                try:
                    rotate_world2lidar, trans_world2lidar = \
                        common_utils.compute_world2lidar_from_annotation(
                            str(annotation_path), vehicle_id
                        )

                    # 处理帧
                    frame = processor.process_single_frame(
                        frame, annotation, rotate_world2lidar,
                        trans_world2lidar, cam_id, vehicle_id
                    )
                    processed_frames += 1
                except Exception as e:
                    pass  # 跳过无法处理的帧

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"  ✓ {video_path.name} → {output_path.name} ({processed_frames}/{frame_idx} 帧带bbox)")
    return True


def process_folder(folder_path, output_root, processor, num_segs, frames_per_seg,
                   frame_selection, fps=29):
    """
    处理单个文件夹（包含一个seg的所有相机视频）

    Args:
        folder_path: 输入文件夹路径
        output_root: 输出根目录
        processor: BboxOverlayProcessor实例
        num_segs: 总seg数
        frames_per_seg: 每seg帧数
        frame_selection: 帧选择方式
        fps: 输出视频帧率
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    # 解析文件夹名
    scene_id, vehicle_id, seg_num = parse_folder_name(folder_name)
    if scene_id is None:
        print(f"⚠️ 无法解析文件夹名: {folder_name}")
        return False

    print(f"\n📂 处理: {folder_name}")
    print(f"   场景: {scene_id}, 车辆ID: {vehicle_id}, Seg: {seg_num}")

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
                scene_id, vehicle_id, seg_num = parse_folder_name(item.name)
                if scene_id is not None:
                    folders.append(item)
        folders = sorted(folders)

    print(f"\n找到 {len(folders)} 个文件夹待处理")

    # 处理每个文件夹
    success_count = 0
    for folder in folders:
        if process_folder(folder, output_dir, processor, args.num_segs,
                         args.frames_per_seg, args.frame_selection, args.fps):
            success_count += 1

    print(f"\n✅ 完成! 成功处理 {success_count}/{len(folders)} 个文件夹")
    print(f"📂 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
