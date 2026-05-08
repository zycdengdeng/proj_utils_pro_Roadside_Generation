#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车端标注 HDMap 投影：车端 3D bbox → 车端 7 相机图像

和路转车 HDMap投影 渲染风格完全一致（实心填充、类别着色、backface culling、painter's algorithm），
区别在于：标注已在车载 lidar 系，不需要 world2lidar 变换。

分段逻辑：每个 clip 取中间 3×29=87 帧，切成 3 段。

输出:
  {output_root}/{clip_id}/seg{00-02}/{ts}/overlay/{cam}.jpg    黑底 + 实心 bbox
  {output_root}/{clip_id}/seg{00-02}/{ts}/gt/{cam}.jpg         去畸变 GT
  {output_root}/{clip_id}/seg{00-02}/{ts}/bbox_on_gt/{cam}.jpg bbox 叠加在 GT 上

用法:
  python generate_carside_hdmap.py
  python generate_carside_hdmap.py --clips 004 006 053
  python generate_carside_hdmap.py --output /mnt/zihanw/car_side_bbox --num-threads 7
"""

import json
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_utils

warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# 配置
# ============================================================

DATASET_ROOT = Path(common_utils.DATASET_ROOT)
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
FISHEYE_CAM_IDS = {2, 3, 4}

SEGMENT_LENGTH = 29
NUM_SEGMENTS = 3
SKIP_CLIPS = {"018", "071"}

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
    "unknown": [255, 0, 128],
}

BBOX_3D_FACES = {
    'bottom': [0, 3, 2, 1],
    'top': [4, 5, 6, 7],
    'front': [0, 1, 5, 4],
    'back': [2, 3, 7, 6],
    'left': [0, 4, 7, 3],
    'right': [1, 2, 6, 5],
}

FACE_BRIGHTNESS = {
    'top': 1.0, 'front': 0.7, 'back': 0.5,
    'left': 0.6, 'right': 0.6, 'bottom': 0.3,
}

GT_TS_RE = re.compile(r'_(\d+)\.(\d+)\.jpg$')


# ============================================================
# 渲染函数 (复用自 HDMap投影)
# ============================================================

def adjust_color_brightness(color, brightness):
    return tuple(int(c * brightness) for c in color)


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def get_3d_bbox_corners(obj):
    x, y, z = obj['x'], obj['y'], obj['z']
    l, w, h = obj['length'], obj['width'], obj['height']
    yaw = obj['yaw']
    corners = np.array([
        [-l/2, -w/2, -h/2], [l/2, -w/2, -h/2],
        [l/2, w/2, -h/2], [-l/2, w/2, -h/2],
        [-l/2, -w/2, h/2], [l/2, -w/2, h/2],
        [l/2, w/2, h/2], [-l/2, w/2, h/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (R @ corners.T).T + np.array([x, y, z])


def is_face_visible(corners_cam, face_indices, corners_valid):
    valid_corners = []
    for idx in face_indices:
        if corners_valid[idx]:
            valid_corners.append(corners_cam[idx])
    if len(valid_corners) < 3:
        return False
    p0, p1, p2 = np.array(valid_corners[0]), np.array(valid_corners[1]), np.array(valid_corners[2])
    normal = np.cross(p1 - p0, p2 - p0)
    face_center = np.mean(valid_corners, axis=0)
    return np.dot(normal, face_center) < 0


def get_face_center_depth(corners_cam, face_indices, corners_valid):
    depths = [corners_cam[idx][2] for idx in face_indices if corners_valid[idx]]
    return np.mean(depths) if depths else float('inf')


def draw_3d_bbox_solid(img, corners_2d, corners_valid, corners_cam, color):
    if corners_2d is None or len(corners_2d) != 8:
        return

    visible_faces = []
    for face_name, face_indices in BBOX_3D_FACES.items():
        if not all(corners_valid[idx] for idx in face_indices):
            continue
        if not is_face_visible(corners_cam, face_indices, corners_valid):
            continue
        depth = get_face_center_depth(corners_cam, face_indices, corners_valid)
        face_pts = np.array([[int(corners_2d[idx][0]), int(corners_2d[idx][1])]
                             for idx in face_indices], dtype=np.int32)
        face_color = adjust_color_brightness(color, FACE_BRIGHTNESS[face_name])
        visible_faces.append({'pts': face_pts, 'color': face_color, 'depth': depth})

    visible_faces.sort(key=lambda x: x['depth'], reverse=True)
    for face in visible_faces:
        cv2.fillPoly(img, [face['pts']], face['color'])
    for face in visible_faces:
        cv2.polylines(img, [face['pts']], isClosed=True,
                      color=adjust_color_brightness(color, 0.3), thickness=1)


# ============================================================
# 标定 & 图像
# ============================================================

def load_all_camera_calibs():
    calibs = {}
    for cam_id in range(1, 8):
        intr = yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_intrinsics.yaml"))
        K = np.array(intr['K'], dtype=np.float64).reshape(3, 3)
        D = np.array(intr['D'], dtype=np.float64).flatten()

        extr = yaml.safe_load(open(VEHICLE_CALIB_DIR / f"camera_{cam_id:02d}_extrinsics.yaml"))
        tr = extr['transform']
        q = [tr['rotation']['x'], tr['rotation']['y'], tr['rotation']['z'], tr['rotation']['w']]
        t = np.array([tr['translation']['x'], tr['translation']['y'], tr['translation']['z']], dtype=np.float64)
        R_cam2lidar = quaternion_to_rotation_matrix(q)

        w, h = VEHICLE_CAMERAS[cam_id]['resolution']
        is_fisheye = cam_id in FISHEYE_CAM_IDS and np.max(np.abs(D)) > 1.0
        if is_fisheye:
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K, D[:4], (w, h), np.eye(3), balance=0.0)
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))

        calibs[cam_id] = {
            'K': K, 'D': D, 'new_K': new_K,
            'R_cam2lidar': R_cam2lidar, 't_cam2lidar': t,
            'w': w, 'h': h,
            'is_fisheye': is_fisheye,
            'name': VEHICLE_CAMERAS[cam_id]['name'],
        }
    return calibs


def build_gt_cache(gt_images_dir, cam_ids=None):
    if cam_ids is None:
        cam_ids = range(1, 8)
    cache = {}
    gt_dir = Path(gt_images_dir)
    for cam_id in cam_ids:
        cam_name = VEHICLE_CAMERAS[cam_id]['name']
        cam_dir = gt_dir / cam_name
        items = []
        if cam_dir.exists():
            for f in cam_dir.glob('*.jpg'):
                m = GT_TS_RE.search(f.name)
                if m:
                    ts_us = int(m.group(1)) * 1_000_000 + int(m.group(2))
                    items.append((ts_us, f))
            items.sort()
        cache[cam_id] = items
    return cache


def find_nearest_gt(gt_items, target_ts_us):
    if not gt_items:
        return None
    best_path, best_diff = None, float('inf')
    for ts_us, path in gt_items:
        diff = abs(ts_us - target_ts_us)
        if diff < best_diff:
            best_diff = diff
            best_path = path
    return best_path


def undistort_image(img, calib):
    K, D, w, h = calib['K'], calib['D'], calib['w'], calib['h']
    if calib['is_fisheye']:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D[:4], np.eye(3), calib['new_K'], (w, h), cv2.CV_16SC2)
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    else:
        return cv2.undistort(img, K, D, None, calib['new_K'])


# ============================================================
# 投影 (无 world2lidar，直接 lidar → cam → image)
# ============================================================

def project_bbox_to_camera(corners_lidar, calib):
    """
    corners_lidar: (8, 3) 已在 lidar 系
    返回 (corners_2d, corners_valid, corners_cam, obj_depth) 或全 None
    """
    R_lidar2cam = calib['R_cam2lidar'].T
    t_lidar2cam = -calib['R_cam2lidar'].T @ calib['t_cam2lidar']
    new_K = calib['new_K']
    img_w, img_h = calib['w'], calib['h']
    margin = 500

    points_cam = (R_lidar2cam @ corners_lidar.T).T + t_lidar2cam
    valid_mask = points_cam[:, 2] > 0.1
    if not valid_mask.any():
        return None, None, None, None

    obj_depth = float(np.mean(points_cam[valid_mask, 2]))

    corners_2d = []
    corners_valid = []
    corners_cam_list = []
    for i in range(8):
        corners_cam_list.append(points_cam[i].tolist())
        if valid_mask[i]:
            uv_h = new_K @ points_cam[i]
            uv = uv_h[:2] / uv_h[2]
            corners_2d.append([float(uv[0]), float(uv[1])])
            in_margin = -margin <= uv[0] <= img_w + margin and -margin <= uv[1] <= img_h + margin
            corners_valid.append(in_margin)
        else:
            corners_2d.append([0.0, 0.0])
            corners_valid.append(False)

    if not any(corners_valid):
        return None, None, None, None
    valid_pts = np.array([corners_2d[i] for i in range(8) if corners_valid[i]])
    x1, y1 = valid_pts.min(axis=0)
    x2, y2 = valid_pts.max(axis=0)
    if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
        return None, None, None, None

    return corners_2d, corners_valid, corners_cam_list, obj_depth


# ============================================================
# 单帧 × 单相机
# ============================================================

def process_camera(cam_id, objects_data, calib, gt_img_path,
                   overlay_dir, gt_dir, bbox_on_gt_dir):
    cam_name = calib['name']
    img_w, img_h = calib['w'], calib['h']

    # 投影所有物体
    bboxes = []
    for obj_data in objects_data:
        corners_2d, corners_valid, corners_cam, obj_depth = project_bbox_to_camera(
            obj_data['corners'], calib)
        if corners_2d is not None:
            bboxes.append({
                'color': obj_data['color'],
                'corners_2d': corners_2d,
                'corners_valid': corners_valid,
                'corners_cam': corners_cam,
                'obj_depth': obj_depth,
            })
    bboxes.sort(key=lambda x: x['obj_depth'], reverse=True)

    # overlay: 黑底 + 实心 bbox
    bbox_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for b in bboxes:
        draw_3d_bbox_solid(bbox_img, b['corners_2d'], b['corners_valid'],
                           b['corners_cam'], b['color'])
    cv2.imwrite(str(overlay_dir / f"{cam_name}.jpg"), bbox_img,
                [cv2.IMWRITE_JPEG_QUALITY, 100])

    # GT 去畸变 + bbox_on_gt
    if gt_img_path is not None:
        img = cv2.imread(str(gt_img_path))
        if img is not None:
            und = undistort_image(img, calib)
            cv2.imwrite(str(gt_dir / f"{cam_name}.jpg"), und,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])
            gt_with_bbox = und.copy()
            for b in bboxes:
                draw_3d_bbox_solid(gt_with_bbox, b['corners_2d'], b['corners_valid'],
                                   b['corners_cam'], b['color'])
            cv2.imwrite(str(bbox_on_gt_dir / f"{cam_name}.jpg"), gt_with_bbox,
                        [cv2.IMWRITE_JPEG_QUALITY, 100])

    return cam_name, len(bboxes)


# ============================================================
# 分段逻辑
# ============================================================

def list_annotation_files(clip_dir):
    label_dir = Path(clip_dir) / 'car_labels' / 'interpolation_labels'
    files = []
    for f in label_dir.glob('*.json'):
        try:
            ts = float(f.stem)
            files.append((ts, f))
        except ValueError:
            continue
    files.sort()
    return files


def select_middle_segments(sorted_files, seg_len=SEGMENT_LENGTH, n_seg=NUM_SEGMENTS):
    total_needed = seg_len * n_seg
    n = len(sorted_files)
    if n < total_needed:
        return []
    start = (n - total_needed) // 2
    selected = sorted_files[start:start + total_needed]
    segments = []
    for i in range(n_seg):
        seg = selected[i * seg_len:(i + 1) * seg_len]
        segments.append(seg)
    return segments


# ============================================================
# 主流程
# ============================================================

def process_clip(clip_id, clip_dir, calibs, output_root, num_threads):
    ann_files = list_annotation_files(clip_dir)
    if len(ann_files) < SEGMENT_LENGTH * NUM_SEGMENTS:
        print(f"  [跳过] {clip_id}: 只有 {len(ann_files)} 帧，不够 {SEGMENT_LENGTH * NUM_SEGMENTS}")
        return 0

    segments = select_middle_segments(ann_files)
    if not segments:
        return 0

    cam_id = 2  # FW only
    gt_images_dir = Path(clip_dir) / 'car' / 'images'
    gt_cache = build_gt_cache(gt_images_dir, cam_ids=[cam_id])

    total_imgs = 0
    for seg_idx, seg_frames in enumerate(segments):
        seg_name = f"seg{seg_idx:02d}"
        for frame_ts, ann_path in seg_frames:
            ts_str = ann_path.stem  # "1742878202.100021"
            ts_us = int(frame_ts * 1_000_000)

            frame_dir = output_root / clip_id / seg_name / ts_str
            overlay_dir = frame_dir / "overlay"
            gt_dir = frame_dir / "gt"
            bbox_on_gt_dir = frame_dir / "bbox_on_gt"
            overlay_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
            bbox_on_gt_dir.mkdir(parents=True, exist_ok=True)

            # 读标注
            with open(ann_path) as f:
                ann = json.load(f)

            objects_data = []
            for obj in ann.get('object', []):
                corners = get_3d_bbox_corners(obj)
                color = LABEL_COLORS.get(obj.get('label', ''), LABEL_COLORS['unknown'])
                objects_data.append({'corners': corners, 'color': color})

            gt_path = find_nearest_gt(gt_cache[cam_id], ts_us)
            process_camera(cam_id, objects_data, calibs[cam_id],
                           gt_path, overlay_dir, gt_dir, bbox_on_gt_dir)
            total_imgs += 1

        print(f"    {seg_name}: {len(seg_frames)} 帧完成")
    return total_imgs


def find_all_clips():
    clips = []
    for d in sorted(DATASET_ROOT.glob('*')):
        if not d.is_dir():
            continue
        clip_id = d.name[:3]
        label_dir = d / 'car_labels' / 'interpolation_labels'
        if label_dir.exists():
            clips.append((clip_id, d))
    return clips


def main():
    import argparse
    p = argparse.ArgumentParser(description='车端标注 HDMap 投影 (实心 3D bbox)')
    p.add_argument('--clips', nargs='*', default=None,
                   help='指定 clip ID (如 004 006)，不指定则处理全部')
    p.add_argument('--output', default='/mnt/zihanw/car_side_bbox',
                   help='输出根目录')
    p.add_argument('--num-threads', type=int, default=7,
                   help='每帧并行处理的相机数 (默认 7)')
    args = p.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print("加载相机标定...")
    calibs = load_all_camera_calibs()

    all_clips = find_all_clips()
    if args.clips:
        all_clips = [(cid, cdir) for cid, cdir in all_clips if cid in args.clips]

    # 过滤跳过的 clip
    all_clips = [(cid, cdir) for cid, cdir in all_clips if cid not in SKIP_CLIPS]

    print(f"共 {len(all_clips)} 个 clip，每个取中间 {NUM_SEGMENTS}×{SEGMENT_LENGTH}={NUM_SEGMENTS*SEGMENT_LENGTH} 帧\n")

    t0 = time.time()
    total = 0
    for clip_id, clip_dir in all_clips:
        print(f"[{clip_id}] {clip_dir.name}")
        n = process_clip(clip_id, clip_dir, calibs, output_root, args.num_threads)
        total += n

    dt = time.time() - t0
    print(f"\n完成: {total} 张图，用时 {dt:.0f}s")


if __name__ == '__main__':
    main()
