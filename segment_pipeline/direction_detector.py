#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
行驶方向检测模块
从 ego 轨迹推算行驶方向，使用参考车辆标定坐标轴朝向
"""

import json
import numpy as np
from pathlib import Path


# 4 个标准方向
DIRECTIONS = {
    'W2E': 'west to east',
    'E2W': 'east to west',
    'N2S': 'north to south',
    'S2N': 'south to north',
}


def compute_displacement_vector(poses):
    """
    从轨迹计算位移方向向量

    Args:
        poses: pose 列表, 每个含 {'x', 'y', ...}

    Returns:
        direction_vector: (dx, dy) 归一化方向向量
    """
    if len(poses) < 2:
        return np.array([0.0, 0.0])

    dx = poses[-1]['x'] - poses[0]['x']
    dy = poses[-1]['y'] - poses[0]['y']

    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6:
        return np.array([0.0, 0.0])

    return np.array([dx / norm, dy / norm])


def build_reference_vectors(reference_vehicles, label_dir_func):
    """
    从 intersection_filter 的参考车辆构建方向参考向量

    Args:
        reference_vehicles: intersection_filter.py 中的 REFERENCE_VEHICLES 列表
        label_dir_func: 函数 (scene_prefix) → 标注目录路径

    Returns:
        ref_vectors: {direction_key: (dx, dy)} 归一化方向向量
    """
    ref_vectors = {}

    for ref in reference_vehicles:
        direction = ref['direction']
        scene_prefix = ref['scene_prefix']
        vehicle_id = ref['vehicle_id']
        entry_ts = ref['entry_ts']
        exit_ts = ref['exit_ts']

        # 查找 entry 和 exit 时间戳对应的标注文件
        label_dir = label_dir_func(scene_prefix)
        if label_dir is None:
            print(f"  警告: 场景 {scene_prefix} 标注目录不存在, 跳过参考车辆 {direction}")
            continue

        entry_pos = _find_vehicle_position(label_dir, vehicle_id, entry_ts)
        exit_pos = _find_vehicle_position(label_dir, vehicle_id, exit_ts)

        if entry_pos is None or exit_pos is None:
            print(f"  警告: 无法获取参考车辆 {direction} 的 entry/exit 位置")
            continue

        dx = exit_pos[0] - entry_pos[0]
        dy = exit_pos[1] - entry_pos[1]
        norm = np.sqrt(dx**2 + dy**2)

        if norm < 1e-6:
            continue

        ref_vectors[direction] = np.array([dx / norm, dy / norm])
        print(f"  参考向量 {direction}: ({dx/norm:.3f}, {dy/norm:.3f})")

    return ref_vectors


def _find_vehicle_position(label_dir, vehicle_id, target_ts):
    """
    在标注目录中找到最接近目标时间戳的车辆位置

    Args:
        label_dir: 标注目录路径
        vehicle_id: 车辆 ID
        target_ts: 目标时间戳 (毫秒)

    Returns:
        (x, y) or None
    """
    label_dir = Path(label_dir)
    if not label_dir.exists():
        return None

    # 查找所有标注文件, 按时间戳排序
    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        return None

    # 提取时间戳并找最近的
    import re
    closest_file = None
    min_diff = float('inf')

    for jf in json_files:
        match = re.search(r'(\d+)', jf.stem)
        if match:
            file_ts = int(match.group(1))
            diff = abs(file_ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                closest_file = jf

    if closest_file is None or min_diff > 5000:  # 5秒容差
        return None

    # 读取标注找车辆位置
    with open(closest_file, 'r') as f:
        data = json.load(f)

    for obj in data.get('object', []):
        if obj['id'] == vehicle_id:
            return (obj['x'], obj['y'])

    return None


def detect_direction(poses, ref_vectors):
    """
    根据轨迹位移向量和参考向量匹配行驶方向

    Args:
        poses: pose 列表
        ref_vectors: {direction_key: (dx, dy)} 参考向量

    Returns:
        direction_key: 'W2E' / 'E2W' / 'N2S' / 'S2N'
        direction_text: 'west to east' 等
        confidence: 余弦相似度 (0~1)
    """
    disp_vec = compute_displacement_vector(poses)

    if np.linalg.norm(disp_vec) < 1e-6:
        return 'unknown', 'unknown direction', 0.0

    best_key = 'unknown'
    best_sim = -1.0

    for key, ref_vec in ref_vectors.items():
        # 余弦相似度
        sim = np.dot(disp_vec, ref_vec)
        if sim > best_sim:
            best_sim = sim
            best_key = key

    direction_text = DIRECTIONS.get(best_key, 'unknown direction')
    confidence = max(0.0, best_sim)

    return best_key, direction_text, confidence


def save_direction(direction_key, direction_text, confidence, output_path):
    """
    保存方向信息到 JSON

    Args:
        direction_key: 'W2E' 等
        direction_text: 'west to east' 等
        confidence: 匹配置信度
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'direction_key': direction_key,
        'direction': direction_text,
        'confidence': round(confidence, 4),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  direction: {direction_text} (confidence={confidence:.3f})")
