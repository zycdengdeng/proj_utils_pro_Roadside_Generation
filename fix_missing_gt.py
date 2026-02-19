#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复缺失的GT相机图像

针对 083-086 等场景中某些时间戳缺少特定相机GT图像的问题，
使用最近的有效帧来填充缺失的相机图像。
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# 缺失记录：{scene_id: [(timestamp, missing_camera), ...]}
MISSING_GT = {
    "083": [
        (1743650746943, "FL"),
        (1743650754054, "RL"),
    ],
    "084": [
        (1743651154074, "FL"),
    ],
    "085": [
        (1743651916381, "FL"),
        (1743651918857, "FL"),
    ],
    "086": [
        (1743652307891, "RL"),
        (1743652308169, "FN"),
        (1743652310078, "FL"),
    ],
}

# 所有相机列表
ALL_CAMERAS = ["FL", "FN", "FR", "FW", "RL", "RN", "RR"]


def find_nearest_valid_timestamp(scene_dir: Path, target_ts: int, camera: str, subdir: str = "gt") -> Tuple[int, Path]:
    """
    找到最近的有该相机图像的时间戳

    Args:
        scene_dir: 场景输出目录 (如 blur投影/083_id22/083/)
        target_ts: 目标时间戳
        camera: 相机名称 (如 "FL")
        subdir: 子目录名 (如 "gt", "proj", "overlay")

    Returns:
        (nearest_ts, image_path) 或 (None, None) 如果找不到
    """
    # 获取所有时间戳目录
    ts_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]

    min_diff = float('inf')
    nearest_ts = None
    nearest_path = None

    for ts_dir in ts_dirs:
        ts = int(ts_dir.name)
        if ts == target_ts:
            continue  # 跳过目标时间戳本身

        # 检查该时间戳是否有目标相机的图像
        img_path = ts_dir / subdir / f"{camera}.jpg"
        if img_path.exists():
            diff = abs(ts - target_ts)
            if diff < min_diff:
                min_diff = diff
                nearest_ts = ts
                nearest_path = img_path

    return nearest_ts, nearest_path


def fix_missing_gt_for_clip(proj_type: str, clip_id: str, vehicle_id: int, missing_list: List[Tuple[int, str]], base_dir: Path):
    """
    修复单个clip的缺失GT

    Args:
        proj_type: 投影类型 (blur投影, depth投影, HDMap投影)
        clip_id: 场景ID (如 "083")
        vehicle_id: 车辆ID
        missing_list: [(timestamp, camera), ...]
        base_dir: 项目根目录
    """
    scene_dir = base_dir / proj_type / f"{clip_id}_id{vehicle_id}" / clip_id

    if not scene_dir.exists():
        print(f"  ⚠️  目录不存在: {scene_dir}")
        return

    # 需要填充的子目录
    subdirs = ["gt", "proj", "overlay", "compare"]
    if proj_type == "depth投影":
        subdirs.append("depth")
    if proj_type == "HDMap投影":
        subdirs = ["gt", "overlay", "bbox_on_gt"]

    for ts, camera in missing_list:
        print(f"  处理 {clip_id}/{ts}/{camera}...")

        for subdir in subdirs:
            target_path = scene_dir / str(ts) / subdir / f"{camera}.jpg"

            # 检查是否确实缺失
            if target_path.exists():
                continue

            # 查找最近的有效帧
            nearest_ts, source_path = find_nearest_valid_timestamp(scene_dir, ts, camera, subdir)

            if source_path is None:
                print(f"    ⚠️  {subdir}/{camera}: 找不到有效的替代帧")
                continue

            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(source_path, target_path)
            print(f"    ✓ {subdir}/{camera}: 从 {nearest_ts} 复制 (时差 {abs(nearest_ts - ts)}ms)")


def get_vehicle_id(clip_id: str, base_dir: Path, proj_type: str = "blur投影") -> int:
    """从目录名获取车辆ID"""
    proj_dir = base_dir / proj_type
    for d in proj_dir.iterdir():
        if d.name.startswith(f"{clip_id}_id"):
            # 提取 id 后面的数字
            vid = d.name.split("_id")[1]
            return int(vid)
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="修复缺失的GT相机图像")
    parser.add_argument("--base-dir", type=str,
                        default="/mnt/zihanw/merge_dyn_stat/proj_utils_pro_Roadside_transfer",
                        help="项目根目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="只检查不实际复制")

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    print("=" * 60)
    print("修复缺失GT相机图像")
    print("=" * 60)

    # 投影类型列表
    proj_types = ["blur投影", "depth投影", "HDMap投影"]

    for clip_id, missing_list in MISSING_GT.items():
        print(f"\n===== 场景 {clip_id} =====")
        print(f"缺失记录: {len(missing_list)} 帧")

        for proj_type in proj_types:
            print(f"\n  [{proj_type}]")

            # 获取车辆ID
            vehicle_id = get_vehicle_id(clip_id, base_dir, proj_type)
            if vehicle_id is None:
                print(f"  ⚠️  未找到 {clip_id} 的输出目录")
                continue

            print(f"  车辆ID: {vehicle_id}")

            if args.dry_run:
                print("  [DRY RUN] 跳过实际操作")
                continue

            fix_missing_gt_for_clip(proj_type, clip_id, vehicle_id, missing_list, base_dir)

    print("\n" + "=" * 60)
    print("✓ 修复完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
