#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理视频2D bbox叠加工具 - 交互式入口

交互式选择：
1. 输入/输出目录
2. 场景选择（全部/特定场景）
3. 帧选择方式（middle/start/end/自定义范围）
4. seg数量和每seg帧数
"""

import sys
import re
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference_bbox_overlay.bbox_overlay import (
    parse_folder_name, load_carid_mapping, BboxOverlayProcessor, process_folder
)
import common_utils


def get_input_directory():
    """获取输入目录"""
    print("\n" + "="*60)
    print("📁 输入推理视频目录")
    print("="*60)

    default_dir = "/mnt/zihanw/Output_R2V_world_foundation_model_v1/inference"
    user_input = input(f"输入目录路径 [{default_dir}]: ").strip()

    input_dir = Path(user_input) if user_input else Path(default_dir)

    if not input_dir.exists():
        print(f"❌ 目录不存在: {input_dir}")
        sys.exit(1)

    return input_dir


def discover_folders(input_dir):
    """发现所有符合格式的文件夹"""
    folders = []
    scenes = {}  # scene_id -> list of folders

    for item in input_dir.iterdir():
        if item.is_dir():
            scene_id, seg_num = parse_folder_name(item.name)
            if scene_id is not None:
                folders.append({
                    'path': item,
                    'name': item.name,
                    'scene_id': scene_id,
                    'seg_num': seg_num
                })
                if scene_id not in scenes:
                    scenes[scene_id] = []
                scenes[scene_id].append(item)

    return sorted(folders, key=lambda x: x['name']), scenes


def select_scenes(folders, scenes):
    """选择要处理的场景"""
    print("\n" + "="*60)
    print("🎬 选择要处理的场景")
    print("="*60)

    # 显示发现的场景
    scene_list = sorted(scenes.keys())
    print(f"\n找到 {len(scene_list)} 个场景（共 {len(folders)} 个seg）：")

    for i, scene_key in enumerate(scene_list, 1):
        num_segs = len(scenes[scene_key])
        print(f"  {i}) {scene_key} ({num_segs} 个seg)")

    print(f"  {len(scene_list) + 1}) 全部处理")
    print("  0) 退出")

    # 获取用户选择
    choice = input(f"\n请选择 [1-{len(scene_list) + 1}, 0, 或输入场景序号如 031]: ").strip()

    if choice == '0':
        print("已退出")
        sys.exit(0)

    try:
        choice_num = int(choice)
        if choice_num == len(scene_list) + 1:
            # 全部处理
            selected_folders = [f['path'] for f in folders]
            print(f"\n✓ 已选择: 全部 {len(selected_folders)} 个seg")
        elif 1 <= choice_num <= len(scene_list):
            # 选择单个场景
            scene_key = scene_list[choice_num - 1]
            selected_folders = scenes[scene_key]
            print(f"\n✓ 已选择: {scene_key} ({len(selected_folders)} 个seg)")
        else:
            print("无效选择")
            sys.exit(1)
    except ValueError:
        # 尝试解析为场景序号
        scene_matches = [key for key in scene_list if choice in key]
        if scene_matches:
            selected_folders = []
            for key in scene_matches:
                selected_folders.extend(scenes[key])
            print(f"\n✓ 已选择: {', '.join(scene_matches)} ({len(selected_folders)} 个seg)")
        else:
            print(f"未找到匹配的场景: {choice}")
            sys.exit(1)

    return selected_folders


def get_frame_selection():
    """获取帧选择方式"""
    print("\n" + "="*60)
    print("🎞️ 选择帧范围方式")
    print("="*60)

    print("\n选项：")
    print("  1) middle - 从中间开始选取（推荐）")
    print("  2) start  - 从开头开始选取")
    print("  3) end    - 从末尾开始选取")
    print("  4) 自定义 - 指定起始帧索引")

    choice = input("\n请选择 [1-4, 默认1]: ").strip() or "1"

    if choice == '1':
        return 'middle'
    elif choice == '2':
        return 'start'
    elif choice == '3':
        return 'end'
    elif choice == '4':
        start = input("起始帧索引: ").strip()
        try:
            start_idx = int(start)
            return (start_idx, None)
        except:
            print("无效输入，使用默认middle")
            return 'middle'
    else:
        return 'middle'


def get_seg_config():
    """获取seg配置"""
    print("\n" + "="*60)
    print("⚙️ Seg配置")
    print("="*60)

    # seg数量
    num_segs_str = input("seg数量 [默认3]: ").strip()
    try:
        num_segs = int(num_segs_str) if num_segs_str else 3
    except:
        num_segs = 3

    # 每seg帧数
    frames_per_seg_str = input("每seg帧数 [默认29]: ").strip()
    try:
        frames_per_seg = int(frames_per_seg_str) if frames_per_seg_str else 29
    except:
        frames_per_seg = 29

    # 帧率
    fps_str = input("输出视频帧率 [默认29]: ").strip()
    try:
        fps = int(fps_str) if fps_str else 29
    except:
        fps = 29

    print(f"\n✓ 配置: {num_segs} segs × {frames_per_seg} 帧/seg @ {fps} FPS")

    return num_segs, frames_per_seg, fps


def get_output_directory(input_dir):
    """获取输出目录"""
    print("\n" + "="*60)
    print("📤 输出目录")
    print("="*60)

    default_output = input_dir / 'proj_hdmap'
    user_input = input(f"输出目录路径 [{default_output}]: ").strip()

    output_dir = Path(user_input) if user_input else default_output

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🎯 推理视频2D bbox叠加工具")
    print("="*60)
    print("功能：将路侧标注的动态物体投影为2D bbox，叠加到推理视频上")

    # 1. 获取输入目录
    input_dir = get_input_directory()

    # 2. 发现文件夹
    folders, scenes = discover_folders(input_dir)
    if not folders:
        print(f"\n❌ 在 {input_dir} 中未找到符合格式的文件夹")
        print("   格式要求: {scene_id}_seg{seg_num}")
        print("   例如: 031_seg01")
        sys.exit(1)

    # 3. 选择场景
    selected_folders = select_scenes(folders, scenes)

    # 4. 获取帧选择方式
    frame_selection = get_frame_selection()

    # 5. 获取seg配置
    num_segs, frames_per_seg, fps = get_seg_config()

    # 6. 获取输出目录
    output_dir = get_output_directory(input_dir)

    # 7. 确认
    print("\n" + "="*60)
    print("📋 配置确认")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"处理文件夹: {len(selected_folders)} 个")
    print(f"帧选择: {frame_selection}")
    print(f"Seg配置: {num_segs} segs × {frames_per_seg} 帧/seg @ {fps} FPS")

    confirm = input("\n确认开始处理? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消")
        sys.exit(0)

    # 8. 初始化处理器
    print("\n" + "="*60)
    print("🚀 开始处理")
    print("="*60)

    vehicle_calib = common_utils.VEHICLE_CALIB_DIR
    processor = BboxOverlayProcessor(vehicle_calib)

    # 加载自车ID映射
    carid_mapping = load_carid_mapping()

    # 9. 处理每个文件夹
    success_count = 0
    total = len(selected_folders)

    for i, folder in enumerate(selected_folders, 1):
        print(f"\n[{i}/{total}] ", end="")
        try:
            if process_folder(folder, output_dir, processor, num_segs,
                             frames_per_seg, frame_selection, carid_mapping, fps):
                success_count += 1
        except Exception as e:
            print(f"❌ 处理失败: {e}")

    # 10. 完成
    print("\n" + "="*60)
    print("✅ 处理完成")
    print("="*60)
    print(f"成功: {success_count}/{total} 个文件夹")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
