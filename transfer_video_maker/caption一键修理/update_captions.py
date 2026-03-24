#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量更新 Transfer2 数据集的 caption

支持：
- 自动识别所有7个相机目录
- 支持模板变量：{camera}, {scene}, {seg}, {view_prefix}, {direction}
- 支持新数据格式：scene{id}/ 子目录，文件名 {scene_id}_id{vehicle_id}_seg{nn}.json
- 自动从 direction.json 读取行驶方向（segment_pipeline 生成）
- 预览更改后再应用
- 可选择单个或多个数据集
"""

import json
import re
from pathlib import Path
import argparse
import sys

# 相机名称列表（用于遍历文件夹）
CAMERA_NAMES = [
    'ftheta_camera_front_tele_30fov',
    'ftheta_camera_front_wide_120fov',
    'ftheta_camera_cross_left_120fov',
    'ftheta_camera_cross_right_120fov',
    'ftheta_camera_rear_left_70fov',
    'ftheta_camera_rear_right_70fov',
    'ftheta_camera_rear_tele_30fov'
]

# 相机视角映射（用于生成 caption 前缀）
CAMERA_VIEW_PREFIX = {
    'ftheta_camera_front_tele_30fov': 'Front telephoto view',
    'ftheta_camera_front_wide_120fov': 'Front wide view',
    'ftheta_camera_cross_left_120fov': 'Left cross view',
    'ftheta_camera_cross_right_120fov': 'Right cross view',
    'ftheta_camera_rear_left_70fov': 'Rear left view',
    'ftheta_camera_rear_right_70fov': 'Rear right view',
    'ftheta_camera_rear_tele_30fov': 'Rear telephoto view'
}

# 详细场景描述（描述目标 RGB 输出）
SCENE_DESCRIPTION = (
    "Northern Chinese suburban intersection captured in early spring. "
    "Clear daytime conditions with bright blue sky and soft natural sunlight casting gentle shadows. "
    "Wide multi-lane asphalt road surface in good condition with crisp white lane markings, "
    "directional arrows, and crosswalk patterns. Beige and tan colored high-rise residential "
    "apartment buildings line both sides of the street, typical of Chinese suburban architecture. "
    "Rows of bare deciduous trees with leafless branches stand along the sidewalks, characteristic "
    "of late winter to early spring season. White painted metal safety railings separate the road "
    "from pedestrian areas. Green traffic signals mounted on overhead poles with directional signs. "
    "Street lamp posts visible along the road. Occasional mixed traffic including sedans, SUVs, "
    "buses, trucks, and non-motorized road users such as pedestrians, cyclists, and electric "
    "tricycles. Clean urban environment with well-maintained infrastructure."
)

# 预定义caption模板（统一描述目标 RGB 输出，包含朝向）
PRESET_TEMPLATES = {
    'unified': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
    'depth': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
    'depth_dense': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
    'hdmap': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
    'blur': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
    'blur_dense': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
    'basic': '{view_prefix}. The ego vehicle is traveling from {direction}. ' + SCENE_DESCRIPTION,
}

# 数据集名称到模板的自动映射
DATASET_TEMPLATE_MAPPING = {
    'DepthSparse': 'unified',
    'DepthDense': 'unified',
    'HDMapBbox': 'unified',
    'BlurProjection': 'unified',
    'BlurDense': 'unified',
    'BasicProjection': 'unified'
}


# ============================================================
# direction.json 自动查找
# ============================================================

# 缓存已加载的 direction 信息
_direction_cache = {}


def load_direction_from_segment_output(segment_output_dir):
    """
    从 segment_pipeline 输出目录加载 direction.json

    Args:
        segment_output_dir: segment 输出目录 (含 direction.json)

    Returns:
        direction_text: 如 'west to east', 找不到则返回 None
    """
    cache_key = str(segment_output_dir)
    if cache_key in _direction_cache:
        return _direction_cache[cache_key]

    direction_file = Path(segment_output_dir) / "direction.json"
    if direction_file.exists():
        with open(direction_file, 'r') as f:
            data = json.load(f)
        direction = data.get('direction', None)
        _direction_cache[cache_key] = direction
        return direction

    _direction_cache[cache_key] = None
    return None


def find_direction_for_segment(json_path, segment_pipeline_output=None):
    """
    为 caption 文件查找对应的行驶方向

    尝试查找策略：
    1. 从 segment_pipeline 输出目录的 direction.json 读取
    2. 从 caption JSON 文件本身已有的 direction 字段读取
    3. 返回 'unknown direction'

    Args:
        json_path: caption JSON 文件路径
        segment_pipeline_output: segment_pipeline 输出根目录

    Returns:
        direction_text: 如 'west to east'
    """
    # 策略1: 从 segment_pipeline 输出读取
    if segment_pipeline_output:
        # 从文件名解析 scene_id 和 vehicle_id
        info = parse_caption_filename(json_path)
        scene_id = info.get('scene', '')
        vehicle_id = info.get('vehicle_id', '')
        seg = info.get('seg', '')

        if scene_id and vehicle_id and seg:
            seg_dir = (Path(segment_pipeline_output) /
                       f"scene{scene_id}" /
                       f"vehicle{vehicle_id}_{seg}")
            direction = load_direction_from_segment_output(seg_dir)
            if direction:
                return direction

    # 策略2: 从 caption JSON 已有的 direction 字段读取
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'direction' in data and data['direction']:
            return data['direction']
    except Exception:
        pass

    return 'unknown direction'


# ============================================================
# 核心函数
# ============================================================

def find_datasets(base_dir):
    """查找所有数据集目录"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"错误: 目录不存在: {base_dir}")
        return []

    dataset_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            captions_dir = item / 'captions'
            if captions_dir.exists() and captions_dir.is_dir():
                dataset_dirs.append(item)

    return sorted(dataset_dirs)


def get_caption_files(dataset_dir):
    """
    获取数据集中所有caption JSON文件

    支持两种目录结构：
    1. 旧格式：captions/{camera}/{scene}_seg{nn}.json
    2. 新格式：captions/{camera}/scene{id}/{scene_id}_id{vehicle_id}_seg{nn}.json
    """
    captions_dir = dataset_dir / 'captions'
    caption_files = []

    for cam_name in CAMERA_NAMES:
        cam_dir = captions_dir / cam_name
        if cam_dir.exists():
            # 旧格式：直接在相机目录下的JSON文件
            json_files = sorted(cam_dir.glob('*.json'))
            caption_files.extend(json_files)

            # 新格式：scene{id}/ 子目录下的JSON文件
            for scene_subdir in cam_dir.iterdir():
                if scene_subdir.is_dir() and scene_subdir.name.startswith('scene'):
                    json_files = sorted(scene_subdir.glob('*.json'))
                    caption_files.extend(json_files)

    return caption_files


def parse_caption_filename(json_path):
    """
    从JSON文件名解析信息

    支持两种格式：
    1. 旧格式：002_seg01.json -> scene=002, seg=seg01
    2. 新格式：004_id45_seg01.json -> scene=004, vehicle_id=45, seg=seg01
    """
    filename = json_path.stem

    # 确定相机名称
    parent = json_path.parent
    if parent.name.startswith('scene'):
        camera = parent.parent.name
    else:
        camera = parent.name

    # 解析文件名
    parts = filename.split('_')
    scene = parts[0]

    # 检查是否为新格式（包含 idXX）
    if len(parts) >= 3 and parts[1].startswith('id'):
        vehicle_id = parts[1][2:]
        seg = parts[2] if len(parts) > 2 else 'seg01'
    else:
        vehicle_id = None
        seg = parts[1] if len(parts) > 1 else 'seg01'

    # 视角前缀
    view_prefix = CAMERA_VIEW_PREFIX.get(
        camera, camera.replace('ftheta_', '').replace('_', ' ')
    )

    return {
        'camera': camera,
        'view_prefix': view_prefix,
        'scene': scene,
        'vehicle_id': vehicle_id,
        'seg': seg,
        'direction': 'unknown direction',  # placeholder, 将被覆盖
    }


def generate_caption(template, info):
    """根据模板生成caption"""
    return template.format(**info)


def preview_changes(caption_files, template, segment_pipeline_output=None, max_preview=5):
    """预览将要做的更改"""
    print(f"\n预览更改（显示前 {max_preview} 个）:")
    print("=" * 80)

    for i, json_path in enumerate(caption_files[:max_preview]):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        old_caption = data.get('caption', '')

        info = parse_caption_filename(json_path)
        info['direction'] = find_direction_for_segment(json_path, segment_pipeline_output)
        new_caption = generate_caption(template, info)

        rel_path = json_path.name
        try:
            rel_path = json_path.relative_to(json_path.parents[3])
        except (ValueError, IndexError):
            pass
        print(f"\n文件: {rel_path}")
        print(f"  方向: {info['direction']}")
        print(f"  旧: {str(old_caption)[:100]}...")
        print(f"  新: {new_caption[:100]}...")

    if len(caption_files) > max_preview:
        print(f"\n... 还有 {len(caption_files) - max_preview} 个文件")

    print(f"\n总计: {len(caption_files)} 个文件将被更新")
    print("=" * 80)


def update_captions(caption_files, template, segment_pipeline_output=None, dry_run=False):
    """
    更新所有caption

    Args:
        caption_files: JSON文件路径列表
        template: 新的caption模板
        segment_pipeline_output: segment_pipeline 输出目录 (用于读取 direction.json)
        dry_run: 是否只是预览，不实际修改

    Returns:
        success_count: 成功更新的文件数
    """
    success_count = 0

    for json_path in caption_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            info = parse_caption_filename(json_path)
            info['direction'] = find_direction_for_segment(json_path, segment_pipeline_output)
            new_caption = generate_caption(template, info)

            data['caption'] = new_caption
            # 同时保存 direction 到 JSON 供后续使用
            data['direction'] = info['direction']

            if not dry_run:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            success_count += 1

        except Exception as e:
            print(f"错误: 处理文件 {json_path} 失败: {e}")

    return success_count


# ============================================================
# 交互式模式
# ============================================================

def interactive_mode(base_dir, segment_pipeline_output=None):
    """交互式模式"""
    print("=" * 80)
    print("Transfer2 Caption 批量更新工具")
    print("=" * 80)

    if segment_pipeline_output:
        print(f"方向数据来源: {segment_pipeline_output}")

    datasets = find_datasets(base_dir)
    if not datasets:
        print(f"错误: 在 {base_dir} 下未找到任何数据集")
        return

    print(f"\n找到 {len(datasets)} 个数据集:")
    for i, ds in enumerate(datasets, 1):
        num_files = len(get_caption_files(ds))
        print(f"  {i}) {ds.name} ({num_files} 个caption文件)")

    print(f"  {len(datasets) + 1}) 全部数据集")
    print("  0) 退出")

    choice = input(f"\n请选择数据集 [1-{len(datasets) + 1}, 0]: ").strip()
    if choice == '0':
        print("已退出")
        return

    try:
        choice_num = int(choice)
        if choice_num == len(datasets) + 1:
            selected_datasets = datasets
        elif 1 <= choice_num <= len(datasets):
            selected_datasets = [datasets[choice_num - 1]]
        else:
            print("无效选择")
            return
    except ValueError:
        print("无效输入")
        return

    all_caption_files = []
    for ds in selected_datasets:
        all_caption_files.extend(get_caption_files(ds))

    if not all_caption_files:
        print("错误: 未找到任何caption文件")
        return

    print(f"\n总计: {len(all_caption_files)} 个caption文件")

    # 模板选择
    print("\n预设caption模板:")
    if len(selected_datasets) > 1:
        print(f"  0) 自动匹配（每个数据集使用对应的预设模板）")

    presets = list(PRESET_TEMPLATES.items())
    for i, (key, template) in enumerate(presets, 1):
        print(f"  {i}) {key}: \"{template[:60]}...\"")
    print(f"  {len(presets) + 1}) 自定义模板")

    if len(selected_datasets) > 1:
        template_choice = input(f"\n请选择模板 [0-{len(presets) + 1}]: ").strip()
    else:
        template_choice = input(f"\n请选择模板 [1-{len(presets) + 1}]: ").strip()

    try:
        template_num = int(template_choice)

        if template_num == 0 and len(selected_datasets) > 1:
            # 自动匹配模式
            print("\n使用自动匹配模式")
            print("\n匹配结果:")
            for ds in selected_datasets:
                matched = DATASET_TEMPLATE_MAPPING.get(ds.name, None)
                if matched:
                    print(f"  {ds.name} -> {matched}")
                else:
                    print(f"  {ds.name} -> [无匹配，跳过]")

            confirm = input("\n确认? [y/N]: ").strip().lower()
            if confirm != 'y':
                return

            print("\n正在更新...")
            total_success = 0
            for ds in selected_datasets:
                matched = DATASET_TEMPLATE_MAPPING.get(ds.name)
                if not matched:
                    continue
                template = PRESET_TEMPLATES[matched]
                files = get_caption_files(ds)
                if files:
                    success = update_captions(files, template, segment_pipeline_output)
                    total_success += success
                    print(f"  {ds.name}: {success}/{len(files)}")

            print(f"\n完成: {total_success}/{len(all_caption_files)}")
            return

        elif 1 <= template_num <= len(presets):
            template = presets[template_num - 1][1]
        elif template_num == len(presets) + 1:
            template = input("\n自定义模板 (变量: {camera}, {view_prefix}, {scene}, {seg}, {direction}): ").strip()
            if not template:
                print("错误: 模板不能为空")
                return
        else:
            print("无效选择")
            return
    except ValueError:
        print("无效输入")
        return

    preview_changes(all_caption_files, template, segment_pipeline_output)

    confirm = input("\n确认更新? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    print("\n正在更新...")
    success_count = update_captions(all_caption_files, template, segment_pipeline_output)
    print(f"\n完成: {success_count}/{len(all_caption_files)}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='批量更新 Transfer2 数据集的 caption')

    parser.add_argument('--base-dir', type=str,
                        default='/mnt/zihanw/proj_utils_pro_Roadside_Generation/transfer_video_maker/output',
                        help='数据集基础目录')

    parser.add_argument('--segment-output', type=str, default=None,
                        help='segment_pipeline 输出目录 (含 direction.json)')

    parser.add_argument('--dataset', type=str,
                        help='指定数据集名称')

    parser.add_argument('--template', type=str,
                        help='Caption模板')

    parser.add_argument('--preset', type=str,
                        choices=list(PRESET_TEMPLATES.keys()),
                        help='使用预设模板')

    parser.add_argument('--dry-run', action='store_true',
                        help='只预览，不实际修改')

    parser.add_argument('--interactive', action='store_true',
                        help='交互式模式（默认）')

    args = parser.parse_args()

    # 自动查找 segment_pipeline 输出目录
    segment_output = args.segment_output
    if segment_output is None:
        # 尝试默认路径
        default_seg_output = Path(__file__).resolve().parent.parent.parent / "segment_pipeline" / "output"
        if default_seg_output.exists():
            segment_output = str(default_seg_output)
            print(f"自动检测到 segment_pipeline 输出: {segment_output}")

    if len(sys.argv) == 1 or args.interactive:
        interactive_mode(args.base_dir, segment_output)
        return

    # 命令行模式
    if args.preset:
        template = PRESET_TEMPLATES[args.preset]
    elif args.template:
        template = args.template
    else:
        print("错误: 请指定 --template 或 --preset")
        return

    base_path = Path(args.base_dir)
    if args.dataset:
        datasets = [base_path / args.dataset]
    else:
        datasets = find_datasets(base_path)

    if not datasets:
        print("错误: 未找到任何数据集")
        return

    all_caption_files = []
    for ds in datasets:
        if ds.exists():
            all_caption_files.extend(get_caption_files(ds))

    if not all_caption_files:
        print("错误: 未找到任何caption文件")
        return

    preview_changes(all_caption_files, template, segment_output)

    if args.dry_run:
        print("\n[Dry-run] 未修改文件")
        return

    print("\n正在更新...")
    success_count = update_captions(all_caption_files, template, segment_output)
    print(f"\n完成: {success_count}/{len(all_caption_files)}")


if __name__ == "__main__":
    main()
