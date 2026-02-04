#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量模式包装脚本
负责收集参数并协调多个投影项目的执行
"""

import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_utils


def main():
    print("\n" + "="*60)
    print("🚀 批量投影模式 - 参数配置")
    print("="*60)
    print("提示：接下来输入的参数将应用于所有选中的投影项目")
    print("="*60)

    # 清除旧的批量配置
    common_utils.clear_batch_config()

    # 交互式输入（首次）
    config = common_utils.interactive_input(batch_mode_enabled=False)

    if not config:
        print("❌ 配置输入失败")
        sys.exit(1)

    # 保存配置供后续项目使用
    # 同时预加载自车ID映射（HDMap投影需要）
    ego_vehicle_mapping = common_utils.get_ego_vehicle_id(
        scene_ids=config['scene_ids'],
        batch_mode_enabled=False,
        default_id=45
    )
    if ego_vehicle_mapping:
        config['ego_vehicle_mapping'] = ego_vehicle_mapping

    common_utils.save_batch_config(config)

    print("\n✅ 配置已保存，准备执行批量投影...")
    print("   后续项目将自动使用这些参数\n")


if __name__ == "__main__":
    main()
