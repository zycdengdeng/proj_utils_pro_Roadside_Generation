#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case C - Step 2: 围绕 ego 在正交 4 方向放虚拟观察车 pose（朝向指向 ego）。

  N: (xe, ye+R)  yaw = -π/2
  E: (xe+R, ye)  yaw =  π
  S: (xe, ye-R)  yaw =  π/2
  W: (xe-R, ye)  yaw =  0

输入: case_C/output/ego_pose_t_star.json
输出: case_C/output/virtual_observers.json
"""

import argparse
import json
import math
from pathlib import Path

# R2A 红框（来自任务描述，仅用于 sanity check）
R2A_REGION = {'x_min': -95.1, 'x_max': -30.7, 'y_min': -30.6, 'y_max': 9.8}


def in_region(x, y, region=R2A_REGION):
    return (region['x_min'] <= x <= region['x_max']
            and region['y_min'] <= y <= region['y_max'])


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Case C: 生成 4 虚拟观察车 pose")
    ap.add_argument('--ego-pose', default=str(here / 'output' / 'ego_pose_t_star.json'))
    ap.add_argument('--radius', type=float, default=12.0, help='距离 ego 多少米，默认 12')
    ap.add_argument('--output', default=str(here / 'output' / 'virtual_observers.json'))
    args = ap.parse_args()

    with open(args.ego_pose) as f:
        ego = json.load(f)
    xe, ye, ze = ego['x'], ego['y'], ego['z']
    R = args.radius

    # (name, dx_dir, dy_dir, yaw_pointing_to_ego)
    spec = [
        ('N',  0.0,  1.0, -math.pi / 2),
        ('E',  1.0,  0.0,  math.pi),
        ('S',  0.0, -1.0,  math.pi / 2),
        ('W', -1.0,  0.0,  0.0),
    ]

    observers = []
    for name, dx, dy, yaw in spec:
        x = xe + R * dx
        y = ye + R * dy
        ok = in_region(x, y)
        observers.append({
            'name': name,
            'x': float(x),
            'y': float(y),
            'z': float(ze),
            'yaw': float(yaw),
            'roll': 0.0,
            'pitch': 0.0,
            'vehicle_height': 1.6,
            'timestamp': ego['timestamp'],
            'in_R2A_region': bool(ok),
        })

    out = {
        'ego_reference': {
            'x': xe, 'y': ye, 'z': ze,
            'timestamp': ego['timestamp'],
            'scene': ego['scene'],
            'ego_vehicle_id': ego['ego_vehicle_id'],
        },
        'radius_m': R,
        'observers': observers,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"✓ 写入 {out_path}")
    for ob in observers:
        flag = 'OK' if ob['in_R2A_region'] else '⚠ 出框'
        print(f"  {ob['name']}: ({ob['x']:.2f}, {ob['y']:.2f})  yaw={ob['yaw']:+.3f}  [{flag}]")
    if any(not ob['in_R2A_region'] for ob in observers):
        print("\n⚠️  有观察车落在 R2A 红框外，请检查 ego pose 或缩小 --radius。")


if __name__ == '__main__':
    main()
