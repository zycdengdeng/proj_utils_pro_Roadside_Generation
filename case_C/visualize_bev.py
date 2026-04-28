#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case C - BEV 可视化（保存 PNG）
画出 R2A 红框 + ego (红) + 4 虚拟观察车 (蓝, 箭头指向 ego)。
可选：叠加场景 t* 时刻的 PCD 点云作灰底。

Usage:
    python case_C/visualize_bev.py
    python case_C/visualize_bev.py --with-pcd
    python case_C/visualize_bev.py --output xxx.png
"""

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common_utils import get_scene_paths

R2A_REGION = {'x_min': -95.1, 'x_max': -30.7, 'y_min': -30.6, 'y_max': 9.8}


def load_pcd_xyz(path):
    """优先 open3d, 回退到手动 ASCII PCD 解析"""
    try:
        import open3d as o3d
        return np.asarray(o3d.io.read_point_cloud(path).points)
    except Exception:
        pass
    pts, in_data = [], False
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            if in_data:
                p = line.strip().split()
                if len(p) >= 3:
                    try:
                        pts.append([float(p[0]), float(p[1]), float(p[2])])
                    except ValueError:
                        continue
            elif line.strip().startswith('DATA'):
                in_data = True
    return np.array(pts) if pts else np.zeros((0, 3))


def find_pcd_for_timestamp(pcd_dir, ts_ms):
    best, best_d = None, float('inf')
    for f in glob.glob(str(Path(pcd_dir) / '*.pcd')):
        stem = Path(f).stem
        nums = ''.join(c for c in stem if c.isdigit())
        try:
            t = int(nums)
        except ValueError:
            continue
        d = abs(t - ts_ms)
        if d < best_d:
            best_d, best = d, f
    return best


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--ego-pose', default=str(here / 'output' / 'ego_pose_t_star.json'))
    ap.add_argument('--observers', default=str(here / 'output' / 'virtual_observers.json'))
    ap.add_argument('--output', default=str(here / 'output' / 'bev_observers.png'))
    ap.add_argument('--with-pcd', action='store_true', help='叠加 t* 时刻的 PCD 灰底')
    ap.add_argument('--margin', type=float, default=10.0,
                    help='坐标范围相对 R2A 框的扩展边距, 默认 10m')
    args = ap.parse_args()

    with open(args.ego_pose) as f:
        ego = json.load(f)
    with open(args.observers) as f:
        observers = json.load(f)['observers']

    fig, ax = plt.subplots(figsize=(12, 10))

    # 可选 PCD 灰底
    if args.with_pcd:
        scene_paths = get_scene_paths(ego['scene'])
        pcd_dir = scene_paths.get('pcd') if scene_paths else None
        pcd_file = find_pcd_for_timestamp(pcd_dir, ego['timestamp']) if pcd_dir else None
        if pcd_file:
            print(f'PCD: {pcd_file}')
            pts = load_pcd_xyz(pcd_file)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=0.1, c='gray', alpha=0.25, rasterized=True)

    # R2A 红框
    rx = R2A_REGION['x_min']; ry = R2A_REGION['y_min']
    rw = R2A_REGION['x_max'] - rx; rh = R2A_REGION['y_max'] - ry
    ax.add_patch(patches.Rectangle((rx, ry), rw, rh,
                                   linewidth=2.5, edgecolor='red',
                                   facecolor='red', alpha=0.08,
                                   label=f"R2A region "
                                         f"x[{R2A_REGION['x_min']},{R2A_REGION['x_max']}] "
                                         f"y[{R2A_REGION['y_min']},{R2A_REGION['y_max']}]"))

    # ego (红圆 + yaw 箭头)
    ex, ey = ego['x'], ego['y']
    ax.scatter([ex], [ey], s=220, c='red', zorder=8,
               edgecolors='black', linewidths=1.2, label='ego (real capture car)')
    yaw_e = ego.get('yaw', 0.0)
    ax.arrow(ex, ey, 3 * math.cos(yaw_e), 3 * math.sin(yaw_e),
             head_width=0.6, head_length=0.8, fc='red', ec='red', zorder=9)
    ax.annotate(f"ego id={ego['ego_vehicle_id']}",
                (ex, ey), textcoords='offset points', xytext=(8, 8),
                fontsize=9, color='red', fontweight='bold')

    # 4 观察车 (蓝箭头指向 ego)
    for ob in observers:
        ox, oy, yaw_o = ob['x'], ob['y'], ob['yaw']
        in_box = ob['in_R2A_region']
        color = 'blue' if in_box else 'orange'
        ax.scatter([ox], [oy], s=160, c=color, marker='^', zorder=7,
                   edgecolors='black', linewidths=1)
        # 朝向箭头 (5 m 长, 指向 ego 那一侧)
        ax.arrow(ox, oy, 5 * math.cos(yaw_o), 5 * math.sin(yaw_o),
                 head_width=0.7, head_length=0.9, fc=color, ec=color, zorder=8)
        # 与 ego 的连线 (虚线)
        ax.plot([ox, ex], [oy, ey], '--', color=color, lw=0.8, alpha=0.5)
        flag = '' if in_box else ' [OUT OF REGION]'
        ax.annotate(f"{ob['name']}{flag}\n({ox:.1f}, {oy:.1f})",
                    (ox, oy), textcoords='offset points', xytext=(8, -12),
                    fontsize=9, color=color)

    # 范围
    m = args.margin
    ax.set_xlim(R2A_REGION['x_min'] - m, R2A_REGION['x_max'] + m)
    ax.set_ylim(R2A_REGION['y_min'] - m, R2A_REGION['y_max'] + m)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m, world)')
    ax.set_ylabel('Y (m, world)')
    ax.set_title(f"Case C BEV  scene={ego['scene']}  t*={ego['timestamp']}  "
                 f"R={ ((observers[0]['x']-ex)**2 + (observers[0]['y']-ey)**2)**0.5 :.1f}m")
    ax.legend(loc='upper left', fontsize=9)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'✓ 写入 {out}')


if __name__ == '__main__':
    main()
