#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 tracks_in_region.json 生成 BEV 轨迹动画

Usage:
    python visualize_tracks_video.py --tracks output/tracks_in_region.json
    python visualize_tracks_video.py --tracks output/tracks_in_region.json --region output/intersection_region.json
    python visualize_tracks_video.py --tracks output/tracks_in_region.json --fps 10
"""

import os
import sys
import json
import io
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_tracks(tracks_path):
    """加载 tracks_in_region.json"""
    with open(tracks_path, 'r') as f:
        raw = json.load(f)

    # 结构: {scene: {vid: [{ts, x, y}, ...]}}
    tracks = {}
    for scene, vids in raw.items():
        tracks[scene] = {}
        for vid, points in vids.items():
            tracks[scene][vid] = [(p["ts"], p["x"], p["y"]) for p in points]
    return tracks


def load_region(region_path):
    """加载 intersection_region.json，返回 region dict 或 None"""
    if region_path and os.path.exists(region_path):
        with open(region_path, 'r') as f:
            data = json.load(f)
        return data.get("region", None)
    return None


def compute_bounds(tracks, region=None, margin=10):
    """计算所有轨迹的坐标范围"""
    all_x, all_y = [], []
    for scene, vids in tracks.items():
        for vid, pts in vids.items():
            all_x.extend(p[1] for p in pts)
            all_y.extend(p[2] for p in pts)

    if region:
        all_x.extend([region["x_min"], region["x_max"]])
        all_y.extend([region["y_min"], region["y_max"]])

    return (min(all_x) - margin, max(all_x) + margin,
            min(all_y) - margin, max(all_y) + margin)


def fig_to_pil(fig):
    """Convert matplotlib figure to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def create_tracks_video(tracks, region=None, output_path=None, fps=10):
    """
    生成 BEV 轨迹动画 GIF

    每帧显示到当前时间戳为止的所有轨迹（渐进绘制），
    当前帧位置用大圆点标出。
    """
    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUT_DIR / "tracks_animation.gif")

    # 收集所有时间戳并排序
    all_ts = set()
    for scene, vids in tracks.items():
        for vid, pts in vids.items():
            for ts, x, y in pts:
                all_ts.add(ts)
    all_ts = sorted(all_ts)

    if not all_ts:
        print("无轨迹数据")
        return None

    print(f"总时间戳数: {len(all_ts)}")
    print(f"时间范围: {all_ts[0]} ~ {all_ts[-1]} ({(all_ts[-1] - all_ts[0]) / 1000:.1f}s)")

    # 为每条轨迹分配颜色
    cmap = plt.cm.get_cmap('tab20')
    track_list = []  # (scene, vid, pts, color)
    idx = 0
    for scene, vids in tracks.items():
        for vid, pts in vids.items():
            if pts:
                track_list.append((scene, vid, pts, cmap(idx % 20)))
                idx += 1

    print(f"轨迹数: {len(track_list)}")

    # 坐标范围
    x_min, x_max, y_min, y_max = compute_bounds(tracks, region)

    # 生成每帧图像
    frames = []
    total = len(all_ts)

    for frame_idx, current_ts in enumerate(all_ts):
        if frame_idx % 10 == 0:
            print(f"  渲染帧 {frame_idx+1}/{total}...")

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 画矩形区域
        if region:
            rx = region["x_min"]
            ry = region["y_min"]
            rw = region["x_max"] - region["x_min"]
            rh = region["y_max"] - region["y_min"]
            rect = patches.Rectangle(
                (rx, ry), rw, rh, linewidth=2, edgecolor='red',
                facecolor='red', alpha=0.1)
            ax.add_patch(rect)

        # 画每条轨迹（到当前时间戳为止）
        for scene, vid, pts, color in track_list:
            visible = [(ts, x, y) for ts, x, y in pts if ts <= current_ts]
            if not visible:
                continue

            xs = [p[1] for p in visible]
            ys = [p[2] for p in visible]

            # 画轨迹线
            ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.7)
            # 起点
            ax.scatter(xs[0], ys[0], c=[color], s=40, marker='o',
                       edgecolors='black', linewidths=0.5, zorder=5)
            # 当前位置（大圆点）
            ax.scatter(xs[-1], ys[-1], c=[color], s=120, marker='o',
                       edgecolors='black', linewidths=1.5, zorder=10)
            # 标签
            ax.annotate(f"S{scene}_V{vid}", (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color=color, fontweight='bold')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        elapsed_ms = current_ts - all_ts[0]
        ax.set_title(f'BEV Tracks Animation  |  t={elapsed_ms/1000:.2f}s  '
                     f'({frame_idx+1}/{total})')
        ax.grid(True, alpha=0.3)

        img = fig_to_pil(fig)
        frames.append(img)
        plt.close(fig)

    # 保存为 GIF
    print(f"\n保存动画: {output_path}  ({len(frames)} frames, fps={fps})")
    duration_ms = 1000 // fps
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0
    )

    # 同时保存最终帧为静态PNG
    png_path = output_path.replace('.gif', '_final.png')
    frames[-1].save(png_path)
    print(f"最终帧静态图: {png_path}")

    print(f"动画已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="从轨迹数据生成BEV动画")
    parser.add_argument("--tracks", type=str,
                        default=str(OUTPUT_DIR / "tracks_in_region.json"),
                        help="tracks_in_region.json 路径")
    parser.add_argument("--region", type=str,
                        default=str(OUTPUT_DIR / "intersection_region.json"),
                        help="intersection_region.json 路径（可选，用于画矩形区域）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径（默认 output/tracks_animation.gif）")
    parser.add_argument("--fps", type=int, default=10,
                        help="帧率（默认10）")
    args = parser.parse_args()

    tracks = load_tracks(args.tracks)
    region = load_region(args.region)
    create_tracks_video(tracks, region=region, output_path=args.output, fps=args.fps)


if __name__ == "__main__":
    main()
