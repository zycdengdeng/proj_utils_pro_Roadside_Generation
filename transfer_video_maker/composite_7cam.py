#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 transfer_video_maker/output/{BlurProjection,DepthSparse,HDMapBbox}/control_input_*/
下的 7 个相机 mp4 按 3x3 网格合成成一个对照视频, 方便人眼检查每辆车 (segment) 的
control 输入质量。

布局:
    +-------+-------+-------+
    |  FL   |  FW   |  FR   |   row 1 (前左 / 前广角 / 前右)
    +-------+-------+-------+
    |  RL   |  RN   |  RR   |   row 2 (后左 / 后窄角 / 后右)
    +-------+-------+-------+
    |       |  FN   |       |   row 3 (前窄角, 居中)
    +-------+-------+-------+

输出: transfer_video_maker/output/<ProjectType>/composite_7cam/<seg>.mp4
共 4 段 (virtN/E/S/W) × 3 投影 = 12 个 mp4。

Usage:
    python transfer_video_maker/composite_7cam.py
    # 仅 case C 4 段:
    python transfer_video_maker/composite_7cam.py --segments 008_idvirtN_seg01 008_idvirtE_seg01 ...
    # 仅 blur:
    python transfer_video_maker/composite_7cam.py --project-types BlurProjection
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

CAM_LONG = {
    'FN': 'ftheta_camera_front_tele_30fov',
    'FW': 'ftheta_camera_front_wide_120fov',
    'FL': 'ftheta_camera_cross_left_120fov',
    'FR': 'ftheta_camera_cross_right_120fov',
    'RL': 'ftheta_camera_rear_left_70fov',
    'RR': 'ftheta_camera_rear_right_70fov',
    'RN': 'ftheta_camera_rear_tele_30fov',
}

# 3 行 × 3 列, None 表示空白; 用户指定:
#   row 1: FL FW FR
#   row 2: RL RN RR
#   row 3: __ FN __
LAYOUT = [
    ['FL', 'FW', 'FR'],
    ['RL', 'RN', 'RR'],
    [None, 'FN', None],
]

DEFAULT_PROJECT_TYPES = ['BlurProjection', 'DepthSparse', 'HDMapBbox']


def find_control_input_dir(ptype_dir):
    """每个 ProjectType 下找唯一的 control_input_* 子目录。"""
    cands = [c for c in ptype_dir.glob('control_input_*') if c.is_dir()]
    if not cands:
        return None
    if len(cands) > 1:
        print(f'  warn: multiple control_input_* dirs in {ptype_dir}, using {cands[0].name}',
              file=sys.stderr)
    return cands[0]


def list_segments(control_dir):
    """7 个相机都存在 mp4 的 seg 名 (取交集)。"""
    common = None
    for short, long_name in CAM_LONG.items():
        cam_dir = control_dir / long_name
        if not cam_dir.exists():
            print(f'  warn: missing camera dir: {cam_dir}', file=sys.stderr)
            continue
        names = {f.stem for f in cam_dir.glob('*.mp4')}
        common = names if common is None else common & names
    return sorted(common or [])


def open_capture(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'cannot open video: {path}')
    return cap


def composite_segment(control_dir, seg_name, out_path, cell_w, cell_h, fps,
                      label_cells=True):
    """合成单个 seg 的 7-cam 视频, 写到 out_path。返回 (n_frames_written, ok)。"""
    caps = {}
    for short, long_name in CAM_LONG.items():
        f = control_dir / long_name / f'{seg_name}.mp4'
        if not f.exists():
            print(f'    skip {seg_name}: missing {short} ({f})', file=sys.stderr)
            for c in caps.values():
                c.release()
            return 0, False
        caps[short] = open_capture(f)

    rows, cols = len(LAYOUT), len(LAYOUT[0])
    grid_w, grid_h = cell_w * cols, cell_h * rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (grid_w, grid_h))
    if not writer.isOpened():
        for c in caps.values():
            c.release()
        raise RuntimeError(f'cannot open writer: {out_path}')

    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    n_frames = 0
    while True:
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        all_done = True
        for r, row in enumerate(LAYOUT):
            for c, short in enumerate(row):
                if short is None:
                    cell = blank
                else:
                    ok, fr = caps[short].read()
                    if not ok:
                        cell = blank
                    else:
                        all_done = False
                        if fr.shape[1] != cell_w or fr.shape[0] != cell_h:
                            fr = cv2.resize(fr, (cell_w, cell_h),
                                            interpolation=cv2.INTER_AREA)
                        cell = fr.copy()
                        if label_cells:
                            cv2.putText(cell, short, (10, 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                        (0, 255, 255), 2, cv2.LINE_AA)
                canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = cell
        if all_done:
            break
        writer.write(canvas)
        n_frames += 1

    for c in caps.values():
        c.release()
    writer.release()
    return n_frames, True


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-base', default=str(here / 'output'),
                    help='transfer_video_maker/output/ 根目录, 默认相对脚本同级')
    ap.add_argument('--project-types', nargs='+', default=DEFAULT_PROJECT_TYPES,
                    help='要合成的 ProjectType 子目录 (默认 3 个)')
    ap.add_argument('--segments', nargs='*', default=None,
                    help='指定 seg 名; 不传则取所有 7 相机都齐的 seg')
    ap.add_argument('--cell-width', type=int, default=640)
    ap.add_argument('--cell-height', type=int, default=360)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--subdir', default='composite_7cam',
                    help='输出子目录名, 写在每个 ProjectType 下')
    ap.add_argument('--no-label', action='store_true',
                    help='不在每个 cell 左上角画 FL/FW/... 标签')
    args = ap.parse_args()

    base = Path(args.output_base)
    if not base.exists():
        sys.exit(f'❌ output base 不存在: {base}')

    total = 0
    fail = 0
    written_paths = []
    for ptype in args.project_types:
        pdir = base / ptype
        if not pdir.exists():
            print(f'⚠️  跳过 {ptype}: 目录不存在 ({pdir})')
            continue
        cdir = find_control_input_dir(pdir)
        if cdir is None:
            print(f'⚠️  跳过 {ptype}: 没找到 control_input_* 子目录')
            continue

        segs_avail = list_segments(cdir)
        if args.segments:
            wanted = set(args.segments)
            missing = wanted - set(segs_avail)
            if missing:
                print(f'⚠️  {ptype}: 这些 seg 在 control_input 里缺相机或不存在: {sorted(missing)}')
            segs = [s for s in args.segments if s in segs_avail]
        else:
            segs = segs_avail

        out_root = pdir / args.subdir
        print(f'\n=== {ptype} ({len(segs)} segs) ===')
        print(f'  src: {cdir}')
        print(f'  out: {out_root}')
        for seg in segs:
            out_mp4 = out_root / f'{seg}.mp4'
            n_frames, ok = composite_segment(
                cdir, seg, out_mp4,
                args.cell_width, args.cell_height, args.fps,
                label_cells=not args.no_label,
            )
            total += 1
            if ok:
                print(f'  ✓ {seg}: {n_frames} 帧 → {out_mp4}')
                written_paths.append(out_mp4)
            else:
                fail += 1
                print(f'  ✗ {seg}: 失败')

    print(f'\nDone: {total - fail}/{total} composites generated.')
    if written_paths:
        print('outputs:')
        for p in written_paths:
            print(f'  {p}')


if __name__ == '__main__':
    main()
