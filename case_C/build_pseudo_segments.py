#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case C - Step 3: 把 4 虚拟观察车 pose 包装成 segment_pipeline 能消费的伪 segments。

每个 observer → 一个 segment，timestamps 与 label_files 都是同一个 t* 重复 29 次（静止观察）。
关键字段 'virtual_pose' 让 segment_pipeline 走虚拟观察车分支，跳过 ego id 查找。

输入:
  case_C/output/ego_pose_t_star.json
  case_C/output/virtual_observers.json
输出:
  intersection_filter/output/filtered_segments_caseC.json
"""

import argparse
import json
from pathlib import Path

SEGMENT_LENGTH = 29


def main():
    here = Path(__file__).resolve().parent
    project_root = here.parent
    ap = argparse.ArgumentParser(description="Case C: 构造 4 个 pseudo segments")
    ap.add_argument('--ego-pose', default=str(here / 'output' / 'ego_pose_t_star.json'))
    ap.add_argument('--observers', default=str(here / 'output' / 'virtual_observers.json'))
    ap.add_argument('--output',
                    default=str(project_root / 'intersection_filter' / 'output' / 'filtered_segments_caseC.json'))
    ap.add_argument('--n-frames', type=int, default=SEGMENT_LENGTH,
                    help='每段帧数，默认 29')
    args = ap.parse_args()

    with open(args.ego_pose) as f:
        ego = json.load(f)
    with open(args.observers) as f:
        observers = json.load(f)['observers']

    ts = int(ego['timestamp'])
    label_file = ego['label_file']
    scene = ego['scene']

    timestamps = [ts] * args.n_frames
    label_files = [label_file] * args.n_frames

    segments = []
    for obs in observers:
        seg = {
            'scene': scene,
            'vehicle_id': f"virt{obs['name']}",          # 字符串 id, 走虚拟分支
            'segment_index': 0,
            'n_frames': args.n_frames,
            'ts_start': ts,
            'ts_end': ts,
            'timestamps': timestamps,
            'label_files': label_files,
            'virtual_pose': {
                'x': obs['x'],
                'y': obs['y'],
                'z': obs['z'],
                'yaw': obs['yaw'],
                'roll': obs.get('roll', 0.0),
                'pitch': obs.get('pitch', 0.0),
                'vehicle_height': obs.get('vehicle_height', 1.6),
            },
            'case_C': {
                'observer_name': obs['name'],
                'ego_reference_id': ego['ego_vehicle_id'],
            },
        }
        segments.append(seg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(segments, f, indent=2)

    print(f"✓ 写入 {out_path}  共 {len(segments)} 段，每段 {args.n_frames} 帧")
    for s in segments:
        vp = s['virtual_pose']
        print(f"  {s['vehicle_id']}: x={vp['x']:.2f} y={vp['y']:.2f} yaw={vp['yaw']:+.3f}  "
              f"label={Path(s['label_files'][0]).name}")


if __name__ == '__main__':
    main()
