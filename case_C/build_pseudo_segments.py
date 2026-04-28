#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case C - Step 3: 把 4 虚拟观察车 + 29 帧 ego 轨迹合成 4 个伪 segments。

每帧 observer pose:
    位置:  observer_world(t) = ego_world(t) + constant_world_offset
    朝向:  observer_yaw(t)   = ego_yaw(t)         (与 ego 平行行驶)
           observer_roll(t)  = ego_roll(t)
           observer_pitch(t) = ego_pitch(t)

→ 4 辆车以 ego 为圆心, 与 ego 一起平移、一起旋转, 始终保持相对静止。
→ Observer 不"正视" ego, 而是与 ego 同向行驶, 用原车端 7 相机内外参。

每个 observer → 一个 segment, 含 29 帧 timestamps/label_files (取自 ego 轨迹) 和
一个 'virtual_poses' 列表 (每帧一个 dict)。

输入:
  case_C/output/ego_trajectory.json    29 帧 ego 世界 pose
  case_C/output/virtual_observers.json 4 个 observer 在 t* 时的位置 (隐含 offset)
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
    ap = argparse.ArgumentParser(description="Case C: 构造 4 个跟随观察车的 segments")
    ap.add_argument('--ego-trajectory', default=str(here / 'output' / 'ego_trajectory.json'))
    ap.add_argument('--observers', default=str(here / 'output' / 'virtual_observers.json'))
    ap.add_argument('--output',
                    default=str(project_root / 'intersection_filter' / 'output' / 'filtered_segments_caseC.json'))
    args = ap.parse_args()

    with open(args.ego_trajectory) as f:
        traj = json.load(f)
    with open(args.observers) as f:
        observers_data = json.load(f)

    frames = traj['frames']
    if not frames:
        raise SystemExit("❌ ego_trajectory.json 帧列表为空")

    scene = traj['scene']
    ego_ref = observers_data['ego_reference']  # ego 在 t* 的位置, 用于反推 offset
    observers = observers_data['observers']

    # 反推每个 observer 的 world 系恒定 offset
    for ob in observers:
        ob['offset_x'] = ob['x'] - ego_ref['x']
        ob['offset_y'] = ob['y'] - ego_ref['y']
        ob['offset_z'] = ob['z'] - ego_ref['z']

    timestamps = [fr['timestamp'] for fr in frames]
    label_files = [fr['label_file'] for fr in frames]

    segments = []
    for ob in observers:
        # 逐帧观察车 pose:
        #   位置 = ego(t).xyz + 世界系恒定 offset
        #   朝向 = ego(t).{yaw,roll,pitch}  → 与 ego 同向行驶, 不指向 ego
        virtual_poses = []
        for fr in frames:
            virtual_poses.append({
                'x': fr['x'] + ob['offset_x'],
                'y': fr['y'] + ob['offset_y'],
                'z': fr['z'] + ob['offset_z'],
                'yaw': fr.get('yaw', 0.0),
                'roll': fr.get('roll', 0.0),
                'pitch': fr.get('pitch', 0.0),
                'vehicle_height': ob.get('vehicle_height', 1.6),
            })

        seg = {
            'scene': scene,
            'vehicle_id': f"virt{ob['name']}",
            'segment_index': 0,
            'n_frames': len(frames),
            'ts_start': timestamps[0],
            'ts_end': timestamps[-1],
            'timestamps': timestamps,
            'label_files': label_files,
            'virtual_poses': virtual_poses,
            'case_C': {
                'observer_name': ob['name'],
                'ego_reference_id': traj['ego_vehicle_id'],
                't_star': traj['t_star'],
                'offset_world': [ob['offset_x'], ob['offset_y'], ob['offset_z']],
                'mode': 'follow_ego_parallel_orientation',
            },
        }
        segments.append(seg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(segments, f, indent=2)

    print(f"✓ 写入 {out_path}  共 {len(segments)} 段, 每段 {len(frames)} 帧 "
          f"(mode=follow_ego_parallel_orientation)")
    for s in segments:
        vp0, vpN = s['virtual_poses'][0], s['virtual_poses'][-1]
        off = s['case_C']['offset_world']
        print(f"  {s['vehicle_id']}: offset=({off[0]:+.2f}, {off[1]:+.2f}, {off[2]:+.2f})  "
              f"pose[0]=({vp0['x']:.2f},{vp0['y']:.2f}) yaw={vp0['yaw']:+.3f}  "
              f"pose[-1]=({vpN['x']:.2f},{vpN['y']:.2f}) yaw={vpN['yaw']:+.3f}")


if __name__ == '__main__':
    main()
