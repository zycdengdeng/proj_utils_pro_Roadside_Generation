#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case C - Step 1: 选 t* 并提取 ego 在 008 场景的"29 帧轨迹"。

后续 Case C 流水线让 4 辆虚拟观察车跟随真 ego 一起平移（offset 在世界系中恒定 →
观察车与 ego 保持相对静止），所以这一步不只取单帧 t*，还要把 t* 周围连续 29 帧
的 ego pose 一并写出，供 build_pseudo_segments 用作每帧 observer 位置的基准。

ego 车辆 ID 来源 (优先级)：
  1) --ego-id 命令行参数
  2) /mnt/car_road_data_fix/support_info/carid.json 中场景对应的 nearest_carid

t* 选择 (优先级)：
  1) --fit-radius R   扫描所有时间戳，挑 ego 离 R2A 红框四壁均 >= R 米的 t*
  2) --timestamp <ms>
  3) 默认：标注目录所有时间戳的中间一帧

输出:
  case_C/output/ego_pose_t_star.json   单帧 t* 的 ego 世界 pose (向后兼容)
  case_C/output/ego_trajectory.json    29 帧 ego 世界 pose 列表 (供后续步骤用)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common_utils import get_scene_paths, load_carid_mapping


def find_label_file(label_dir, timestamp_ms):
    """精确匹配 (label_dir/{ts}.json)，否则取最接近的"""
    exact = Path(label_dir) / f"{timestamp_ms}.json"
    if exact.exists():
        return exact, 0
    best, best_diff = None, 10**18
    for f in sorted(glob.glob(str(Path(label_dir) / '*.json'))):
        try:
            ts = int(Path(f).stem)
        except ValueError:
            continue
        d = abs(ts - timestamp_ms)
        if d < best_diff:
            best_diff, best = d, Path(f)
    return best, best_diff


def list_label_timestamps(label_dir):
    out = []
    for f in sorted(glob.glob(str(Path(label_dir) / '*.json'))):
        try:
            out.append(int(Path(f).stem))
        except ValueError:
            continue
    out.sort()
    return out


# R2A 红框（与 build_virtual_observers.py 保持一致）
R2A_REGION = {'x_min': -95.1, 'x_max': -30.7, 'y_min': -30.6, 'y_max': 9.8}


def fit_radius_score(x, y, radius, region=R2A_REGION):
    """如果 ego 周围 ±radius 完全落在区域内，返回离最近一面墙的距离 (越大越好)；否则返回负值"""
    margins = (
        x - region['x_min'] - radius,
        region['x_max'] - x - radius,
        y - region['y_min'] - radius,
        region['y_max'] - y - radius,
    )
    return min(margins)


def find_ego_at_timestamp(label_dir, ts, ego_id):
    f, _ = find_label_file(label_dir, ts)
    if f is None:
        return None
    with open(f) as fh:
        anno = json.load(fh)
    obj = next((o for o in anno.get('object', []) if o.get('id') == ego_id), None)
    if obj is None:
        return None
    return f, obj


def main():
    ap = argparse.ArgumentParser(description="Case C: 找 ego 在 t* 的世界 pose")
    ap.add_argument('--scene', default='008', help='场景前缀，默认 008')
    ap.add_argument('--ego-id', type=int, default=None,
                    help='ego 车辆 ID (默认从 carid.json 读取)')
    ap.add_argument('--timestamp', type=int, default=None,
                    help='t* 毫秒时间戳 (默认取该场景标注的中间一帧)')
    ap.add_argument('--fit-radius', type=float, default=None,
                    help='自动扫描所有时间戳, 选 ego 离 R2A 红框四壁都 >= 此半径的 t*; '
                         '若指定则覆盖 --timestamp')
    ap.add_argument('--output', default=str(Path(__file__).resolve().parent / 'output' / 'ego_pose_t_star.json'),
                    help='单帧 t* 输出 JSON 路径')
    ap.add_argument('--num-frames', type=int, default=29,
                    help='提取的 ego 轨迹帧数, 默认 29 (segment_pipeline 要求)')
    ap.add_argument('--trajectory-output', default=None,
                    help='29 帧 ego 轨迹输出 JSON 路径, 默认 case_C/output/ego_trajectory.json')
    args = ap.parse_args()

    scene_paths = get_scene_paths(args.scene)
    if not scene_paths:
        sys.exit(f"❌ 找不到场景 {args.scene}")
    label_dir = scene_paths['roadside_labels']

    # 1) 决定 ego_id
    ego_id = args.ego_id
    if ego_id is None:
        carid_map = load_carid_mapping()
        ego_id = carid_map.get(args.scene)
        if ego_id is None:
            sys.exit(f"❌ carid.json 中没有 {args.scene} 的 nearest_carid，请用 --ego-id 指定")
    print(f"ego_vehicle_id = {ego_id}")

    # 2) 决定 t*
    all_ts = list_label_timestamps(label_dir)
    if not all_ts:
        sys.exit(f"❌ 标注目录为空: {label_dir}")

    if args.fit_radius is not None:
        # 扫描所有时间戳, 找 ego 离红框四壁都至少 fit_radius 米的 t* (取 margin 最大的)
        print(f"扫描 {len(all_ts)} 帧, 寻找 ego 离 R2A 红框四壁 >= {args.fit_radius}m 的 t* ...")
        best = None  # (margin, ts, obj, file)
        for ts in all_ts:
            res = find_ego_at_timestamp(label_dir, ts, ego_id)
            if res is None:
                continue
            f, obj = res
            margin = fit_radius_score(obj['x'], obj['y'], args.fit_radius)
            if margin >= 0 and (best is None or margin > best[0]):
                best = (margin, ts, obj, f)
        if best is None:
            sys.exit(f"❌ 没有任何一帧能让 R={args.fit_radius}m 的观察车全在红框内, "
                     f"请减小 --fit-radius 或换场景")
        ts_star = best[1]
        print(f"  → 命中 t*={ts_star}  ego=({best[2]['x']:.2f}, {best[2]['y']:.2f})  "
              f"min margin to wall = {best[0]:.2f}m")
    elif args.timestamp is None:
        ts_star = all_ts[len(all_ts) // 2]
        print(f"未指定 --timestamp，自动选中间帧 t* = {ts_star} (共 {len(all_ts)} 帧)")
    else:
        ts_star = args.timestamp

    # 3) 找标注文件并提取 ego pose
    label_file, diff = find_label_file(label_dir, ts_star)
    if label_file is None:
        sys.exit(f"❌ 标注目录中没找到接近 {ts_star} 的文件")
    print(f"label_file = {label_file}  (diff={diff} ms)")

    with open(label_file) as f:
        anno = json.load(f)

    ego_obj = next((o for o in anno.get('object', []) if o.get('id') == ego_id), None)
    if ego_obj is None:
        ids = sorted({o.get('id') for o in anno.get('object', [])})
        sys.exit(f"❌ 在 {label_file.name} 里找不到 id={ego_id} 的 ego；该帧出现的 id: {ids[:30]}{'...' if len(ids)>30 else ''}")

    # 用文件名时间戳当作 t*（精确，可被 build_pseudo_segments 复用）
    ts_label = int(label_file.stem)

    out = {
        'scene': args.scene,
        'ego_vehicle_id': ego_id,
        'timestamp': ts_label,
        'label_file': str(label_file),
        'x': float(ego_obj['x']),
        'y': float(ego_obj['y']),
        'z': float(ego_obj['z']),
        'yaw': float(ego_obj.get('yaw', 0.0)),
        'roll': float(ego_obj.get('roll', 0.0)),
        'pitch': float(ego_obj.get('pitch', 0.0)),
        'length': float(ego_obj.get('length', 0.0)),
        'width': float(ego_obj.get('width', 0.0)),
        'height': float(ego_obj.get('height', 0.0)),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ 写入 {out_path}")
    print(json.dumps(out, indent=2))

    # ---- 4) 提取 29 帧 ego 轨迹 (以 t* 为中心) ----
    n = args.num_frames
    try:
        idx = all_ts.index(ts_label)
    except ValueError:
        # ts_label 来自 label_file.stem, 必在 all_ts 中; 兜底
        idx = min(range(len(all_ts)), key=lambda i: abs(all_ts[i] - ts_label))

    half = n // 2
    start = max(0, idx - half)
    end = start + n
    if end > len(all_ts):
        end = len(all_ts)
        start = max(0, end - n)
    window = all_ts[start:end]

    if len(window) < n:
        print(f"⚠️  场景仅有 {len(all_ts)} 帧, 取到 {len(window)} 帧 (要求 {n})")

    frames = []
    missing = []
    for ts in window:
        f, _ = find_label_file(label_dir, ts)
        if f is None:
            missing.append(ts)
            continue
        with open(f) as fh:
            anno = json.load(fh)
        obj = next((o for o in anno.get('object', []) if o.get('id') == ego_id), None)
        if obj is None:
            missing.append(ts)
            continue
        frames.append({
            'timestamp': int(f.stem),
            'label_file': str(f),
            'x': float(obj['x']),
            'y': float(obj['y']),
            'z': float(obj['z']),
            'yaw': float(obj.get('yaw', 0.0)),
            'roll': float(obj.get('roll', 0.0)),
            'pitch': float(obj.get('pitch', 0.0)),
        })

    if missing:
        print(f"⚠️  {len(missing)} 帧没有 ego={ego_id} 的标注: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if len(frames) < n:
        print(f"⚠️  只提取到 {len(frames)} 帧 ego pose (要求 {n}); 流水线仍按这些有效帧跑")

    traj = {
        'scene': args.scene,
        'ego_vehicle_id': ego_id,
        't_star': ts_label,
        't_star_index_in_window': window.index(ts_label) if ts_label in window else -1,
        'num_frames': len(frames),
        'frames': frames,
    }
    traj_path = Path(args.trajectory_output) if args.trajectory_output else (out_path.parent / 'ego_trajectory.json')
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with open(traj_path, 'w') as f:
        json.dump(traj, f, indent=2)
    print(f"\n✓ 写入 {traj_path}  共 {len(frames)} 帧 ego 轨迹")
    if frames:
        print(f"  起点: ({frames[0]['x']:.2f}, {frames[0]['y']:.2f})  ts={frames[0]['timestamp']}")
        print(f"  终点: ({frames[-1]['x']:.2f}, {frames[-1]['y']:.2f})  ts={frames[-1]['timestamp']}")
        dx = frames[-1]['x'] - frames[0]['x']
        dy = frames[-1]['y'] - frames[0]['y']
        print(f"  位移: dx={dx:+.2f}m, dy={dy:+.2f}m, |d|={(dx*dx+dy*dy)**0.5:.2f}m")


if __name__ == '__main__':
    main()
