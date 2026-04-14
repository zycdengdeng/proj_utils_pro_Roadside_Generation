#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车端 LiDAR 点云 → 车端相机图像 投影叠加

输入:
  - 车端 LiDAR PCD: {scene_root}/car/pcds/main/{ts}.pcd        (已在车载 lidar 系)
  - 车端 GT 图像:    {scene_root}/car/images/{cam_name}/*.jpg
  - 车端标定:        support_info/NoEER705_v3/camera/camera_{XX}_{intrinsics,extrinsics}.yaml

输出:
  - {output_root}/{scene_id}/{cam_name}/{ts}.jpg
    GT 图像 (去畸变) 叠加上 JET 着色的 LiDAR 投影点

变换链:
  p_lidar  ──(R_lidar2cam, t_lidar2cam)──>  p_cam  ──(K, D / fisheye)──>  (u, v)

颜色:
  按 z_cam (相机坐标系下的深度) 用 JET colormap 着色,
  默认深度范围 [near=1m, far=50m], 可命令行调整或用 --auto-range 自适应

用法示例:
  # 投影场景 004 的所有 PCD 到所有 7 个相机, 输出到 ./output
  python project_vehicle_lidar.py --scene 004 --output ./output

  # 只投到前视广角 (FW) 和前视窄角 (FN), 帧范围限制为前 30 帧, 自动深度范围
  python project_vehicle_lidar.py --scene 004 --cameras FW FN --max-frames 30 --auto-range

  # 调点大小和透明度
  python project_vehicle_lidar.py --scene 004 --point-size 4 --alpha 0.85
"""

import argparse
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

# 复用项目根目录的 common_utils (场景路径查找等)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common_utils  # noqa: E402

warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================
# 配置
# ============================================================

# 车端相机配置 (与其他模块保持一致)
VEHICLE_CAMERAS = {
    1: {"name": "FN", "desc": "前视窄角30°",  "resolution": (3840, 2160)},
    2: {"name": "FW", "desc": "前视广角120°", "resolution": (3840, 2160)},
    3: {"name": "FL", "desc": "左前视120°",   "resolution": (3840, 2160)},
    4: {"name": "FR", "desc": "右前视120°",   "resolution": (3840, 2160)},
    5: {"name": "RL", "desc": "左后视60°",    "resolution": (1920, 1080)},
    6: {"name": "RR", "desc": "右后视60°",    "resolution": (1920, 1080)},
    7: {"name": "RN", "desc": "后视60°",      "resolution": (1920, 1080)},
}
NAME2ID = {v["name"]: k for k, v in VEHICLE_CAMERAS.items()}
FISHEYE_CAM_IDS = {2, 3, 4}  # FW/FL/FR 是鱼眼

# GT 图像时间戳匹配容差 (微秒, 命令行可覆盖)
DEFAULT_GT_MATCH_TOL_US = 200_000  # 200ms


# ============================================================
# 标定加载
# ============================================================

def quaternion_to_rotation_matrix(q):
    """[qx, qy, qz, qw] → 3x3"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def load_camera_calib(calib_dir, cam_id):
    """
    加载车端单个相机的内外参

    Returns:
        dict {K, D, R_cam2lidar, t_cam2lidar, resolution, is_fisheye, name}
    """
    calib_dir = Path(calib_dir)
    intr_path = calib_dir / f"camera_{cam_id:02d}_intrinsics.yaml"
    extr_path = calib_dir / f"camera_{cam_id:02d}_extrinsics.yaml"

    with open(intr_path, 'r') as f:
        intr = yaml.safe_load(f)
    K = np.array(intr['K'], dtype=np.float64).reshape(3, 3)
    D = np.array(intr['D'], dtype=np.float64).flatten()

    with open(extr_path, 'r') as f:
        extr = yaml.safe_load(f)
    tr = extr['transform']
    q = [tr['rotation']['x'], tr['rotation']['y'], tr['rotation']['z'], tr['rotation']['w']]
    t = np.array([tr['translation']['x'], tr['translation']['y'], tr['translation']['z']],
                 dtype=np.float64)
    R_cam2lidar = quaternion_to_rotation_matrix(q)

    return {
        'K': K,
        'D': D,
        'R_cam2lidar': R_cam2lidar,
        't_cam2lidar': t.reshape(3, 1),
        'resolution': VEHICLE_CAMERAS[cam_id]['resolution'],
        'is_fisheye': cam_id in FISHEYE_CAM_IDS and np.max(np.abs(D)) > 1.0,
        'name': VEHICLE_CAMERAS[cam_id]['name'],
        'cam_id': cam_id,
    }


# ============================================================
# 文件匹配
# ============================================================

PCD_TS_RE = re.compile(r'^(\d+)\.(\d+)\.pcd$')
GT_TS_RE = re.compile(r'_(\d+)\.(\d+)\.jpg$')


def list_vehicle_pcds(scene_root):
    """列出场景下的车端 PCD, 按时间戳排序"""
    pcd_dir = Path(scene_root) / 'car' / 'pcds' / 'main'
    if not pcd_dir.exists():
        raise FileNotFoundError(f"车端 PCD 目录不存在: {pcd_dir}")

    files = []
    for f in pcd_dir.glob('*.pcd'):
        m = PCD_TS_RE.match(f.name)
        if m:
            ts_us = int(m.group(1)) * 1_000_000 + int(m.group(2))
            files.append((ts_us, f))
    files.sort()
    return files  # [(ts_us, Path), ...]


def list_gt_timestamps(gt_cam_dir):
    """列出某相机 GT 目录下所有文件的时间戳 (按 ts_us 排序). 返回 [(ts_us, Path), ...]"""
    gt_cam_dir = Path(gt_cam_dir)
    if not gt_cam_dir.exists():
        return []
    items = []
    for f in gt_cam_dir.glob('*.jpg'):
        m = GT_TS_RE.search(f.name)
        if not m:
            continue
        ts_us = int(m.group(1)) * 1_000_000 + int(m.group(2))
        items.append((ts_us, f))
    items.sort()
    return items


def find_gt_image_from_cache(gt_items, target_ts_us, tol_us):
    """
    从已缓存的 (ts_us, Path) 列表里找最接近 target_ts_us 的 GT 图像

    tol_us = 0 或 None 时不做容差检查, 总返回最近的
    """
    if not gt_items:
        return None, None

    best_path = None
    best_diff = float('inf')
    for ts_us, path in gt_items:
        diff = abs(ts_us - target_ts_us)
        if diff < best_diff:
            best_diff = diff
            best_path = path
    if best_path is None:
        return None, None
    if tol_us and tol_us > 0 and best_diff > tol_us:
        return None, best_diff
    return best_path, best_diff


# ============================================================
# 投影核心
# ============================================================

def load_pcd_xyz(pcd_path):
    """读取 PCD, 返回 (N, 3) 的 xyz"""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    return pts


def undistort_image(img, calib):
    """对 GT 图像去畸变, 并返回 (undistorted_img, new_K)"""
    K = calib['K']
    D = calib['D']
    w, h = calib['resolution']

    if calib['is_fisheye']:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D[:4], (w, h), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D[:4], np.eye(3), new_K, (w, h), cv2.CV_16SC2
        )
        und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
        und = cv2.undistort(img, K, D, None, new_K)
    return und, new_K


def project_lidar_to_image(points_lidar, calib, new_K):
    """
    把 lidar 系下的点投到去畸变后的相机像素平面

    Returns:
        uv: (M, 2) int32 像素坐标
        z_cam: (M,) float32 相机坐标系下的深度 (用于上色)
    """
    R_cam2lidar = calib['R_cam2lidar']
    t_cam2lidar = calib['t_cam2lidar']

    # lidar → cam: p_cam = R_cam2lidar.T @ (p_lidar - t_cam2lidar)
    R_lidar2cam = R_cam2lidar.T
    t_lidar2cam = -R_cam2lidar.T @ t_cam2lidar  # (3,1)

    pts_cam = (R_lidar2cam @ points_lidar.T).T + t_lidar2cam.reshape(1, 3)

    # 过滤背后点
    front = pts_cam[:, 2] > 0.1
    if not np.any(front):
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    pts_cam = pts_cam[front]
    z_cam = pts_cam[:, 2].astype(np.float32)

    # 用 new_K 直接做针孔投影 (因为图像已去畸变, 不再需要 D)
    uv_h = (new_K @ pts_cam.T).T  # (M, 3)
    uv = (uv_h[:, :2] / uv_h[:, 2:3]).astype(np.int32)
    return uv, z_cam


def colorize_by_depth(z_cam, near, far, colormap=cv2.COLORMAP_JET):
    """
    用给定的 colormap 把深度映射成 BGR 颜色

    z_cam < near → 全部映到最近色
    z_cam > far  → 全部映到最远色
    """
    z_clip = np.clip(z_cam, near, far)
    norm = (z_clip - near) / (far - near + 1e-9)  # [0, 1]
    norm_u8 = (norm * 255).astype(np.uint8).reshape(-1, 1)
    bgr = cv2.applyColorMap(norm_u8, colormap).reshape(-1, 3)  # (M, 3) uint8
    return bgr


def overlay_points_on_image(img, uv, colors, point_size=2, alpha=1.0):
    """
    在 img 上绘制带颜色的投影点

    Args:
        img: HxWx3 BGR
        uv:  (M, 2) int32, 已经裁剪到图像范围内
        colors: (M, 3) uint8 BGR
        point_size: 点的边长 (像素)
        alpha: 1.0 = 完全覆盖, <1 = 与底图混合
    """
    h, w = img.shape[:2]
    if alpha >= 1.0:
        canvas = img
    else:
        canvas = img.copy()

    if point_size <= 1:
        # 单像素, 矢量化最快
        canvas[uv[:, 1], uv[:, 0]] = colors
    else:
        # 多像素, 用 cv2.circle 逐点画 (cv2 内部 C 实现, 速度可接受)
        # filled circle 半径 = point_size // 2
        r = max(1, point_size // 2)
        for (u, v), c in zip(uv, colors):
            cv2.circle(canvas, (int(u), int(v)), r,
                       (int(c[0]), int(c[1]), int(c[2])), thickness=-1,
                       lineType=cv2.LINE_AA)

    if alpha < 1.0:
        return cv2.addWeighted(canvas, alpha, img, 1.0 - alpha, 0)
    return canvas


# ============================================================
# 单帧 × 单相机 处理
# ============================================================

def process_one_camera(points_lidar, ts_us, calib, gt_items, out_dir,
                       near, far, point_size, alpha, auto_range,
                       jpg_quality, gt_tol_us):
    """处理单帧的单个相机, 写出叠加图. 返回 (cam_name, num_points, ok, ts_diff_us)"""
    cam_name = calib['name']

    # 1. 找 GT 图像
    gt_path, ts_diff = find_gt_image_from_cache(gt_items, ts_us, gt_tol_us)
    if gt_path is None:
        return cam_name, 0, False, ts_diff

    img = cv2.imread(str(gt_path))
    if img is None:
        return cam_name, 0, False, ts_diff

    # 2. 去畸变
    und_img, new_K = undistort_image(img, calib)
    h, w = und_img.shape[:2]

    # 3. 投影
    uv, z_cam = project_lidar_to_image(points_lidar, calib, new_K)
    if uv.shape[0] == 0:
        # 没有点落在前方, 仍然把去畸变 GT 写出去 (方便调试)
        out_path = out_dir / f"{ts_us // 1000}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), und_img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        return cam_name, 0, True, ts_diff

    # 4. 裁剪到图像范围
    in_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[in_img]
    z_cam = z_cam[in_img]
    if uv.shape[0] == 0:
        out_path = out_dir / f"{ts_us // 1000}.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), und_img, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        return cam_name, 0, True, ts_diff

    # 5. 排序: 先画远的, 后画近的, 这样近处的点不会被远处点遮盖
    order = np.argsort(-z_cam)  # 从远到近
    uv = uv[order]
    z_cam = z_cam[order]

    # 6. 确定上色范围
    if auto_range:
        cam_near = float(np.percentile(z_cam, 2))
        cam_far = float(np.percentile(z_cam, 98))
        cam_near = max(0.5, cam_near)
        cam_far = max(cam_near + 1.0, cam_far)
    else:
        cam_near, cam_far = near, far

    # 7. 着色 + 绘制
    colors = colorize_by_depth(z_cam, cam_near, cam_far)
    overlay = overlay_points_on_image(und_img, uv, colors,
                                      point_size=point_size, alpha=alpha)

    # 8. 写出
    out_path = out_dir / f"{ts_us // 1000}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
    return cam_name, int(uv.shape[0]), True, ts_diff


# ============================================================
# 主流程
# ============================================================

def run_scene(scene_id, args):
    paths = common_utils.get_scene_paths(scene_id)
    if not paths:
        print(f"[ERROR] 场景 {scene_id} 不存在")
        return

    scene_root = paths['root']
    gt_root = Path(paths['vehicle_images'])  # car/images/
    if not gt_root.exists():
        print(f"[ERROR] 车端 GT 图像目录不存在: {gt_root}")
        return

    # 加载相机标定
    calib_dir = Path(paths['vehicle_calib'])
    cam_ids = [NAME2ID[n] for n in args.cameras]
    calibs = {cid: load_camera_calib(calib_dir, cid) for cid in cam_ids}

    # 列出 PCD
    pcd_files = list_vehicle_pcds(scene_root)
    if len(pcd_files) == 0:
        print(f"[ERROR] 场景 {scene_id} 下没有 PCD 文件")
        return

    # 预先列出每个相机的 GT 图像 (避免每帧都 glob 一次)
    gt_cache = {}
    for cid in cam_ids:
        cam_name = VEHICLE_CAMERAS[cid]['name']
        items = list_gt_timestamps(gt_root / cam_name)
        gt_cache[cid] = items
        if not items:
            print(f"[WARN] 相机 {cam_name} 的 GT 目录为空或不存在: {gt_root / cam_name}")

    # 把容差从 ms 转成 us
    gt_tol_us = int(args.gt_tolerance_ms * 1000) if args.gt_tolerance_ms > 0 else 0

    # 如果要求对齐, 先截断 PCD 列表到 GT 覆盖区间
    align_info = ""
    if args.align_to_gt:
        all_gt_ts = [ts for cid in cam_ids for ts, _ in gt_cache[cid]]
        if all_gt_ts:
            gt_min = min(all_gt_ts)
            gt_max = max(all_gt_ts)
            pad = gt_tol_us if gt_tol_us > 0 else 0
            before = len(pcd_files)
            pcd_files = [(ts, p) for ts, p in pcd_files
                         if gt_min - pad <= ts <= gt_max + pad]
            align_info = f"  [align-to-gt] {before} → {len(pcd_files)} 帧 (截断到 GT 覆盖区间)"

    # 应用 start / max-frames
    if args.start_frame > 0:
        pcd_files = pcd_files[args.start_frame:]
    if args.max_frames > 0:
        pcd_files = pcd_files[:args.max_frames]

    output_root = Path(args.output) / scene_id
    output_root.mkdir(parents=True, exist_ok=True)

    # 打印概览, 包括时间范围
    pcd_ts_list = [ts for ts, _ in pcd_files]
    all_gt_ts = [ts for cid in cam_ids for ts, _ in gt_cache[cid]]
    print(f"\n[场景 {scene_id}] {paths['scene_name']}")
    print(f"  相机:        {args.cameras}")
    print(f"  PCD:         {len(pcd_files)} 帧 (筛选后)")
    if pcd_ts_list:
        print(f"               时间戳 {pcd_ts_list[0]/1e6:.3f} ~ {pcd_ts_list[-1]/1e6:.3f} s")
    if all_gt_ts:
        print(f"  GT 图像:     共 {len(all_gt_ts)} 张 (各相机合计)")
        print(f"               时间戳 {min(all_gt_ts)/1e6:.3f} ~ {max(all_gt_ts)/1e6:.3f} s")
    if align_info:
        print(align_info)
    print(f"  GT 容差:     " + (f"{args.gt_tolerance_ms:.0f} ms" if gt_tol_us > 0
                                else "无 (总是取最近图像)"))
    print(f"  深度范围:    " +
          (f"自适应 (per-frame 2%~98% 分位)" if args.auto_range
           else f"固定 [{args.near}, {args.far}] m"))
    print(f"  点大小:      {args.point_size} px,  alpha = {args.alpha}")
    print(f"  输出到:      {output_root}")
    print()

    t0 = time.time()
    total_pts = 0
    total_imgs = 0

    for idx, (ts_us, pcd_path) in enumerate(pcd_files, 1):
        # 读 PCD (一帧一次, 7 个相机共用)
        try:
            points_lidar = load_pcd_xyz(pcd_path)
        except Exception as e:
            print(f"  [{idx}/{len(pcd_files)}] {pcd_path.name}: 读 PCD 失败 ({e})")
            continue

        if points_lidar.shape[0] == 0:
            print(f"  [{idx}/{len(pcd_files)}] {pcd_path.name}: 空点云, 跳过")
            continue

        # 7 个相机并行处理
        per_cam_results = {}
        with ThreadPoolExecutor(max_workers=len(cam_ids)) as ex:
            futures = {}
            for cid in cam_ids:
                cam_name = VEHICLE_CAMERAS[cid]['name']
                out_dir = output_root / cam_name
                fut = ex.submit(
                    process_one_camera,
                    points_lidar, ts_us, calibs[cid], gt_cache[cid], out_dir,
                    args.near, args.far, args.point_size, args.alpha,
                    args.auto_range, args.jpg_quality, gt_tol_us,
                )
                futures[fut] = cam_name
            for fut in futures:
                cam_name, npts, ok, ts_diff = fut.result()
                per_cam_results[cam_name] = (npts, ok, ts_diff)

        # 进度打印
        cams_ok = [c for c, (_, ok, _) in per_cam_results.items() if ok]
        miss_parts = []
        for c, (_, ok, diff) in per_cam_results.items():
            if not ok:
                if diff is None:
                    miss_parts.append(f"{c}(无GT目录)")
                else:
                    miss_parts.append(f"{c}(Δ{diff/1000:.0f}ms)")
        npts_total = sum(n for n, _, _ in per_cam_results.values())
        total_pts += npts_total
        total_imgs += len(cams_ok)
        print(f"  [{idx:3d}/{len(pcd_files)}] {pcd_path.name}  "
              f"pts={points_lidar.shape[0]:6d}  proj={npts_total:6d}  "
              f"OK={len(cams_ok)}/{len(cam_ids)}" +
              (f"  缺GT=[{', '.join(miss_parts)}]" if miss_parts else ""))

    dt = time.time() - t0
    print(f"\n[完成] 用时 {dt:.1f}s, 共写出 {total_imgs} 张图, 总投影点 {total_pts}")


def parse_args():
    p = argparse.ArgumentParser(
        description="车端 LiDAR PCD 投影到车端相机 GT 图像 (JET 深度着色)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--scene', '-s', required=True, nargs='+',
                   help='场景ID, 可多个 (如: 004 006)')
    p.add_argument('--cameras', '-c', nargs='+',
                   default=['FN', 'FW', 'FL', 'FR', 'RL', 'RR', 'RN'],
                   help='相机名列表, 默认全部 7 个')
    p.add_argument('--output', '-o', default='./output',
                   help='输出根目录 (默认 ./output)')

    p.add_argument('--near', type=float, default=1.0,
                   help='固定深度范围近端 (米), 默认 1.0')
    p.add_argument('--far', type=float, default=50.0,
                   help='固定深度范围远端 (米), 默认 50.0')
    p.add_argument('--auto-range', action='store_true',
                   help='每帧自适应深度范围 (用 2%%~98%% 分位), 覆盖 --near/--far')

    p.add_argument('--point-size', type=int, default=2,
                   help='投影点的边长 (像素), 默认 2')
    p.add_argument('--alpha', type=float, default=1.0,
                   help='点对底图的不透明度 (0~1), 默认 1.0 = 完全覆盖')

    p.add_argument('--start-frame', type=int, default=0,
                   help='从第 N 帧开始 (0-based, 默认 0); 在 --align-to-gt 之后生效')
    p.add_argument('--max-frames', type=int, default=0,
                   help='最多处理多少帧 (0 = 不限制); 在 --align-to-gt 之后生效')
    p.add_argument('--gt-tolerance-ms', type=float, default=200.0,
                   help='PCD 和 GT 图像时间戳匹配容差 (毫秒, 默认 200); '
                        '设 0 表示不做容差检查, 总是用最近的图像 (和其他投影模块一致)')
    p.add_argument('--align-to-gt', action='store_true',
                   help='自动把 PCD 列表截断到 GT 图像覆盖的时间区间内 '
                        '(避免处理没有对应图像的 PCD)')

    p.add_argument('--jpg-quality', type=int, default=92,
                   help='输出 JPG 质量 (默认 92)')

    args = p.parse_args()

    # 验证相机名
    bad = [c for c in args.cameras if c not in NAME2ID]
    if bad:
        p.error(f"未知相机名: {bad}, 可选: {list(NAME2ID.keys())}")
    return args


def main():
    args = parse_args()
    for scene_id in args.scene:
        run_scene(scene_id, args)


if __name__ == '__main__':
    main()
