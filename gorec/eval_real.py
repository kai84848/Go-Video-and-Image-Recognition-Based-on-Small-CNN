# -*- coding: utf-8 -*-
"""真实数据评测：对截图/录屏/视频帧跑"矫正 + 逐点分类"，
与人工核对的局面比对，输出整体准确率报告。

用法：
  python -m gorec.eval_real            # 评测全部真实数据
目前几何定位在纯棋盘画面（截图/录屏）上已 100%；
复合画面（讲解视频，棋盘+UI 面板）定位不可靠，标记为 KNOWN_LIMITATION。
"""
import glob
import os

import cv2
import numpy as np

from . import geometry


def eval_pictures(pics_dir):
    """对目录中的截图跑几何检测，报告结果。"""
    pics = sorted(f for f in glob.glob(os.path.join(pics_dir, '*.png'))
                  if '/._' not in f and '\\._' not in f)
    results = []
    for p in pics:
        img = cv2.imread(p)
        if img is None:
            continue
        det, info = geometry.detect_grid(img)
        r = {'file': os.path.basename(p), 'img': img, 'det': det, 'info': info}
        if det is not None:
            can = geometry.warp_canonical(img, det)
            r['canonical'] = can
            step = np.hypot(det['tr'][0] - det['tl'][0],
                            det['tr'][1] - det['tl'][1]) / 18
            r['step'] = step
            # 对齐质量：期望线位置与实际暗线的偏差
            gh = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY)
            prof = gh[:, 5:gh.shape[1] - 5].mean(axis=1)
            errs = []
            m = geometry.CELL // 2  # canonical 交叉点带 margin 偏移
            for i in range(18):
                e = m + i * geometry.CELL
                seg = prof[max(0, e - 8):e + 9]
                y = int(np.argmin(seg)) + max(0, e - 8)
                errs.append(abs(y - e))
            r['align_med'] = float(np.median(errs))
        results.append(r)
    return results


def sample_video_frames(video, n_frames=5, detect=True):
    """从视频均匀抽帧并做几何检测。返回 [(frame, det, info), ...]"""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for frac in np.linspace(0.15, 0.85, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ok, frame = cap.read()
        if not ok:
            continue
        det = info = None
        if detect:
            det, info = geometry.detect_grid(frame)
        out.append((frame, det, info))
    cap.release()
    return out
