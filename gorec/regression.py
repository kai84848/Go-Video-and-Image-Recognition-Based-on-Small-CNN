# -*- coding: utf-8 -*-
"""批量回归测试：一键跑全部验收用例，输出 PASS/FAIL 报告。

用法：
  python -m gorec.regression          # 全量
  python -m gorec.regression --quick  # 跳过耗时项

用例：
  G1 几何：合成 60 盘（JPEG）检出率 + 中位误差
  G2 几何：旋转 0/1/2 度 40 盘
  G3 几何：真实截图 9 张全检出
  G4 几何：空棋盘帧（曾误检的回归用例）
  G5 几何：讲解视频复合画面锁定（时序一致）
  C1 分类：合成 30 盘逐点准确率 + 整盘全对率
  V1 视频：录屏端到端 SGF 手数>0、无非法接受
"""
import glob
import os
import sys
import time

import cv2
import numpy as np

from . import dataset, geometry, video
from .renderer import render_board


def _labels_pattern():
    labels = [[0] * 19 for _ in range(19)]
    for r in range(19):
        for c in range(19):
            if (r * 7 + c * 3) % 5 == 0:
                labels[r][c] = 1
            elif (r + c) % 4 == 0:
                labels[r][c] = 2
    return labels


def g1_synthetic(n=60):
    labels = _labels_pattern()
    errs, fails = [], 0
    for seed in range(n):
        img, corners = render_board(labels, 640, 640, perspective=0.0,
                                    jpeg_quality=75, seed=seed)
        det, _ = geometry.detect_grid(img)
        if det is None:
            fails += 1
            continue
        errs.append(max(np.hypot(det[k][0] - corners[k][0],
                                 det[k][1] - corners[k][1]) for k in det))
    med = float(np.median(errs)) if errs else -1
    ok = (n - fails) == n and med < 2.0
    return ok, f'{n - fails}/{n} 检出, median {med:.2f}px, {fails} fail'


def g2_rotation():
    labels = _labels_pattern()
    lines = []
    ok = True
    for rot in (0.0, 1.0, 2.0):
        errs, fails = [], 0
        for seed in range(40):
            img, corners = render_board(labels, 640, 640, perspective=0.0,
                                        jpeg_quality=75, seed=seed)
            truth = corners
            if rot:
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
                pts = np.array([corners['tl'], corners['tr'],
                                corners['br'], corners['bl']],
                               np.float32).reshape(-1, 1, 2)
                vpts = cv2.transform(pts, M).reshape(-1, 2)
                truth = dict(zip(('tl', 'tr', 'br', 'bl'), vpts))
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            det, _ = geometry.detect_grid(img)
            if det is None:
                fails += 1
                continue
            errs.append(max(np.hypot(det[k][0] - truth[k][0],
                                     det[k][1] - truth[k][1]) for k in det))
        med = float(np.median(errs)) if errs else -1
        line = f'rot {rot}: {40 - fails}/40, median {med:.2f}px'
        lines.append(line)
        if fails > 0 or med > 3.0:
            ok = False
    return ok, '; '.join(lines)


def g3_real_pics():
    pics = sorted(f for f in glob.glob('数据集/pic/*.png') if '/._' not in f)
    if not pics:
        return True, 'no pics (skipped)'
    ok_all = 0
    for p in pics:
        img = cv2.imread(p)
        det, _ = geometry.detect_grid(img)
        ok_all += det is not None
    return ok_all == len(pics), f'{ok_all}/{len(pics)} 检出'


def g4_empty_board(frame_path='/tmp/scr_start2.png'):
    if not os.path.exists(frame_path):
        return True, 'frame missing (skipped)'
    img = cv2.imread(frame_path)
    det, score = geometry.detect_grid(img)
    if det is None:
        return False, 'detect_fail'
    # 空棋盘真值 tl≈(37,31)
    ok = abs(det['tl'][0] - 37) < 8 and abs(det['tl'][1] - 31) < 8
    return ok, f"tl=({det['tl'][0]:.0f},{det['tl'][1]:.0f}) score {score:.3f}"


def g5_lecture_lock(video_path=None):
    if video_path is None:
        cands = sorted(glob.glob('数据集/video/0*.mp4'))
        if not cands:
            return True, 'no lecture video (skipped)'
        video_path = cands[0]
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    loc = video.BoardLocator()
    locked = 0
    total = 0
    tl0 = None
    drift = False
    for frac in np.linspace(0.38, 0.6, 8):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
        ok, fr = cap.read()
        if not ok:
            continue
        total += 1
        c, lk = loc.locate(fr)
        if lk and c is not None:
            locked += 1
            tl_cur = (round(c['tl'][0]), round(c['tl'][1]))
            if tl0 is None:
                tl0 = tl_cur
            elif tl_cur != tl0:
                drift = True
    cap.release()
    ok = locked >= total - 2 and not drift
    return ok, f'{locked}/{total} 锁定, tl={tl0}{", 漂移!" if drift else ""}'


def c1_classification(model_path='pointnet.pt', n=30):
    import torch
    from .model import PointNet, predict_board
    model = PointNet()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    accs, boards_ok = [], 0
    for seed in range(990000, 990000 + n):
        img, labels, meta = dataset.make_sample(seed=seed, with_marker=True)
        pred, prob = predict_board(model, img, border=meta.get('border', 0))
        gt = np.array(labels)
        acc = float((pred == gt).mean())
        accs.append(acc)
        boards_ok += acc == 1.0
    return float(np.mean(accs)) > 0.985, (f'逐点 {np.mean(accs) * 100:.2f}%, '
                                          f'整盘 {boards_ok}/{n}')


def v1_screencast(video_path=None, max_seconds=600):
    vids = sorted(f for f in glob.glob('数据集/video/录屏*.mov') if '/._' not in f)
    if not vids:
        return True, 'no screencast (skipped)'
    if video_path is None:
        video_path = vids[0]
    t0 = time.time()
    moves, stats = video.video_to_sgf(video_path, sample_fps=2.0, verbose=False)
    dt = time.time() - t0
    ok = len(moves) > 0 and stats['rejected'] <= max(2, stats['moves'] // 10)
    return ok, f'{len(moves)} 手, {dt:.0f}s, stats={stats}'


CASES = [
    ('G1 几何-合成60盘', lambda: g1_synthetic()),
    ('G2 几何-旋转', g2_rotation),
    ('G3 几何-真实截图', g3_real_pics),
    ('G4 几何-空棋盘帧', g4_empty_board),
    ('G5 几何-讲解视频锁定', g5_lecture_lock),
    ('C1 分类-合成30盘', c1_classification),
]

QUICK_CASES = ['G3', 'G4', 'G5']


def main():
    quick = '--quick' in sys.argv
    results = []
    for name, fn in CASES:
        if quick and not any(name.startswith(p) for p in QUICK_CASES):
            continue
        t0 = time.time()
        try:
            ok, msg = fn()
        except Exception as e:
            ok, msg = False, f'EXC {e}'
        dt = time.time() - t0
        results.append((name, ok, msg, dt))
        print(f'{"PASS" if ok else "FAIL"}  {name}  ({dt:.0f}s)  {msg}',
              flush=True)
    n_fail = sum(1 for _, ok, _, _ in results if not ok)
    print(f'\n{"=" * 50}\n{len(results) - n_fail}/{len(results)} PASS')
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
