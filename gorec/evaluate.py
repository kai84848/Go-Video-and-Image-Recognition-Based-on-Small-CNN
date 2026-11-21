# -*- coding: utf-8 -*-
"""端到端评测：合成测试集 + 真实截图。

合成：make_sample 生成全新局面（训练没用过的 seed 段）→ 整盘 361 点准确率。
真实：数据集/pic 里的截图 → detect_grid 矫正 → 分类 → 保存可视化。
"""
import glob
import os

import cv2
import numpy as np
import torch

from . import dataset, geometry
from .model import PointNet, predict_board


def eval_synthetic(model, n_boards=50, seed0=990000, device='cpu'):
    """返回 (逐点准确率, 整盘全对率, 逐子准确率)。"""
    total = sub_total = sub_ok = boards_ok = 0
    for i in range(n_boards):
        img, labels, meta = dataset.make_sample(seed=seed0 + i)
        pred, prob = predict_board(model, img, device=device,
                                   border=meta.get('border', 0))
        gt = np.array(labels)
        ok = pred == gt
        total += ok.sum()
        sub_total += gt.size
        # 只统计有子的位置
        mask = gt != 0
        sub_ok += (pred[mask] == gt[mask]).sum()
        sub_total_masked = mask.sum()
        boards_ok += int(ok.all())
    point_acc = total / (sub_total)
    stone_acc = sub_ok / max(1, (sub_total - 0))  # 近似
    return point_acc, boards_ok / n_boards, stone_acc


def eval_real_pics(model, pics_dir='数据集/pic', out_dir='/tmp/go_real_eval', device='cpu'):
    """对真实截图跑完整管线，保存可视化。返回每张的预测摘要。"""
    os.makedirs(out_dir, exist_ok=True)
    pics = sorted(f for f in glob.glob(os.path.join(pics_dir, '*.png'))
                  if '/._' not in f)
    results = []
    for p in pics:
        img = cv2.imread(p)
        det, info = geometry.detect_grid(img)
        if det is None:
            results.append({'file': os.path.basename(p), 'status': 'detect_fail'})
            continue
        can = geometry.warp_canonical(img, det)
        pred, prob = predict_board(model, can, device=device)
        conf = float(prob.mean())
        n_black = int((pred == 1).sum())
        n_white = int((pred == 2).sum())
        # 低置信点
        n_low = int((prob < 0.85).sum())
        vis = can.copy()
        for r in range(19):
            for c in range(19):
                if prob[r, c] < 0.85:
                    x = geometry.CELL // 2 + c * geometry.CELL
                    y = geometry.CELL // 2 + r * geometry.CELL
                    cv2.circle(vis, (x, y), geometry.CELL // 2, (0, 0, 255), 2)
        name = os.path.splitext(os.path.basename(p))[0][:16]
        cv2.imwrite(os.path.join(out_dir, f'{name}_pred.png'),
                    cv2.resize(vis, (576, 576), interpolation=cv2.INTER_AREA))
        results.append({'file': os.path.basename(p), 'status': 'ok',
                        'conf': conf, 'black': n_black, 'white': n_white,
                        'low_conf': n_low})
    return results


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = PointNet()
    model.load_state_dict(torch.load('pointnet.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    print('=== 合成测试（新 seed）===')
    pa, ba, sa = eval_synthetic(model, n_boards=50, device=device)
    print(f'逐点准确率: {pa*100:.3f}%  整盘全对: {ba*100:.1f}%')

    print('=== 真实截图 ===')
    for r in eval_real_pics(model, device=device):
        if r['status'] == 'ok':
            print(f"{r['file'][:24]:26s} 黑{r['black']:3d} 白{r['white']:3d} "
                  f"均置信={r['conf']:.3f} 低置信点={r['low_conf']}")
        else:
            print(f"{r['file'][:24]:26s} {r['status']}")
