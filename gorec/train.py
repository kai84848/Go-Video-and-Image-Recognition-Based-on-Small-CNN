# -*- coding: utf-8 -*-
"""训练逐点分类网络。

数据：dataset.make_sample 合成棋盘 → 提取全部交叉点 patch。
用法：python -m gorec.train [epochs] [boards_per_epoch]
"""
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import dataset
from .model import PointNet, extract_patches, patches_to_tensor


def gen_batch_samples(n_boards, seed0, with_marker=True, hard_star=False):
    """生成 n_boards 张棋盘，返回 (patches, labels) 两个大数组。"""
    all_p = []
    all_y = []
    for i in range(n_boards):
        img, labels, meta = dataset.make_sample(seed=seed0 + i, with_marker=with_marker)
        patches = extract_patches(img, border=meta.get('border', 0))
        all_p.append(patches)
        all_y.append(np.array(labels).reshape(-1))
    return np.concatenate(all_p), np.concatenate(all_y)


def train(epochs=3, boards_per_epoch=250, batch_size=512, device=None):
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'device: {device}')
    model = PointNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * boards_per_epoch // 4)

    seed0 = 10000
    for ep in range(epochs):
        t0 = time.time()
        X, Y = gen_batch_samples(boards_per_epoch, seed0, with_marker=(ep > 0))
        seed0 += boards_per_epoch
        # 类别加权（空点远多于子）
        counts = np.bincount(Y, minlength=3).astype(np.float64)
        w = counts.sum() / (3 * np.maximum(counts, 1))
        w_t = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
        perm = np.random.permutation(len(X))
        tot_loss = tot_acc = seen = 0
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size]
            xb = patches_to_tensor(X[idx]).to(device)
            yb = torch.from_numpy(Y[idx]).to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=w_t)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += float(loss.detach()) * len(idx)
            tot_acc += float((logits.argmax(1) == yb).sum())
            seen += len(idx)
        print(f'ep{ep}: loss={tot_loss/seen:.4f} acc={tot_acc/seen:.4f} '
              f'({seen} 点, {time.time()-t0:.0f}s)')
        torch.save(model.state_dict(), 'pointnet.pt')
    print('saved pointnet.pt')
    return model


if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    boards = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    train(epochs=epochs, boards_per_epoch=boards)
