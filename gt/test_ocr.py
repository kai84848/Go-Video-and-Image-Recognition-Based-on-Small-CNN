# -*- coding: utf-8 -*-
"""单帧 OCR 验证：700s 帧自建 1-9 模板，识别该帧全部有子点。"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec.model import PointNet, predict_board
from gorec import geometry
from gt.ocr_proto import lock_corners, grid_points_original
from gt.digit_ocr import DigitOCR

SAMPLES = [  # (r, c, digit) —— 700s 帧人工读数
    (5, 14, 7), (6, 11, 9), (6, 12, 8), (6, 13, 4), (6, 14, 2),
    (6, 15, 1), (6, 16, 5), (7, 13, 3), (7, 16, 6),
]


def get_patches(video_path, t_sec):
    corners, img = lock_corners(video_path, t_sec)
    assert corners is not None and img is not None, 'lock fail'
    can = geometry.warp_canonical(img, corners)
    model = PointNet()
    model.load_state_dict(torch.load('pointnet.pt', map_location='cpu'))
    labels, prob = predict_board(model, can, device='cpu')
    pts = grid_points_original(corners)
    half = 22
    patches = {}
    for r in range(19):
        for c in range(19):
            if labels[r, c] == 0:
                continue
            x, y = pts[r, c]
            xi, yi = int(round(x)), int(round(y))
            patch = img[max(0, yi - half):yi + half, max(0, xi - half):xi + half]
            if patch.shape[0] == 2 * half and patch.shape[1] == 2 * half:
                patches[(r, c)] = (int(labels[r, c]), patch)
    return img, pts, labels, patches


def main():
    video_path = glob.glob('数据集/gt_candidates/BV1rnsS6uEzy*.mp4')[0]
    img, pts, labels, patches = get_patches(video_path, 700)
    ocr = DigitOCR()
    for r, c, d in SAMPLES:
        color, patch = patches[(r, c)]
        ocr.add_sample(patch, color, d)
    print(f'模板库: {len(ocr.templates)} 类')

    hits = []
    for (r, c), (color, patch) in sorted(patches.items()):
        num, score = ocr.recognize(patch, color)
        if num is not None:
            hits.append((r, c, color, num, round(score, 3)))
    print(f'识别出数字的点: {len(hits)} / 有子点 {len(patches)}')
    for h in hits:
        print('  r=%d c=%d color=%d num=%s score=%.3f' % h)
    expect = {(r, c): d for r, c, d in SAMPLES}
    got = {(r, c): num for r, c, _, num, _ in hits}
    print('与人工读数一致:', got == expect)
    if got != expect:
        for k in set(expect) | set(got):
            if expect.get(k) != got.get(k):
                print('  差异', k, 'expect', expect.get(k), 'got', got.get(k))


if __name__ == '__main__':
    main()
