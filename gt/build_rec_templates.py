# -*- coding: utf-8 -*-
"""录屏客户端字体模板库构建：60s 帧取 1-9 样本，全帧识别验证后保存。"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec import geometry
from gorec.model import PointNet, predict_board
from gt.digit_ocr import DigitOCR

IMG = '/tmp/rec_num_录屏2026-11-21_16.24.44_60.png'
SAMPLES = [  # (r, c, digit) —— 从 patch 拼图人工读数
    (3, 15, 1), (3, 3, 2), (15, 15, 3), (16, 3, 4), (14, 3, 5),
    (14, 2, 6), (13, 2, 7), (15, 2, 8), (13, 3, 9),
]


def patches_from_image(img_path):
    img = cv2.imread(img_path)
    corners, score = geometry.detect_grid(img)
    assert corners is not None, 'detect fail'
    can = geometry.warp_canonical(img, corners)
    model = PointNet()
    model.load_state_dict(torch.load('pointnet.pt', map_location='cpu'))
    labels, prob = predict_board(model, can, device='cpu')
    pts = geometry.grid_points_original(corners)
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
    return img, corners, labels, patches


def main():
    img, corners, labels, patches = patches_from_image(IMG)
    ocr = DigitOCR()
    for r, c, d in SAMPLES:
        color, patch = patches[(r, c)]
        ocr.add_sample(patch, color, d)
    # 0 模板：从 (16,5) 白子"10"的第二位提取
    from gt.digit_ocr import digit_mask, _components
    color, patch = patches[(16, 5)]
    mask = digit_mask(patch, color)
    chars = _components(mask)
    assert len(chars) == 2, f'10 应为 2 位，实际 {len(chars)}'
    ocr.templates.setdefault((0, color), []).append(chars[1])
    # 多位数逐位样本扩充：0/6 易混淆，需要覆盖渲染变体（人工读数确认）
    EXTRA = [(5, 5, 20), (14, 5, 30), (10, 5, 40), (5, 3, 16), (2, 4, 26),
             (11, 3, 28)]
    for r, c, num in EXTRA:
        color, patch = patches[(r, c)]
        mask = digit_mask(patch, color)
        chars = _components(mask)
        assert len(chars) == len(str(num)), f'{(r, c)} 位数不符'
        for ch, d in zip(chars, str(num)):
            ocr.templates.setdefault((int(d), color), []).append(ch)
    print(f'模板 {len(ocr.templates)} 类（含 0）；开始全帧识别')
    results = []
    for (r, c), (color, patch) in patches.items():
        num, score = ocr.recognize(patch)
        if num is not None:
            results.append((num, r, c, color, round(score, 2)))
    results.sort()
    for num, r, c, color, score in results:
        print(f'  {num:3d} @({r},{c}) {"黑" if color == 1 else "白"} score={score}')
    print(f'共识别 {len(results)} 个数字子 / {len(patches)} 有子点')
    ocr.save('gt/templates_rec.pkl')
    print('已保存 gt/templates_rec.pkl')


if __name__ == '__main__':
    main()
