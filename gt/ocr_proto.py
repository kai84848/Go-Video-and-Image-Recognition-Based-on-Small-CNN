# -*- coding: utf-8 -*-
"""OCR 原型：单帧 → 定位 → 原图分辨率逐点 patch 拼图，供人工确认数字。"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec import geometry
from gorec.model import PointNet, predict_board

CELL = 32
MARGIN = 16

# 生产实现已移至 gorec.geometry；此处保留导入兼容旧调用
from gorec.geometry import grid_points_original  # noqa: E402,F401


def lock_corners(video_path, t_sec, search_back=10.0):
    """从 t_sec-search_back 起顺序喂帧，返回 (锁定的 corners, 该时刻帧)。"""
    from gorec.video import BoardLocator
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start = max(0.0, t_sec - search_back)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    loc = BoardLocator()
    target_fi = int(round(t_sec * fps))
    corners = None
    target_frame = None
    for _ in range(int(search_back * fps) + int(3 * fps)):
        ok, frame = cap.read()
        if not ok:
            break
        fi = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 * fps))
        c, locked = loc.locate(frame)
        if locked:
            corners = c
        if fi >= target_fi and corners is not None:
            target_frame = frame
            break
    cap.release()
    return corners, target_frame


def main():
    if sys.argv[1] == '--video':
        video_path, t_sec = sys.argv[2], float(sys.argv[3])
        out_path = sys.argv[4] if len(sys.argv) > 4 else '/tmp/gt_patches.png'
        corners, img = lock_corners(video_path, t_sec)
        if corners is None or img is None:
            print('lock_fail')
            return
        print('locked corners:', {k: [round(float(v[0]), 1), round(float(v[1]), 1)]
                                  for k, v in corners.items()})
    else:
        img_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/gt_patches.png'
        img = cv2.imread(img_path)
        corners, score = geometry.detect_grid(img)
        if corners is None:
            print('detect_fail:', score)
            return
        print('grid score:', score)
    can = geometry.warp_canonical(img, corners)
    import torch
    model = PointNet()
    model.load_state_dict(torch.load('pointnet.pt', map_location='cpu'))
    labels, prob = predict_board(model, can, device='cpu')
    pts = grid_points_original(corners)

    # 有子点切原图 patch
    half = 22
    items = []
    for r in range(19):
        for c in range(19):
            if labels[r, c] == 0:
                continue
            x, y = pts[r, c]
            xi, yi = int(round(x)), int(round(y))
            patch = img[max(0, yi - half):yi + half, max(0, xi - half):xi + half]
            if patch.shape[0] != 2 * half or patch.shape[1] != 2 * half:
                continue
            items.append((r, c, int(labels[r, c]), float(prob[r, c]), patch))
    print(f'有子点: {len(items)} (黑{int((labels==1).sum())} 白{int((labels==2).sum())})')

    cols = 10
    cell = 52
    rows = (len(items) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 240, np.uint8)
    for i, (r, c, lab, p, patch) in enumerate(items):
        y0 = (i // cols) * cell
        x0 = (i % cols) * cell
        sheet[y0 + 6:y0 + 6 + 2 * half, x0 + 4:x0 + 4 + 2 * half] = patch
        cv2.putText(sheet, f'{r},{c}', (x0 + 2, y0 + cell - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 200), 1)
    cv2.imwrite(out_path, sheet)
    print('拼图已写:', out_path)


if __name__ == '__main__':
    main()
