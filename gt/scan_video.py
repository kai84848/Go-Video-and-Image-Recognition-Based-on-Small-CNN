# -*- coding: utf-8 -*-
"""全片扫描：抽帧 → 定位 → CNN 局面 → 数字 OCR → 合并为带数字的画面段。

输出：
  <out>/segments.json  段列表（代表帧局面 + 数字真值）
  <out>/seg_XXXX.png   代表帧标注图（人工抽查用）

用法：.venv/bin/python gt/scan_video.py <video> <out_dir> [stride_sec]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec import geometry
from gorec.model import PointNet, predict_board
from gorec.video import BoardLocator
from gt.ocr_proto import grid_points_original
from gt.digit_ocr import DigitOCR
from gt.test_ocr import get_patches, SAMPLES


def build_ocr(template_path='gt/templates_extra.pkl'):
    extra = Path(template_path)
    if extra.exists():
        ocr = DigitOCR()
        ocr.load(extra)
        return ocr
    video_path = sorted(Path('数据集/gt_candidates').glob('BV1rnsS6uEzy*.mp4'))[0]
    _, _, _, patches = get_patches(str(video_path), 700)
    ocr = DigitOCR()
    for r, c, d in SAMPLES:
        color, patch = patches[(r, c)]
        ocr.add_sample(patch, color, d)
    return ocr


def annotate(frame, pts, digits):
    vis = frame.copy()
    for d in digits:
        x, y = pts[d['r'], d['c']]
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(vis, (xi, yi), 26, (0, 220, 0), 2)
        cv2.putText(vis, str(d['num']), (xi - 12, yi - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    return vis


def main():
    video_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    stride_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    template_path = sys.argv[4] if len(sys.argv) > 4 else 'gt/templates_extra.pkl'
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = build_ocr(template_path)
    model = PointNet()
    model.load_state_dict(torch.load('pointnet.pt', map_location='cpu'))
    device = 'cpu'  # 361 点/帧 CPU 已够快；多进程并行时避免 MPS 互相拖慢

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps * stride_sec)))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'fps={fps:.2f} frames={n_frames} stride={stride}', flush=True)

    loc = BoardLocator()
    segments = []
    cur = None  # 当前段
    half = 22
    fi = -1
    n_done = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fi += 1
        if fi % stride:
            continue
        n_done += 1
        if n_done % 200 == 0:
            print(f'  ..{fi}/{n_frames} frames, {len(segments)} segments', flush=True)
        t = fi / fps
        corners, locked = loc.locate(frame, frame_index=fi)
        if not locked or corners is None:
            if cur is not None:
                segments.append(cur)
                cur = None
            continue
        can = geometry.warp_canonical(frame, corners)
        labels, prob = predict_board(model, can, device=device)
        pts = grid_points_original(corners)
        digits = []
        patches = {}
        for r in range(19):
            for c in range(19):
                if labels[r, c] == 0:
                    continue
                x, y = pts[r, c]
                xi, yi = int(round(x)), int(round(y))
                patch = frame[max(0, yi - half):yi + half, max(0, xi - half):xi + half]
                if patch.shape[0] != 2 * half or patch.shape[1] != 2 * half:
                    continue
                num, score = ocr.recognize(patch)  # 颜色自动：CNN 可能误判带字子的颜色
                if num is not None:
                    digits.append({'r': r, 'c': c, 'color': int(labels[r, c]),
                                   'num': int(num), 'score': round(float(score), 4)})
        key = tuple(sorted((d['r'], d['c'], d['color'], d['num']) for d in digits))
        if cur is not None and cur['key'] == key:
            cur['t1'] = t
            cur['n_frames'] += 1
            continue
        if cur is not None:
            segments.append(cur)
        cur = None
        if not digits:
            continue
        # 新段：保存代表帧信息
        idx = len(segments)
        annot_name = f'seg_{idx:04d}.png'
        vis = annotate(frame, pts, digits)
        cv2.imwrite(str(out_dir / annot_name), cv2.resize(vis, (1280, 720)))
        cur = {'key': key, 't0': t, 't1': t, 'n_frames': 1, 'rep_t': t,
               'digits': digits, 'labels': labels.tolist(), 'annot': annot_name}
    if cur is not None:
        segments.append(cur)
    cap.release()

    for i, s in enumerate(segments):
        s.pop('key', None)
        s['index'] = i
    # 单帧段多为 OCR 抖动/转场，单独标记（不删，供人工判断）
    for s in segments:
        s['single_frame'] = s['n_frames'] < 2
    meta = {'video': video_path, 'fps': fps, 'stride_sec': stride_sec,
            'n_segments': len(segments),
            'n_stable': sum(1 for s in segments if not s['single_frame'])}
    with open(out_dir / 'segments.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
        f.write('\n')
    # 段数据单独文件，避免 meta 过大
    with open(out_dir / 'segments_data.json', 'w') as f:
        json.dump(segments, f, ensure_ascii=False)
    print(f"DONE segments={len(segments)} stable={meta['n_stable']} -> {out_dir}", flush=True)


if __name__ == '__main__':
    main()
