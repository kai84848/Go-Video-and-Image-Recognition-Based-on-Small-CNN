# -*- coding: utf-8 -*-
"""真实流比对：模拟现有管线的逐帧处理，校验数字段的落子顺序与终局。

与 compare.py（一次拆 k 手的压力测试）不同，这里按真实视频流逐帧喂入，
回答"实际使用中，带数字的变化图段落手序/局面是否与现实一致"。

用法：.venv/bin/python gt/stream_check.py <video> <segments_dir> [max_segments]
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
from gorec.stream import BoardStream
from gt.scan_video import build_ocr


def check_segment(cap, model, seg, fps, ocr):
    t0, t1 = seg['t0'], seg['t1']
    start = max(0.0, t0 - 6.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    loc = BoardLocator()
    stream = BoardStream(19)
    stride = max(1, int(round(fps / 2)))
    n_total = int((t1 + 1.0 - start) * fps)
    accepted = []  # (t, color, pos)
    for i in range(n_total):  # 顺序解码：cap.set 逐帧定位有偏差，只在起点跳一次
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride:
            continue
        t = start + i / fps
        corners, locked = loc.locate(frame)
        if not locked or corners is None:
            continue
        can = geometry.warp_canonical(frame, corners)
        labels, prob = predict_board(model, can, device='cpu')
        res = stream.accept_grid(labels, prob, point_conf=prob)
        if res.get('accepted'):
            seq = res.get('moves_seq') or [(res['color'], res['move'])]
            for color, pos in seq:
                accepted.append((t, color, tuple(pos)))
    final_grid = np.asarray(stream.tracker.board.grid)
    gt_labels = np.asarray(seg['labels'])
    board_diff = int((final_grid != gt_labels).sum())

    # GT 数字子在窗口内被接受的落子（可能落在 resync 基底里 → missing）
    gt_digits = sorted(seg['digits'], key=lambda d: d['num'])
    seen = []
    for d in gt_digits:
        hits = [t for t, color, pos in accepted if pos == (d['r'], d['c'])]
        seen.append((d['num'], hits[0] if hits else None))
    accepted_nums = [n for n, t in seen if t is not None]
    order_ok = accepted_nums == sorted(accepted_nums)
    return {
        'index': seg['index'], 't0': t0, 'n_digits': len(gt_digits),
        'n_accepted_moves': len(accepted),
        'gt_seen_as_moves': len(accepted_nums),
        'gt_missing': [n for n, t in seen if t is None],
        'order_ok': order_ok,
        'board_diff': board_diff,
        'n_resync': len(stream.resyncs),
    }


def main():
    video_path = sys.argv[1]
    seg_dir = Path(sys.argv[2])
    max_segments = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    segs = json.load(open(seg_dir / 'segments_data.json'))
    stable = [s for s in segs if not s['single_frame'] and len(s['digits']) >= 2
              and s['t1'] - s['t0'] >= 1.0][:max_segments]
    model = PointNet()
    model.load_state_dict(torch.load('pointnet.pt', map_location='cpu'))
    ocr = build_ocr()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    reports = []
    for i, seg in enumerate(stable):
        r = check_segment(cap, model, seg, fps, ocr)
        reports.append(r)
        if (i + 1) % 10 == 0:
            print(f'  ..{i + 1}/{len(stable)}', flush=True)
    cap.release()
    out_path = seg_dir / 'stream_report.json'
    with open(out_path, 'w') as f:
        json.dump(reports, f, ensure_ascii=False, indent=1)
    n = len(reports)
    order_ok = sum(1 for r in reports if r['order_ok'])
    board_ok = sum(1 for r in reports if r['board_diff'] == 0)
    print(f"== {seg_dir.name}: 段 {n}")
    print(f"   落子时间序与数字序一致: {order_ok}/{n} = {order_ok / max(1, n) * 100:.1f}%")
    print(f"   终局局面逐点一致: {board_ok}/{n} = {board_ok / max(1, n) * 100:.1f}%")
    print(f"   报告: {out_path}")


if __name__ == '__main__':
    main()
