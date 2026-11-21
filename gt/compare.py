# -*- coding: utf-8 -*-
"""GT 比对基准：带数字画面段 → 校验多手拆解顺序。

对每个稳定数字段：
1. GT 自洽：从终局局面去掉数字子得到 base，按数字顺序重放，
   必须全程合法且终局与画面完全一致（否则段本身不可用，人工复核）。
2. 管线比对：BoardTracker 以 base 为基准，一次喂入终局局面
   （k 手一次拆解 = 最难情形，真实逐帧流只会更容易），
   比较拆解手序与 GT 数字顺序。

用法：.venv/bin/python gt/compare.py gt/out/<BV> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec.rules import Board, IllegalMoveError
from gorec.stream import BoardTracker


def eval_segment(seg, use_anchor=False, hint_override=None, strict=False):
    digits = seg['digits']
    labels = np.asarray(seg['labels'], dtype=np.int64)
    gt = sorted(digits, key=lambda d: d['num'])
    nums = [d['num'] for d in gt]
    out = {'index': seg['index'], 't0': seg['t0'], 'n_digits': len(gt),
           'nums_contiguous': nums == list(range(nums[0], nums[0] + len(nums))),
           'num_start': nums[0]}
    base = labels.copy()
    for d in gt:
        base[d['r'], d['c']] = 0
    base_list = base.tolist()  # 深拷贝：numpy 行切片是视图，直接赋给 Board 会被重放污染

    # 1. GT 自洽重放
    sim = Board(19)
    sim.grid = [row[:] for row in base_list]
    for d in gt:
        try:
            sim.apply_move(d['color'], (d['r'], d['c']))
        except IllegalMoveError as e:
            out.update(gt_ok=False, reason=f"illegal@num{d['num']}: {e}")
            return out
    if not np.array_equal(np.asarray(sim.grid), labels):
        diff = int((np.asarray(sim.grid) != labels).sum())
        out.update(gt_ok=False, reason=f'final_mismatch({diff}点)')
        return out
    out['gt_ok'] = True

    # 2. 管线拆解
    tracker = BoardTracker(19)
    tracker.board.grid = [row[:] for row in base_list]
    tracker.confirmed = 0 if gt[0]['color'] == 1 else 1  # 交替约束对齐 GT 首手颜色
    tracker.strict_alternation = strict
    hint = hint_override
    if use_anchor and hint_override is None:
        # 模拟完美三角检测：变化图中最后一手（数字最大）必有标记
        last = gt[-1]
        hint = (last['r'], last['c'])
    res = tracker.accept_grid(labels, confidence=1.0, last_move_hint=hint)
    if use_anchor:
        out['hint'] = list(hint) if hint is not None else None
        out['hint_correct'] = hint == (gt[-1]['r'], gt[-1]['c'])
    out['pipeline_status'] = res['status']
    if not res.get('accepted'):
        out.update(match=False, reason=f"pipeline_{res['status']}:{res.get('reason', '')}")
        return out
    seq = res.get('moves_seq') or [(res['color'], res['move'])]
    gt_seq = [('b' if d['color'] == 1 else 'w', (d['r'], d['c'])) for d in gt]
    got_seq = [(c, tuple(p)) for c, p in seq]
    out['pipeline_moves'] = len(got_seq)
    n_prefix = 0
    for a, b in zip(gt_seq, got_seq):
        if a == b:
            n_prefix += 1
        else:
            break
    out['prefix_match'] = n_prefix
    out['match'] = got_seq == gt_seq
    # 缺号段（提子致数字消失）子序列比对：拆解序列中 GT 数字点的相对顺序
    if not out['nums_contiguous']:
        gt_pos = {(d['r'], d['c']) for d in gt}
        got_sub = [(c, p) for c, p in got_seq if p in gt_pos]
        out['subseq_match'] = got_sub == gt_seq
    if not out['match']:
        out['gt_seq'] = [(c, list(p)) for c, p in gt_seq]
        out['got_seq'] = [(c, list(p)) for c, p in got_seq]
    return out


def summarize(reports, name):
    n = len(reports)
    gt_bad = [r for r in reports if not r.get('gt_ok')]
    gaps = [r for r in reports if r.get('gt_ok') and not r['nums_contiguous']]
    usable = [r for r in reports if r.get('gt_ok') and r['nums_contiguous']]
    if gaps:
        n_sub = sum(1 for r in gaps if r.get('subseq_match'))
        n_rej = sum(1 for r in gaps if r.get('pipeline_status') != 'move')
        print(f"   缺号段 {len(gaps)}：子序列一致 {n_sub} = {n_sub / len(gaps) * 100:.1f}%"
              f"（管线未接受 {n_rej}）")
    matched = [r for r in usable if r.get('match')]
    n_gt_moves = sum(r['n_digits'] for r in usable)
    n_prefix = sum(r.get('prefix_match', 0) for r in usable)
    print(f"== {name}: 段 {n} | GT 自洽失败 {len(gt_bad)} | 缺号剔除 {len(gaps)}"
          f" | 有效段 {len(usable)}")
    if usable:
        print(f"   拆解顺序全对 {len(matched)}/{len(usable)}"
              f" = {len(matched) / len(usable) * 100:.1f}%"
              f" | 手级前缀一致 {n_prefix}/{n_gt_moves}"
              f" = {n_prefix / max(1, n_gt_moves) * 100:.1f}%")
        buckets = {}
        for r in usable:
            k = r['n_digits']
            b = '2' if k == 2 else '3-4' if k <= 4 else '5-8' if k <= 8 else '9+'
            buckets.setdefault(b, [0, 0])
            buckets[b][1] += 1
            if r.get('match'):
                buckets[b][0] += 1
        for b in ['2', '3-4', '5-8', '9+']:
            if b in buckets:
                m, t = buckets[b]
                print(f"   段手数 {b:>3}: {m}/{t} = {m / t * 100:.0f}% 顺序全对")
    if gt_bad:
        print(f"   GT 异常: " + '; '.join(f"#{r['index']}({r['t0']:.0f}s,{r.get('reason', '')})"
                                          for r in gt_bad[:6]))
    mism = [r for r in usable if not r.get('match')]
    if mism:
        print(f"   不一致段（前 8）: " +
              '; '.join(f"#{r['index']}({r['t0']:.0f}s,{r['n_digits']}手,前缀{r.get('prefix_match', 0)})"
                        for r in mism[:8]))


def compute_real_hints(out_dir, segs):
    """真实三角检测：每段取代表帧，返回 {段 index: hint|None}。"""
    import cv2
    from gt.ocr_proto import lock_corners, grid_points_original
    from gorec.marker import find_last_move
    meta = json.load(open(out_dir / 'segments.json'))
    video = meta['video']
    hints = {}
    if not segs:
        return hints
    corners, _ = lock_corners(video, segs[0]['rep_t'])
    if corners is None:
        return hints
    pts = grid_points_original(corners)
    cap = cv2.VideoCapture(video)
    for s in segs:
        cap.set(cv2.CAP_PROP_POS_MSEC, s['rep_t'] * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        labels = np.asarray(s['labels'])
        hints[s['index']] = find_last_move(frame, pts, labels)
    cap.release()
    return hints


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    use_anchor = '--anchor' in sys.argv or '--real-anchor' in sys.argv
    real_anchor = '--real-anchor' in sys.argv
    strict = '--strict' in sys.argv
    all_reports = []
    for out_dir in args:
        out_dir = Path(out_dir)
        segs = json.load(open(out_dir / 'segments_data.json'))
        stable = [s for s in segs if not s['single_frame'] and len(s['digits']) >= 2]
        hints = compute_real_hints(out_dir, stable) if real_anchor else {}
        reports = [eval_segment(s, use_anchor=use_anchor, strict=strict,
                                hint_override=hints.get(s['index']) if real_anchor else None)
                   for s in stable]
        for r, s in zip(reports, stable):
            r['video'] = out_dir.name
            r['annot'] = str(out_dir / s['annot'])
        all_reports.extend(reports)
        summarize(reports, out_dir.name)
        print()
    out_path = Path('gt/out/compare_report_real_anchor_strict.json' if real_anchor and strict else
                    'gt/out/compare_report_real_anchor.json' if real_anchor else
                    'gt/out/compare_report_anchor.json' if use_anchor else
                    'gt/out/compare_report.json')
    with open(out_path, 'w') as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=1)
    if use_anchor:
        usable = [r for r in all_reports if r.get('gt_ok') and r['nums_contiguous']]
        n_hint = sum(1 for r in usable if r.get('hint') is not None)
        n_correct = sum(1 for r in usable if r.get('hint_correct'))
        print(f"锚点: 检出 {n_hint}/{len(usable)}，与 GT 最大数字一致 {n_correct}")
    if len(args) > 2:
        summarize(all_reports, '总计' + ('（真实三角锚定）' if real_anchor else
                                        '（锚定最后一手）' if use_anchor else ''))
    print(f"报告: {out_path}")


if __name__ == '__main__':
    main()
