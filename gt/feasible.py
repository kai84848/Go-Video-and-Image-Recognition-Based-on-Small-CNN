# -*- coding: utf-8 -*-
"""不一致段归因：GT 顺序在管线的单手迁移校验框架下是否可行。

- feasible：GT 顺序每步 check 都合法 → 管线本可拆对，是贪心选择错误
- infeasible：某步 check 失败 → 校验框架与真值有差距（如提子处理）
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec.rules import Board, TransitionChecker, IllegalMoveError


def gt_feasibility(seg):
    labels = np.asarray(seg['labels'], dtype=np.int64)
    gt = sorted(seg['digits'], key=lambda d: d['num'])
    base = labels.copy()
    for d in gt:
        base[d['r'], d['c']] = 0
    sim = Board(19)
    sim.grid = base.tolist()
    checker = TransitionChecker(19)
    for d in gt:
        trial = np.array(sim.grid, copy=True)
        trial[d['r'], d['c']] = d['color']
        res = checker.check(sim.grid, trial)
        if not res.get('ok'):
            return False, f"num{d['num']}:{res.get('reason')}"
        try:
            sim.apply_move(d['color'], (d['r'], d['c']))
        except IllegalMoveError as e:
            return False, f"num{d['num']}:apply:{e}"
    return True, ''


def main():
    segs = json.load(open(Path(sys.argv[1]) / 'segments_data.json'))
    video = Path(sys.argv[1]).name
    reports = [r for r in json.load(open('gt/out/compare_report.json'))
               if r.get('video') == video]
    by_idx = {s['index']: s for s in segs}
    stats = {'feasible': 0, 'infeasible': 0}
    examples = []
    for r in reports:
        if not r.get('gt_ok') or not r['nums_contiguous'] or r.get('match'):
            continue
        seg = by_idx[r['index']]
        ok, reason = gt_feasibility(seg)
        key = 'feasible' if ok else 'infeasible'
        stats[key] += 1
        if not ok and len(examples) < 10:
            examples.append(f"#{r['index']}({r['n_digits']}手) {reason}")
    print(f"不一致段 {stats['feasible'] + stats['infeasible']}："
          f"GT 顺序可行(贪心选错) {stats['feasible']}，不可行(校验框架差异) {stats['infeasible']}")
    for e in examples:
        print('  ', e)


if __name__ == '__main__':
    main()
