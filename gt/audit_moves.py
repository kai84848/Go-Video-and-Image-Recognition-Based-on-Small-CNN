# -*- coding: utf-8 -*-
"""识别错误归因：GT 漏失手 vs 多出手，逐手定位是 CNN 识别问题还是流问题。

对 GT 漏失的每一手：查它在所有扫描段中的 labels 颜色与 OCR 数字，
区分：颜色误判 / 位置不稳定 / 识别正确但被流（resync 等）丢弃。
对 SGF 多出的每一手：查该位置是否从未有过数字（假子候选）。

用法：.venv/bin/python gt/audit_moves.py gt/out/rec2 default
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gt.sgf_compare import build_gt


def main():
    out_dir = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'default'
    segs = json.load(open(out_dir / 'segments_data.json'))
    rep = json.load(open(out_dir / f'sgf_compare_{mode}.json'))
    gt, conflicts = build_gt(segs)
    first_color = gt[1][2] if 1 in gt else 1

    def true_color(num):
        return first_color if (num - 1) % 2 == 0 else 3 - first_color

    pos2num = {}
    for num, (r, c, color) in gt.items():
        pos2num.setdefault((r, c), []).append(num)

    print('== GT 漏失手归因（管线未记录）==')
    for (color, pos), _ in rep['unmatched_gt']:
        pos = tuple(pos)
        nums = pos2num.get(pos, [])
        info = []
        for num in nums:
            occurrences = []
            for s in segs:
                for d in s['digits']:
                    if d['num'] == num and (d['r'], d['c']) == pos:
                        occurrences.append(s)
                        break
            n = len(occurrences)
            tc = true_color(num)
            cnn_colors = {s['labels'][pos[0]][pos[1]] for s in occurrences}
            info.append(f"num{num}(真值色{tc},CNN色{sorted(cnn_colors)},出现{n}段)")
        print(f"  {color}@{pos}: " + ('; '.join(info) if info else 'GT 中无此位置'))

    print('== SGF 多出手归因（GT 缺号或假子）==')
    for (color, pos), _ in rep['unmatched_sgf']:
        pos = tuple(pos)
        # 该位置在所有段中的数字与 labels 历史
        nums_here = set()
        n_stone = 0
        for s in segs:
            if s['labels'][pos[0]][pos[1]] != 0:
                n_stone += 1
            for d in s['digits']:
                if (d['r'], d['c']) == pos:
                    nums_here.add(d['num'])
        print(f"  {color}@{pos}: 数字历史={sorted(nums_here)} 有子段数={n_stone}")


if __name__ == '__main__':
    main()
