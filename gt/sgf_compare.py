# -*- coding: utf-8 -*-
"""录屏全局手序 GT → SGF 逐手比对。

与 compare.py（讲解视频变化图）不同：录屏客户端显示的是全局手数，
合并整片所有段的数字即得完整手序真值（被提的手在提子前的帧里也有数字）。

用法：
  .venv/bin/python gt/sgf_compare.py gt/out/rec1 [precise]
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_gt(segments):
    """合并所有段的数字 → {num: (r,c,color)}（按段帧数加权投票）。"""
    votes = defaultdict(Counter)
    for s in segments:
        for d in s['digits']:
            if d['num'] <= 0:
                continue  # 客户端的 0 是特殊标记，非手序
            votes[d['num']][(d['r'], d['c'], d['color'])] += max(1, s['n_frames'])
    gt = {}
    conflicts = {}
    for num, counter in votes.items():
        (best, n_best) = counter.most_common(1)[0]
        gt[num] = best
        if len(counter) > 1:
            conflicts[num] = dict(counter)
    return gt, conflicts


def compare(gt, moves):
    """GT 手序 vs 管线 moves：SequenceMatcher 序列对齐（容忍 GT 缺号、
    SGF 星位 setup 前缀、漏手/多手），返回报告 dict。"""
    from difflib import SequenceMatcher
    gt_seq = [(num, 'b' if color == 1 else 'w', (r, c))
              for num, (r, c, color) in sorted(gt.items())]
    sgf_seq = [(color, tuple(pos)) for color, pos in
               [(c, p) for c, p in moves if c != 'resync']]
    n_resync = sum(1 for c, _ in moves if c == 'resync')
    a = [(color, pos) for _, color, pos in gt_seq]
    sm = SequenceMatcher(a=a, b=sgf_seq, autojunk=False)
    matched = 0
    diffs = []
    ins_counter = Counter()
    del_counter = Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            matched += i2 - i1
            continue
        for i in range(i1, i2):
            del_counter[(gt_seq[i][1], gt_seq[i][2])] += 1
        for j in range(j1, j2):
            ins_counter[sgf_seq[j]] += 1
        diffs.append({'op': tag,
                      'gt': [{'num': gt_seq[i][0], 'color': gt_seq[i][1], 'pos': gt_seq[i][2]}
                             for i in range(i1, min(i2, i1 + 6))],
                      'sgf': [{'color': sgf_seq[j][0], 'pos': sgf_seq[j][1]}
                              for j in range(j1, min(j2, j1 + 6))],
                      'gt_span': [i1 + 1, i2], 'sgf_span': [j1 + 1, j2]})
    # 同子对（GT 缺号致对齐错位的 insert+delete）不算真实差异
    pairs = ins_counter & del_counter
    n_pairs = sum(pairs.values())
    unmatched_sgf = list((ins_counter - pairs).items())
    unmatched_gt = list((del_counter - pairs).items())
    return {'gt_moves': len(gt_seq), 'sgf_moves': len(sgf_seq),
            'matched': matched, 'n_resync': n_resync,
            'aligned_pairs': n_pairs,
            'effective_matched': matched + n_pairs,
            'unmatched_sgf': unmatched_sgf,
            'unmatched_gt': unmatched_gt,
            'first_diff': diffs[0] if diffs else None,
            'diffs': diffs[:40]}


def main():
    out_dir = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'default'
    segs = json.load(open(out_dir / 'segments_data.json'))
    gt, conflicts = build_gt(segs)
    max_num = max(gt) if gt else 0
    missing = [n for n in range(1, max_num + 1) if n not in gt]
    print(f'GT: {len(gt)} 手（1..{max_num}），缺号 {len(missing)} 个')
    # 复盘分支歧义：同一手数在不同段位置不同 → 从 GT 剔除（不计入准确率）
    gt_clean = {n: v for n, v in gt.items() if n not in conflicts}
    print(f'分支冲突剔除 {len(conflicts)} 手，干净 GT {len(gt_clean)} 手')

    from gorec.video import video_to_sgf
    meta = json.load(open(out_dir / 'segments.json'))
    video = meta['video']
    moves_cache = out_dir / f'moves_{mode}.json'
    sgf_path = out_dir / f'pipeline_{mode}.sgf'
    if moves_cache.exists():
        moves = [(c, tuple(p) if p else None) for c, p in json.load(open(moves_cache))]
        stats = {'cached': True}
    elif sgf_path.exists():
        # 复用已生成的 SGF（避免重复跑视频）
        import re
        moves = []
        for m in re.finditer(r';([BW])\[([a-z]{2})\]|;C\[RESYNC', sgf_path.read_text()):
            if m.group(0).startswith(';C'):
                moves.append(('resync', None))
            else:
                col = ord(m.group(2)[0]) - ord('a')
                row = ord(m.group(2)[1]) - ord('a')
                moves.append((m.group(1).lower(), (row, col)))
        stats = {'cached_sgf': True}
        with open(moves_cache, 'w') as f:
            json.dump([(c, list(p) if p else None) for c, p in moves], f)
    else:
        moves, stats = video_to_sgf(video, str(out_dir / f'pipeline_{mode}.sgf'),
                                    precise=(mode == 'precise'), verbose=False)
        with open(moves_cache, 'w') as f:
            json.dump([(c, list(p) if p else None) for c, p in moves], f)
    print(f'管线: {len(moves)} 条目 stats={stats}')
    rep = compare(gt_clean, moves)
    rep['video'] = video
    rep['mode'] = mode
    rep['missing_gt_nums'] = missing
    with open(out_dir / f'sgf_compare_{mode}.json', 'w') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(f"逐手一致 {rep['matched']}/{rep['gt_moves']}"
          f"（SGF {rep['sgf_moves']} 手，resync {rep['n_resync']}）")
    print(f"同子错位对 {rep['aligned_pairs']}（GT 缺号所致，不算错误）→ "
          f"有效一致 {rep['effective_matched']}")
    print(f"SGF 多出（GT 缺号或假手）: {rep['unmatched_sgf']}")
    print(f"GT 漏失（管线未记录）: {rep['unmatched_gt']}")
    if rep['first_diff']:
        print('首个分歧:', rep['first_diff'])
        for d in rep['diffs'][:12]:
            print('  ', d)
    else:
        print('全部一致 ✅')


if __name__ == '__main__':
    main()
