# -*- coding: utf-8 -*-
"""手数驱动 SGF：客户端显示手数的录屏 → 直接读数字生成棋谱。

绕开多手拆解与 resync：数字 n 的子 = 第 n 手（真值）。
分支（复盘试变化）按 last_seen 策略合并：每个手数取最后出现的位置。
自洽验证：setup + 数字顺序规则重放，报告非法手与交替异常。

用法：.venv/bin/python gt/rec_to_sgf.py gt/out/rec1_fine [out.sgf]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gorec.rules import Board, IllegalMoveError

SGF_COL = 'abcdefghijklmnopqrstuvwxyz'


def build_moves_last_seen(segments):
    """{num: (r,c,color)}：位置按段帧数加权投票（抗单帧 OCR 误读），
    颜色由手数奇偶决定（num1 的 CNN 颜色投票定先手，之后严格交替）。"""
    from collections import Counter
    pos_votes = defaultdict(Counter)
    color_votes = defaultdict(Counter)
    for s in segments:
        for d in s['digits']:
            if d['num'] <= 0:
                continue
            w = max(1, s['n_frames'])
            pos_votes[d['num']][(d['r'], d['c'])] += w
            color_votes[d['num']][d['color']] += w
    first_color = 1
    if 1 in color_votes:
        first_color = color_votes[1].most_common(1)[0][0]
    seen = {}
    for num, counter in pos_votes.items():
        (r, c), _ = counter.most_common(1)[0]
        color = first_color if (num - 1) % 2 == 0 else (3 - first_color)
        seen[num] = (r, c, color, 0.0)
    return seen


def detect_setup(segments):
    """让子/初始摆子显示为 0：只取"数字 1 出现之前"就存在的 0 子。
    对局中被提的死子也显示 0，但其首次出现在数字 1 之后，须排除。"""
    from collections import Counter
    t_first = min((s['t0'] for s in segments if any(d['num'] == 1 for d in s['digits'])),
                  default=0.0)
    votes = defaultdict(Counter)
    first_seen = {}
    for s in segments:
        for d in s['digits']:
            if d['num'] != 0:
                continue
            key = (d['r'], d['c'])
            first_seen[key] = min(first_seen.get(key, 1e18), s['t0'])
            votes[key][d['color']] += max(1, s['n_frames'])
    ab = sorted(p for p, cnt in votes.items()
                if cnt.most_common(1)[0][0] == 1 and first_seen[p] <= t_first)
    aw = sorted(p for p, cnt in votes.items()
                if cnt.most_common(1)[0][0] == 2 and first_seen[p] <= t_first)
    return ab, aw


def main():
    out_dir = Path(sys.argv[1])
    out_sgf = sys.argv[2] if len(sys.argv) > 2 else str(out_dir / 'number_driven.sgf')
    segs = json.load(open(out_dir / 'segments_data.json'))
    seen = build_moves_last_seen(segs)
    max_num = max(seen) if seen else 0
    missing = [n for n in range(1, max_num + 1) if n not in seen]
    ab, aw = detect_setup(segs)

    # 自洽验证：setup + 数字顺序重放
    board = Board(19)
    for r, c in ab:
        board.grid[r][c] = 1
    for r, c in aw:
        board.grid[r][c] = 2
    illegal = []
    alternation = []
    prev_color = None
    color_fixed = []
    first_color = seen[1][2] if 1 in seen else 1
    for num in sorted(seen):
        r, c, color, _ = seen[num]
        # 颜色奇偶校验：全局手数决定应手颜色（setup 后 num1 定先手），
        # CNN 对带数字子的颜色误判可按奇偶修正
        expect = first_color if (num - 1) % 2 == 0 else (3 - first_color)
        if color != expect:
            color_fixed.append((num, color, expect))
            color = expect
            seen[num] = (r, c, expect, seen[num][3])
        ch = 'b' if color == 1 else 'w'
        if prev_color == ch:
            alternation.append(num)
        prev_color = ch
        try:
            board.apply_move(color, (r, c))
        except IllegalMoveError as e:
            illegal.append((num, str(e)))

    def pt(pos):
        r, c = pos
        return SGF_COL[c] + SGF_COL[r]

    parts = ['(;GM[1]FF[4]SZ[19]KM[7.5]']
    if ab:
        parts.append('AB' + ''.join(f'[{pt(p)}]' for p in ab))
    if aw:
        parts.append('AW' + ''.join(f'[{pt(p)}]' for p in aw))
    parts.append('C[手数驱动模式：客户端手数数字直接生成，未经多手拆解]')
    for num in sorted(seen):
        r, c, color, _ = seen[num]
        ch = 'B' if color == 1 else 'W'
        parts.append(f';{ch}[{pt((r, c))}]C[#{num}]')
    parts.append(')')
    with open(out_sgf, 'w') as f:
        f.write(''.join(parts))

    print(f'手数驱动 SGF: {len(seen)} 手（1..{max_num}），缺号 {len(missing)} 个 {missing[:15]}')
    print(f'setup: 黑 {len(ab)} 白 {len(aw)}')
    print(f'颜色奇偶修正 {len(color_fixed)} 处: {[n for n, _, _ in color_fixed][:15]}')
    print(f'同色连手（分支/OCR 异常）: {alternation[:15]}')
    print(f'重放非法手: {illegal[:15]}')
    print(f'输出: {out_sgf}')


if __name__ == '__main__':
    main()
