# -*- coding: utf-8 -*-
"""最小示例：视频 → SGF（含统计信息）。

在仓库根目录运行：
  python examples/video_to_sgf.py game.mp4 out.sgf            # 默认 2fps（推荐）
  python examples/video_to_sgf.py game.mp4 out.sgf --precise  # 逐帧精确模式
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gorec.video import video_to_sgf


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    video_path = sys.argv[1]
    out_sgf = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'out.sgf'
    precise = '--precise' in sys.argv
    moves, stats = video_to_sgf(video_path, out_sgf, precise=precise, verbose=False)
    n_resync = sum(1 for color, _ in moves if color == 'resync')
    print(f'完成: {len(moves)} 条目（含 resync {n_resync}）')
    print('统计:', stats)
    if stats.get('multi_jump', 0) >= 3:
        print('提示: multi_jump 较多，若关心逐手顺序可尝试 --precise')
    print('SGF 已保存:', out_sgf)


if __name__ == '__main__':
    main()
