# -*- coding: utf-8 -*-
"""SGF 读取工具：把 sgfmill 的棋谱变成 (color, pos) 序列。"""
from sgfmill import sgf, boards as sgf_boards


def load_sgf_moves(path):
    """返回 (moves, size)。moves: [(color 'b'/'w', (row,col)), ...]，pos 用 SGF 行列。"""
    with open(path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    size = game.get_size()
    moves = []
    for node in game.get_main_sequence()[1:]:
        colour, move = node.get_move()
        if colour is None:
            continue
        if move is None:  # pass
            moves.append((colour, None))
        else:
            moves.append((colour, move))
    return moves, size


def replay(moves, size=19):
    """按规则重放，返回 (board, error_index)。error_index 为 None 表示全部合法。"""
    b = sgf_boards.Board(size)
    for i, (color, pos) in enumerate(moves):
        if pos is None:
            b.play(-1, -1, color) if hasattr(b, 'play') else None
            continue
        try:
            b.play(pos[0], pos[1], color)
        except ValueError:
            return b, i
    return b, None
