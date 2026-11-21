# -*- coding: utf-8 -*-
"""围棋规则引擎：棋盘状态、落子、提子、合法性校验。

坐标约定：pos = (row, col)，row 0 在上，col 0 在左（与 SGF 的 [row][col] 一致）。
player: 'b' 黑 / 'w' 白。
"""
from collections import deque

EMPTY, BLACK, WHITE = 0, 1, 2
CHAR = {EMPTY: '.', BLACK: 'x', WHITE: 'o'}


class IllegalMoveError(Exception):
    pass


class Board:
    """维护当前局面，支持 apply_move / is_legal / diff 对比。

    superko: 可选的禁全局同形模式。
      None      仅单劫禁着（默认，与主流客户端一致的行为）
      'positional'  落子后不得与任意历史局面相同（中国规则）
      'situational' 落子后不得重复"自己走完后的局面"
    历史局面以 (color, tuple(flat_grid)) 记录；Board 不会自动记录，
    由上层在确知起点后调用 start_history() 初始化。
    """

    def __init__(self, size=19, superko=None):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.captures = {BLACK: 0, WHITE: 0}
        self.ko_point = None  # 单纯劫：被提单子后，对方不能立即回提的那一点
        self.superko = superko
        self._history = None  # set of (color, flat_bytes)

    def start_history(self):
        """以当前局面为起点开始记录历史（供 superko 校验）。"""
        self._history = {self._pos_key(None)}

    def _pos_key(self, color):
        flat = bytes(v for row in self.grid for v in row)
        return (color, flat) if color is not None else flat

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                yield nr, nc

    def group_and_liberties(self, r, c):
        """返回 (石头集合, 气数)。grid[r][c] 必须非空。"""
        color = self.grid[r][c]
        stones = {(r, c)}
        libs = set()
        q = deque([(r, c)])
        while q:
            cr, cc = q.popleft()
            for nr, nc in self.neighbors(cr, cc):
                v = self.grid[nr][nc]
                if v == EMPTY:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in stones:
                    stones.add((nr, nc))
                    q.append((nr, nc))
        return stones, libs

    def _place(self, r, c, color):
        self.grid[r][c] = color

    def _remove(self, stones):
        for r, c in stones:
            self.grid[r][c] = EMPTY

    def apply_move(self, color, pos):
        """落子。color 接受 'b'/'w' 或 BLACK/WHITE。不合法时抛 IllegalMoveError，棋盘状态不变。"""
        if isinstance(color, str):
            color = BLACK if color.lower().startswith('b') else WHITE
        else:
            color = int(color)  # numpy 标量统一转 int（captures dict 的 key 是 Python int）
        r, c = pos
        if not self.in_bounds(r, c):
            raise IllegalMoveError(f"落点越界: {pos}")
        if self.grid[r][c] != EMPTY:
            raise IllegalMoveError(f"落点已有子: {pos}")
        if self.ko_point == pos:
            raise IllegalMoveError(f"劫争禁着点: {pos}")

        opponent = WHITE if color == BLACK else BLACK
        self._place(r, c, color)

        # 先提对方无气棋块
        captured = set()
        for nr, nc in self.neighbors(r, c):
            if self.grid[nr][nc] == opponent:
                stones, libs = self.group_and_liberties(nr, nc)
                if not libs:
                    captured |= stones
        self._remove(captured)
        self.captures[color] += len(captured)

        # 自杀手不合法
        _, own_libs = self.group_and_liberties(r, c)
        if not own_libs:
            self._remove({(r, c)})
            raise IllegalMoveError(f"自杀手: {pos}")

        # 劫：恰提一子且自身为单子且该点仅剩这一口气
        self.ko_point = None
        if len(captured) == 1 and len(self.group_and_liberties(r, c)[0]) == 1:
            cap_r, cap_c = next(iter(captured))
            _, cap_libs_after = self._libs_if_removed((r, c), cap_r, cap_c)
            if len(cap_libs_after) == 1:
                self.ko_point = (cap_r, cap_c)

        # superko：落子后的局面不得与历史重复。
        # 单劫禁着已覆盖最常见的重复；superko 拦截三劫/循环劫等
        # 全局同形（中国规则为 positional：任何一方落子后不得重现
        # 任意历史局面）。
        if self._history is not None:
            key = self._pos_key(color)
            if key in self._history:
                # 回滚本次落子（上面已 apply，此处撤销）
                self._remove({(r, c)})
                for cr, cc in captured:
                    self.grid[cr][cc] = opponent
                self.captures[color] -= len(captured)
                raise IllegalMoveError(f"禁全局同形(superko): {pos}")
            self._history.add(key)
        return captured

    def _libs_if_removed(self, keep, remove_r, remove_c):
        """临时移除 (remove_r, remove_c) 处的子，返回 keep 点所属棋块的气。"""
        saved = self.grid[remove_r][remove_c]
        self.grid[remove_r][remove_c] = EMPTY
        stones, libs = self.group_and_liberties(*keep)
        self.grid[remove_r][remove_c] = saved
        return stones, libs

    def is_legal(self, color, pos):
        trial = self.copy()
        try:
            trial.apply_move(color, pos)
            return True
        except IllegalMoveError:
            return False

    def undo_move(self, color, pos, captured):
        """回滚一次 apply_move（需传入其返回的 captured 集合）。"""
        if isinstance(color, str):
            color = BLACK if color.lower().startswith('b') else WHITE
        else:
            color = int(color)
        r, c = pos
        # 历史弹出须在撤子前：key 是落子后的局面
        if self._history is not None:
            self._history.discard(self._pos_key(color))
        self._remove({(r, c)})
        opponent = WHITE if color == BLACK else BLACK
        for cr, cc in captured:
            self.grid[cr][cc] = opponent
        self.captures[color] -= len(captured)
        self.ko_point = None  # 简化：回滚后劫点重算交给上层，实时场景不依赖深度回滚

    def to_positions(self):
        """返回 {(row,col): BLACK/WHITE}，跳过空点。"""
        out = {}
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] != EMPTY:
                    out[(r, c)] = self.grid[r][c]
        return out

    def diff(self, other_grid):
        """与另一 grid 对比，返回 (added, removed) 位置集合。numpy 向量化。"""
        import numpy as np
        a = np.asarray(self.grid)
        b = np.asarray(other_grid)
        mask = a != b
        added = set(map(tuple, np.argwhere(mask & (b != EMPTY)).tolist()))
        removed = set(map(tuple, np.argwhere(mask & (b == EMPTY)).tolist()))
        return added, removed

    def __str__(self):
        return '\n'.join(''.join(CHAR[v] for v in row) for row in self.grid)

    def copy(self):
        b = Board(self.size, superko=self.superko)
        b.grid = [row[:] for row in self.grid]
        b.captures = dict(self.captures)
        b.ko_point = self.ko_point
        if self._history is not None:
            b._history = set(self._history)
        return b


class TransitionChecker:
    """校验两个局面之间是否为合法的一手变化（落一子 + 可能的提子）。"""

    def __init__(self, size=19):
        self.size = size

    def check(self, prev_grid, next_grid, color_hint=None):
        """prev/next 均为 size x size 的 0/1/2 数组。

        返回 dict：
          ok            是否为合法单手变化
          reason        不合法原因（ok=False 时）
          color/player  落子方
          move          落点 (row,col)
          captured      被提子位置集合
        """
        prev = Board(self.size)
        prev.grid = [list(row) for row in prev_grid]
        added, removed = prev.diff(next_grid)
        removed = removed - added

        if not added:
            return {'ok': False, 'reason': 'no_stone_added'}
        if len(added) > 1:
            return {'ok': False, 'reason': 'multi_stone_added', 'added': added}

        pos = next(iter(added))
        # 判断落子方：落点在 next 中必为该方颜色
        color = next_grid[pos[0]][pos[1]]
        if color == EMPTY:
            return {'ok': False, 'reason': 'added_point_empty_in_next'}

        captured = removed
        # 模拟落子验证提子位置完全一致
        sim = prev.copy()
        try:
            sim_captured = sim.apply_move(color, pos)
        except IllegalMoveError as e:
            return {'ok': False, 'reason': f'illegal: {e}'}
        if sim_captured != captured:
            return {'ok': False, 'reason': 'capture_mismatch',
                    'expected': sim_captured, 'got': captured}
        return {'ok': True, 'color': color, 'move': pos, 'captured': sim_captured}
