# -*- coding: utf-8 -*-
"""识别结果 → 合法棋谱：把逐帧的 19x19 标签图流变成 SGF。

核心思路（"100%" 的来源）：
1. 相邻两个被接受的局面之间必须恰好是"落一子+提子"的合法变化
   （TransitionChecker 校验），不合法的帧被丢弃重试；
2. 低置信度点触发重采样/标记人工复核，不硬猜；
3. 输出流式 SGF，支持 pass。
"""
import numpy as np

from .rules import Board, TransitionChecker, EMPTY, BLACK, WHITE, IllegalMoveError


class BoardTracker:
    """维护"当前可信局面"，吃进候选局面，输出合法变化序列。"""

    def __init__(self, size=19):
        self.size = size
        self.board = Board(size)
        self.checker = TransitionChecker(size)
        self.confirmed = 0  # 已确认的手数
        self.pending = []   # (grid, confidence, source)
        # 严格黑白交替拆解：缩小合法序列空间（讲解变化图/正常对局均严格
        # 交替）；摆谱等同色连放场景应关闭，否则拆解失败走 resync。
        self.strict_alternation = False

    def accept_grid(self, grid, confidence=1.0, min_conf=0.85, point_conf=None,
                    last_move_hint=None):
        """提交一帧识别结果。

        point_conf: 可选的 19x19 每点置信度，用于迁移修复时挑选要剔除的点。
        last_move_hint: 可选的 (row,col)——最后一手标记（三角）所在点。
            多手拆解时该点必须排在最后（外部真值锚点），拆解失败自动回退无锚。

        返回 dict：
          accepted: 是否产生了一手合法棋
          status: 'move' | 'same' | 'rejected' | 'low_conf'
          move/captured/color: 同 TransitionChecker
        """
        grid = np.asarray(grid)
        conf = float(np.mean(confidence)) if np.ndim(confidence) else float(confidence)
        if conf < min_conf:
            return {'accepted': False, 'status': 'low_conf', 'confidence': conf}
        cur = self.board.grid
        # 无变化
        if np.array_equal(np.asarray(cur), grid):
            return {'accepted': False, 'status': 'same'}
        res = self.checker.check(cur, grid)
        if not res.get('ok') and res.get('reason') == 'multi_stone_added':
            # 两类 multi_add：
            # a) 识别噪声（真迁移恰加一子，多余是假子）→ 贪心剔除修复
            # b) 跳帧多手（快速翻谱/低采样率，一次出现 k 手真子）
            #    → 迭代拆解：反复从 added 中找可合法落的单手并应用。
            moves_out = self._extract_multi_moves(cur, grid, res['added'], point_conf,
                                                  last_move_hint=last_move_hint)
            if not moves_out and last_move_hint is not None:
                # 锚点可能有误检：回退无锚拆解，不牺牲可用性
                moves_out = self._extract_multi_moves(cur, grid, res['added'], point_conf)
            if moves_out:
                # 数子/形势判断画面守门：成片提子(死子标记) + 大批量变化
                # 是终局数子环节的特征，不属于对局手序。
                n_captured = sum(len(c) for _, _, c in moves_out)
                if len(moves_out) >= 8 and n_captured >= len(moves_out) * 0.5:
                    return {'accepted': False, 'status': 'rejected',
                            'reason': 'scoring_screen_suspected'}
                last = None
                seq = []
                try:
                    for color, move, captured in moves_out:
                        self.board.apply_move(color, move)
                        self.confirmed += 1
                        last = (color, move, captured)
                        seq.append(('b' if color == BLACK else 'w', move))
                except IllegalMoveError:
                    # 拆解顺序在劫争等约束下不可行 — 本帧交 resync/丢弃
                    return {'accepted': False, 'status': 'rejected',
                            'reason': 'multi_apply_illegal'}
                color, move, captured = last
                return {'accepted': True, 'status': 'move',
                        'color': 'b' if color == BLACK else 'w',
                        'move': move, 'captured': captured,
                        'confidence': conf, 'n_moves': len(moves_out),
                        'moves_seq': seq}
            repaired = self._repair_multi_add(cur, grid, res['added'], point_conf)
            if repaired is not None:
                grid = repaired
                res = self.checker.check(cur, grid)
        if not res.get('ok'):
            return {'accepted': False, 'status': 'rejected', 'reason': res.get('reason'),
                    'detail': res}
        # 落子动画渐显保护：新子置信度不足时挂起，等完全显现的帧再接受。
        # defer 不计入拒绝连击（不触发 resync）。
        if point_conf is not None:
            mv = res['move']
            if float(np.asarray(point_conf)[mv[0], mv[1]]) < 0.80:
                return {'accepted': False, 'status': 'defer',
                        'reason': 'faint_stone'}
        try:
            self.board.apply_move(res['color'], res['move'])
        except IllegalMoveError:
            # 校验通过但应用失败（劫点状态差异等边缘情况）——丢弃该帧
            return {'accepted': False, 'status': 'rejected', 'reason': 'apply_failed'}
        self.confirmed += 1
        return {'accepted': True, 'status': 'move',
                'color': 'b' if res['color'] == BLACK else 'w',
                'move': res['move'], 'captured': res['captured'],
                'confidence': conf}

    def _extract_multi_moves(self, cur, grid, added, point_conf, max_moves=30,
                             last_move_hint=None):
        """从多手帧中迭代拆解出合法单手序列。

        每轮：对 added 中每个候选点 p（其余 added 视为"已出现但未处理"——
        用 cur+p 的局面校验单手迁移），找 check 合法者应用之（真实应用到
        self 副本），直到 added 耗尽或一轮无进展。
        注意候选校验用"增量局面"：正确性由 apply_move 的提子一致性保证。
        last_move_hint: 最后一手标记点，必须留在最后拆（锚定外部真值）。
        返回 [(color, move, captured), ...]，失败返回 []。
        """
        import numpy as np
        sim = Board(self.size)
        sim.grid = [list(row) for row in cur]
        remaining = set(map(tuple, added))
        anchor = None
        if last_move_hint is not None:
            hint = tuple(last_move_hint)
            if hint in remaining:
                anchor = hint
        out = []
        # 交替颜色约束：对局黑白交替（黑先）。拆解时优先落"下一手应有
        # 颜色"的点，避免拆出黑黑白白的非法顺序。下一手颜色 = 上一手反色
        # （初始黑先；若 sim 已有局面则按 confirmed 奇偶推）。
        n_prior = self.confirmed
        next_color = BLACK if n_prior % 2 == 0 else WHITE
        for _ in range(max_moves):
            progressed = False
            for p in sorted(remaining):
                if anchor is not None and p == anchor and len(remaining) > 1:
                    continue  # 锚点（最后一手标记）必须排在最后
                r, c = p
                color = grid[r][c]
                if color == EMPTY:
                    continue
                # 交替约束：先试应有颜色；不符则跳过（本轮其他点可能符合）
                if color != next_color:
                    if self.strict_alternation:
                        continue
                    if any(grid[qr][qc] == next_color for qr, qc in remaining):
                        continue
                trial = np.array(sim.grid, copy=True)
                trial[r][c] = color
                # trial 之外, remaining 的其他点视为"暂不存在"
                res = self.checker.check(sim.grid, trial)
                if res.get('ok'):
                    try:
                        sim.apply_move(res['color'], res['move'])
                    except IllegalMoveError:
                        # 校验通过但劫点状态差异导致不可应用 — 跳过该点
                        continue
                    remaining.discard(p)
                    out.append((res['color'], res['move'], res['captured']))
                    next_color = WHITE if color == BLACK else BLACK
                    progressed = True
                    break
            if not progressed or not remaining:
                break
        # 全部拆解完才算成功；剩余点视为噪声交上层 repair。
        # 另须验证拆解结果与目标局面一致（removed 死子须被真实提掉）。
        if not remaining and out:
            if np.array_equal(np.asarray(sim.grid), np.asarray(grid)):
                return out
        return []

    def _repair_multi_add(self, cur, grid, added, point_conf):
        """added 多于一子时，尝试剔除低置信点使迁移合法。

        贪心策略（O(k) 次 check）：按置信度升序逐个剔除，每剔一个立即验证；
        全部剔除仍失败再尝试"仅保留最高置信的一个"（单子迁移兜底）。
        指数级 combinations 在 added>10 时不可行（录屏手数标记帧可达 18 个
        假 added，曾单帧 59 秒）。
        返回修复后的 grid，失败返回 None。
        """
        if point_conf is not None:
            added = sorted(added,
                           key=lambda rc: float(np.asarray(point_conf)[rc[0], rc[1]]))
        else:
            added = sorted(added)
        trial = np.array(grid, copy=True)
        # 逐个剔除（累积）
        for r, c in added:
            trial[r, c] = EMPTY
            res = self.checker.check(cur, trial)
            if res.get('ok'):
                return trial
        # 累积剔除失败：只保留置信度最高的一个点
        keep = added[-1] if added else None
        if keep is not None and point_conf is not None:
            for cand in reversed(added):
                trial = np.array(grid, copy=True)
                for r, c in added:
                    if (r, c) != cand:
                        trial[r, c] = EMPTY
                res = self.checker.check(cur, trial)
                if res.get('ok'):
                    return trial
        return None

    def board_grid(self):
        return [row[:] for row in self.board.grid]


class BoardStream:
    """带历史的流式棋谱构建器。"""

    def __init__(self, size=19, resync_after=4):
        self.size = size
        self.tracker = BoardTracker(size)
        self.moves = []  # [('b'|'w', (r,c) | None)]
        self.resync_after = resync_after  # 连续 N 次拒绝后重同步（讲解视频）
        self._reject_streak = 0
        self.resyncs = []  # [(帧内手数, 局面)]，局面跳变点
        self.rewinds = []  # 回退翻谱：每次撤销的手数 k

    def accept_grid(self, grid, confidence=1.0, min_conf=0.85, point_conf=None,
                    last_move_hint=None):
        res = self.tracker.accept_grid(grid, confidence, min_conf, point_conf=point_conf,
                                       last_move_hint=last_move_hint)
        if res.get('accepted'):
            self._reject_streak = 0
            seq = res.get('moves_seq')
            if seq:
                self.moves.extend(seq)
            else:
                self.moves.append((res['color'], res['move']))
        elif res['status'] == 'rejected':
            # 回退翻谱优先于 resync：用户往回翻时局面恰好是"撤销尾部 k 手"，
            # 截尾即可保住此前全部手序，而不是重置丢手。
            if self._try_rewind(grid):
                self._reject_streak = 0
                return {'accepted': False, 'status': 'rewind'}
            self._reject_streak += 1
            # 讲解视频（复盘摆谱）常出现无法拆解的局面跳变。
            # 连续多次拒绝后，认为 tracker 已过时，重置到当前识别局面。
            if self.resync_after and self._reject_streak >= self.resync_after:
                # 重同步用 pass 表达（SGF 无清盘语义）：后续手序在播发器中
                # 显示为新局面起点。resyncs 记录跳变信息供上层输出。
                self.moves.append(('resync', None))
                self.resyncs.append((len(self.moves), len(grid[grid != 0])))
                self.tracker.board = Board(self.size, superko=self.tracker.board.superko)
                self.tracker.board.grid = [list(row) for row in grid]
                if self.tracker.board.superko is not None:
                    self.tracker.board.start_history()
                self.tracker.board.ko_point = None
                self._reject_streak = 0
        return res

    def _try_rewind(self, grid):
        """检测回退翻谱：cur 与 grid 的差异恰为"撤销尾部连续 k 手"。

        cur 比 grid 多出最近 k 手的落子（removed == 尾部 k 手落点集合），
        被提子随之恢复（added 不约束）。命中则 moves 截尾、局面直接同步，
        返回 True；否则返回 False 走原 resync。
        正常对局的提子：removed 是被提的旧子而非"尾部 k 手"落点，不会误判。
        """
        cur_board = Board(self.size)
        cur_board.grid = [list(row) for row in self.tracker.board.grid]
        added, removed = cur_board.diff(grid)
        removed = removed - added
        if not removed:
            return False
        tail = []
        for color, pos in reversed(self.moves):
            if color == 'resync' or pos is None:
                break
            tail.append(pos)
        tail_set = set()
        for k, pos in enumerate(tail, 1):
            tail_set.add(pos)
            if tail_set == removed:
                for _ in range(k):
                    self.moves.pop()
                self.rewinds.append(k)
                self.tracker.board = Board(self.size, superko=self.tracker.board.superko)
                self.tracker.board.grid = [list(row) for row in grid]
                if self.tracker.board.superko is not None:
                    self.tracker.board.start_history()
                self.tracker.board.ko_point = None
                self.tracker.confirmed = max(0, self.tracker.confirmed - k)
                return True
        return False

    def to_sgf(self, komi=7.5, event=None, black_player=None, white_player=None):
        def pt(pos):
            if pos is None:
                return ''
            r, c = pos
            return chr(ord('a') + c) + chr(ord('a') + r)
        head = [f'(;GM[1]FF[4]SZ[{self.size}]KM[{komi}]']
        if event:
            head.append(f'EV[{event}]')
        if black_player:
            head.append(f'PB[{black_player}]')
        if white_player:
            head.append(f'PW[{white_player}]')
        body = []
        n = 0
        for color, pos in self.moves:
            if color == 'resync':
                body.append(f';C[RESYNC: 局面重置（讲解视频摆谱跳变），此后局面为新起点]')
                continue
            n += 1
            body.append(f';{color.upper()}[{pt(pos)}]C[#{n}]')
        return ''.join(head) + ''.join(body) + ')'
