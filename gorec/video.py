# -*- coding: utf-8 -*-
"""视频流式识别：录屏/直播视频 → 合法棋谱 SGF。

变化检测（低成本）触发完整识别；规则校验兜底。
用法：
  python -c "from gorec.video import video_to_sgf; \
             video_to_sgf('数据集/video/录屏xxx.mov', 'out.sgf')"
"""
import cv2
import numpy as np

from . import geometry
from . import marker
from .model import predict_board
from .pipeline import Pipeline


# 采样点坐标（canonical 交叉点，全 361 点）。
# 曾只用偶数行×偶数列（100 点，1/4 覆盖）：奇数行列上的落子/提子对
# 触发器完全不可见，变化积压到某个可见点变化时一次"补账"多手 ——
# 实测录屏 86 个新增点中 76% 落在盲区，是"跳步"的根源。
_SAMPLE_XY = [(16 + c * 32, 16 + r * 32)
              for r in range(19) for c in range(19)]


def _board_changed(prev_gray, gray, tol=12):
    """检查 canonical 棋盘区域是否变化（仅采样交叉点邻域，便宜）。

    gray 是 canonical 矫正图：交叉点在 (margin + c*CELL, margin + r*CELL)。
    向量化：一次性取出全部采样点 5x5 邻域。
    """
    if prev_gray is None or prev_gray.shape != gray.shape:
        return True
    g_old = prev_gray.astype(np.int16)
    g_new = gray.astype(np.int16)
    diffs = []
    for x, y in _SAMPLE_XY:
        if y < 2 or y > gray.shape[0] - 3 or x < 2 or x > gray.shape[1] - 3:
            return True
        diffs.append(float(np.abs(g_old[y - 2:y + 3, x - 2:x + 3]
                                  - g_new[y - 2:y + 3, x - 2:x + 3]).mean()))
    # max 判定：落子动画的累积帧 diff ~33+，光标 <30 —— 阈值 30 分离两者。
    return max(diffs) > (tol if tol != 12 else 30)


def _ui_overlay(canonical, min_px=3000):
    """检测深蓝色 UI 覆盖层（客户端提示框），命中则跳过识别。

    客户端"已经是最后一手了"提示框是深蓝底色块，CNN 会把它误认成
    成片黑子（实测一帧 31 颗假黑子 → 拆解失败 → resync 把假局面
    当基准，提示框消失后又触发第二次 resync，尾部手序全部污染）。
    木纹棋盘（偏黄）、黑白棋子、坐标尺都不满足深蓝判据：
    B > R+30 且 B > G+15 且亮度居中。整框约 4 万像素，阈值 3000
    可过滤光标等小面积深蓝元素。
    """
    b = canonical[:, :, 0].astype(np.int16)
    g = canonical[:, :, 1].astype(np.int16)
    r = canonical[:, :, 2].astype(np.int16)
    navy = (b > r + 30) & (b > g + 15) & (b > 60) & (b < 200)
    return int(navy.sum()) >= min_px


class BoardLocator:
    """时序一致的棋盘定位器。

    讲解视频等复合画面单帧定位易锁定部分网格（缺坐标尺）或被 UI 干扰；
    棋盘在镜头切换之间位置固定，因此：
    1. 未锁定时连续 STABLE_N 帧检测窗口互相接近（<2px）才接受；
    2. 锁定后每帧用 reproj 分数验证，跌破阈值（画面切换/遮挡）才重检测；
    3. 重检测的新窗口需连续稳定才替换锁定。

    性能护栏：失锁期间 detect_grid 全量网格检测很贵（复合画面约 1.4s/帧，
    实测 06 号视频终局复盘画面 1000 帧耗 1426s，占全程 94%）。连续失败后
    按 DETECT_EVERY 帧间隔重试一次，其余帧跳过——画面乱跳时识别本就该停。
    """

    STABLE_N = 3
    STABLE_TOL = 2.0    # 帧间窗口一致性（px）
    LOCK_MIN_SCORE = 0.12   # reproj 分数锁定阈（对齐分不足时的后备）
    LOCK_ALIGN = 0.65       # grid_align_score 锁定阈
    REALIGN_SCORE = 0.55    # 锁定维持阈：跌破即重检测
    DETECT_EVERY = 15       # 失锁期间每 N 帧重试一次全量检测

    def __init__(self):
        self.corners = None      # 当前锁定的四角
        self._pending = []       # 连续一致的检测窗口
        self._lock_score = 0.0
        self._last_detect_fi = None  # 上次全量检测的帧号（失锁限流用）

    @staticmethod
    def _close(a, b, tol):
        return all(abs(a[k][0] - b[k][0]) < tol and abs(a[k][1] - b[k][1]) < tol
                   for k in a)

    def locate(self, frame, frame_index=None):
        """返回 (corners, locked)。corners 为 None 表示本帧无法定位。

        frame_index: 可选的真实帧号。传入时失锁限流按帧号差计算
        （连续流语义：失败后 15 帧内不重试）；不传（单帧/跳帧调用）
        则不限流，保持"每个独立帧都可检测"的语义（regression G5）。
        """
        if self.corners is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            s, _ = geometry.grid_align_score(gray, self.corners)
            if s >= self.REALIGN_SCORE:
                self._last_detect_fi = frame_index
                return self.corners, True
            # 锁定失效（镜头切换等）：清空重检测
            self.corners = None
            self._pending = []
        if frame_index is not None and self._last_detect_fi is not None and \
                frame_index - self._last_detect_fi < self.DETECT_EVERY:
            # 失锁限流（仅连续流）：距上次检测尝试不足 N 帧不重试
            return None, False
        self._last_detect_fi = frame_index
        det, info = geometry.detect_grid(frame)
        if det is None:
            self._pending = []
            return None, False
        self._pending = [c for c in self._pending if self._close(c, det, 6.0)]
        self._pending.append(det)
        if len(self._pending) >= self.STABLE_N:
            # 用对齐分决定是否值得锁定（reproj 幅度在大格/低对比场景不可靠）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            a, n_measured = geometry.grid_align_score(gray, det)
            if self.corners is None:
                if (a >= self.LOCK_ALIGN or info >= self.LOCK_MIN_SCORE) and self._pending_stable():
                    self.corners = det
                    self._lock_score = max(a, info)
                    self._pending = []
                    return self.corners, True
            else:
                if self._close(det, self.corners, 6.0) and a >= self.LOCK_ALIGN:
                    self.corners = det
                    self._lock_score = max(a, info)
                self._pending = []
        elif self.corners is not None:
            # 锁定刚失效：重新稳定前不输出（旧窗口已证伪）
            self.corners = None
        return self.corners, self.corners is not None

    def _pending_stable(self):
        return all(self._close(c, self._pending[-1], self.STABLE_TOL)
                   for c in self._pending)


def video_to_sgf(video_path, sgf_out=None, sample_fps=2.0, verbose=True,
                 precise=False, progress=None):
    """离线处理视频：均匀采样 → 变化检测 → 识别 → 规则校验 → SGF。

    precise=True: 逐帧处理（不跳采样），并挂起"半透明落子动画"帧，
    只在棋子完全显现时接受 —— 用于逐手精确记录。
    progress: 可选回调 progress(done_frames, total_frames)。

    返回 (moves, stats)。
    """
    pipe = Pipeline()
    locator = BoardLocator()
    geometry._ANGLE_CACHE.clear()  # 不同视频角度可能不同，避免跨视频缓存污染
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if precise:
        step = 1
        total = n
    else:
        step = max(1, int(round(fps / sample_fps)))
        total = n // step
    stats = {'frames': 0, 'recognized': 0, 'moves': 0, 'rejected': 0,
             'low_conf': 0, 'same': 0, 'relocated': 0, 'deferred': 0}
    prev_gray = None    # 相邻帧（捕捉跳变）
    ref_gray = None     # 参考帧 = 上次触发识别的帧（捕捉渐进动画的累积变化）
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            stats['frames'] += 1
            if progress and fi % (step * 50) == 0:
                progress(fi, n)
            corners, locked = locator.locate(frame, frame_index=fi)
            if corners is not None:
                can = geometry.warp_canonical(frame, corners)
                gray = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY)
                if _ui_overlay(can):
                    # 深蓝 UI 提示框盖住棋盘：识别必产生假子。
                    # 更新 prev_gray 保持触发器连续，等提示框消失再识别
                    # （期间落下的手会在之后由拆解器补上）。
                    prev_gray = gray
                    if precise:
                        ref_gray = gray
                elif precise:
                    # 逐帧模式：相对上一帧的任何变化都触发（阈值调低，
                    # 动画渐显期就持续跟踪），但接受时机由 defer 机制把关
                    trig = _board_changed(prev_gray, gray, tol=6)
                else:
                    # 客户端落子是淡入动画：相邻采样帧 diff 小(~33@2fps)，
                    # 相对参考帧的累积 diff 大。两者任一超阈即触发。
                    trig = (_board_changed(prev_gray, gray)
                            or _board_changed(ref_gray, gray, tol=40))
                if trig:
                    labels, prob = predict_board(pipe.model, can, device=pipe.device)
                    # 最后一手三角标记 → 多手拆解的外部真值锚点。
                    # 检测不到（无标记/多标记异常）则为 None，行为与之前一致。
                    hint = marker.find_last_move(
                        frame, geometry.grid_points_original(corners), labels)
                    res = pipe.stream.accept_grid(labels, prob,
                                                  min_conf=pipe.min_conf,
                                                  point_conf=prob,
                                                  last_move_hint=hint)
                    stats['recognized'] += 1
                    if res.get('n_moves', 0) >= 3:
                        # 一次拆 3+ 手 = 采样不足的快速翻谱段：GT 基线显示
                        # 这类段的内部顺序只能部分保证（锚定下 3-4 手 80%），
                        # 关心逐手顺序时应对本视频改用 precise 逐帧模式。
                        stats['multi_jump'] = stats.get('multi_jump', 0) + 1
                    if res.get('accepted'):
                        ref_gray = gray
                        stats['moves'] += 1
                    elif res['status'] == 'rejected':
                        stats['rejected'] += 1
                    elif res['status'] == 'rewind':
                        stats['rewind'] = stats.get('rewind', 0) + 1
                        ref_gray = gray
                    elif res['status'] == 'defer':
                        stats['deferred'] += 1
                        ref_gray = gray  # 动画跟踪基准也推进
                    elif res['status'] == 'low_conf':
                        stats['low_conf'] += 1
                    elif res['status'] == 'same':
                        stats['same'] += 1
                    elif res['status'] == 'resync' or len(pipe.stream.resyncs) and \
                            stats.get('_last_resyncs', 0) != len(pipe.stream.resyncs):
                        ref_gray = gray
                        stats['_last_resyncs'] = len(pipe.stream.resyncs)
                    else:
                        ref_gray = gray
                prev_gray = gray
        fi += 1
    cap.release()
    stats.pop('_last_resyncs', None)
    if verbose:
        print('stats:', stats)
    if sgf_out:
        with open(sgf_out, 'w') as f:
            f.write(pipe.sgf())
    return pipe.stream.moves, stats
