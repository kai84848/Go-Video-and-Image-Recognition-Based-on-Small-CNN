# -*- coding: utf-8 -*-
"""数据集生成：从随机对局/局面渲染带标注的训练数据。

每条样本 = 矫正后的棋盘图 + 每交叉点标签(0空/1黑/2白)。
直接在 canonical 空间渲染（模拟已矫正的输入），并做亮度/对比度/
JPEG/噪点扰动。棋盘定位训练数据由 renderer + geometry 独立生成。
"""
import numpy as np
import cv2

from .rules import Board, BLACK, WHITE, EMPTY
from .renderer import SKINS, _draw_stone, _draw_grid

CELL = 32


def random_position(size=19, moves=120, rng=None):
    """随机走出一盘真实合法的棋（用规则引擎），返回 19x19 标签图。"""
    if rng is None:
        rng = np.random.default_rng()
    b = Board(size)
    empties = [(r, c) for r in range(size) for c in range(size)]
    passed = 0
    color = 'b'
    for _ in range(moves):
        if not empties:
            break
        i = rng.integers(len(empties))
        pos = empties[i]
        if b.is_legal(color, pos):
            b.apply_move(color, pos)
            empties.pop(i)
            passed = 0
            color = 'w' if color == 'b' else 'b'
        else:
            empties.pop(i)
            if not empties:
                break
    return [row[:] for row in b.grid]


def render_canonical(labels, skin_idx=None, seed=None, last_move=None,
                     move_number=None, last_marker='circle', cursor=None):
    """在 canonical 空间（576x576，交叉点在 c*CELL）渲染棋盘。"""
    rng = np.random.default_rng(seed)
    size = len(labels)
    if skin_idx is None:
        skin_idx = int(rng.integers(len(SKINS)))
    skin = SKINS[skin_idx]
    m = CELL // 2  # 边距半格：边缘交叉点外侧有真实底色
    w = CELL * size + 2 * m
    img = np.full((w, w, 3), skin['bg'], dtype=np.uint8)

    # 木纹
    if rng.random() < 0.6:
        yy = np.arange(w, dtype=np.float32).reshape(-1, 1)
        bands = np.sin(yy * rng.uniform(0.05, 0.15)) * rng.uniform(3, 8)
        img = np.clip(img.astype(np.float32) + np.repeat(bands, w, axis=1)[..., None],
                      0, 255).astype(np.uint8)

    corners = {'tl': (m, m), 'tr': (m + CELL * (size - 1), m),
               'br': (m + CELL * (size - 1), m + CELL * (size - 1)),
               'bl': (m, m + CELL * (size - 1))}
    lw = max(1, int(round(CELL * 0.006 * (640 / 32 / 0.006 * 0.006))))
    lw = max(1, int(round(CELL * 0.075)))  # CELL=32 时线宽约 2px
    _draw_grid(img, corners, CELL, size, skin['line'], lw, skin['line'],
               max(2, int(CELL * 0.14)))

    radius = CELL * skin['stone_pad'] / 2
    black_stone = (35, 35, 38)
    white_stone = (242, 242, 240)
    for r in range(size):
        for c in range(size):
            v = labels[r][c]
            if v == EMPTY:
                continue
            col = black_stone if v == BLACK else white_stone
            img = _draw_stone(img, m + c * CELL, m + r * CELL, radius, col)

    if last_move is not None:
        r, c = last_move
        x, y = m + c * CELL, m + r * CELL
        v = labels[r][c]
        marker_col = (255, 255, 255) if v == BLACK else (0, 0, 0)
        if move_number is not None:
            cv2.putText(img, str(move_number), (int(x - CELL * 0.3), int(y + CELL * 0.24)),
                        cv2.FONT_HERSHEY_SIMPLEX, CELL / 44, marker_col,
                        max(1, int(CELL * 0.055)), cv2.LINE_AA)
        elif last_marker == 'triangle':
            pts = np.array([[x, y - radius * 0.75], [x - radius * 0.72, y + radius * 0.55],
                            [x + radius * 0.72, y + radius * 0.55]], np.int32)
            cv2.polylines(img, [pts], True, marker_col, max(1, int(CELL * 0.05)), cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), int(radius * 0.55), marker_col,
                       max(1, int(CELL * 0.05)), cv2.LINE_AA)

    if cursor is not None:
        from .renderer import _draw_pointer
        r, c = cursor
        x, y = m + c * CELL, m + r * CELL
        _draw_pointer(img, x, y, CELL, rng)
    return img


def augment(img, seed=None, jpeg=True):
    """亮度/对比度/gamma/噪点/模糊/JPEG 扰动（覆盖真实截图的光照范围）。"""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    alpha = rng.uniform(0.6, 1.4)    # 对比度
    beta = rng.uniform(-45, 45)      # 亮度
    out = out * alpha + beta
    out = np.clip(out, 0, 255)
    if rng.random() < 0.5:           # gamma：屏幕亮度差异的主项
        g = rng.uniform(0.6, 1.45)
        out = 255.0 * (out / 255.0) ** g
    # 防过曝：白子区域应保留纹理。若 >60% 像素 >245 则整体压暗
    frac_bright = float((out > 245).mean())
    if frac_bright > 0.6:
        out = out * (0.92 / max(frac_bright, 0.61))
    if rng.random() < 0.4:
        noise = rng.normal(0, rng.uniform(2, 9), out.shape)
        out = out + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if jpeg and rng.random() < 0.7:
        q = int(rng.integers(45, 92))
        ok, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def make_sample(size=19, seed=None, with_marker=True):
    """生成一条训练样本：返回 (img, labels, meta)。

    img 是 canonical 棋盘图外加了随机边框带（模拟真实截图的框边），
    但交叉点仍按 c*CELL 对齐 —— 提取 patch 时以坐标直接取样即可。
    """
    rng = np.random.default_rng(seed)
    labels = random_position(size, moves=int(rng.integers(10, 300)), rng=rng)
    stones = [(r, c) for r in range(size) for c in range(size) if labels[r][c] != EMPTY]
    last_move = None
    move_number = None
    cursor = None
    if with_marker and stones:
        i = int(rng.integers(len(stones)))
        last_move = stones[i]
        if rng.random() < 0.4:
            move_number = int(rng.integers(1, 300))
    if rng.random() < 0.15:
        # 光标优先悬停在棋子上（真实场景：用户指着刚下的子），
        # 也允许空点。出现率保持低位（15%）：过高会把"被遮挡"分布
        # 学得过重，反而让干净棋子的边界变模糊（实测 40% 时录屏
        # 端到端 rejected 0→28，识别退化）。
        if stones and rng.random() < 0.6:
            cursor = stones[int(rng.integers(len(stones)))][:2]
        else:
            cursor = (int(rng.integers(19)), int(rng.integers(19)))
    img = render_canonical(labels, seed=seed, last_move=last_move,
                           move_number=move_number, cursor=cursor)
    # 边框带：真实截图棋盘外围有 0~1 格宽的框边。
    # 交叉点坐标从 (bw, bw) 开始 —— meta 里带出偏移。
    bw = int(rng.integers(0, CELL))  # 0~31px
    if bw > 0:
        # 用纯底色填充（真实截图的框边是均匀木色/深色底），不含棋子
        base = img[2, 2].astype(np.uint8).copy()
        band_h = np.repeat(base.reshape(1, 1, 3), bw, axis=0)
        band_h = np.repeat(band_h, img.shape[1], axis=1)
        img = np.concatenate([band_h, img, band_h], axis=0)
        band_w = np.repeat(base.reshape(1, 1, 3), bw, axis=1)
        band_w = np.repeat(band_w, img.shape[0], axis=0)
        img = np.concatenate([band_w, img, band_w], axis=1)
    img = augment(img, seed=None if seed is None else seed + 1)
    return img, labels, {'last_move': last_move, 'cursor': cursor, 'border': bw}
