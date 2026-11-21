# -*- coding: utf-8 -*-
"""合成数据渲染器：把任意局面渲染成"客户端截图风格"的训练/测试图片。

设计目标（阶段 1）：
- 多种皮肤风格（模拟野狐/OGS 等不同木纹、棋子、标记、星位）
- 随机扰动：缩放、平移、轻微旋转、透视、JPEG 压缩、亮度对比度、噪点
- 可选：最后一手标记（圆圈/三角/手数数字）、光标遮挡
输出：图片 + 19x19 标签图（0 空/1 黑/2 白），以及棋盘四角坐标（供矫正模块用）
"""
import numpy as np
import cv2

EMPTY_L, BLACK_L, WHITE_L = 0, 1, 2


def _rng(seed=None):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- 星位
STAR_POINTS_19 = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15),
                  (15, 3), (15, 9), (15, 9), (15, 15)]


def board_corners(width, height, size=19, margin=None):
    """计算"正对视角"棋盘线交叉点的四角坐标（像素）。"""
    if margin is None:
        margin = width * 0.04
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    step = min(usable_w, usable_h) / (size - 1)
    ox = (width - step * (size - 1)) / 2
    oy = (height - step * (size - 1)) / 2
    return {
        'tl': (ox, oy), 'tr': (ox + step * (size - 1), oy),
        'br': (ox + step * (size - 1), oy + step * (size - 1)),
        'bl': (ox, oy + step * (size - 1)),
    }, step


def _draw_grid(img, corners, step, size, line_color, line_width, star_color, star_radius):
    tl, tr, br, bl = corners['tl'], corners['tr'], corners['br'], corners['bl']
    for i in range(size):
        p0 = (tl[0] + step * i, tl[1])
        p1 = (bl[0] + step * i, bl[1])
        cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), line_color, line_width, cv2.LINE_AA)
        p0 = (tl[0], tl[1] + step * i)
        p1 = (tr[0], tr[1] + step * i)
        cv2.line(img, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), line_color, line_width, cv2.LINE_AA)
    for (r, c) in {(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15),
                   (15, 3), (15, 9), (15, 15)}:
        x = tl[0] + step * c
        y = tl[1] + step * r
        cv2.circle(img, (int(round(x)), int(round(y))), star_radius, star_color, -1, cv2.LINE_AA)


def _draw_stone(img, x, y, radius, color_bgr, edge_dark=0.55, highlight=True):
    """带高光的棋子。color_bgr: 黑子 (30,30,30)，白子 (245,245,245)。

    edge_dark: 边缘压暗系数；highlight: 是否画高光（风格差异大时可关）。
    """
    radius = int(round(radius))
    overlay = img.copy()
    edge = tuple(int(v * edge_dark) for v in color_bgr)
    cv2.circle(overlay, (int(x), int(y)), radius, edge, -1, cv2.LINE_AA)
    cv2.circle(overlay, (int(x - radius * 0.18), int(y - radius * 0.18)),
               max(1, int(radius * 0.72)), color_bgr, -1, cv2.LINE_AA)
    if highlight:
        hx, hy = int(x - radius * 0.25), int(y - radius * 0.30)
        if color_bgr[0] > 128:
            # 白子小高光
            cv2.circle(overlay, (hx, hy), max(1, int(radius * 0.18)),
                       (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # 黑子边缘反光：只占左上一角，中心保持纯黑
            cv2.circle(overlay, (int(x - radius * 0.55), int(y - radius * 0.55)),
                       max(1, int(radius * 0.30)), (185, 185, 190), -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(x - radius * 0.60), int(y - radius * 0.60)),
                       max(1, int(radius * 0.15)), (245, 245, 248), -1, cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------- 风格预设
SKINS = [
    # (bg_color, line_color, black, white, name)
    # line 与 bg 的灰度差保持 >= 40；stone_pad 是棋子直径/格长（真实客户端 ~0.9-1.0）
    {'name': 'fox_wood',   'bg': (60, 130, 200),  'line': (70, 70, 90),   'stone_pad': 0.92},
    {'name': 'tygem_dark', 'bg': (40, 60, 90),    'line': (170, 170, 190), 'stone_pad': 0.95},
    {'name': 'ogs_plain',  'bg': (150, 190, 220), 'line': (60, 60, 60),   'stone_pad': 0.90},
    {'name': 'green_mat',  'bg': (70, 120, 90),   'line': (40, 40, 40),   'stone_pad': 0.93},
    # 讲解视频风格：深黄木底、暗棕线、大棋子、乳白/墨绿黑棋子（明玥读棋等）
    {'name': 'lec_yellow', 'bg': (11, 141, 181),  'line': (60, 80, 95),   'stone_pad': 1.00,
     'black_stone': (16, 36, 20), 'white_stone': (196, 220, 194),
     'edge_dark': 0.82, 'stone_highlight': False},
]


def render_board(labels, width=640, height=640, skin_idx=None, seed=None,
                 last_move=None, last_marker='circle', move_number=None,
                 cursor=None, perspective=0.0, jpeg_quality=None):
    """渲染一张棋盘图。

    labels: 19x19 list，0/1/2
    last_move: (row, col) 或 None，画最后一手标记
    move_number: 若给数字则用手数标记替代圆圈
    cursor: (row, col) 遮挡某个点（模拟鼠标光标盖住交叉点）
    perspective: 0~0.08 透视强度
    """
    rng = _rng(seed)
    if skin_idx is None:
        skin_idx = rng.integers(len(SKINS))
    skin = SKINS[skin_idx]

    img = np.full((height, width, 3), skin['bg'], dtype=np.uint8)

    # 木纹纹理：低频水平条带，不含高频噪声（真实截图纹理是平滑的）
    if rng.random() < 0.6:
        yy = np.arange(height, dtype=np.float32).reshape(-1, 1)
        bands = np.sin(yy * rng.uniform(0.05, 0.15)) * rng.uniform(3, 8)
        bands = np.repeat(bands, width, axis=1)
        img = np.clip(img.astype(np.float32) + bands[..., None], 0, 255).astype(np.uint8)

    size = len(labels)
    corners, step = board_corners(width, height, size)

    line_color = skin['line']
    lw = max(1, int(round(step * 0.006)))  # 真实客户端线宽约 1-3px
    _draw_grid(img, corners, step, size, line_color, lw, line_color, max(2, int(step * 0.05)))

    radius = step * skin['stone_pad'] / 2
    black_stone = skin.get('black_stone', (35, 35, 38))
    white_stone = skin.get('white_stone', (242, 242, 240))

    stone_cells = []
    for r in range(size):
        for c in range(size):
            v = labels[r][c]
            if v == EMPTY_L:
                continue
            x = corners['tl'][0] + step * c
            y = corners['tl'][1] + step * r
            col = black_stone if v == BLACK_L else white_stone
            img = _draw_stone(img, x, y, radius, col,
                              edge_dark=skin.get('edge_dark', 0.55),
                              highlight=skin.get('stone_highlight', True))
            stone_cells.append((r, c, v))

    # 最后一手标记
    if last_move is not None:
        r, c = last_move
        x = corners['tl'][0] + step * c
        y = corners['tl'][1] + step * r
        v = labels[r][c] if labels[r][c] != EMPTY_L else BLACK_L
        marker_col = (255, 255, 255) if v == BLACK_L else (0, 0, 0)
        if move_number is not None:
            cv2.putText(img, str(move_number), (int(x - step * 0.28), int(y + step * 0.22)),
                        cv2.FONT_HERSHEY_SIMPLEX, step / 42, marker_col, max(1, int(step * 0.055)), cv2.LINE_AA)
        elif last_marker == 'triangle':
            pts = np.array([[x, y - radius * 0.75], [x - radius * 0.72, y + radius * 0.55],
                            [x + radius * 0.72, y + radius * 0.55]], np.int32)
            cv2.polylines(img, [pts], True, marker_col, max(1, int(step * 0.05)), cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), int(radius * 0.55), marker_col, max(1, int(step * 0.05)), cv2.LINE_AA)

    # 光标遮挡（在某个交叉点画一个箭头状光标块）
    if cursor is not None:
        r, c = cursor
        x = corners['tl'][0] + step * c
        y = corners['tl'][1] + step * r
        _draw_pointer(img, x, y, step, rng)

    # 扰动：透视 + 亮度对比度 + 噪点 + JPEG
    img, corners = _apply_perspective(img, corners, perspective, rng)
    if jpeg_quality:
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img, corners


def _draw_pointer(img, x, y, step, rng, offset=None):
    """在 (x, y) 附近画一个鼠标指针。

    真实客户端录屏的光标是黑色小三角+白描边（macOS 指针样式），悬停时
    会盖住棋子的一角 —— 曾让 CNN 把被盖的白子读成 0.52 低置信黑子。
    三种形态随机：macOS 黑三角箭头 / 白色大箭头（旧样式，保留）/
    小手型指针。offset 允许光标悬停在交叉点之间（真实场景大多如此）。
    """
    if rng is not None and offset is None:
        offset = rng.uniform(-0.3, 0.3, 2) * step
    elif offset is None:
        offset = (0.0, 0.0)
    x = x + offset[0]
    y = y + offset[1]
    kind = rng.integers(3) if rng is not None else 0
    if kind == 0:
        # macOS 风格黑色三角箭头（含白描边）
        s = step / 32.0
        arrow = np.array([
            [0, 0], [0, 18 * s], [4.5 * s, 14 * s], [7.5 * s, 20.5 * s],
            [10.5 * s, 19 * s], [7.5 * s, 12.5 * s], [13 * s, 12.5 * s],
        ], np.float32)
        arrow += np.array([x - 2 * s, y - 2 * s])
        cv2.fillPoly(img, [arrow.astype(np.int32)], (255, 255, 255))
        cv2.polylines(img, [arrow.astype(np.int32)], True, (255, 255, 255),
                      max(2, int(2 * s)), cv2.LINE_AA)
        # 白描边大一号再画黑色主体
        cv2.fillPoly(img, [arrow.astype(np.int32)], (30, 30, 30))
    elif kind == 1:
        # 旧样式白色大箭头
        arrow = np.array([[x, y], [x, y + step * 0.9], [x + step * 0.25, y + step * 0.62],
                          [x + step * 0.42, y + step * 0.95], [x + step * 0.55, y + step * 0.87],
                          [x + step * 0.38, y + step * 0.56], [x + step * 0.68, y + step * 0.56]], np.float32)
        cv2.fillPoly(img, [arrow.astype(np.int32)], (250, 250, 250))
        cv2.polylines(img, [arrow.astype(np.int32)], True, (30, 30, 30), 1)
    else:
        # 小手型/圆点指针（触控板点按反馈）
        rr = step * 0.22
        cv2.circle(img, (int(x), int(y)), int(rr), (250, 250, 250), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), int(rr), (60, 60, 60), max(1, int(step * 0.03)), cv2.LINE_AA)


def _apply_perspective(img, corners, strength, rng):
    """对棋盘做小透视/旋转变换，并返回变换后的四角真值。"""
    if strength <= 0:
        return img, corners
    h, w = img.shape[:2]
    src = np.float32([corners['tl'], corners['tr'], corners['br'], corners['bl']])
    jitter = rng.uniform(-strength, strength, (4, 2)) * np.array([w, h])
    dst = src + jitter.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    out = np.zeros_like(img)
    # 保留棋盘区域：把四边形映射回全图后填充原背景色
    bg = img[2, 2].copy()
    out[:] = bg
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    mask = np.zeros((h, w), np.uint8)
    poly = (dst + np.array([[0, 0]])).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    out[mask > 0] = warped[mask > 0]
    corners_new = {'tl': tuple(dst[0]), 'tr': tuple(dst[1]),
                   'br': tuple(dst[2]), 'bl': tuple(dst[3])}
    return out, corners_new
