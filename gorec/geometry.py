# -*- coding: utf-8 -*-
"""棋盘几何定位与矫正：从图像中定位 19 路网格并矫正到标准俯视图。

矫正后约定（canonical）：画布 = CELL*size + 2*margin（margin=CELL//2=16，
CELL=32 → 640x640），交叉点位于 (margin + c*CELL, margin + r*CELL)。
⚠️ canonical 空间内的所有采样坐标都必须加 margin 偏移。
"""
import numpy as np
import cv2

CELL = 32


def warp_canonical(image, corners, size=19):
    """按四角（tl,tr,br,bl，为网格交叉点坐标）把图像矫正到标准俯视图。

    输出尺寸 CELL*(size-1)，交叉点恰好落在 (c*CELL, r*CELL)。
    """
    # 输出 CELL*size + 2*MARGIN：交叉点在 MARGIN + c*CELL，
    # 边缘交叉点四周有真实底色（MARGIN=CELL/2），patch 采样无需 replicate
    w = CELL * size
    m = CELL // 2
    src = np.float32([corners['tl'], corners['tr'], corners['br'], corners['bl']])
    dst = np.float32([[m, m], [m + CELL * (size - 1), m],
                      [m + CELL * (size - 1), m + CELL * (size - 1)], [m, m + CELL * (size - 1)]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w + 2 * m, w + 2 * m), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def grid_points_original(corners, size=19):
    """canonical 交叉点 → 原图坐标（warp_canonical 的逆映射）。

    返回 (size,size,2) float64，用于在原图分辨率下切 patch
    （数字 OCR、三角标记检测等需要原始像素的场景）。
    """
    m = CELL // 2
    src = np.float32([corners['tl'], corners['tr'], corners['br'], corners['bl']])
    dst = np.float32([[m, m], [m + CELL * (size - 1), m],
                    [m + CELL * (size - 1), m + CELL * (size - 1)],
                    [m, m + CELL * (size - 1)]])
    M_inv = np.linalg.inv(cv2.getPerspectiveTransform(src, dst))
    pts = np.array([[m + c * CELL, m + r * CELL, 1.0]
                    for r in range(size) for c in range(size)])
    out = pts @ M_inv.T
    return (out[:, :2] / out[:, 2:3]).reshape(size, size, 2)


def _board_in_frame(corners, w, h, slack=0.006):
    """棋盘四角必须落在画面内（允许 0.6% 亚像素级出格余量）。

    精修/外推跑到画面外时 warp 的 BORDER_REPLICATE 会拉伸边缘像素，
    产生虚高分数 —— 用物理约束直接否决。"""
    for x, y in corners.values():
        if x < -slack * w or x > w * (1 + slack) or y < -slack * h or y > h * (1 + slack):
            return False
    return True


def detect_grid(image, size=19):
    """检测棋盘网格四角（容忍小角度旋转、亮/暗线、棋子遮挡）。

    返回 (corners, score) 或 (None, reason)。
    corners: {tl,tr,br,bl}，坐标为网格首末线交点。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    h, w = gray.shape

    # 先估计主方向角并矫正到 0 度附近，再跑小步长候选
    est = _estimate_angle(gray)
    cands = [None, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0]
    if est is not None and abs(est) > 0.1:
        # est 为需要矫正的角度：主角度偏离 0 度 est 度
        cands = [est] + cands
    best = None
    for rot in cands:
        g = gray if rot is None else _rotate(gray, rot)
        for flag in (cv2.THRESH_BINARY_INV, cv2.THRESH_BINARY):
            got_list = _detect_on(g, flag, size)
            if not got_list:
                continue
            # 先过滤画外窗口再取前 8（_detect_on 按 quad 分数排序，
            # 出画窗口可能占据前列挤掉真窗口）
            got_list = [c for c in got_list if _board_in_frame(c[0], w, h)]
            for corners, _q in got_list[:8]:
                if rot:
                    corners = _rotate_corners_back(corners, rot, w, h)
                # 混合评分：reproj 度量"线上对比强度"，align 度量"线位是否对齐"。
                # 大格满子 + JPEG 场景 reproj 会被棋子边缘强对比带偏，
                # align 的 exact-0 特征判别窗口真伪更稳。
                score = _grid_reproj_score(gray, corners, size)
                align, _ = grid_align_score(gray, corners, size)
                mixed = 0.5 * score + 0.5 * align
                if mixed < 0.08:
                    cand2, score2 = _refine_corners(gray, corners, size, iters=1)
                    if score2 > score and _board_in_frame(cand2, w, h):
                        corners, score = cand2, score2
                        mixed = 0.5 * score + 0.5 * grid_align_score(gray, corners, size)[0]
                if best is None or mixed > best[0]:
                    best = (mixed, corners)
        if best is not None and best[0] >= 0.25 and rot is None:
            break
    if best is None:
        return None, 'detect_fail'
    # 低置信时才做全局精修（高置信时精修反而引入噪声）
    if best[0] < 0.15:
        cand2, score = _refine_corners(gray, best[1], size, iters=2)
        if score > best[0] and _board_in_frame(cand2, w, h):
            return cand2, score
    score, corners = best
    return corners, score


def _refine_corners(gray, corners, size=19, iters=2):
    """在标准坐标系里对每条网格线做 ±CELL/2 内的峰值吸附，迭代精修四角。

    向量化实现；返回精修后的四角与 reproj 分数。
    """
    cur = {k: tuple(map(float, v)) for k, v in corners.items()}
    score = _grid_reproj_score(gray, cur, size)
    for _ in range(iters):
        can = warp_canonical(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cur, size)
        gh = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY).astype(np.float32)
        n = gh.shape[0]
        half = CELL // 2

        def line_offsets(fixed_axis):
            """fixed_axis=0: 找每行的偏移（扫行带）；=1: 找每列的偏移。"""
            offs = []
            for i in range(size):
                p = CELL // 2 + i * CELL  # canonical 交叉点带 margin 偏移
                lo, hi = max(1, p - half), min(n - 2, p + half)
                if lo >= hi:
                    offs.append(0.0)
                    continue
                if fixed_axis == 0:
                    band = gh[lo:hi + 1, :]
                    bg_u = gh[max(0, lo - 3):lo, :]
                    bg_d = gh[hi + 1:min(n, hi + 4), :]
                else:
                    band = gh[:, lo:hi + 1]
                    bg_u = gh[:, max(0, lo - 3):lo]
                    bg_d = gh[:, hi + 1:min(n, hi + 4)]
                contrast = np.zeros(hi - lo + 1, np.float64)
                if fixed_axis == 0:
                    # band: (K, W)，与上方/下方背景(每列均值)比较，再对列取均值
                    if bg_u.shape[0]:
                        mu = bg_u.mean(axis=0)
                        contrast += np.abs(band - mu).mean(axis=1)
                    if bg_d.shape[0]:
                        mu = bg_d.mean(axis=0)
                        contrast += np.abs(band - mu).mean(axis=1)
                else:
                    # band: (H, K)，与左/右背景(每行均值)比较，再对行取均值
                    if bg_u.shape[1]:
                        mu = bg_u.mean(axis=1)
                        contrast += np.abs(band - mu[:, None]).mean(axis=0)
                    if bg_d.shape[1]:
                        mu = bg_d.mean(axis=1)
                        contrast += np.abs(band - mu[:, None]).mean(axis=0)
                offs.append((lo + int(np.argmax(contrast))) - p)
            return offs

        row_off = line_offsets(0)
        col_off = line_offsets(1)

        def linfit(offs):
            xs = np.arange(len(offs), dtype=np.float64)
            ys = np.array(offs, dtype=np.float64)
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
            return float(slope), float(intercept)

        sx, ix = linfit(col_off)
        sy, iy = linfit(row_off)
        last = size - 1
        m2 = CELL // 2  # 偏移以 canonical 交叉点 (m2+i*CELL) 为参考，角点须加回
        tl = (m2 + ix, m2 + iy)
        tr = (m2 + ix + last * CELL + last * sx, m2 + iy + last * sy)
        bl = (m2 + ix, m2 + iy + last * CELL + last * sy)
        br = (tr[0], bl[1])
        src = np.float32([cur['tl'], cur['tr'], cur['br'], cur['bl']])
        wpx = CELL * (size - 1)
        dst = np.float32([[m2, m2], [m2 + wpx, m2],
                          [m2 + wpx, m2 + wpx], [m2, m2 + wpx]])
        M = cv2.getPerspectiveTransform(dst, src)  # canonical -> 原图
        newc = {}
        for name, pt in (('tl', tl), ('tr', tr), ('br', br), ('bl', bl)):
            v = M @ np.array([pt[0], pt[1], 1.0])
            newc[name] = (float(v[0] / v[2]), float(v[1] / v[2]))
        cur = newc
        new_score = _grid_reproj_score(gray, cur, size)
        if new_score <= score:
            break
        score = new_score
    return cur, score


def _grid_reproj_score(gray, corners, size=19):
    """网格对齐分 + 边界纯度分的加权综合。

    对齐分：每条期望网格线与邻近背景的对比度，取低分位（薄弱线决定
    成败）与均值加权。正确网格的所有线都有响应；错误窗口只有部分线。
    边界分：四角外扩 1 格的环带应接近单一底色。
    """
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    can = warp_canonical(gray3, corners, size)
    gh = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n = gh.shape[0]
    line_scores = []
    for i in range(size):
        p = CELL // 2 + i * CELL  # canonical 交叉点带 margin 偏移
        if p >= n:
            line_scores.append(0.0)
            continue
        row = gh[p, :]
        col = gh[:, p]
        best_row = 0.0
        best_col = 0.0
        for off in (-3, 3):
            q = p + off
            if 0 <= q < n:
                best_row = max(best_row, float(np.abs(row - gh[q, :]).mean()))
                best_col = max(best_col, float(np.abs(col - gh[:, q]).mean()))
        line_scores.append(min(best_row, best_col))
    ls = np.array(line_scores)
    align = float(np.percentile(ls, 25) * 0.7 + ls.mean() * 0.3) / 255.0

    # 边界纯度：四角向外扩 1 格的环带
    c = {k: np.array(v, np.float64) for k, v in corners.items()}
    center = (c['tl'] + c['tr'] + c['br'] + c['bl']) / 4
    expand = {k: center + (v - center) * (size / (size - 2.0)) for k, v in c.items()}
    expand = {k: (float(v[0]), float(v[1])) for k, v in expand.items()}
    can2 = warp_canonical(gray3, expand, size)
    g2 = cv2.cvtColor(can2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n2 = g2.shape[0]
    # can2 中外扩四角落在 [CELL//2, CELL//2 + wpx]；真网格按 17/19 缩放回
    # 中心，每侧环带恰为一格宽：[CELL//2, CELL//2+CELL)
    b0, b1 = CELL // 2, CELL // 2 + CELL
    bands = np.concatenate([
        g2[b0:b1, :].ravel(), g2[n2 - b1:n2 - b0, :].ravel(),
        g2[:, b0:b1].ravel(), g2[:, n2 - b1:n2 - b0].ravel(),
    ])
    if bands.size > 10:
        purity = 1.0 - min(1.0, float(np.std(bands)) / 60.0)
    else:
        purity = 0.5
    return float(align * 0.8 + purity * 0.2)
_ANGLE_CACHE = {}


def _estimate_angle(gray):
    """估计图像中棋盘水平线相对水平轴的平均角度（度），用于预矫正。

    视频流中棋盘角度恒定，按 (h, w) 缓存（首帧后近零成本）。
    """
    h, w = gray.shape
    key = (h, w)
    if key in _ANGLE_CACHE:
        return _ANGLE_CACHE[key]
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 51, 6)
    kh = max(9, w // 55)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    lines = cv2.HoughLinesP(horiz, 1, np.pi / 1440, threshold=int(w * 0.1),
                            minLineLength=w * 0.25, maxLineGap=8)
    if lines is None:
        return None
    angs = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        if ang < 30 or ang > 150:
            angs.append(ang if ang < 90 else ang - 180.0)
    if len(angs) < 5:
        return None
    a = float(np.median(angs))
    result = a if abs(a) > 0.1 else None
    _ANGLE_CACHE[key] = result
    return result


def _rotate(image, deg):
    hh, ww = image.shape[:2]
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), deg, 1.0)
    return cv2.warpAffine(image, M, (ww, hh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _rotate_corners_back(corners, deg, w, h):
    """把 _rotate(gray, deg) 后坐标系里的角点映射回原图坐标。

    _rotate 的正向：p' = R(-deg)·(p-c)+c（cv2 正角度=逆时针视觉，坐标系 y 向下）。
    逆映射需用 R(+deg)：p = R(deg)·(p'-c)+c。
    """
    th = np.radians(deg)
    cx, cy = w / 2, h / 2
    cos_t, sin_t = np.cos(th), np.sin(th)
    out = {}
    for k, (x, y) in corners.items():
        dx, dy = x - cx, y - cy
        out[k] = (cx + dx * cos_t - dy * sin_t,
                  cy + dx * sin_t + dy * cos_t)
    return out


def _detect_on(gray, flag, size):
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               flag, 51, 6)
    # 方向性开运算：保留长直线，去掉棋子等团块
    kh = max(9, w // 55)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh)))
    # 去掉贴图像边缘的边框线干扰
    m = max(2, int(0.01 * w))
    hp = _line_peaks(horiz[:, m:w - m], axis=1)
    vp = _line_peaks(vert[m:h - m, :], axis=0)
    hs_list = _fit_grid_lines(hp, size, h) or []
    vs_list = _fit_grid_lines(vp, size, w) or []
    # 单边补救：一个方向的峰被棋子/标记大面积遮挡时，借用另一方向的
    # 步长生成候选（棋盘是正方形，横竖步长一致）。
    if not hs_list and vs_list:
        hs_list = _complete_from_other(hp, vs_list, size, h)
    if not vs_list and hs_list:
        vs_list = _complete_from_other(vp, hs_list, size, w)
    if not hs_list or not vs_list:
        return None
    out = []
    for hs in hs_list:
        for vs in vs_list:
            corners = {
                'tl': (vs[0], hs[0]), 'tr': (vs[-1], hs[0]),
                'br': (vs[-1], hs[-1]), 'bl': (vs[0], hs[-1]),
            }
            out.append((corners, _quad_score(corners)))
    return out


def _complete_from_other(peaks, other_list, size, dim):
    """借用另一方向已拟合窗口的步长，从本方向峰里对齐出候选线位。

    other_list: 另一方向的线位列表 [(起点...终点, step)]。对本方向每个
    假想起点（峰位与其减步长的组合），生成等差序列，峰位落在格点上的
    数量达标者作为候选返回。
    """
    out = []
    seen_steps = []
    for seq in other_list[:3]:
        step = (seq[-1] - seq[0]) / (len(seq) - 1)
        if any(abs(step - s) < 1.5 for s in seen_steps):
            continue
        seen_steps.append(step)
        pos = np.array(sorted(peaks), dtype=np.float64)
        if len(pos) < size // 3:
            continue
        # 起点候选：每个峰位以及峰位 - k*step（k=0..size-1）
        starts = set()
        for p in pos:
            for k in range(size):
                s = p - k * step
                if -step < s < dim:
                    starts.add(round(s, 1))
        scored = []
        for s0 in sorted(starts):
            # 窗口必须落在画面内（外推出画的窗口靠杂峰也能凑高支撑，喧宾夺主）
            if s0 < -step * 0.3 or s0 + (size - 1) * step > dim + step * 0.3:
                continue
            grid_idx = np.round((pos - s0) / step)
            resid = pos - s0 - grid_idx * step
            inlier = np.abs(resid) < step * 0.22
            k_span = grid_idx[inlier].max() - grid_idx[inlier].min()
            if inlier.sum() >= max(size // 3, 7) and k_span >= size - 3:
                seq = [s0 + i * step for i in range(size)]
                # 按峰支撑数排序（支撑多=起点更可信），缺线多的交给上层评分
                scored.append((-int(inlier.sum()), seq))
        scored.sort(key=lambda t: t[0])
        out.extend(seq for _, seq in scored[:3])
        if len(out) >= 6:
            break
    return out[:6]


def _line_peaks(mask, axis):
    """沿 axis 方向投影并取峰。返回峰中心（未加偏移）。"""
    proj = mask.sum(axis=axis).astype(np.float64)
    if proj.max() <= 0:
        return []
    bg = np.percentile(proj, 60)
    span = proj.max() - bg
    if span <= 0:
        return []
    thresh = bg + 0.3 * span
    peaks = []
    i, n = 0, len(proj)
    while i < n:
        if proj[i] > thresh:
            j = i
            while j < n and proj[j] > thresh:
                j += 1
            seg = proj[i:j]
            idx = np.arange(i, j)
            peaks.append(float((idx * seg).sum() / seg.sum()))
            i = j
        else:
            i += 1
    return peaks


def _fit_grid_lines(peak_positions, size, dim):
    """把峰位拟合为 size 条等距网格线；容忍缺线与孤立噪声峰。

    间距估计用相邻峰间距的众数（对孤立噪声峰鲁棒），
    再做格点对齐，内点占多数者胜出。"""
    if len(peak_positions) < size // 2:
        return None
    pos = np.array(sorted(peak_positions), dtype=np.float64)
    merged = []
    for p in pos:
        if merged and p - merged[-1] < dim * 0.01:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)
    pos = np.array(merged)
    if len(pos) < size // 2:
        return None

    # d 候选：峰间距众数 d_mode（可能因缺线是真实 d 的 2 倍），
    # 搜索范围覆盖 d_mode/2.2 ~ d_mode*1.3，同时受物理约束夹紧
    gaps = np.diff(pos)
    gaps = gaps[(gaps > 8) & (gaps < dim * 0.12)]
    if len(gaps) < size // 3:
        return None
    hist = np.bincount(np.round(gaps).astype(int))
    # 取直方图的前几个局部极大作为锚（真间距票分散，假间距集中，
    # 只取 argmax 会被假间距劫持）
    anchors = []
    h = hist.astype(float)
    kernel = np.array([1, 2, 3, 2, 1], float)
    h_sm = np.convolve(h, kernel / kernel.sum(), mode='same')
    order = np.argsort(h_sm)[::-1]
    for i in order:
        v = i
        if h_sm[v] <= 0:
            break
        if all(abs(v - a) > 6 for a in anchors):
            anchors.append(float(v))
        if len(anchors) >= 3:
            break
    if not anchors:
        anchors = [float(np.argmax(hist))]
    top_anchor = max(anchors, key=lambda a: h_sm[int(a)])
    cands = []  # (crit, d, k, phi0)
    for d_mode in anchors:
        d_lo = max(6.0, d_mode * 0.78)
        d_hi = max(d_lo + 1.0, d_mode * 1.3)
        for d in np.linspace(d_lo, d_hi, 40):
            # 细分格陷阱：d 是最高票 anchor 的 1/2 或 1/3 时跳过 —— 这类 d 的
            # 相位直方图 inlier 多、crit 好，会挤掉真 d。
            if any(abs(d * m - top_anchor) < 6 for m in (2, 3)):
                continue
            phase = pos % d
            nb = max(8, int(d))
            hist_p, edges = np.histogram(phase, bins=nb, range=(0, d))
            imax = int(np.argmax(hist_p))
            phi0 = float((edges[imax] + edges[imax + 1]) / 2)
            k = np.round((pos - phi0) / d)
            resid = pos - phi0 - k * d
            inlier = np.abs(resid) < d * 0.25
            if inlier.sum() < size // 2:
                continue
            err = float(np.sqrt((resid[inlier] ** 2).mean()))
            crit = err - int(inlier.sum()) * (2.0 / size)
            cands.append((crit, d, k, phi0))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    # 只保留不同 d 的前 4 个候选
    seen = []
    picked = []
    for crit, d, k, phi0 in cands:
        if any(abs(d - dd) < 1.5 for dd in seen):
            continue
        seen.append(d)
        picked.append((crit, d, k, phi0))
        if len(picked) >= 4:
            break

    results = []
    for crit, d, k, phi0 in picked:
        seq_out = _complete_grid(pos, k, size)
        if seq_out is not None:
            for e, seq in seq_out:
                results.append((crit + e * 0.1, seq))
    if not results:
        return None
    results.sort(key=lambda t: t[0])
    return [r[1] for r in results]


def _complete_grid(pos, k, size):
    """按格点序号 k 分组、等差补齐缺失线，返回所有合法窗口的线位列表。

    返回 [(e, [线位...]), ...]（按 e 升序）。等差误差 e 只统计有峰支撑的
    相邻线对外推段（纯外推区天然等差、e→0，会包装出假窗口），
    由上层 reproj/align 评分做最终裁决。
    """
    idx = k.astype(int)
    groups = {}
    for i, ki in enumerate(idx):
        groups.setdefault(int(ki), []).append(pos[i])
    kmin, kmax = int(idx.min()), int(idx.max())
    if kmax - kmin + 1 < size - 1:
        return None
    results = []
    # 允许窗口向外延一格（峰跨度 kmax-kmin+1 == size-1 时，一端外推）：
    # 旋转/棋子遮挡常让最边缘线的峰缺失，不能因此整窗拒绝。
    for start in range(kmin - 1, kmax - size + 3):
        # 等差步长初值：组代表值相邻差的中位数（对杂峰鲁棒）
        reps = [float(np.mean(v)) for ki, v in sorted(groups.items())
                if start <= ki < start + size]
        spacing_est = None
        if len(reps) >= 3:
            spacing_est = float(np.median(np.diff(reps)))
        # 组内多峰（棋盘附近的杂线/边缘）时均值会被拉偏；
        # 用相邻已定线位的等差预测选最近的峰。
        chosen = {}
        for j in range(size):
            ki = start + j
            if ki in groups:
                chosen[j] = list(groups[ki])
        resolved = {}
        for j in sorted(chosen):
            cand = chosen[j]
            if len(cand) == 1:
                resolved[j] = cand[0]
                continue
            pred = None
            if j - 1 in resolved and spacing_est:
                pred = resolved[j - 1] + spacing_est
            elif (j + 1) in chosen and len(chosen[j + 1]) == 1 and spacing_est:
                pred = chosen[j + 1][0] - spacing_est
            resolved[j] = (min(cand, key=lambda v: abs(v - pred)) if pred is not None
                           else float(np.mean(cand)))
        chosen = resolved
        seq = []
        missing = 0
        have = []
        for j in range(size):
            if j in chosen:
                v = float(chosen[j])
                seq.append(v)
                have.append((j, v))
            else:
                missing += 1
                seq.append(None)
        if missing > max(2, size // 5):
            continue
        if missing:
            xs = np.array([j for j, _ in have])
            ys = np.array([v for _, v in have])
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
            seq = [seq[j] if seq[j] is not None else slope * j + intercept
                   for j in range(size)]
        arr = np.array(seq)
        # 只对"两端都有真实峰"的相邻线对计算间距误差
        errs = []
        have_set = set(j for j, _ in have)
        for j in range(size - 1):
            if j in have_set and (j + 1) in have_set:
                errs.append(arr[j + 1] - arr[j])
        if not errs:
            continue
        spacing = float(np.median(errs))
        if spacing <= 1:
            continue
        errs = np.array(errs)
        e = float(np.sqrt(np.mean((errs - spacing) ** 2)) / spacing)
        e += 0.08 * missing / size  # 外推线无峰支撑，按比例惩罚
        results.append((e, arr))
    if not results:
        return None
    results.sort(key=lambda t: t[0])
    return [(e, seq.tolist()) for e, seq in results]
def _quad_score(corners):
    """四角越接近正方形分数越高，0~1。"""
    pts = np.array([corners['tl'], corners['tr'], corners['br'], corners['bl']])
    side = np.array([np.hypot(*(pts[i] - pts[(i + 1) % 4])) for i in range(4)])
    if side.min() <= 1:
        return 0.0
    even = side.min() / side.max()
    # 对角线中点应重合（平行四边形性）
    c1 = (pts[0] + pts[2]) / 2
    c2 = (pts[1] + pts[3]) / 2
    para = max(0.0, 1 - float(np.hypot(*(c1 - c2))) / (side.mean() * 0.05 + 1e-9))
    return float(even * 0.7 + para * 0.3)


def _classify_lines(segs):
    """按角度把线段分为近水平和近垂直两组，返回 [(x1,y1,x2,y2), ...]。"""
    horiz, vert = [], []
    for x1, y1, x2, y2 in segs:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
        length = np.hypot(x2 - x1, y2 - y1)
        if ang < 45 or ang > 135:
            horiz.append((x1, y1, x2, y2, length))
        elif 45 <= ang <= 135:
            vert.append((x1, y1, x2, y2, length))
    return horiz, vert


def _cluster_parallel(lines, size):
    """把一组线段聚成 size 簇代表直线 (theta, rho)，容忍缺线。

    思路：网格线间距均匀。对候选间距 d（总跨度/(size-1) 的若干倍数），
    把所有线段的 rho 按 round((rho-rho_min)/d) 归入格点；被棋子完全
    遮挡的网格线缺失时，由等差外推补齐。"""
    if len(lines) < size // 2:
        return None
    angs = np.array([np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
                     for x1, y1, x2, y2, _ in lines])
    med = np.median(angs)
    keep = np.abs(angs - med) < 12
    lines = [l for l, k in zip(lines, keep) if k]
    if len(lines) < size // 2:
        return None

    thetas, rhos, weights = [], [], []
    for x1, y1, x2, y2, ln in lines:
        theta = float(np.arctan2(y2 - y1, x2 - x1) % np.pi)
        rho = x1 * (-np.sin(theta)) + y1 * np.cos(theta)
        thetas.append(theta)
        rhos.append(rho)
        weights.append(ln)
    thetas = np.array(thetas)
    rhos = np.array(rhos)
    weights = np.array(weights)

    rho_min = rhos.min()
    span = rhos.max() - rho_min
    if span < size * 4:
        return None
    d0 = span / (size - 1)
    best = None
    for scale in np.linspace(0.7, 1.4, 15):
        d = d0 * scale
        k = np.round((rhos - rho_min) / d)
        resid = rhos - rho_min - k * d
        err = float(np.sqrt((resid ** 2 * weights).sum() / weights.sum()))
        if best is None or err < best[0]:
            best = (err, d, k)
    err, d, k = best
    if err > 3.0:
        return None

    idx = k.astype(int)
    groups = {}
    for i, ki in enumerate(idx):
        groups.setdefault(ki, []).append(i)
    kmin, kmax = int(idx.min()), int(idx.max())
    if kmax - kmin + 1 < size:
        return None

    result = None
    for start in range(kmin, kmax - size + 2):
        seq = []
        missing = []
        for j, ki in enumerate(range(start, start + size)):
            if ki in groups:
                members = groups[ki]
                th = float(np.median([thetas[i] for i in members]))
                rho = float(np.average([rhos[i] for i in members],
                                       weights=[weights[i] for i in members]))
                seq.append((th, rho))
            else:
                missing.append(j)
                seq.append(None)
        if len(missing) > max(1, size // 5):
            continue
        if missing:
            have = [(j, s[1]) for j, s in enumerate(seq) if s is not None]
            xs = np.array([j for j, _ in have])
            ys = np.array([r for _, r in have])
            A = np.vstack([xs, np.ones(len(xs))]).T
            slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
            ref_th = seq[[j for j, s in enumerate(seq) if s is not None][0]][0]
            for j in missing:
                seq[j] = (ref_th, slope * j + intercept)
        rhos_w = np.array([s[1] for s in seq])
        gaps = np.diff(rhos_w)
        spacing = float(np.median(gaps))
        if spacing <= 1:
            continue
        e = float(np.sqrt(np.mean((gaps - spacing) ** 2)) / spacing)
        if result is None or e < result[0]:
            result = (e, seq)
    if result is None or result[0] > 0.25:
        return None
    return result[1]


def _intersect(l1, l2):
    """两条 (theta方向角, rho有符号距离) 直线求交点。

    rho = x*(-sin t) + y*(cos t)，即法线方向 (-sin t, cos t)（可为负）。
    """
    t1, r1 = l1
    t2, r2 = l2
    A = np.array([[-np.sin(t1), np.cos(t1)], [-np.sin(t2), np.cos(t2)]])
    b = np.array([r1, r2])
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    return tuple(np.linalg.solve(A, b))


def _grid_corners(h_group, v_group):
    # 垂直线 rho = -x，转回 x 以便左右排序一致
    v_sorted = sorted(v_group, key=lambda g: -g[1])
    tl = _intersect(h_group[0], v_sorted[0])
    tr = _intersect(h_group[0], v_sorted[-1])
    br = _intersect(h_group[-1], v_sorted[-1])
    bl = _intersect(h_group[-1], v_sorted[0])
    if any(p is None for p in (tl, tr, br, bl)):
        return None
    return {'tl': tl, 'tr': tr, 'br': br, 'bl': bl}


def grid_align_score(gray, corners, size=19, half=16, bg=5):
    """交叉点对齐分（0~1）：canonical 矫正后期望线位的吸附质量。

    对每条期望网格线在 ±half 窗口内找最强对比位置（offset）：
    - 覆盖率：有响应（对比峰值>15）的线占比
    - 对齐率：offset 恰在 ±3px 内的线占比（含完全居中）
    - 居中率：offset 恰为 0 的占比（棋子盖线时吸附会跳到棋子边缘 ±16，
      正确窗口里被盖的线是少数，因此 exact-0 比例是强判别特征）
    大格（格宽>40px、棋子近满格）场景下 reproj 对比幅度失真，
    本指标按"位置是否对"打分而非"对比多强"。
    返回 (score, n_measured)。
    """
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    can = warp_canonical(gray3, corners, size)
    gh = cv2.cvtColor(can, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n = gh.shape[0]
    offs = []
    for i in range(size):
        p = CELL // 2 + i * CELL
        if p < half + 1 or p >= n - half - 1:
            continue
        # 横线 y=p：行带内与上下背景比（吸附窗两端 3px 是 warp 边缘伪峰，置零）
        band = gh[p - half:p + half + 1, :]
        cu = np.abs(band - gh[max(0, p - bg - 2):p - bg, :].mean(axis=0)).mean(axis=1)
        cd = np.abs(band - gh[p + bg:min(n, p + bg + 2), :].mean(axis=0)).mean(axis=1)
        c = cu + cd
        c[:3] = 0
        c[-3:] = 0
        if c.max() > 15:
            offs.append(int(np.argmax(c)) - half)
        # 竖线 x=p：列带内与左右背景比
        band = gh[:, p - half:p + half + 1]
        cl = np.abs(band - gh[:, max(0, p - bg - 2):p - bg].mean(axis=1, keepdims=True)).mean(axis=0)
        cr = np.abs(band - gh[:, p + bg:min(n, p + bg + 2)].mean(axis=1, keepdims=True)).mean(axis=0)
        c = cl + cr
        c[:3] = 0
        c[-3:] = 0
        if c.max() > 15:
            offs.append(int(np.argmax(c)) - half)
    o = np.array(offs)
    if len(o) < 25:
        return 0.0, len(o)
    within = float(np.mean(np.abs(o) <= 3.0))
    exact = float(np.mean(o == 0))
    return 0.6 * within + 0.4 * exact, len(o)
