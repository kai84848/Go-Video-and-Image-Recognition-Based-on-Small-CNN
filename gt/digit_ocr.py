# -*- coding: utf-8 -*-
"""棋子上的手数数字 OCR：黑子白字 / 白子黑字 → 数字。

同 UP 主视频字体/尺寸固定，用二值连通域分割 + 模板 NCC 匹配。
patch 为原图分辨率（棋子直径 ~50px，数字高 ~22px），居中 44x44。
"""
import cv2
import numpy as np

CROP = 32          # 中心裁剪边长（只含数字，不含棋子边缘/网格线）
NORM_H = 32        # 模板归一化高度
MIN_INK = 25       # 最少笔画像素（低于此视为无数字）
MAX_INK_FRAC = 0.55  # 笔画占比上限（高于此视为光标/遮挡等异常）


def digit_mask(patch, color):
    """提取数字笔画二值图（前景=255=字）。

    color: 1 黑子（白字）/ 2 白子（黑字）。无数字返回 None。
    """
    h, w = patch.shape[:2]
    y0 = (h - CROP) // 2
    x0 = (w - CROP) // 2
    gray = cv2.cvtColor(patch[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_BGR2GRAY)
    if color == 1:
        mask = (gray > 140).astype(np.uint8) * 255
    else:
        mask = (gray < 115).astype(np.uint8) * 255
    ink = int((mask > 0).sum())
    if ink < MIN_INK or ink > MAX_INK_FRAC * CROP * CROP:
        return None
    return mask


def _components(mask):
    """笔画连通域 → 按 x 排序的归一化字符图列表。"""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 6:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return []
    # 数字同字体同高；边缘杂物（网格线/光标碎块）偏矮，按主字高过滤
    max_h = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= 0.6 * max_h and b[2] * b[3] >= 20]
    if not boxes:
        return []
    # 该字体每个数字都是单连通域，无需跨连通域合并；
    # 按 x 排序即得从左到右的字符序列（多位数逐位识别）
    boxes.sort()
    chars = []
    for b in boxes:
        x0, y0, w0, h0 = b
        ch = mask[y0:y0 + h0, x0:x0 + w0]
        scale = NORM_H / ch.shape[0]
        nw = max(1, int(round(ch.shape[1] * scale)))
        ch = cv2.resize(ch, (nw, NORM_H), interpolation=cv2.INTER_AREA)
        chars.append(ch)
    return chars


def _align(a, b):
    """两字符图中心对齐到同宽画布（高均为 NORM_H）。"""
    w = max(a.shape[1], b.shape[1])
    ca = np.zeros((NORM_H, w), np.uint8)
    cb = np.zeros((NORM_H, w), np.uint8)
    ca[:, (w - a.shape[1]) // 2:(w - a.shape[1]) // 2 + a.shape[1]] = a
    cb[:, (w - b.shape[1]) // 2:(w - b.shape[1]) // 2 + b.shape[1]] = b
    return ca, cb


def _ncc(a, b):
    ca, cb = _align(a, b)
    ca = ca.astype(np.float32) / 255.0
    cb = cb.astype(np.float32) / 255.0
    va = ca - ca.mean()
    vb = cb - cb.mean()
    denom = float(np.sqrt((va * va).sum() * (vb * vb).sum()))
    if denom < 1e-6:
        return 0.0
    return float((va * vb).sum() / denom)


def _chamfer(a, b):
    """对称 chamfer 距离 → (0,1] 分数。对笔画粗细差异鲁棒。"""
    ca, cb = _align(a, b)
    ia = ca > 0
    ib = cb > 0
    if not ia.any() or not ib.any():
        return 0.0
    dta = cv2.distanceTransform((~ia).astype(np.uint8), cv2.DIST_L2, 3)
    dtb = cv2.distanceTransform((~ib).astype(np.uint8), cv2.DIST_L2, 3)
    d = (float(dtb[ia].mean()) + float(dta[ib].mean())) / 2
    return 1.0 / (1.0 + d)


def _match(a, b):
    return 0.4 * _ncc(a, b) + 0.6 * _chamfer(a, b)


class DigitOCR:
    """模板库：{(digit, color): [归一化字符图]}。匹配时忽略颜色（前景已统一）。"""

    def __init__(self):
        self.templates = {}

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.templates, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            extra = pickle.load(f)
        for k, v in extra.items():
            self.templates.setdefault(k, []).extend(v)

    def add_sample(self, patch, color, digit):
        mask = digit_mask(patch, color)
        if mask is None:
            raise ValueError(f'样本无数字笔画: digit={digit}')
        chars = _components(mask)
        if len(chars) != 1:
            raise ValueError(f'样本字符数={len(chars)}，应为 1: digit={digit}')
        self.templates.setdefault((digit, color), []).append(chars[0])

    def recognize(self, patch, color=None):
        """返回 (number, score)。无数字 → (None, 0)；低置信 → (None, best_score)。

        color 传 None 时黑白两种前景都试（CNN 可能把带白字的黑子误判为白子）。
        """
        colors = [color] if color in (1, 2) else [1, 2]
        best_num, best_score = None, 0.0
        for col in colors:
            mask = digit_mask(patch, col)
            if mask is None:
                continue
            chars = _components(mask)
            if not chars or len(chars) > 3:
                continue
            digits = []
            scores = []
            for ch in chars:
                best, bs = None, -1.0
                for (d, tc), temps in self.templates.items():
                    # 不按模板颜色过滤：二值化已统一前景，黑白字只是反相
                    for t in temps:
                        s = _match(ch, t)
                        if s > bs:
                            best, bs = d, s
                if best is None:
                    continue
                digits.append((str(best), bs))
                scores.append(bs)
            if not digits:
                continue
            score = min(scores)
            if score > best_score:
                best_score = score
                best_num = int(''.join(d for d, _ in digits)) if min(scores) >= 0.62 else None
        if best_num is None:
            return None, best_score
        return best_num, best_score
