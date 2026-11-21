# -*- coding: utf-8 -*-
"""最后一手三角标记检测（多手拆解的外部真值锚点）。

客户端/讲解软件在最后一手上画小三角：彩色（红/蓝）或黑白反色，
位于棋子顶部（数字在中心，位置可分离）。
patch 为原图分辨率（棋子直径 ~50px），44x44 居中。
"""
import cv2
import numpy as np


def has_marker(patch, color):
    """检测有子点 patch 上是否有最后一手三角标记。

    color: 1 黑子 / 2 白子（仅用于黑白反色三角的阈值方向）。
    """
    if patch is None or patch.shape[0] < 30 or patch.shape[1] < 30:
        return False
    p = patch.astype(np.int16)
    b, g, r = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    # 彩色三角：红/蓝。木色棋盘 R-G ~30 不达阈；黑白棋子无彩色。
    red = (r > 120) & (r > g + 50) & (r > b + 50)
    blue = (b > 120) & (b > r + 40) & (b > g + 20)
    if int((red | blue).sum()) >= 8:
        return True
    # 黑白反色三角：要求连通域整体位于棋子顶部（y+h<=11）。
    # 数字虽上探到 y~6，但其连通域必然延伸到中心（y>20）→ 排除。
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if color == 1:
        m = (gray > 180).astype(np.uint8)  # 黑子上白三角（黑子高光一般 <150）
    else:
        m = (gray < 70).astype(np.uint8)   # 白子上黑三角（白子阴影一般 >100）
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # 排除 patch 边缘：角落亮块是棋子外的木纹背景/邻子，不是标记
        if x < 6 or x + w > patch.shape[1] - 6:
            continue
        if area >= 6 and w >= 4 and h >= 3 and y + h <= 11:
            return True
    return False


def find_last_move(frame, pts, labels, half=22):
    """扫描全部有子点，返回带三角标记的 (row,col)；无或不止一个返回 None。

    pts: grid_points_original 的 (19,19,2) 原图坐标；labels: 19x19。
    """
    hits = []
    for r in range(19):
        for c in range(19):
            if labels[r, c] == 0:
                continue
            x, y = pts[r, c]
            xi, yi = int(round(x)), int(round(y))
            patch = frame[max(0, yi - half):yi + half, max(0, xi - half):xi + half]
            if patch.shape[0] != 2 * half or patch.shape[1] != 2 * half:
                continue
            if has_marker(patch, int(labels[r, c])):
                hits.append((r, c))
    if len(hits) == 1:
        return hits[0]
    return None
