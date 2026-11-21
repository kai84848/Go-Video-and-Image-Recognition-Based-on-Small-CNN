# -*- coding: utf-8 -*-
"""逐点分类小 CNN：交叉点 patch → 空/黑/白 三分类。

输入 40x40 BGR patch（以交叉点为中心），3 层卷积 + 池化 + 全连接。
模型刻意做小：单帧 361 点推理在 CPU 上也 <5ms。
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH = 40
NUM_CLASSES = 3  # 0 空 1 黑 2 白
CELL = 32  # canonical 网格步长（与 geometry 一致）


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=2), nn.ReLU(),       # 40x40
            nn.MaxPool2d(2),                                  # 20x20
            nn.Conv2d(24, 48, 3, padding=1), nn.ReLU(),      # 20x20
            nn.MaxPool2d(2),                                  # 10x10
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),      # 10x10
            nn.AdaptiveAvgPool2d(1),                          # 1x1
        )
        self.classifier = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def extract_patches(canonical, border=0):
    """从矫正后棋盘图提取全部交叉点 patch。

    canonical: 尺寸 >= (size-1)*CELL + 2*border，交叉点位于
    (c*CELL + border, r*CELL + border)。
    返回 (N, PATCH, PATCH, 3) uint8 数组，顺序 row-major。
    """
    h, w = canonical.shape[:2]
    size = 19
    margin = CELL // 2  # warp/render 约定：交叉点从 MARGIN 开始
    cell = (h - 2 * margin - 2 * border) // size
    half = PATCH // 2
    pad = half + border + margin
    big = cv2_copy_border(canonical, half)
    patches = np.empty((size * size, PATCH, PATCH, 3), dtype=np.uint8)
    i = 0
    for r in range(size):
        y = r * cell + pad
        for c in range(size):
            x = c * cell + pad
            patches[i] = big[y - half:y + half, x - half:x + half]
            i += 1
    return patches


def cv2_copy_border(img, pad):
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)


def patches_to_tensor(patches):
    """(N,H,W,3) uint8 → 归一化 float tensor (N,3,H,W)。"""
    x = patches.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x).permute(0, 3, 1, 2)
    return x


def predict_board(model, canonical, device='cpu', batch=1024, border=0):
    """对矫正后棋盘图预测 19x19 标签图。返回 (labels, probs)。

    canonical 的交叉点位于 (c*CELL+border, r*CELL+border)。
    """
    patches = extract_patches(canonical, border=border)
    model.eval()
    labels = np.empty(19 * 19, dtype=np.int64)
    probs = np.empty(19 * 19, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(patches), batch):
            xb = patches_to_tensor(patches[s:s + batch]).to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1)
            labels[s:s + len(xb)] = p.argmax(dim=1).cpu().numpy()
            probs[s:s + len(xb)] = p.max(dim=1).values.cpu().numpy()
    return labels.reshape(19, 19), probs.reshape(19, 19)


class OnnxPredictor:
    """ONNX Runtime 推理后端（部署用，无需 torch）。

    用法：
      pred = OnnxPredictor('pointnet.onnx')
      labels, probs = pred.predict_board(canonical)
    """

    def __init__(self, model_path='pointnet.onnx'):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name

    def predict_patches(self, patches, batch=1024):
        """(N,H,W,3) uint8 → (labels(N,), probs(N,))。"""
        labels = np.empty(len(patches), dtype=np.int64)
        probs = np.empty(len(patches), dtype=np.float32)
        for s in range(0, len(patches), batch):
            x = patches_to_tensor(patches[s:s + batch]).numpy()
            logits = self.sess.run(None, {self.input_name: x})[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            labels[s:s + len(x)] = p.argmax(axis=1)
            probs[s:s + len(x)] = p.max(axis=1)
        return labels, probs

    def predict_board(self, canonical, border=0):
        patches = extract_patches(canonical, border=border)
        labels, probs = self.predict_patches(patches)
        return labels.reshape(19, 19), probs.reshape(19, 19)
