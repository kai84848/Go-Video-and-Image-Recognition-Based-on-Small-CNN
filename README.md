# gorec — 围棋棋盘识别

将围棋视频和图片识别为 SGF 棋谱。小型 CNN + 围棋规则校验层，支持.mp4视频和单张截图。

神经网络**不直接生成棋谱**，而是把非法步骤挡在输出之外：

1. **棋盘定位**（`geometry.py`、`video.py`）：网格线检测 + 等差拟合 → 四角 → 透视矫正为标准俯视图；视频端用 `BoardLocator` 时序锁定棋盘位置。
2. **逐点分类**（`model.py`）：361 个交叉点各取 40×40 patch，约 160KB 的小 CNN 三分类（空/黑/白）。
3. **手序重建**（`stream.py`）：相邻两局面必须恰好是"落一子+提子"的合法变化；快速翻谱/动画造成的多手帧按规则约束拆解，有**最后一手三角标记**时作为外部真值锚点。
4. **时间线处理**：**回退翻谱检测**（用户往回翻时截尾同步，不再整段丢手）、任意局面跳变走 `resync` 分段、数子画面守门。

**模型权重 ~160KB 已随仓库提供。**

---

## 安装

Python 3.9+。

```bash
pip install numpy opencv-python torch        # 核心依赖
pip install onnxruntime                      # 可选：免 torch 的 ONNX 推理
```

## 快速开始

### 单张图片 → SGF

```bash
python -m gorec.pipeline board.png out.sgf
```

### 视频 → SGF

```bash
python -c "from gorec.video import video_to_sgf; \
  video_to_sgf('game.mp4', 'out.sgf')"
```

- 默认 2fps 采样——速度最快，也是通用最优配置。
- `precise=True` 逐帧处理（更慢；短视频或落子极快时更有保障）。
- 返回 `(moves, stats)`。`stats` 中 `multi_jump` 表示"一帧含 3 手以上"的次数（提示可尝试 `precise=True`），`rewind` 表示回退翻谱被检测并保住手序的次数。

### Python API

```python
import cv2
from gorec.pipeline import Pipeline

pipe = Pipeline('pointnet.pt')               # 默认 CPU；有 Apple Silicon 时可用 MPS
result = pipe.process_image(cv2.imread('board.png'))
labels, prob = result['labels'], result['prob']   # 19x19：0 空、1 黑、2 白
print(result['grid_result']['status'])            # 本帧规则校验结果
print(pipe.sgf())                                 # 累积棋谱
```

`labels` 按 `[row][col]` 索引，row 0 在上、col 0 在左（与 SGF 坐标一致）。

### 手动指定棋盘四角（可选）（推荐，可以提高对复杂场景的识别精准度）

自动定位在特殊版式上失败时，可手动给出四个网格角点（**线交叉点**的像素坐标，不是棋子边缘）：

```python
corners = {'tl': (x0, y0), 'tr': (x1, y1), 'br': (x2, y2), 'bl': (x3, y3)}
result = pipe.process_image(image, corners=corners)
```

### 免 torch 的 ONNX 推理

```python
from gorec.model import OnnxPredictor
from gorec import geometry
import cv2

pred = OnnxPredictor('pointnet.onnx')
img = cv2.imread('board.png')
corners, _ = geometry.detect_grid(img)
can = geometry.warp_canonical(img, corners)
labels, probs = pred.predict_board(can)
```

ONNX 与 PyTorch 输出已验证完全一致。

## 评测

### 内置回归（无需下载数据）

```bash
python -m gorec.regression
```

合成棋盘自动生成。最近一次结果：**6/6 PASS**——定位中位角点误差 **0.60px**，容忍 0–2° 旋转，30 盘合成对局逐点分类 **100%**。需要可选真实数据的用例在数据缺失时跳过。

### 外部真值基准（`gt/`）（训练数据主要来源于网棋客户端录屏与网上批量下载）

`gt/` 工具链从"棋子上带手数数字"的视频中 OCR 出真值，再对管线打分：

- 讲解视频 877 个数字段：位置+颜色识别 **100%**（GT 重放自洽）。
- 真实视频流手序：**508/508** 段与数字真值一致。
- 多手拆解顺序（最难情形，单帧含 k 手）：总体 **47.7%**——2 手 **100%**、3-4 手 **80%**，5 手以上受信息量限制（见"已知限制"）。
- 两段快速复盘录屏（默认+回退检测）：有效逐手一致率 **93.5%** 和 **88.6%**；加入回退检测后 `resync` 分别由 2→0、8→4。

复现方式：`gt/scan_video.py` → `gt/compare.py` / `gt/sgf_compare.py`（OCR 模板已含在 `gt/` 内）。

## 从头训练

```python
from gorec.train import train
train(epochs=10, boards_per_epoch=600)  
```

`renderer.py` + `dataset.py` 生成随机合法对局，含多种客户端皮肤、光照与几何增广，权重保存为 `pointnet.pt`。渲染分布变更后需要重新训练。

## 仓库结构

| 路径 | 职责 |
|---|---|
| `gorec/geometry.py` | 网格检测、canonical 矫正、对齐评分、角点↔原图映射 |
| `gorec/model.py` | PointNet 小 CNN、patch 提取、ONNX 推理 |
| `gorec/rules.py` | 围棋规则引擎：落子/提子/劫/合法性、局面迁移校验 |
| `gorec/stream.py` | 帧流 → 合法手序：多手拆解、三角锚定、回退检测、resync |
| `gorec/marker.py` | 最后一手三角标记检测 |
| `gorec/video.py` | 视频流水线、时序定位、precise 逐帧模式 |
| `gorec/pipeline.py` | 单图入口 |
| `gorec/renderer.py`、`dataset.py`、`train.py` | 合成数据与训练 |
| `gorec/regression.py` | 一键回归验收 |
| `pointnet.pt`、`pointnet.onnx` | 模型权重（各 ~160KB） |
| `gt/` | 真值 OCR 工具链与基准 |



