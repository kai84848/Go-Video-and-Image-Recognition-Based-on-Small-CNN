# -*- coding: utf-8 -*-
"""最小示例：单张棋盘图片 → SGF。

在仓库根目录运行：
  python examples/image_to_sgf.py board.png out.sgf
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from gorec.pipeline import Pipeline


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    img_path = sys.argv[1]
    out_sgf = sys.argv[2] if len(sys.argv) > 2 else 'out.sgf'
    img = cv2.imread(img_path)
    if img is None:
        print(f'无法读取图片: {img_path}')
        return
    pipe = Pipeline('pointnet.pt')
    result = pipe.process_image(img)
    if result['status'] != 'ok':
        print('识别失败:', result.get('reason'))
        return
    labels = result['labels']
    n_black = int((labels == 1).sum())
    n_white = int((labels == 2).sum())
    n_low = int((result['prob'] < 0.85).sum())
    print(f'局面: 黑 {n_black} 白 {n_white}，低置信点 {n_low}')
    print('规则校验:', result['grid_result']['status'])
    with open(out_sgf, 'w') as f:
        f.write(pipe.sgf())
    print('SGF 已保存:', out_sgf)


if __name__ == '__main__':
    main()
