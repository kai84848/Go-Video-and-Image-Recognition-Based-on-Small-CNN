# -*- coding: utf-8 -*-
"""单图/视频帧全流水线：图像 → 定位矫正 → 逐点分类 → 规则校验。

用法（命令行）：
  python -m gorec.pipeline 图片.png 输出.sgf
"""
import sys

import cv2
import numpy as np
import torch

from . import geometry
from .model import PointNet, predict_board
from .stream import BoardStream

DEFAULT_MIN_CONF = 0.85


class Pipeline:
    def __init__(self, model_path='pointnet.pt', device=None, min_conf=DEFAULT_MIN_CONF):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = PointNet()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.to(self.device).eval()
        self.min_conf = min_conf
        self.stream = BoardStream(19)

    def process_image(self, image_bgr, corners=None):
        """处理单张图。返回 dict：canonical, labels, prob, grid_result。"""
        if corners is None:
            corners, score = geometry.detect_grid(image_bgr)
            if corners is None:
                return {'status': 'detect_fail', 'reason': score}
        can = geometry.warp_canonical(image_bgr, corners)
        labels, prob = predict_board(self.model, can, device=self.device)
        res = self.stream.accept_grid(labels, prob, min_conf=self.min_conf)
        return {'status': 'ok', 'canonical': can, 'labels': labels,
                'prob': prob, 'grid_result': res}

    def process_frame(self, frame_bgr, corners=None):
        """同 process_image，别名用于视频流。"""
        return self.process_image(frame_bgr, corners)

    def sgf(self):
        return self.stream.to_sgf()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    img = cv2.imread(sys.argv[1])
    pipe = Pipeline()
    r = pipe.process_image(img)
    if r['status'] != 'ok':
        print('FAIL:', r.get('reason'))
        return
    res = r['grid_result']
    n_low = int((r['prob'] < DEFAULT_MIN_CONF).sum())
    print(f"局面: 黑{int((r['labels']==1).sum())} 白{int((r['labels']==2).sum())} "
          f"低置信点={n_low}")
    print(f"规则校验: {res['status']}" +
          (f" move={res.get('move')} by {res.get('color')}" if res['status'] == 'move' else ''))
    cv2.imwrite('/tmp/pipeline_canonical.png', r['canonical'])


if __name__ == '__main__':
    main()
