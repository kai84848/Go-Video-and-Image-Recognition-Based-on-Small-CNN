# -*- coding: utf-8 -*-
"""从 seg_0017（10 被误读为 16）提取 0 模板，保存完整模板库并复验。"""
import glob
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gt.test_ocr import get_patches, SAMPLES
from gt.digit_ocr import DigitOCR, digit_mask, _components


def base_ocr():
    video_path = glob.glob('数据集/gt_candidates/BV1rnsS6uEzy*.mp4')[0]
    _, _, _, patches = get_patches(video_path, 700)
    ocr = DigitOCR()
    for r, c, d in SAMPLES:
        color, patch = patches[(r, c)]
        ocr.add_sample(patch, color, d)
    return ocr


def main():
    ocr = base_ocr()
    video_path = glob.glob('数据集/gt_candidates/BV1rnsS6uEzy*.mp4')[0]
    segs = json.load(open('gt/out/BV1rnsS6uEzy/segments_data.json'))
    seg = next(s for s in segs if s['index'] == 17)
    t = seg['rep_t']
    img, pts, labels, patches = get_patches(video_path, t)
    print('修复前该帧识别:')
    for d in seg['digits']:
        key = (d['r'], d['c'])
        if key not in patches:
            continue
        color, patch = patches[key]
        mask = digit_mask(patch, color)
        chars = _components(mask) if mask is not None else []
        num, score = ocr.recognize(patch)
        print(f"  {key} gt_num={d['num']} chars={len(chars)} -> {num} ({score:.2f})")

    # num=16 的点即被误读的 10：第二位是 0
    d16 = next(d for d in seg['digits'] if d['num'] == 16)
    color, patch = patches[(d16['r'], d16['c'])]
    mask = digit_mask(patch, color)
    chars = _components(mask)
    assert len(chars) == 2, f'期望 2 位，实际 {len(chars)}'
    ocr.templates.setdefault((0, color), []).append(chars[1])
    print(f"已加入 0 模板（来自 {d16['r']},{d16['c']}，宽 {chars[1].shape[1]}）")

    print('修复后该帧全部点识别:')
    for key, (color, patch) in sorted(patches.items()):
        num, score = ocr.recognize(patch)
        if num is not None:
            print(f'  {key} -> {num} ({score:.2f})')
    ocr.save('gt/templates_extra.pkl')
    print('模板库已保存 gt/templates_extra.pkl，共', len(ocr.templates), '类')


if __name__ == '__main__':
    main()
