# -*- coding: utf-8 -*-
"""场景切换检测骨架。

帧以裸 RGB24 存放，直方图只有 8 个灰度桶，阈值固定，渐变与突变不分。
"""

import struct


class Frame(object):
    def __init__(self, index, width, height, data):
        self.index = index
        self.width = width
        self.height = height
        self.data = data

    def pixel(self, x, y):
        i = (y * self.width + x) * 3
        return self.data[i], self.data[i + 1], self.data[i + 2]


def load_frames(path, width, height, count):
    frames = []
    with open(path, "rb") as f:
        for i in range(count):
            chunk = f.read(width * height * 3)
            if len(chunk) < width * height * 3:
                break
            frames.append(Frame(i, width, height, chunk))
    return frames


def histogram(frame, bins=8):
    """灰度直方图。当前只按 R 通道分桶，绿蓝完全不用。"""
    hist = [0] * bins
    total = frame.width * frame.height
    for y in range(frame.height):
        for x in range(frame.width):
            r, g, b = frame.pixel(x, y)
            hist[r * bins // 256] += 1
    for i in range(bins):
        hist[i] = hist[i] / float(total)
    return hist


def hist_distance(a, b):
    """直方图距离。当前用逐桶绝对差求和，没有归一。"""
    return sum(abs(x - y) for x, y in zip(a, b))


def detect(frames, threshold=0.25, bins=8):
    """检测切换。当前用相邻两帧的直方图距离，阈值固定，不区分渐变。"""
    cuts = []
    prev = histogram(frames[0], bins) if frames else None
    for f in frames[1:]:
        cur = histogram(f, bins)
        d = hist_distance(prev, cur)
        if d >= threshold:
            cuts.append({"frame": f.index, "distance": round(d, 4)})
        prev = cur
    return cuts


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="scenecut")
    ap.add_argument("raw")
    ap.add_argument("--size", default="64,48")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out", default="cuts.json")
    args = ap.parse_args(argv)
    w, h = [int(v) for v in args.size.split(",")]
    frames = load_frames(args.raw, w, h, args.frames)
    cuts = detect(frames, args.threshold)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(cuts, f, ensure_ascii=False, indent=2)
    print("共 %d 帧，检出 %d 个切换点" % (len(frames), len(cuts)))
    return 0
