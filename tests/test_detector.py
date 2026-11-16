# -*- coding: utf-8 -*-
"""检测器行为测试：三通道、归一区间、渐变、闪光、距离曲线、阈值可调。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenecut.core import (  # noqa: E402
    Frame,
    detect,
    distance_curve,
    hist_distance,
    histogram,
    load_frames,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "frames.raw")


def solid(index, color, width=8, height=6):
    """构造整帧同色的 Frame。"""
    r, g, b = color
    return Frame(index, width, height, bytes([r, g, b]) * (width * height))


def sample_frames():
    return load_frames(RAW, 64, 48, 40)


def test_three_channels_participate():
    bins = 8
    red = histogram(solid(0, (255, 0, 0)), bins)
    green = histogram(solid(1, (0, 255, 0)), bins)
    blue = histogram(solid(2, (0, 0, 255)), bins)

    def channel(hist, c):
        return hist[c * bins:(c + 1) * bins]

    # 红到绿：R 通道从高端桶移到低端桶，G 通道相反。
    assert channel(red, 0)[7] > 0 and channel(red, 0)[0] == 0
    assert channel(green, 0)[0] > 0 and channel(green, 0)[7] == 0
    assert channel(red, 1)[0] > 0 and channel(green, 1)[7] > 0
    # 红、绿两帧在 B 通道上完全一样，但 R/G 差异足以拉开距离。
    assert channel(red, 2) == channel(green, 2)
    assert hist_distance(red, green) > 0.5
    # 蓝通道也参与距离：蓝到红、蓝到绿都要能被测到。
    assert hist_distance(blue, red) > 0.5
    assert hist_distance(blue, green) > 0.5


def test_distance_is_normalized():
    a = histogram(solid(0, (0, 0, 0)), 16)
    b = histogram(solid(1, (255, 255, 255)), 16)
    c = histogram(solid(2, (10, 200, 30)), 16)
    assert hist_distance(a, a) == 0.0
    assert abs(hist_distance(a, b) - 1.0) < 1e-9
    assert 0.0 <= hist_distance(a, c) <= 1.0
    assert hist_distance(a, c) == hist_distance(c, a)


def test_more_bins_improve_discrimination():
    dark = solid(0, (0, 0, 0))
    almost = solid(1, (31, 31, 31))
    # 粗桶把 0 与 31 放进同一桶，看不见差异。
    assert hist_distance(histogram(dark, 4), histogram(almost, 4)) == 0.0
    # 细桶能把它们分开。
    assert hist_distance(histogram(dark, 32), histogram(almost, 32)) > 0.05

    # 样本数据上，细桶让“切换”相对“渐变”的区分度更高。
    frames = sample_frames()
    coarse = distance_curve(frames, 4)
    fine = distance_curve(frames, 64)
    coarse_ratio = coarse[10] / max(coarse[20:30])
    fine_ratio = fine[10] / max(fine[20:30])
    assert fine_ratio > coarse_ratio


def test_histogram_bins_configurable():
    frame = solid(0, (128, 128, 128))
    for bins in (2, 8, 16, 32):
        hist = histogram(frame, bins)
        assert len(hist) == 3 * bins
        assert abs(sum(hist) - 1.0) < 1e-9


def test_distance_curve_length():
    frames = sample_frames()
    curve = distance_curve(frames, 8)
    assert len(curve) == len(frames)
    assert curve[0] == 0.0
    assert len(distance_curve([], 8)) == 0


def test_sample_cuts_at_default_threshold():
    frames = sample_frames()
    cuts = detect(frames, 0.25)
    assert [cut["frame"] for cut in cuts] == [10, 30]
    assert all(cut["kind"] == "突变" for cut in cuts)


def test_gradient_not_reported_as_series():
    frames = sample_frames()
    for threshold in (0.25, 0.10, 0.049):
        frames_hit = [cut["frame"] for cut in detect(frames, threshold)]
        assert 10 in frames_hit
        assert 30 in frames_hit
        assert not [f for f in frames_hit if 20 <= f <= 28]


def test_gradient_end_is_labeled():
    frames = sample_frames()
    cuts = detect(frames, 0.02)
    ramp = [cut for cut in cuts if 20 <= cut["frame"] <= 29]
    # 低阈值下渐变会露出，但只能合并成一个点，且标为渐变末尾。
    assert len(ramp) == 1
    assert ramp[0]["frame"] == 29
    assert ramp[0]["kind"] == "渐变末尾"
    assert [cut["frame"] for cut in cuts if cut["kind"] == "突变"] == [10, 30]


def test_single_flash_suppressed():
    normal = (40, 90, 200)
    flash = (255, 255, 255)
    frames = [
        solid(0, normal),
        solid(1, normal),
        solid(2, flash),
        solid(3, normal),
        solid(4, normal),
    ]
    assert detect(frames, 0.25) == []


def test_real_cut_not_suppressed():
    a = (40, 90, 200)
    b = (200, 40, 40)
    frames = [solid(0, a), solid(1, a), solid(2, b), solid(3, b)]
    cuts = detect(frames, 0.25)
    assert [cut["frame"] for cut in cuts] == [2]
    assert cuts[0]["kind"] == "突变"


def test_threshold_changes_count():
    frames = sample_frames()
    assert len(detect(frames, 0.60)) == 0
    assert len(detect(frames, 0.25)) == 2
    assert len(detect(frames, 0.02)) == 3


def test_empty_input():
    assert detect([], 0.25) == []
    assert distance_curve([], 8) == []
