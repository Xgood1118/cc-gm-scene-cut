# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scenecut.core import Frame, histogram, load_frames  # noqa: E402


def test_frame_pixel():
    data = bytes([10, 20, 30, 40, 50, 60])
    f = Frame(0, 2, 1, data)
    assert f.pixel(0, 0) == (10, 20, 30)
    assert f.pixel(1, 0) == (40, 50, 60)


def test_histogram_sum():
    f = Frame(0, 4, 4, bytes([128, 128, 128] * 16))
    hist = histogram(f, 8)
    assert abs(sum(hist) - 1.0) < 1e-8


def test_load_frames():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "frames.raw")
    if not os.path.exists(path):
        return
    frames = load_frames(path, 64, 48, 40)
    assert len(frames) == 40
