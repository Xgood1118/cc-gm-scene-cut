# -*- coding: utf-8 -*-
"""生成 40 帧裸 RGB：0-9 蓝天，10-19 突变到红色，20-29 缓慢变绿，30-39 突变到暗场。"""
import struct
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 64, 48
FRAMES = 40


def main():
    out = bytearray()
    for i in range(FRAMES):
        for y in range(H):
            for x in range(W):
                if i < 10:
                    r, g, b = 40, 90, 200
                elif i < 20:
                    r, g, b = 200, 40, 40
                elif i < 30:
                    k = i - 20
                    r, g, b = 200 - k * 12, 40 + k * 12, 40
                else:
                    r, g, b = 12, 12, 12
                out.extend((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "frames.raw")
    with open(path, "wb") as f:
        f.write(bytes(out))
    print("已生成", path, FRAMES, "帧", W, "x", H)


if __name__ == "__main__":
    main()
