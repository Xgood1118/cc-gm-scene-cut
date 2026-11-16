# -*- coding: utf-8 -*-
"""基于 RGB 三维直方图的场景切换检测。

核心约定：

* ``histogram`` 返回长度为 ``3 * bins`` 的扁平列表：R、G、B 三个通道各
  ``bins`` 个桶，每个桶的值是该通道内像素的占比（三通道合计为 1）。
* ``hist_distance`` 计算归一化的逐通道一维推土机距离（EMD，也称
  Wasserstein-1 距离），结果位于 ``[0, 1]``。0 表示两个直方图完全相同，
  1 表示两个分布完全不相交。
* ``detect`` 使用相邻帧距离曲线找切换点，并做两项时间连续性检查：
  抑制单帧闪光造成的“进入/退出”两次假切换；把连续多帧都超阈值的
  缓慢变化合并成一次切换（标注为渐变末尾）。
"""


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
    """返回三个通道拼接的归一化直方图，长度为 ``3 * bins``。

    R、G、B 各自独立分桶，每个通道内像素占比之和为 1，因此整个列表
    的元素之和也为 1。``bins`` 越大，颜色量化越细，相邻颜色的区分度
    越高。
    """
    if bins < 2:
        raise ValueError("bins 必须 >= 2")
    width, height = frame.width, frame.height
    total = width * height
    channels = [[0] * bins for _ in range(3)]
    data = frame.data
    # 裸 RGB24，按像素交错存放：R G B R G B ...
    for p in range(total):
        i = p * 3
        channels[0][data[i] * bins // 256] += 1
        channels[1][data[i + 1] * bins // 256] += 1
        channels[2][data[i + 2] * bins // 256] += 1

    scale = 1.0 / (3.0 * total) if total else 0.0
    flat = []
    for channel in channels:
        flat.extend(value * scale for value in channel)
    return flat


def _channel_emd(a, b):
    """单通道的一维 EMD：把 a 变成 b 需要搬运的概率质量（按值域归一）。

    对于分布在桶上的直方图，一维 EMD 等于两个累积分布函数差值的积分。
    桶索引 0..bins-1 对应值域 0..255，除以 ``bins - 1`` 后最大值为 1。
    """
    size = len(a)
    if size < 2:
        return 0.0
    mass = sum(a)
    if mass <= 0:
        return 0.0
    cumulative = 0.0
    earth = 0.0
    for k in range(size - 1):
        cumulative += a[k] - b[k]
        earth += abs(cumulative)
    earth /= mass
    earth /= float(size - 1)
    # 浮点误差可能让完全相同（或极端）的分布略微越界。
    if earth < 0.0:
        return 0.0
    if earth > 1.0:
        return 1.0
    return earth


def hist_distance(a, b):
    """归一化直方图距离，范围 ``[0, 1]``。

    输入是 ``histogram`` 产生的扁平列表（长度 ``3 * bins``，按 R、G、B
    依次排列）。函数对每个通道分别计算一维 EMD，再对三个通道取平均：

        d = (emd_R + emd_G + emd_B) / 3

    因此阈值 0.25 的物理含义是：平均下来，每个颜色通道的像素强度分布
    需要整体搬移约 25% 的值域（0-255 的 64 个灰阶）才能从 a 变成 b。
    若传入长度不是 3 的倍数的列表，则按单通道直方图处理，便于单独
    分析某个通道。
    """
    if len(a) != len(b):
        raise ValueError("两个直方图长度必须相同")
    if not a:
        return 0.0
    if len(a) % 3 == 0:
        channel_count = 3
        size = len(a) // 3
    else:
        channel_count = 1
        size = len(a)

    total = 0.0
    for c in range(channel_count):
        start = c * size
        total += _channel_emd(a[start:start + size], b[start:start + size])
    distance = total / channel_count
    if distance < 0.0:
        return 0.0
    if distance > 1.0:
        return 1.0
    return distance


def distance_curve(frames, bins=8):
    """返回逐帧距离曲线，长度为 ``len(frames)``。

    ``curve[i]`` 是第 i 帧与前一帧的归一化直方图距离；第 0 帧没有前驱，
    距离记为 0.0。
    """
    curve = [0.0] * len(frames)
    previous = None
    for i, frame in enumerate(frames):
        current = histogram(frame, bins)
        if previous is not None:
            curve[i] = hist_distance(previous, current)
        previous = current
    return curve


def _suppress_flashes(histograms, distances, threshold):
    """找出单帧闪光，返回被抑制的切换下标集合。

    闪光帧 i 的特征：进入（i-1 -> i）与离开（i -> i+1）都超阈值，但闪光
    前后的两帧彼此相似。真正的切换会改变后续场景，``i-1`` 与 ``i+1``
    的距离仍然很大。
    """
    suppressed = set()
    for i in range(1, len(histograms) - 1):
        if distances[i] >= threshold and distances[i + 1] >= threshold:
            before_after = hist_distance(histograms[i - 1], histograms[i + 1])
            if before_after < threshold:
                suppressed.add(i)
                suppressed.add(i + 1)
    return suppressed


def _segments(candidates, merge_gap=2):
    """把候选下标按时间聚成段，段内允许 ``merge_gap`` 帧的低于阈值间隙。

    渐变时每帧的变化量可能不足以稳定超过阈值（例如某些帧恰好落在同一
    桶里），但整体趋势是连续多帧的累积变化。把间隔很小的候选合并成
    一段，才能把它识别为一次渐变，而不是一串独立切换。
    """
    segments = []
    for index in candidates:
        if segments and index - segments[-1][-1] <= merge_gap:
            segments[-1].append(index)
        else:
            segments.append([index])
    return segments


def _classify_segment(indices, distances):
    """把一段候选切换拆成 ``[(下标, 类型), ...]``。

    段内至少 3 个候选说明变化在持续累积，是渐变段；只报告段末一个点
    （类型“渐变末尾”），避免渐变被拆成一串切换。段内只有 1-2 个候选
    时无法确认是持续渐变，逐个按“突变”报告，避免把相距两帧的两次真实
    硬切误合并。

    若段末距离明显高于段内其余步长（至少 1.5 倍），说明这一帧是独立的
    硬切（例如渐变之后紧接黑场），先把它作为“突变”报告，再递归处理
    前面的渐变部分。
    """
    if len(indices) < 3:
        return [(index, "突变") for index in indices]

    last = indices[-1]
    prefix = indices[:-1]
    inner = sorted(distances[i] for i in prefix)
    typical = inner[len(inner) // 2]
    if typical > 0.0 and distances[last] >= 1.5 * typical:
        return _classify_segment(prefix, distances) + [(last, "突变")]
    return [(last, "渐变末尾")]


def detect(frames, threshold=0.25, bins=8):
    """检测切换点。

    参数
    ----
    frames : list[Frame]
    threshold : float
        归一化距离阈值，范围 ``[0, 1]``；含义见 :func:`hist_distance`。
    bins : int
        每个颜色通道的直方图桶数。

    返回
    ----
    list[dict]
        按帧号排序的切换点，元素形如
        ``{"frame": 10, "distance": 0.5238, "kind": "突变"}``。
        ``kind`` 为 ``"突变"`` 或 ``"渐变末尾"``。
    """
    if not frames:
        return []

    histograms = [histogram(frame, bins) for frame in frames]
    count = len(frames)
    distances = [0.0] * count
    for i in range(1, count):
        distances[i] = hist_distance(histograms[i - 1], histograms[i])

    suppressed = _suppress_flashes(histograms, distances, threshold)
    candidates = [
        i for i in range(1, count)
        if distances[i] >= threshold and i not in suppressed
    ]

    cuts = []
    for segment in _segments(candidates):
        for index, kind in _classify_segment(segment, distances):
            cuts.append({
                "frame": frames[index].index,
                "distance": round(distances[index], 4),
                "kind": kind,
            })
    return cuts


def main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="scenecut")
    ap.add_argument("raw")
    ap.add_argument("--size", default="64,48")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.25],
                    help="一个或多个阈值；第一个用于详细输出，全部用于数量对比")
    ap.add_argument("--bins", type=int, default=8, help="每个颜色通道的桶数")
    ap.add_argument("--out", default="cuts.json")
    args = ap.parse_args(argv)

    width, height = [int(v) for v in args.size.split(",")]
    frames = load_frames(args.raw, width, height, args.frames)
    curve = distance_curve(frames, args.bins)

    primary = args.threshold[0]
    cuts = detect(frames, primary, args.bins)
    by_frame = {}
    for cut in cuts:
        by_frame[cut["frame"]] = cut

    print("共 %d 帧，桶数 %d，阈值 %.2f" % (len(frames), args.bins, primary))
    print("帧\t距离")
    for i, frame in enumerate(frames):
        cut = by_frame.get(frame.index)
        marker = ""
        if cut is not None:
            marker = "\t<-- 切换（%s）" % cut["kind"]
        print("%d\t%.6f%s" % (frame.index, curve[i], marker))

    print("")
    print("切换点：")
    if cuts:
        for cut in cuts:
            print("  第 %d 帧  距离=%.4f  类型=%s"
                  % (cut["frame"], cut["distance"], cut["kind"]))
    else:
        print("  无")

    comparisons = []
    for threshold in args.threshold:
        found = detect(frames, threshold, args.bins)
        comparisons.append({
            "threshold": threshold,
            "count": len(found),
            "frames": [cut["frame"] for cut in found],
        })
    if len(comparisons) > 1:
        print("")
        print("不同阈值检出数量对比：")
        for item in comparisons:
            print("  阈值 %.2f -> %d 个 %s"
                  % (item["threshold"], item["count"], item["frames"]))

    payload = {
        "frames": len(frames),
        "bins": args.bins,
        "threshold": primary,
        "distance_curve": [round(value, 6) for value in curve],
        "cuts": cuts,
        "threshold_comparison": comparisons,
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return 0
