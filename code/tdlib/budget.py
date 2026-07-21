# -*- coding: utf-8 -*-
"""预算核算与分辨率选择 (P0.1 / P0.2).

- ALLOWED_SPECS: POT 方图 + 2:1 矩形
- choose_resolution(target, policy): preserve_at_least / hard_cap 两种明确政策
- choose_multi(target, k_max, policy): 联合离散选择(禁止逐图独立吸附)
- rasterize_masks: 最终分辨率下的 chart 像素 mask(union) -> B_signal/B_pad/B_empty/overlap
  (P0.1: 不能用连续三角形 UV 面积之和冒充像素 union)
"""
from itertools import combinations_with_replacement

import numpy as np
from scipy import ndimage


def allowed_specs(min_side=128, max_side=8192, ratios=(1, 2)):
    specs = []
    s = min_side
    while s <= max_side:
        for r in ratios:
            if s * r <= max_side:
                specs.append((s * r, s))          # (W,H), W>=H
        s *= 2
    return sorted(set(specs), key=lambda wh: wh[0] * wh[1])


ALLOWED_SPECS = allowed_specs()


def choose_resolution(target_texels, policy="preserve_at_least", specs=None):
    """单图集规格选择. 返回 (spec, actual, gap_frac)."""
    specs = specs or ALLOWED_SPECS
    areas = [(w * h, (w, h)) for w, h in specs]
    if policy == "preserve_at_least":
        cands = [a for a in areas if a[0] >= target_texels]
        if not cands:
            a = max(areas)
        else:
            a = min(cands)
    elif policy == "hard_cap":
        cands = [a for a in areas if a[0] <= target_texels]
        a = max(cands) if cands else min(areas)
    else:
        raise ValueError(policy)
    actual, spec = a
    return spec, actual, (actual - target_texels) / max(target_texels, 1)


def choose_multi(target_texels, k_max=3, policy="preserve_at_least",
                 specs=None, lambda_k=0.01):
    """多图集联合选择: 最小化 |预算误差比例| + λ·图集数.
    返回 (spec 列表, actual, gap_frac). 满足政策方向约束."""
    specs = specs or ALLOWED_SPECS
    best = None
    for k in range(1, k_max + 1):
        for combo in combinations_with_replacement(specs, k):
            actual = sum(w * h for w, h in combo)
            if policy == "preserve_at_least" and actual < target_texels:
                continue
            if policy == "hard_cap" and actual > target_texels:
                continue
            err = abs(actual - target_texels) / max(target_texels, 1)
            cost = err + lambda_k * k
            if best is None or cost < best[0]:
                best = (cost, list(combo), actual)
    if best is None:                                  # 政策方向上无解, 放宽
        spec, actual, _ = choose_resolution(target_texels, "preserve_at_least", specs)
        best = (0, [spec], actual)
    _, combo, actual = best
    return combo, actual, (actual - target_texels) / max(target_texels, 1)


def rasterize_masks(charts, uvs, W, H):
    """光栅化所有 chart -> (owner map, overlap_count, per-chart texels).
    owner: (H,W) int32, -1=空; overlap: 被 >=2 个 chart 覆盖的纹素数."""
    from . import gpu
    if gpu.available():
        return gpu.rasterize_masks(charts, uvs, W, H)
    owner = np.full((H, W), -1, np.int32)
    overlap = 0
    per_chart = np.zeros(len(charts), np.int64)
    for ci, (c, uvc) in enumerate(zip(charts, uvs)):
        cF = np.asarray(c["F"])
        P_all = np.stack([uvc[:, 0] * W, (1 - uvc[:, 1]) * H], 1)
        for tri in cF:
            P = P_all[tri]
            mn = np.maximum(np.floor(P.min(0)).astype(int), 0)
            mx = np.minimum(np.ceil(P.max(0)).astype(int), [W - 1, H - 1])
            if (mx < mn).any():
                continue
            xs, ys = np.meshgrid(np.arange(mn[0], mx[0] + 1),
                                 np.arange(mn[1], mx[1] + 1))
            pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
            T = np.stack([P[1] - P[0], P[2] - P[0]], 1)
            det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
            if abs(det) < 1e-12:
                continue
            invT = np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]]) / det
            w12 = (pts - P[0]) @ invT.T
            w0 = 1 - w12.sum(1)
            m = (w12[:, 0] >= -1e-6) & (w12[:, 1] >= -1e-6) & (w0 >= -1e-6)
            if not m.any():
                continue
            yy, xx = ys.ravel()[m], xs.ravel()[m]
            prev = owner[yy, xx]
            clash = (prev >= 0) & (prev != ci)
            overlap += int(clash.sum())
            owner[yy, xx] = ci
            per_chart[ci] += int(m.sum()) - int(clash.sum())
    return owner, overlap, per_chart


def budget_accounting(owner, gutter_px=4):
    """B_raw / B_signal / B_pad / B_empty (P0.1)."""
    H, W = owner.shape
    signal = owner >= 0
    dilated = ndimage.binary_dilation(signal, iterations=gutter_px)
    b_signal = int(signal.sum())
    b_pad = int(dilated.sum()) - b_signal
    b_raw = W * H
    return dict(B_raw=b_raw, B_signal=b_signal, B_pad=b_pad,
                B_empty=b_raw - b_signal - b_pad,
                signal_frac=b_signal / b_raw)
