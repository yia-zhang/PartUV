# -*- coding: utf-8 -*-
"""内容信号与预算权重.

luminance_std_heuristic: 面内重心采样的亮度标准差 —— 明确命名为 heuristic
  (P0.4: 不得称为 oracle/frequency; 它只测对比幅值, 不测空间频率)
demand_weights: 面积加权 log 域 z-score -> q(线性TD倍率) -> 预算归一 -> 裁剪 -> w=q²
  β=0 时严格返回全 1 (P0.6 一致性由 tests/test_beta0.py 验证)
"""
import numpy as np


def weighted_quantile(vals, weights, q):
    o = np.argsort(vals)
    cw = np.cumsum(weights[o])
    cw = cw / cw[-1]
    return float(np.interp(q, cw, vals[o]))


def luminance_std_heuristic(texA, uv0, Fo, f2o, ok_map, n_samples=24, seed=0):
    """逐面亮度标准差 (nearest 采样; 已知局限: 频率不敏感/依赖三角化, 见任务书§二)."""
    rng = np.random.RandomState(seed)
    bar = rng.dirichlet((1.2, 1.2, 1.2), n_samples)
    lum = texA @ np.array([0.299, 0.587, 0.114])
    OUV3 = np.asarray(uv0, float)[Fo[f2o]]
    samp = np.einsum("sk,fkd->sfd", bar, OUV3)
    acc = acc2 = None
    for s in range(n_samples):
        x = np.clip(samp[s, :, 0], 0, 1) * (lum.shape[1] - 1)
        y = np.clip(1 - samp[s, :, 1], 0, 1) * (lum.shape[0] - 1)
        v = lum[y.astype(int), x.astype(int)]
        acc = v.copy() if acc is None else acc + v
        acc2 = v ** 2 if acc2 is None else acc2 + v ** 2
    cw = np.sqrt(np.maximum(acc2 / n_samples - (acc / n_samples) ** 2, 0))
    cw[~ok_map] = 0.0
    return cw


def demand_weights(cw, sel, area, beta=0.4, z_max=2.5, q_min=0.5, q_max=2.83,
                   n_norm_iters=3):
    """cw -> (q, w). sel: 参与统计的面 mask; area: 3D 面积权重.
    保证 (数值上) mean_A(q²)≈1; β=0 时精确返回全 1."""
    nF = len(cw)
    q = np.ones(nF)
    if beta == 0.0 or not np.any(sel):
        return q, q ** 2
    lcw = np.log(1e-8 + cw[sel])
    A = area[sel]
    med = weighted_quantile(lcw, A, 0.5)
    mad = max(weighted_quantile(np.abs(lcw - med), A, 0.5), 1e-6)
    z = (lcw - med) / mad
    qs = np.exp(beta * np.clip(z, -z_max, z_max))
    for _ in range(n_norm_iters):
        qs = qs / np.sqrt(np.average(qs ** 2, weights=A))
        qs = np.clip(qs, q_min, q_max)
    q[sel] = qs
    return q, q ** 2
