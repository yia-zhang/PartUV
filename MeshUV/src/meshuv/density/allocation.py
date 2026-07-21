# -*- coding: utf-8 -*-
"""Chart texel allocation(移植已冻结 teacher 公式, β=0.25, 线性 TD 语义).

demand_weights: 面积加权 log 域 z-score(med/MAD) -> q=exp(β·clip(z,±2.5))
-> 预算归一 mean_A(q²)=1(3 迭代) -> clip[0.5,2.83] -> w=q²; β=0 严格全 1。
标签(全部 chart 级):
- chart_target_area_fraction = Σ(area·w)/总和(=需求份额, 恒 >=0 且 sum=1)
- chart_log_density_ratio = mean_centered(0.5·log(demand_share/area_share))
  —— 线性纹素密度 log 比(linear_texel_density_log_ratio_v1)
标签生成禁止 packing/rebake/质量筛选(那些属于 Gold evaluator)。"""
import numpy as np

BETA = 0.25
LABEL_SEMANTICS = "linear_texel_density_log_ratio_v1"


def weighted_quantile(vals, weights, q):
    o = np.argsort(vals)
    cw = np.cumsum(weights[o])
    cw = cw / cw[-1]
    return float(np.interp(q, cw, vals[o]))


def demand_weights(cw, sel, area, beta=BETA, z_max=2.5, q_min=0.5, q_max=2.83,
                   n_norm_iters=3):
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


def chart_targets(chartset, content_score, beta=BETA):
    """ChartSet + 每面内容分 -> chart 级标签 dict(含合法性保证)."""
    f2c, fa = chartset["face_to_chart"], chartset["face_area"]
    sel = chartset["covered"] & chartset["source_uv_valid"]
    nC = chartset["n_charts"]
    _, w = demand_weights(content_score, sel, fa, beta=beta)
    dem = np.zeros(nC)
    A3 = np.zeros(nC)
    m = f2c >= 0
    np.add.at(dem, f2c[m], (fa * w)[m])
    np.add.at(A3, f2c[m], fa[m])
    valid = (dem > 0) & (A3 > 0)
    dshare = dem / max(dem.sum(), 1e-20)
    ashare = A3 / max(A3.sum(), 1e-20)
    logr = np.zeros(nC)
    logr[valid] = 0.5 * np.log(np.maximum(dshare[valid], 1e-20)
                               / np.maximum(ashare[valid], 1e-20))
    logr[valid] -= logr[valid].mean()               # mean-centered
    chart_cs = np.zeros(nC)
    np.add.at(chart_cs, f2c[m], (fa * content_score)[m])
    chart_cs[valid] /= np.maximum(A3[valid], 1e-20)
    lab = dict(chart_surface_area=A3.astype(np.float32),
               chart_target_area_fraction=dshare.astype(np.float32),
               chart_log_density_ratio=logr.astype(np.float32),
               chart_valid_mask=valid,
               chart_content_score=chart_cs.astype(np.float32))
    assert (lab["chart_target_area_fraction"] >= 0).all()
    assert abs(float(lab["chart_target_area_fraction"].sum()) - 1) < 1e-5
    assert np.isfinite(lab["chart_log_density_ratio"][valid]).all()
    assert abs(float(lab["chart_log_density_ratio"][valid].mean())) < 1e-5
    return lab
