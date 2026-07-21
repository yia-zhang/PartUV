# -*- coding: utf-8 -*-
"""指标 (P0.3 / P0.4).

e_face / e_chart / e_irreducible: chart-level 投影天花板诊断.
比较在"等交付预算"归一下进行: 缩放 TD² 使 mean_A(TD²)=1 (w 已满足 mean_A(w)=1),
于是 log(TD²/w) 的绝对值有意义, 且 L1/L2 在同一预算尺度上可比.
"""
import numpy as np

from .signal import weighted_quantile


def cvw(x, w):
    m = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - m) ** 2, weights=w)) / m)


def normalize_td2(td, area, mask):
    td2 = td ** 2
    s = np.average(td2[mask], weights=area[mask])
    return td2 / max(s, 1e-18)


def chart_mean_w(charts, w, area, n_faces):
    """每面所属 chart 的面积加权平均目标 w̄_c(f) (P0.3)."""
    wbar = np.ones(n_faces)
    for c in charts:
        g = c["gidx"]
        A = area[g]
        if A.sum() <= 0:
            continue
        wbar[g] = float(np.average(w[g], weights=A))
    return wbar


def e_metrics(td, w, wbar, area, mask, charts=None):
    """面积加权 log-RMS 诊断 (P0 勘误后定义).

    e_face   = RMS_A[log(TD²/w)]
    e_chart  = RMS_A[log(TD²/w̄_arith)]      w̄_arith = 交付目标(线性均值)
    within_chart_heterogeneity = RMS_A[log(w̄_arith/w)]
        —— 相对交付目标的 chart 内残差 (非 log-RMS 最优, 勘误第3条重命名)
    cross_term: e_face² = e_chart² + within² + 2·cross 的交叉项 (精确恒等式)
    e_irreducible_log (需 charts): RMS_A[log w - mean_A(log w | chart)]
        —— log 空间逐 chart 最优尺度下的真·表达下界 (≤ within)
    等交付预算归一: TD² 与 w(及 w̄) 都归一到 mean_A=1."""
    td2n = normalize_td2(td, area, mask)
    s_w = np.average(w[mask], weights=area[mask])
    wn = w / max(s_w, 1e-18)
    wbarn = wbar / max(s_w, 1e-18)
    A = area[mask]

    def rms(x):
        return float(np.sqrt(np.average(x ** 2, weights=A)))

    lf = np.log(np.maximum(td2n[mask], 1e-18) / np.maximum(wn[mask], 1e-18))
    lc = np.log(np.maximum(td2n[mask], 1e-18) / np.maximum(wbarn[mask], 1e-18))
    li = np.log(np.maximum(wbarn[mask], 1e-18) / np.maximum(wn[mask], 1e-18))
    out = dict(e_face=rms(lf), e_chart=rms(lc),
               within_chart_heterogeneity=rms(li),
               cross_term=float(np.average(lc * li, weights=A)))
    if charts is not None:
        lw = np.log(np.maximum(wn, 1e-18))
        resid = np.zeros(len(w))
        for c in charts:
            g = c["gidx"]
            Ac = area[g]
            if Ac.sum() <= 0:
                continue
            resid[g] = lw[g] - np.average(lw[g], weights=Ac)
        out["e_irreducible_log"] = rms(resid[mask])
    return out


def top_content_gain(td_ours, td_base, cw_n, area, mask, q_top=0.9):
    """top 内容面的 TD 增益 —— P0.4: 阈值用面积加权 quantile."""
    thr = weighted_quantile(cw_n[mask], area[mask], q_top)
    hi = mask & (cw_n >= thr)
    if not hi.any():
        return float("nan")
    return float(np.average(td_ours[hi], weights=area[hi])
                 / np.average(td_base[hi], weights=area[hi]))


def h_c(charts, q, area):
    """chart 内需求异质性 (log q 相对 chart 等效倍率的面积加权 RMS)."""
    out = []
    lq = np.log(np.maximum(q, 1e-9))
    for c in charts:
        g = c["gidx"]
        A = area[g]
        if A.sum() <= 0:
            out.append(0.0)
            continue
        lqc = 0.5 * np.log(max(np.average(q[g] ** 2, weights=A), 1e-18))
        out.append(float(np.sqrt(np.average((lq[g] - lqc) ** 2, weights=A))))
    return np.asarray(out)
