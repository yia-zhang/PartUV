# -*- coding: utf-8 -*-
"""P0 驱动: 实验完整性与因果诊断 (任务书 §四).

资产: 鞋(22b822, 失效案例) + 车轮(92ff6, 成功案例) + synthetic_freq(半平坦半棋盘合成信号).
每资产输出: metrics.json(预算核算/e三指标/gate) ; 汇总 e_face/e_chart/e_irreducible 对比图.
QUALITY_GATE 本阶段一律 NOT_EVALUATED (无直接重建误差, P1 建立).
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tdlib.budget import budget_accounting, choose_resolution, rasterize_masks
from tdlib.gates import evaluate_gates
from tdlib.layout import face_td, chart_scales, layout_with_scales
from tdlib.metrics import chart_mean_w, e_metrics, h_c, top_content_gain
from tdlib.pipeline import load_reference, run_partuv
from tdlib.signal import demand_weights, luminance_std_heuristic

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT_ROOT = "/root/youjiaZhang/PartUV/code/notebook/outputs/p0"
BETA, Z_MAX, Q_MIN, Q_MAX = 0.4, 2.5, 0.5, 2.83
GUTTER_PX = 4

ASSETS = [
    ("shoe_22b822", f"{DATA}/objaverse_22b822c6520d4d49.glb", None),
    ("wheel_92ff6", f"{DATA}/objaverse_92ff65712c62408d.glb", None),
    ("synthetic_halfchecker", f"{DATA}/synthetic_freq.glb", "half_checker"),
]


def synth_texture(kind, size=1024):
    """合成表面信号 (P0 gate 要求至少一个 constant/gradient/checker 类信号)."""
    img = np.zeros((size, size, 3))
    if kind == "half_checker":                        # u<0.5 常量, u>0.5 棋盘
        img[:, : size // 2] = 0.5
        yy, xx = np.mgrid[0:size, size // 2:size]
        img[:, size // 2:] = ((xx // 16 + yy // 16) % 2)[..., None].astype(float)
    return img


def run_asset(name, path, synth):
    out = f"{OUT_ROOT}/{name}/"
    os.makedirs(out, exist_ok=True)
    pu = run_partuv(path, out)
    V, F, charts = pu["V"], pu["F"], pu["charts"]
    area, covered = pu["area"], pu["covered"]

    ref = load_reference(path, V, F, pu["mesh_scale"])
    if not ref["has_tex"]:
        return dict(name=name, error="no texture / uv")
    texA = synth_texture(synth) if synth else ref["texA"]
    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"], ref["ok_map"])
    sel = covered & ref["ok_map"]

    q, w = demand_weights(cw, sel, area, BETA, Z_MAX, Q_MIN, Q_MAX)
    # ---- P0.6 β=0 一致性: L2(β=0) 的 per-chart scale 必须与 L1 完全一致 ----
    q0, w0 = demand_weights(cw, sel, area, beta=0.0)
    s_l1 = chart_scales(charts, np.ones(len(F)))
    s_b0 = chart_scales(charts, w0)
    beta0_pass = bool(np.allclose(s_l1, s_b0, rtol=0, atol=0))

    uv_l1, _ = layout_with_scales(charts, np.ones(len(F)))
    uv_l2, _ = layout_with_scales(charts, w)
    td_l1 = face_td(charts, uv_l1, len(F))
    td_l2 = face_td(charts, uv_l2, len(F))

    # ---- P0.3 e 指标 ----
    wbar = chart_mean_w(charts, w, area, len(F))
    em_l1 = e_metrics(td_l1, w, wbar, area, sel, charts=charts)
    em_l2 = e_metrics(td_l2, w, wbar, area, sel, charts=charts)
    hc = h_c(charts, q, area)

    # ---- P0.1 预算: 目标=原资产被引用纹素(光栅 union, 原始分辨率) ----
    Ht, Wt = texA.shape[:2]
    ch0 = dict(F=ref["Fo"], gidx=np.arange(len(ref["Fo"])))
    owner0, _, _ = rasterize_masks([ch0], [ref["uv0"]], Wt, Ht)
    target = int((owner0 >= 0).sum())
    spec, actual, gap = choose_resolution(target, "preserve_at_least")
    W_at, H_at = spec

    # ---- 最终分辨率下的 rasterized 预算核算 + overlap/NaN 检查 ----
    accs = {}
    overlaps = {}
    for tag, uvs in [("L1", uv_l1), ("L2", uv_l2)]:
        owner, ov, per = rasterize_masks(charts, uvs, W_at, H_at)
        accs[tag] = budget_accounting(owner, GUTTER_PX)
        overlaps[tag] = int(ov)
    nan_count = int(sum(np.isnan(uv).sum() for uv in uv_l2))
    zero_uv = int(sum((np.asarray(td_l2[c["gidx"]]) <= 0).sum() for c in charts))

    meas = dict(
        coverage=float(covered.mean()),
        n_quarantined_reported=True,                      # 未覆盖面显式列入 metrics
        overlap_texels=overlaps["L2"],
        nan_count=nan_count, zero_uv_count=zero_uv,
        budget_gap_frac=float(gap), budget_policy_ok=bool(actual >= target),
        beta0_pass=beta0_pass,
        matcher_unique=bool(pu["match_report"]["unique"]
                            and pu["match_report"]["mismatch"] == 0),
        e_chart_L1=em_l1["e_chart"], e_chart_L2=em_l2["e_chart"],
        quality_evaluated=False, quality_pass=None,
    )
    gates = evaluate_gates(meas)

    metrics = dict(
        asset=name, mesh=os.path.basename(path), signal="synthetic:" + synth if synth else "original_texture",
        faces=int(len(F)), parts=pu["n_parts"], charts=len(charts),
        coverage=meas["coverage"],
        uncovered_faces=int((~covered).sum()),
        uncovered_area_frac=float(area[~covered].sum() / area.sum()),
        matcher=pu["match_report"],
        beta0_consistency=beta0_pass,
        budget=dict(policy="preserve_at_least",
                    target_signal_texels=target, chosen_spec=list(spec),
                    actual=actual, gap_frac=float(gap),
                    accounting_L1=accs["L1"], accounting_L2=accs["L2"],
                    overlap_L1=overlaps["L1"], overlap_L2=overlaps["L2"]),
        e_metrics=dict(L1=em_l1, L2=em_l2),
        h_c=dict(mean=float(hc.mean()), p95=float(np.quantile(hc, 0.95)),
                 n_over_025=int((hc > 0.25).sum()), n_charts=len(hc)),
        aux=dict(cvw_td2w_L1=None, top10_gain=top_content_gain(
            td_l2, td_l1, cw / max(np.median(cw[sel]), 1e-9), area, sel)),
        gates=gates,
    )
    with open(f"{out}/metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)
    return metrics


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for name, path, synth in ASSETS:
        print(f"===== {name}")
        m = run_asset(name, path, synth)
        results.append(m)
        if "error" in m:
            print("  SKIP:", m["error"])
            continue
        g = m["gates"]
        print(f"  e_chart: L1={m['e_metrics']['L1']['e_chart']:.3f} -> "
              f"L2={m['e_metrics']['L2']['e_chart']:.3f}   "
              f"e_face L2={m['e_metrics']['L2']['e_face']:.3f}   "
              f"within={m['e_metrics']['L2']['within_chart_heterogeneity']:.3f}   "
              f"irr_log={m['e_metrics']['L2']['e_irreducible_log']:.3f}   "
              f"cross={m['e_metrics']['L2']['cross_term']:+.4f}")
        print(f"  h_c>0.25: {m['h_c']['n_over_025']}/{m['h_c']['n_charts']}   "
              f"budget: target={m['budget']['target_signal_texels']} "
              f"spec={m['budget']['chosen_spec']} gap={m['budget']['gap_frac']:+.1%}")
        print(f"  GATES: {g['VALIDITY_GATE']} / {g['MECHANISM_GATE']} / "
              f"{g['QUALITY_GATE']} -> {g['FINAL_STATUS']}")
    # ---- 汇总图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [m for m in results if "error" not in m]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(ok))
    bw = 0.22
    for i, (key, lab, col) in enumerate([
            ("e_face", "e_face (L2)", "#c25b4e"),
            ("e_chart", "e_chart (L2)", "#4a7dbd"),
            ("within_chart_heterogeneity", "within-chart het.", "#8a8a8a")]):
        ax.bar(x + (i - 1) * bw, [m["e_metrics"]["L2"][key] for m in ok], bw,
               label=lab, color=col)
    ax.plot(x, [m["e_metrics"]["L1"]["e_chart"] for m in ok], "kv",
            label="e_chart (L1)", markersize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m["asset"] for m in ok], fontsize=9)
    ax.legend(fontsize=8)
    ax.set_title("P0 mechanism diagnosis: e_face / e_chart / e_irreducible", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT_ROOT}/p0_e_metrics.png", dpi=120)
    with open(f"{OUT_ROOT}/summary.json", "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\nsaved: {OUT_ROOT}/summary.json, p0_e_metrics.png")


if __name__ == "__main__":
    main()
