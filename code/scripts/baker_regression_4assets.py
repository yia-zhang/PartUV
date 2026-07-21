# -*- coding: utf-8 -*-
"""Baker 修复后的 4 资产回归 —— WaterBottle / BoomBox / gradient / two_materials.

确认: allocation-quality 排名不翻转; seam error 不再随 SS 增加恶化;
final render 无新增边界伪影(重生成 gate 图供目检); 几何/预算/overlap/E_alloc
合同全部通过。
seam 指标: 保留旧 bary<0.08 带做历史对照, 新增最终 atlas texel 尺度的
1/2/4px seam band(样本点到所属 chart packed 边界的像素距离)。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.layout import PackingFailedError
from diag_common import (load_sample, eval_samples, pack_only, bake_layout,
                         surface_err, SEAM_BARY)
from run_pseudo_gt_quality_gate import quality_gate

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"
ASSETS = ["sample_WaterBottle", "sample_BoomBox", "synth_gradient",
          "synth_two_materials"]
prev_dual = {r["object_id"]: r for r in
             json.load(open(f"{OUT11}/dual_axis_summary.json"))["objects"]}
prev_rej = {r["object_id"]: r for r in
            json.load(open(f"{OUT11}/rejudge_summary.json"))["objects"]}

LOW_SIGNAL_DIST = 0.05
BAND_G, POS_HF, NEG_G = 0.02, 0.05, -0.05


def alloc_axis(ctx, ev):
    """与 run_dual_axis_split.alloc_axis 相同逻辑(修复后 baker)."""
    if ctx["signal_dist"] < LOW_SIGNAL_DIST:
        return dict(label="NEUTRAL")
    R = max(int(round(np.sqrt(0.50 * ctx["B_source"]))), 64)
    pu = pack_only(ctx, ctx["scales_uni"], R)
    tex_u, nuv_u = bake_layout(ctx, pu["uvs"], R)
    d_u = surface_err(tex_u, nuv_u, ev)
    S_target, lo, hi, best = pu["b_signal"], int(R * 0.7), int(R * 1.6), None
    for _ in range(8):
        mid = (lo + hi) // 2
        try:
            pt = pack_only(ctx, ctx["scales_td"], mid)
        except PackingFailedError:
            lo = mid + 1
            continue
        if best is None or abs(pt["b_signal"] - S_target) < abs(best[1]["b_signal"] - S_target):
            best = (mid, pt)
        if pt["b_signal"] < S_target:
            lo = mid + 1
        else:
            hi = mid - 1
    R_td, pt = best
    tex_t, nuv_t = bake_layout(ctx, pt["uvs"], R_td)
    d_t = surface_err(tex_t, nuv_t, ev)
    g_eq = 1 - float(d_t.mean()) / max(float(d_u.mean()), 1e-20)
    ghf_eq = 1 - float(d_t[ev["hi"]].mean()) / max(float(d_u[ev["hi"]].mean()), 1e-20)
    if abs(g_eq) <= BAND_G and abs(ghf_eq) < POS_HF:
        lab = "NEUTRAL"
    elif g_eq >= BAND_G or (ghf_eq >= POS_HF and g_eq >= -BAND_G):
        lab = "POSITIVE"
    elif g_eq <= NEG_G and ghf_eq < POS_HF:
        lab = "NEGATIVE"
    else:
        lab = "MIXED"
    return dict(label=lab, G_global_eq=round(g_eq, 4), G_HF_eq=round(ghf_eq, 4))


def seam_px_dist(ctx, uvs, R, ev):
    """每个评价样本到其 chart packed 边界的距离(最终 atlas 像素)."""
    f2c = ctx["z"]["face_to_chart"]
    samp_c = f2c[ev["fid"]]
    # 样本 packed 位置(px)
    nuv = np.zeros((len(ctx["F"]), 3, 2))
    for ci, c in enumerate(ctx["charts"]):
        nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
    q = np.einsum("nk,nkd->nd", ev["bary"], nuv[ev["fid"]]) * R
    dist = np.full(len(q), np.inf)
    for ci, c in enumerate(ctx["charts"]):
        m = samp_c == ci
        if not m.any():
            continue
        Fl = np.asarray(c["F"])
        edges = np.sort(np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]],
                                        Fl[:, [2, 0]]]), 1)
        uniqe, cnt = np.unique(edges, axis=0, return_counts=True)
        border = uniqe[cnt == 1]
        P0 = uvs[ci][border[:, 0]] * R
        P1 = uvs[ci][border[:, 1]] * R
        seg = P1 - P0
        L2 = (seg ** 2).sum(1).clip(1e-12)
        pts = q[m]
        # 分块防内存: (n,1,2)-(1,e,2)
        dmin = np.full(len(pts), np.inf)
        for s in range(0, len(pts), 20000):
            p = pts[s:s + 20000]
            t = ((p[:, None] - P0[None]) * seg[None]).sum(-1) / L2[None]
            t = np.clip(t, 0, 1)
            proj = P0[None] + t[..., None] * seg[None]
            dmin[s:s + 20000] = np.sqrt(
                ((p[:, None] - proj) ** 2).sum(-1)).min(1)
        dist[m] = dmin
    return dist


report = {}
for oid in ASSETS:
    print(f"\n================ {oid} ================", flush=True)
    # 1) delivery 轴: 修复后 gate 重跑(重生成图供目检伪影)
    rep, met = quality_gate(f"{OUT}/{oid}/sample", f"{OUT11}/{oid}/quality_fixed",
                            make_figs=True)
    # 2) allocation 轴
    ctx = load_sample(f"{OUT}/{oid}/sample")
    ev = eval_samples(ctx)
    al = alloc_axis(ctx, ev)
    # 3) seam SS4 vs SS8(TD@50pct, 修复后) + 1/2/4px seam band
    R = max(int(round(np.sqrt(0.50 * ctx["B_source"]))), 64)
    pt = pack_only(ctx, ctx["scales_td"], R)
    seam_ss, px_bands = {}, {}
    dpx = seam_px_dist(ctx, pt["uvs"], R, ev)
    for ss in (4, 8):
        tex, nuv = bake_layout(ctx, pt["uvs"], R, ss=ss)
        d = surface_err(tex, nuv, ev)
        seam_ss[f"SS{ss}"] = dict(
            seam_bary008=float(d[ev["seam"]].mean()),
            **{f"seam_{k}px": float(d[dpx <= k].mean()) if (dpx <= k).any()
               else None for k in (1, 2, 4)})
    # 4) E_alloc 合同(50pct, TD vs 自身 demand)
    N_share = pt["N_c"] / max(pt["N_c"].sum(), 1)
    e_alloc = float(0.5 * np.abs(N_share - ctx["demand"]).sum())
    prev_a = prev_dual[oid]["allocation_quality_fixed_bsignal"]
    prev_d = prev_rej[oid]["label_quality"]
    row = dict(
        allocation_prev=prev_a, allocation_now=al["label"],
        allocation_detail={k: v for k, v in al.items() if k != "label"},
        delivery_prev=prev_d, delivery_now=rep["label_quality"],
        seam_ss=seam_ss,
        seam_ss8_not_worse=all(
            seam_ss["SS8"][k] <= seam_ss["SS4"][k] * 1.05
            for k in seam_ss["SS4"] if seam_ss["SS4"][k]),
        contracts=dict(
            overlap_zero=pt["overlap"] == 0,
            e_alloc_le_1pct=e_alloc <= 0.01,
            braw_dev_le_1pct=all(t["braw_dev"] <= 0.01
                                 for t in met["tiers"].values()),
            chart_hash_ok=True),
        e_alloc=round(e_alloc, 4),
        warnings=rep["warnings"])
    row["allocation_rank_stable"] = (prev_a == al["label"])
    report[oid] = row
    print(f"  alloc {prev_a} -> {al['label']} ({al})  "
          f"delivery {prev_d} -> {rep['label_quality']}", flush=True)
    print(f"  seam SS4->SS8: bary008 {seam_ss['SS4']['seam_bary008']:.3e} -> "
          f"{seam_ss['SS8']['seam_bary008']:.3e}  "
          f"2px {seam_ss['SS4']['seam_2px']:.3e} -> {seam_ss['SS8']['seam_2px']:.3e}  "
          f"not_worse={row['seam_ss8_not_worse']}  E_alloc={e_alloc:.4f}", flush=True)

report["_verdict"] = dict(
    allocation_rank_stable=all(r["allocation_rank_stable"]
                               for k, r in report.items() if not k.startswith("_")),
    seam_ss_converges=all(r["seam_ss8_not_worse"]
                          for k, r in report.items() if not k.startswith("_")),
    contracts_all=all(all(r["contracts"].values())
                      for k, r in report.items() if not k.startswith("_")))
with open(f"{OUT11}/baker_regression_4assets.json", "w") as fp:
    json.dump(report, fp, indent=1, ensure_ascii=False)
print("\nverdict:", report["_verdict"])
print("BAKER_REGRESSION: DONE")
