# -*- coding: utf-8 -*-
"""Metric Rebaseline —— 修复后评测器(coverage baker + texel-center 约定)复跑
全部 17 个 development cases, 与旧分类逐资产 diff。

不重跑 PartUV/export(processing_status 与 structural_status 沿用 V1.1);
不下载新资产; 输出写新目录 outputs/pilot_v2_rebaseline(不覆盖旧报告)。
产物: summary.json(双轴+diff) + metric_lineage.json(血缘)。
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
from diag_common import load_sample, eval_samples, pack_only, bake_layout, surface_err
from run_pseudo_gt_quality_gate import quality_gate

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"
OUT2 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v2_rebaseline"
os.makedirs(OUT2, exist_ok=True)
rej = {r["object_id"]: r for r in
       json.load(open(f"{OUT11}/rejudge_summary.json"))["objects"]}
dual = {r["object_id"]: r for r in
        json.load(open(f"{OUT11}/dual_axis_summary.json"))["objects"]}

LOW_SIGNAL_DIST = 0.05
BAND_G, POS_HF, NEG_G = 0.02, 0.05, -0.05


def alloc_axis(ctx, ev):
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
    return dict(label=lab, G_global_eq=round(g_eq, 4), G_HF_eq=round(ghf_eq, 4),
                overlap=pt["overlap"] + pu["overlap"])


rows, new_protocol_hash = [], None
for oid, r in rej.items():
    print(f"\n================ {oid} ================", flush=True)
    row = dict(object_id=oid, category=r["category"],
               processing_status=r["processing_status"],
               structural_status=r["structural_status"] or "-",
               allocation_quality_fixed_bsignal="NOT_EVALUATED",
               delivery_quality_fixed_braw=r["label_quality"]
               if r["label_quality"] == "NOT_EVALUATED" else "",
               allocation_detail={}, delivery_borderline=False,
               delivery_warnings=[], signal_dist=r.get("signal_dist"))
    sd = f"{OUT}/{oid}/sample"
    evaluable = (r["processing_status"] == "OK"
                 and r["structural_status"] == "ACCEPTED"
                 and os.path.exists(f"{sd}/manifest.json"))
    if evaluable:
        rep, met = quality_gate(sd, f"{OUT2}/{oid}/quality", make_figs=False)
        new_protocol_hash = rep["protocol_hash"]
        row["delivery_quality_fixed_braw"] = rep["label_quality"]
        row["delivery_borderline"] = rep["label_quality_borderline"]
        row["delivery_warnings"] = rep["warnings"]
        row["G_global"] = {t: met["tiers"][t]["G_global"] for t in met["tiers"]}
        row["G_HF"] = {t: met["tiers"][t]["G_HF"] for t in met["tiers"]}
        row["ssim_delta_mean"] = {t: met["tiers"][t]["ssim_delta_mean"]
                                  for t in met["tiers"]}
        try:
            ctx = load_sample(sd)
            ev = eval_samples(ctx)
            al = alloc_axis(ctx, ev)
            row["allocation_quality_fixed_bsignal"] = al.pop("label")
            row["allocation_detail"] = al
        except Exception as e:
            row["allocation_detail"] = dict(
                error=f"{type(e).__name__}: {str(e)[:120]}")
    else:
        row["delivery_quality_fixed_braw"] = "NOT_EVALUATED"
    ok_geom = evaluable and row["allocation_detail"].get("overlap", 0) == 0
    row["training_eligible"] = dict(
        td_allocation=row["allocation_quality_fixed_bsignal"]
        in ("POSITIVE", "NEUTRAL"),
        local_uv_refinement=False, packed_uv_regression=False)
    row["artifact_valid"] = dict(packed_layout=bool(ok_geom), rebaked_asset=False)
    old_a = dual[oid]["allocation_quality_fixed_bsignal"]
    old_d = rej[oid]["label_quality"]
    row["diff"] = dict(
        allocation_old=old_a, delivery_old=old_d,
        allocation_changed=old_a != row["allocation_quality_fixed_bsignal"],
        delivery_changed=old_d != row["delivery_quality_fixed_braw"])
    print(f"  alloc {old_a} -> {row['allocation_quality_fixed_bsignal']} "
          f"{row['allocation_detail']}", flush=True)
    print(f"  delivery {old_d} -> {row['delivery_quality_fixed_braw']}"
          f"{'(BORDERLINE)' if row['delivery_borderline'] else ''} "
          f"warn={len(row['delivery_warnings'])}", flush=True)
    rows.append(row)

cnt = lambda key: {s: sum(1 for r in rows if r[key] == s) for s in
                   ("POSITIVE", "NEUTRAL", "MIXED", "NEGATIVE", "NOT_EVALUATED")}
changed = [dict(object_id=r["object_id"],
                allocation=f"{r['diff']['allocation_old']} -> "
                           f"{r['allocation_quality_fixed_bsignal']}"
                if r["diff"]["allocation_changed"] else "unchanged",
                delivery=f"{r['diff']['delivery_old']} -> "
                         f"{r['delivery_quality_fixed_braw']}"
                if r["diff"]["delivery_changed"] else "unchanged")
           for r in rows if r["diff"]["allocation_changed"]
           or r["diff"]["delivery_changed"]]
summary = dict(
    semantics=dict(
        schema="pilot_v2_rebaseline",
        evaluator="coverage baker(bake_atlas_ss) + texel-center 采样约定(u*W-0.5)",
        carried_over="processing_status/structural_status 沿用 V1.1(不重跑 PartUV/export)",
        development_set="原 17 资产, 不得用于最终 held-out validation"),
    new_protocol_hash=new_protocol_hash,
    counts_allocation=cnt("allocation_quality_fixed_bsignal"),
    counts_delivery=cnt("delivery_quality_fixed_braw"),
    label_changes=changed, objects=rows)
with open(f"{OUT2}/summary.json", "w") as fp:
    json.dump(summary, fp, indent=1, ensure_ascii=False)

print("\n============ Rebaseline 汇总 ============")
for r in rows:
    d = r["allocation_detail"]
    print(f"{r['object_id']:24s} alloc={r['allocation_quality_fixed_bsignal']:14s} "
          f"delivery={r['delivery_quality_fixed_braw']:14s} "
          f"G_g_eq={d.get('G_global_eq', ''):>7} "
          f"{'<< CHANGED' if r['diff']['allocation_changed'] or r['diff']['delivery_changed'] else ''}")
print("label_changes:", json.dumps(changed, ensure_ascii=False))
print("counts_allocation:", summary["counts_allocation"])
print("counts_delivery:", summary["counts_delivery"])
print("REBASELINE_17: DONE")
