# -*- coding: utf-8 -*-
"""Packing-only 小型 A/B —— 仅 synth_gradient / synth_two_materials.

冻结: chart hash / TD relative scales / B_raw(50pct 档) / padding=4 / baker /
renderer / evaluation samples(150k, seed=2)。
候选 = 当前 packing 基础设施已有的确定性自由度(不实现新 packer/multi-atlas/
chart split): chart 提交顺序 {原序, 缩放面积降序, 缩放面积升序, 最大边降序}
× rotate {True, False} = 8 个; Uniform 与 TD 候选集与搜索预算完全相同,
各自选 B_signal 最高的合法结果(zero overlap)。
成功标准: zero overlap/OOB; E_alloc 合同(<=1%)保持; TD-Uniform fill gap<=5pp;
fixed-B_raw 质量接近 fixed-B_signal 结论。若 xatlas 不敏感或 fill 无明显改善,
立即停止该路线并如实报告。
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

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"
ASSETS = ["synth_gradient", "synth_two_materials"]


def candidates(ctx, scales):
    n = len(ctx["charts"])
    sc = np.asarray(scales)
    area_scaled = ctx["a2"] * sc ** 2
    maxdim = np.array([max(np.ptp(c["UV"][:, 0]), np.ptp(c["UV"][:, 1]))
                       for c in ctx["charts"]]) * sc
    orders = [("orig", np.arange(n)),
              ("area_desc", np.argsort(-area_scaled)),
              ("area_asc", np.argsort(area_scaled)),
              ("maxdim_desc", np.argsort(-maxdim))]
    return [(f"{on}|rot={rot}", od, rot)
            for on, od in orders for rot in (True, False)]


report = {}
for oid in ASSETS:
    print(f"\n================ {oid} ================", flush=True)
    ctx = load_sample(f"{OUT}/{oid}/sample")
    ev = eval_samples(ctx)
    R = max(int(round(np.sqrt(0.50 * ctx["B_source"]))), 64)
    res = dict(R50=R, n_charts=len(ctx["charts"]), methods={})
    for m, scales in [("Uniform", ctx["scales_uni"]), ("TD", ctx["scales_td"])]:
        trials = []
        for name, order, rot in candidates(ctx, scales):
            try:
                p = pack_only(ctx, scales, R, rotate=rot, order=order)
                legal = p["overlap"] == 0
                trials.append(dict(cand=name, fill=round(p["fill"], 4),
                                   overlap=p["overlap"], legal=legal,
                                   _pack=p if legal else None))
            except PackingFailedError as e:
                trials.append(dict(cand=name, fill=None, overlap=None,
                                   legal=False, error=str(e)[:60], _pack=None))
            t = trials[-1]
            print(f"  {m:8s} {t['cand']:22s} fill={t['fill']}", flush=True)
        legal = [t for t in trials if t["legal"]]
        assert legal, f"{oid}/{m}: 无合法候选"
        best = max(legal, key=lambda t: t["fill"])
        p = best["_pack"]
        # E_alloc 合同(对各自 demand: TD=demand, Uniform=面积份额)
        D = ctx["demand"] if m == "TD" else ctx["area_norm"]
        N_share = p["N_c"] / max(p["N_c"].sum(), 1)
        e_alloc = float(0.5 * np.abs(N_share - D).sum())
        tex, nuv = bake_layout(ctx, p["uvs"], R)
        d = surface_err(tex, nuv, ev)
        fills = [t["fill"] for t in legal]
        res["methods"][m] = dict(
            best_cand=best["cand"], fill=best["fill"],
            fill_spread_pp=round((max(fills) - min(fills)) * 100, 2),
            b_signal=p["b_signal"], overlap=p["overlap"],
            E_alloc=round(e_alloc, 4), mse=float(d.mean()),
            mse_hf=float(d[ev["hi"]].mean()),
            trials=[{k: v for k, v in t.items() if k != "_pack"} for t in trials])
    u, t = res["methods"]["Uniform"], res["methods"]["TD"]
    res["fill_gap_pp"] = round((u["fill"] - t["fill"]) * 100, 2)
    res["G_global_braw"] = round(1 - t["mse"] / max(u["mse"], 1e-20), 4)
    res["G_HF_braw"] = round(1 - t["mse_hf"] / max(u["mse_hf"], 1e-20), 4)
    res["success"] = dict(
        zero_overlap=(u["overlap"] == 0 and t["overlap"] == 0),
        e_alloc_contract=(u["E_alloc"] <= 0.01 and t["E_alloc"] <= 0.01),
        fill_gap_le_5pp=res["fill_gap_pp"] <= 5.0,
        xatlas_sensitive=max(u["fill_spread_pp"], t["fill_spread_pp"]) >= 1.0)
    res["verdict"] = ("CONTINUE" if all(res["success"].values())
                      else "STOP_ROUTE(如实报告: 见 success 各项)")
    report[oid] = res
    print(f"  -> best: Uni={u['best_cand']}({u['fill']:.3f}) "
          f"TD={t['best_cand']}({t['fill']:.3f}) gap={res['fill_gap_pp']}pp "
          f"G_global={res['G_global_braw']:+} G_HF={res['G_HF_braw']:+} "
          f"verdict={res['verdict']}", flush=True)

with open(f"{OUT11}/packing_ab_min.json", "w") as fp:
    json.dump(report, fp, indent=1, ensure_ascii=False)
print("\nPACKING_AB: DONE")
