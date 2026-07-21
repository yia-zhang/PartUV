# -*- coding: utf-8 -*-
"""P1a 公平性核查 (六项):
1. 每(资产,预算,方法)报告 B_raw/B_signal/B_pad/B_empty; 固定 R ≠ 固定有效预算(仅报告);
2. R-D 曲线采样(seed=1)与质量评价采样(seed=2)分离, 保存 sample sha1;
3. reference 饱和检查: L1 B_signal ≥ 原资产被引用纹素 ⇒ 该预算点饱和,
   不计入主 AUC(单独报告);
4. R-D 曲线做单调化+下凸包(hull_curves), 报告违反递减性的 chart 数,
   方法标注为 RD_oracle_hull(approximate oracle);
5. (报告措辞修订见 p1_quality_report.md);
6. 高频区误差: 按 reference 自身梯度 top-10% 采样点子集的 MSE(与分配信号无关).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tdlib.budget import rasterize_masks
from tdlib.layout import layout_with_scales
from tdlib.pipeline import load_reference, run_partuv
from tdlib.rd import (bake_atlas_masks, chart_rd_curves, eval_surface_error,
                      hull_curves, oracle_allocate, prepare_face_ref_uv,
                      ref_gradient_at_samples, surface_samples)
from tdlib.signal import demand_weights, luminance_std_heuristic

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT_ROOT = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1a"
BUDGETS = [500_000, 1_000_000, 2_000_000, 4_000_000]
BETA, Z_MAX, Q_MIN, Q_MAX = 0.4, 2.5, 0.5, 2.83
N_SAMPLES = 150_000

ASSETS = [
    ("shoe_22b822", f"{DATA}/objaverse_22b822c6520d4d49.glb"),
    ("wheel_92ff6", f"{DATA}/objaverse_92ff65712c62408d.glb"),
]


def sample_hash(s):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(s["fid"]).tobytes())
    h.update(np.ascontiguousarray(s["bary"]).tobytes())
    return h.hexdigest()[:16]


def auc_log(budgets, errors):
    return float(np.trapz(np.asarray(errors, float), np.log2(np.asarray(budgets, float))))


def eval_subset_error(tex, pu, uvs, samples, face2chart, subset):
    charts = pu["charts"]
    fid, bary = samples["fid"][subset], samples["bary"][subset]
    uv_new = np.zeros((len(fid), 2))
    ci_all, row_all = face2chart[fid, 0], face2chart[fid, 1]
    for ci in np.unique(ci_all):
        m = ci_all == ci
        cF = np.asarray(charts[ci]["F"])
        uv_new[m] = np.einsum("nk,nkd->nd", bary[m], uvs[ci][cF[row_all[m]]])
    from tdlib.rd import bilinear
    rec = bilinear(tex, uv_new)
    return float(((rec - samples["ref_color"][subset]) ** 2).mean())


def run_asset(name, path):
    out = f"{OUT_ROOT}/{name}/"
    os.makedirs(out, exist_ok=True)
    pu = run_partuv(path, out)
    F, area, covered = pu["F"], pu["area"], pu["covered"]
    charts = pu["charts"]
    ref = load_reference(path, pu["V"], F, pu["mesh_scale"])
    texA = ref["texA"]
    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)

    # ---- P1a-2: 曲线采样与评价采样分离 ----
    s_curve = surface_samples(pu, face_refuv, valid, texA, N_SAMPLES, seed=1)
    s_eval = surface_samples(pu, face_refuv, valid, texA, N_SAMPLES, seed=2)
    hashes = dict(curve=sample_hash(s_curve), eval=sample_hash(s_eval))

    # ---- P1a-6: 高频子集(reference 自身梯度 top-10%, 与分配信号无关) ----
    g = ref_gradient_at_samples(texA, face_refuv, s_eval)
    hi = g >= np.quantile(g, 0.9)

    # ---- P1a-3: reference 支持度 ----
    Ht, Wt = texA.shape[:2]
    ch0 = dict(F=ref["Fo"], gidx=np.arange(len(ref["Fo"])))
    owner0, _, _ = rasterize_masks([ch0], [ref["uv0"]], Wt, Ht)
    ref_used = int((owner0 >= 0).sum())

    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"], ref["ok_map"])
    sel = covered & ref["ok_map"]
    _, w_l2 = demand_weights(cw, sel, area, BETA, Z_MAX, Q_MIN, Q_MAX)

    print(f"  [{name}] R-D 曲线(seed=1 采样) ...", flush=True)
    curves_raw = chart_rd_curves(pu, face_refuv, valid, texA, s_curve, face2chart)
    curves, hull_diag = hull_curves(curves_raw)          # P1a-4

    per_budget = {}
    for B in BUDGETS:
        R = int(round(np.sqrt(B)))
        w_or, _, _ = oracle_allocate(curves, pu, B)
        row = {}
        for tag, w in [("L1", np.ones(len(F))), ("L2_heuristic", w_l2),
                       ("RD_oracle_hull", w_or)]:
            uvs, _ = layout_with_scales(charts, w)
            tex, sig, filled = bake_atlas_masks(pu, uvs, R, face_refuv, valid, texA)
            b_raw = R * R
            b_sig = int(sig.sum())
            b_pad = int(filled.sum()) - b_sig
            row[tag] = dict(
                err=eval_surface_error(tex, pu, uvs, s_eval, face2chart),
                err_hifreq=eval_subset_error(tex, pu, uvs, s_eval, face2chart, hi),
                R=R, B_raw=b_raw, B_signal=b_sig, B_pad=b_pad,
                B_empty=b_raw - b_sig - b_pad)
            print(f"    B={B/1e6:.1f}M {tag:15s} MSE={row[tag]['err']:.6f} "
                  f"hiMSE={row[tag]['err_hifreq']:.6f} "
                  f"sig={b_sig/1e6:.2f}M pad={b_pad/1e6:.2f}M", flush=True)
        row["saturated"] = bool(row["L1"]["B_signal"] >= ref_used)   # P1a-3
        per_budget[str(B)] = row

    main_b = [b for b in BUDGETS if not per_budget[str(b)]["saturated"]]
    sat_b = [b for b in BUDGETS if per_budget[str(b)]["saturated"]]
    methods = {}
    for tag in ["L1", "L2_heuristic", "RD_oracle_hull"]:
        methods[tag] = dict(
            auc_main=auc_log(main_b, [per_budget[str(b)][tag]["err"] for b in main_b])
            if len(main_b) >= 2 else None,
            auc_main_hifreq=auc_log(main_b, [per_budget[str(b)][tag]["err_hifreq"]
                                             for b in main_b])
            if len(main_b) >= 2 else None,
            errors_all=[per_budget[str(b)][tag]["err"] for b in BUDGETS])
    qp = (methods["L2_heuristic"]["auc_main"] is not None
          and methods["L2_heuristic"]["auc_main"] < methods["L1"]["auc_main"])
    result = dict(
        asset=name, faces=int(len(F)), charts=len(charts),
        sample_hashes=hashes, ref_used_texels=ref_used,
        budgets=BUDGETS, main_budgets=main_b, saturated_budgets=sat_b,
        hull_diagnostics=hull_diag,
        per_budget=per_budget, methods=methods,
        QUALITY_GATE_main=("PASS" if qp else "FAIL"),
        oracle_beats_uniform_main=bool(
            methods["RD_oracle_hull"]["auc_main"] is not None
            and methods["RD_oracle_hull"]["auc_main"] < methods["L1"]["auc_main"]),
        note="固定 R 并非固定有效纹素预算; 各方法实际 B_signal 见 per_budget (P1a-1)",
    )
    with open(f"{out}/metrics.json", "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    return result


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = [run_asset(n, p) for n, p in ASSETS]
    with open(f"{OUT_ROOT}/summary.json", "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    for r in results:
        print(f"\n{r['asset']}: 饱和预算点={r['saturated_budgets']} "
              f"(ref_used={r['ref_used_texels']/1e6:.2f}M)  "
              f"hull修正: 非单调 {r['hull_diagnostics']['n_nonmonotone']}, "
              f"非凸 {r['hull_diagnostics']['n_nonconvex']} / {r['hull_diagnostics']['n_eval']}")
        print(f"  主AUC(非饱和): L1={r['methods']['L1']['auc_main']:.4f} "
              f"L2={r['methods']['L2_heuristic']['auc_main']:.4f} "
              f"oracle_hull={r['methods']['RD_oracle_hull']['auc_main']:.4f}  "
              f"QUALITY_GATE_main={r['QUALITY_GATE_main']}")
        print(f"  高频AUC(非饱和): L1={r['methods']['L1']['auc_main_hifreq']:.4f} "
              f"L2={r['methods']['L2_heuristic']['auc_main_hifreq']:.4f} "
              f"oracle_hull={r['methods']['RD_oracle_hull']['auc_main_hifreq']:.4f}")
        print(f"  sample hashes: {r['sample_hashes']}")


if __name__ == "__main__":
    main()
