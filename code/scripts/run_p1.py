# -*- coding: utf-8 -*-
"""P1 驱动: 直接质量尺子 + chart-level R-D oracle (协议 A, 最小范围).

固定: PartUV charts / 单 atlas(方图 R=round(√B), 全方法同 R) / shelf packer /
      base-color reference(sRGB 域 MSE, 全方法一致) / 同名义预算.
方法: L1 uniform | L2 heuristic(luminance-std, β=0.4) | chart-level R-D oracle.
预算点: 0.5M / 1M / 2M / 4M texels.  主结果: budget–error 曲线 + AUC(log2 预算域).
QUALITY_GATE: AUC(L2 heuristic) < AUC(L1) ⇒ PASS (oracle AUC 作诊断).
冻结项: 不做 chart split / multi-atlas / UniTEX / PBR / 架构重构.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tdlib.layout import layout_with_scales
from tdlib.pipeline import load_reference, run_partuv
from tdlib.rd import (bake_atlas, chart_rd_curves, eval_surface_error,
                      oracle_allocate, prepare_face_ref_uv, surface_samples)
from tdlib.signal import demand_weights, luminance_std_heuristic

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT_ROOT = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1"
BUDGETS = [500_000, 1_000_000, 2_000_000, 4_000_000]
BETA, Z_MAX, Q_MIN, Q_MAX = 0.4, 2.5, 0.5, 2.83
N_SAMPLES = 150_000

ASSETS = [
    ("shoe_22b822", f"{DATA}/objaverse_22b822c6520d4d49.glb"),
    ("wheel_92ff6", f"{DATA}/objaverse_92ff65712c62408d.glb"),
]


def auc_log(budgets, errors):
    """log2 预算域的梯形 AUC."""
    x = np.log2(np.asarray(budgets, float))
    y = np.asarray(errors, float)
    return float(np.trapz(y, x))


def run_asset(name, path):
    out = f"{OUT_ROOT}/{name}/"
    os.makedirs(out, exist_ok=True)
    pu = run_partuv(path, out)
    F, area, covered = pu["F"], pu["area"], pu["covered"]
    charts = pu["charts"]
    ref = load_reference(path, pu["V"], F, pu["mesh_scale"])
    if not ref["has_tex"]:
        return dict(asset=name, error="no texture")
    texA = ref["texA"]

    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)
    samples = surface_samples(pu, face_refuv, valid, texA, N_SAMPLES)

    # ---- heuristic L2 权重(全预算共用; 相对分配与预算无关) ----
    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"], ref["ok_map"])
    sel = covered & ref["ok_map"]
    _, w_l2 = demand_weights(cw, sel, area, BETA, Z_MAX, Q_MIN, Q_MAX)

    # ---- chart-level R-D 曲线(离线金标准, 每 chart 5 档) ----
    print(f"  [{name}] 计算 {len(charts)} 个 chart 的 R-D 曲线 ...", flush=True)
    curves = chart_rd_curves(pu, face_refuv, valid, texA, samples, face2chart)

    methods = {}
    per_budget = {b: {} for b in BUDGETS}
    for B in BUDGETS:
        R = int(round(np.sqrt(B)))
        w_oracle, lvl, used = oracle_allocate(curves, pu, B)
        for tag, w in [("L1", np.ones(len(F))), ("L2_heuristic", w_l2),
                       ("RD_oracle", w_oracle)]:
            uvs, _ = layout_with_scales(charts, w)
            tex, filled = bake_atlas(pu, uvs, R, face_refuv, valid, texA)
            err = eval_surface_error(tex, pu, uvs, samples, face2chart)
            per_budget[B][tag] = dict(err=err, R=R,
                                      signal_texels=int(filled.sum()))
            print(f"    B={B/1e6:.1f}M R={R} {tag:12s} MSE={err:.6f} "
                  f"signal={filled.sum()/1e6:.2f}M", flush=True)

    for tag in ["L1", "L2_heuristic", "RD_oracle"]:
        errs = [per_budget[b][tag]["err"] for b in BUDGETS]
        methods[tag] = dict(errors=errs, auc=auc_log(BUDGETS, errs))

    q_pass = methods["L2_heuristic"]["auc"] < methods["L1"]["auc"]
    oracle_pass = methods["RD_oracle"]["auc"] < methods["L1"]["auc"]
    result = dict(
        asset=name, faces=int(len(F)), charts=len(charts),
        budgets=BUDGETS, per_budget=per_budget, methods=methods,
        QUALITY_GATE=("PASS" if q_pass else "FAIL"),
        oracle_beats_uniform=bool(oracle_pass),
        protocol="A (fixed charts/packer/single-atlas/R=sqrt(B); sRGB MSE on "
                 f"{N_SAMPLES} area-weighted surface samples)",
    )
    with open(f"{out}/metrics.json", "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    return result


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = [run_asset(n, p) for n, p in ASSETS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in results if "error" not in r]
    fig, axs = plt.subplots(1, len(ok), figsize=(6.0 * len(ok), 4.4))
    for ax, r in zip(np.atleast_1d(axs), ok):
        for tag, style, col in [("L1", "o-", "#8a8a8a"),
                                ("L2_heuristic", "s-", "#c25b4e"),
                                ("RD_oracle", "d-", "#4a7dbd")]:
            ax.plot(np.asarray(r["budgets"]) / 1e6, r["methods"][tag]["errors"],
                    style, color=col,
                    label=f"{tag} (AUC={r['methods'][tag]['auc']:.4f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("texel budget (M)")
        ax.set_ylabel("surface MSE (sRGB)")
        ax.set_title(f"{r['asset']}  QUALITY_GATE={r['QUALITY_GATE']}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_ROOT}/p1_budget_error_curves.png", dpi=120)
    with open(f"{OUT_ROOT}/summary.json", "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\nsaved: {OUT_ROOT}/summary.json, p1_budget_error_curves.png")
    for r in ok:
        print(f"{r['asset']}: QUALITY_GATE={r['QUALITY_GATE']} "
              f"AUC L1={r['methods']['L1']['auc']:.4f} "
              f"L2={r['methods']['L2_heuristic']['auc']:.4f} "
              f"oracle={r['methods']['RD_oracle']['auc']:.4f}")


if __name__ == "__main__":
    main()
