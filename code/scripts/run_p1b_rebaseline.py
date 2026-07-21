# -*- coding: utf-8 -*-
"""P1b Rebaseline: 修复后评测器(coverage baker + texel-center 约定)下的 β 重扫.

与原 P1b 唯一差异: chart decomposition 从原 P1b 缓存加载(不重跑 PartUV,
排除跨运行漂移), 输出写新目录(不覆盖旧报告, 见 metric lineage)。
协议/β 候选/停止条件与原 P1b 逐字相同:

核心算法(全部既有模块, 零新增): PartUV charts(缓存复用) -> luminance-std 内容分数
-> 单一 β (demand_weights, 其余参数用冻结默认) -> 预算归一化 -> chart 缩放 + shelf
packing -> rebake.

协议:
- 每资产 PartUV 只运行一次, chart decomposition 缓存到磁盘(数量/face2chart/sha1);
- 所有 β 复用同一 charts / packer / padding / reference / 评价采样点(seed=2);
- fixed-B_signal: β ∈ {0,0.25,0.5,0.75,1.0}, 目标 B_signal ∈ {0.125M,0.25M,0.5M}
  (双资产均低于 ref_used, 非饱和); 标定 R 使实测 B_signal 逼近目标, 偏差>1% 必须报告;
- 主指标: global surface MSE 的 budget-error AUC(log2 域); 辅助: ref-gradient
  top-10% 高频区 AUC。R-D hull 不参与。
- 预定停止条件: 存在共享 β>0 使两资产 AUC 均不劣化(<1% 记持平)且至少一资产改善(>1%)
  => Simple V1 case-study validated; 否则 not yet validated, 立即暂停。
- 附: 对最佳共享 β 另做 fixed-B_raw 对照(复用 β=0 标定的 R), 供两协议差异报告。
"""
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from run_p1a import auc_log, eval_subset_error, sample_hash
from tdlib.budget import rasterize_masks
from tdlib.geometry import tri_area_2d
from tdlib.layout import layout_with_scales
from tdlib.pipeline import load_reference, run_partuv
from tdlib.rd import (bake_atlas_masks, eval_surface_error,
                      prepare_face_ref_uv, ref_gradient_at_samples,
                      surface_samples)
from tdlib.signal import demand_weights, luminance_std_heuristic

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT_ROOT = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1b_rebaseline"
OLD_ROOT = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1b"
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TARGETS = [125_000, 250_000, 500_000]          # B_signal 目标(纹素)
TIE_BAND = 0.01                                 # <1% 记持平

ASSETS = [
    ("shoe_22b822", f"{DATA}/objaverse_22b822c6520d4d49.glb"),
    ("wheel_92ff6", f"{DATA}/objaverse_92ff65712c62408d.glb"),
]


def bake_measured(pu, uvs, R, face_refuv, valid, texA):
    tex, sig, _ = bake_atlas_masks(pu, uvs, R, face_refuv, valid, texA)
    return tex, int(sig.sum())


def calibrated_bake(pu, uvs, eta, target, face_refuv, valid, texA):
    """标定 R 使实测 B_signal 逼近 target; 最多修正一次, 取更接近者."""
    R0 = max(int(round(np.sqrt(target / eta))), 8)
    tex, sig = bake_measured(pu, uvs, R0, face_refuv, valid, texA)
    best = (R0, tex, sig)
    if abs(sig - target) / target > 0.005:
        R1 = max(int(round(R0 * np.sqrt(target / max(sig, 1)))), 8)
        if R1 != R0:
            tex1, sig1 = bake_measured(pu, uvs, R1, face_refuv, valid, texA)
            if abs(sig1 - target) < abs(sig - target):
                best = (R1, tex1, sig1)
    return best


def run_asset(name, path):
    out = f"{OUT_ROOT}/{name}/"
    os.makedirs(out, exist_ok=True)

    # ---- 复用原 P1b 缓存的 chart decomposition(不重跑 PartUV) ----
    with open(f"{OLD_ROOT}/{name}/charts_cache.pkl", "rb") as fp:
        pu = pickle.load(fp)
    F, area, covered = pu["F"], pu["area"], pu["covered"]
    charts = pu["charts"]
    ref = load_reference(path, pu["V"], F, pu["mesh_scale"])
    texA = ref["texA"]
    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)

    with open(f"{out}/charts_cache.pkl", "wb") as fp:
        pickle.dump(pu, fp)
    f2c_hash = hashlib.sha1(np.ascontiguousarray(face2chart).tobytes()).hexdigest()[:16]
    cache_meta = dict(n_charts=len(charts), n_faces=int(len(F)),
                      face2chart_sha1=f2c_hash)
    with open(f"{out}/cache_meta.json", "w") as fp:
        json.dump(cache_meta, fp, indent=1)
    old_meta = json.load(open(f"{OLD_ROOT}/{name}/cache_meta.json"))
    assert old_meta["face2chart_sha1"] == f2c_hash, "chart 分解与原 P1b 缓存不一致"
    print(f"  [cache] 复用原 P1b charts: {cache_meta['n_charts']} charts, "
          f"sha1={f2c_hash} (与旧缓存一致)", flush=True)

    # ---- 共享评价采样点(seed=2, 同 P1a 协议) + 高频子集 ----
    s_eval = surface_samples(pu, face_refuv, valid, texA, 150_000, seed=2)
    ev_hash = sample_hash(s_eval)
    g = ref_gradient_at_samples(texA, face_refuv, s_eval)
    hi = g >= np.quantile(g, 0.9)

    # ---- 饱和检查(同 P1a 判据基准) ----
    Ht, Wt = texA.shape[:2]
    ch0 = dict(F=ref["Fo"], gidx=np.arange(len(ref["Fo"])))
    owner0, _, _ = rasterize_masks([ch0], [ref["uv0"]], Wt, Ht)
    ref_used = int((owner0 >= 0).sum())
    assert max(TARGETS) < ref_used, f"目标预算点饱和: {max(TARGETS)} >= {ref_used}"

    # ---- 内容分数(既有 luminance-std) ----
    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"], ref["ok_map"])
    sel = covered & ref["ok_map"]

    per_beta = {}
    layouts = {}
    for b in BETAS:
        _, w = demand_weights(cw, sel, area, beta=b)      # 其余参数冻结默认
        uvs, _ = layout_with_scales(charts, w)            # 同 packer/padding
        layouts[b] = uvs
        eta = sum(float(tri_area_2d(uv[np.asarray(c["F"])]).sum())
                  for c, uv in zip(charts, uvs))
        pts = []
        for target in TARGETS:
            R, tex, sig = calibrated_bake(pu, uvs, eta, target,
                                          face_refuv, valid, texA)
            dev = (sig - target) / target
            pts.append(dict(
                target=target, R=R, B_raw=R * R, B_signal=sig,
                dev=round(float(dev), 5), dev_gt_1pct=bool(abs(dev) > 0.01),
                err=eval_surface_error(tex, pu, uvs, s_eval, face2chart),
                err_hifreq=eval_subset_error(tex, pu, uvs, s_eval, face2chart, hi)))
            print(f"    beta={b:.2f} target={target/1e3:.0f}k R={R} "
                  f"sig={sig/1e3:.1f}k dev={dev*100:+.2f}% "
                  f"MSE={pts[-1]['err']:.6f} hi={pts[-1]['err_hifreq']:.6f}",
                  flush=True)
        per_beta[str(b)] = dict(
            points=pts,
            auc=auc_log(TARGETS, [p["err"] for p in pts]),
            auc_hifreq=auc_log(TARGETS, [p["err_hifreq"] for p in pts]))

    # ---- 附: fixed-B_raw 对照(对每个 β>0 复用 β=0 标定的 R) ----
    R_base = [p["R"] for p in per_beta["0.0"]["points"]]
    fixed_braw = {"0.0": dict(auc=per_beta["0.0"]["auc"])}   # β=0 两协议相同
    for b in BETAS[1:]:
        errs = []
        for R in R_base:
            tex, sig = bake_measured(pu, layouts[b], R, face_refuv, valid, texA)
            errs.append(eval_surface_error(tex, pu, layouts[b], s_eval, face2chart))
        fixed_braw[str(b)] = dict(R=R_base, err=errs, auc=auc_log(TARGETS, errs))

    result = dict(asset=name, cache=cache_meta, eval_sample_hash=ev_hash,
                  ref_used_texels=ref_used, targets=TARGETS, betas=BETAS,
                  per_beta=per_beta, fixed_braw=fixed_braw)
    with open(f"{out}/metrics.json", "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    return result


def decide(results):
    """预定停止条件: 共享 β>0, 两资产均不劣化(<1% 持平), ≥1 资产改善(>1%)."""
    base = {r["asset"]: r["per_beta"]["0.0"]["auc"] for r in results}
    rows, qualifying = [], []
    for b in BETAS[1:]:
        ratios = {r["asset"]: r["per_beta"][str(b)]["auc"] / base[r["asset"]]
                  for r in results}
        ok = all(v <= 1 + TIE_BAND for v in ratios.values())
        improved = any(v < 1 - TIE_BAND for v in ratios.values())
        rows.append(dict(beta=b, auc_ratio_vs_beta0=ratios,
                         no_regression=ok, some_improved=improved))
        if ok and improved:
            qualifying.append((b, float(np.mean(list(ratios.values())))))
    best = min(qualifying, key=lambda x: x[1])[0] if qualifying else None
    # 最佳共享 β(不论是否达标): 两资产 AUC 比值均值最小
    best_any = min(BETAS[1:], key=lambda b: np.mean(
        [r["per_beta"][str(b)]["auc"] / base[r["asset"]] for r in results]))
    return dict(
        criterion="两资产 global AUC 均 <=1.01x β=0, 且至少一资产 <0.99x; "
                  "最佳共享 β = 达标者中两资产 AUC 比值均值最小",
        rows=rows, qualifying_betas=[b for b, _ in qualifying],
        best_shared_beta=best, best_shared_beta_any=best_any,
        status=("Simple_V1_case_study_validated" if best is not None
                else "Simple_V1_not_yet_validated"))


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    results = []
    for n, p in ASSETS:
        print(f"[{n}]", flush=True)
        results.append(run_asset(n, p))
    verdict = decide(results)
    with open(f"{OUT_ROOT}/summary.json", "w") as fp:
        json.dump(dict(verdict=verdict, assets=results), fp,
                  indent=2, ensure_ascii=False)

    # ---- 曲线图: 鞋/车轮 fixed-B_signal ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    cmap = plt.cm.viridis
    for ax, r in zip(axs, results):
        for b in BETAS:
            pts = r["per_beta"][str(b)]["points"]
            xs = [p["B_signal"] for p in pts]
            ys = [p["err"] for p in pts]
            kw = dict(color="tab:gray", lw=2.4) if b == 0 else \
                 dict(color=cmap(b * 0.85), lw=1.4)
            ax.plot(xs, ys, "o-", label=f"beta={b:g}" + (" (L1)" if b == 0 else ""),
                    **kw)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("measured B_signal (texels)")
        ax.set_ylabel("surface MSE (sRGB)")
        ax.set_title(f"{r['asset']} | protocol: fixed B_signal", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_ROOT}/p1b_fixed_bsignal_curves.png", dpi=115,
                bbox_inches="tight")

    print("\n===== P1b verdict =====")
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    for r in results:
        print(f"{r['asset']}: " + "  ".join(
            f"beta={b:g}: AUC={r['per_beta'][str(b)]['auc']:.6f}" for b in BETAS))


if __name__ == "__main__":
    main()
