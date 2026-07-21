# -*- coding: utf-8 -*-
"""V1.1 负向/混合案例诊断(只读 pilot 样本, 不改 teacher).

对象: synth_gradient / objav_7f9212 / synth_many_islands / synth_two_materials
逐案例输出:
  - per-chart demand 份额 vs 实际光栅化纹素份额(top 偏差 charts)
  - B_signal / fill 差异(Uniform vs TD @50pct)
  - subpixel chart 占比(packed 高或宽 < 1px)
  - padding/signal 开销比(bbox 估计: Σ((w+2p)(h+2p)-wh)/signal)
  - seam / interior 误差(来自重判 metrics)
  - 需求最高/最低 chart 的 reference 内容裁剪图 + 误差贡献
fixed-B_signal 归因(仅 fill 差 >5pp 的 gradient/two_materials):
  给 TD 单独搜 atlas 边长使其 B_signal ≈ Uniform@R50 的 B_signal,
  等 B_signal 下若 TD 仍差 >5% -> content ranking 错误; 否则 -> packing delivery。
  (fixed-B_raw 仍是最终产品指标, 本诊断仅用于失败归因。)
Signal V2 候选排序验证(仅当 ranking 错误被确认时执行, 只测一个候选):
  surface-domain multiscale downsample -> reconstruction residual,
  四个合成控制: constant ≈ smooth linear ramp < checkerboard / thin text。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.budget import rasterize_masks
from tdlib.layout import xatlas_pack, PackingFailedError
from tdlib.rd import bake_atlas_masks, bilinear, ref_gradient_at_samples, surface_samples
from run_pseudo_gt_quality_gate import (charts_from_sample, srgb2lin, lin2srgb,
                                        SS, SEED_EVAL, N_SAMPLES, SEAM_BARY)

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"
CASES = [("synth_gradient", True), ("objav_7f9212", False),
         ("synth_many_islands", False), ("synth_two_materials", True)]


def build_pack(charts, scales, R, pu_like, face_refuv, valid, texA, ch_masks):
    uvs = xatlas_pack(charts, scales, resolution=R, padding_px=4)
    owner, ov, _ = rasterize_masks(ch_masks, uvs, R, R)
    tex_hi, _, _ = bake_atlas_masks(pu_like, uvs, R * SS, face_refuv, valid, texA)
    lin = srgb2lin(tex_hi).reshape(R, SS, R, SS, 3).mean(axis=(1, 3))
    tex = lin2srgb(lin)
    nuv = np.zeros((len(pu_like["F"]), 3, 2))
    for ci, c in enumerate(charts):
        nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
    N_c = np.bincount(owner[owner >= 0].ravel(), minlength=len(charts)).astype(float)
    return dict(uvs=uvs, tex=tex, nuv=nuv, b_signal=int((owner >= 0).sum()),
                overlap=int(ov), N_c=N_c)


def sample_mse(p, fid, bary, ref_lin, sel=None):
    uvq = np.einsum("nk,nkd->nd", bary, p["nuv"][fid])
    d = ((srgb2lin(bilinear(p["tex"], uvq)) - ref_lin) ** 2).mean(1)
    return d if sel is None else float(d[sel].mean())


def multiscale_residual(img, factor=4):
    """Signal V2 候选(仅测试, 不集成): 灰度多尺度降采样重建残差.
    box 降采样 factor 倍 -> bilinear 回升 -> RMS 残差.
    线性渐变可被 bilinear 精确重建 -> 残差≈0(这是与 luminance-std 的本质区别)."""
    g = np.asarray(img, float)
    if g.ndim == 3:
        g = g[..., :3] @ [0.299, 0.587, 0.114]
    H, W = g.shape
    h, w = max(H // factor, 1), max(W // factor, 1)
    small = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize(
        (w, h), Image.BOX), float) / 255
    up = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR), float) / 255
    return float(np.sqrt(((g - up) ** 2).mean()))


report = {"cases": {}, "signal_v2": None}
ranking_error_confirmed = False

for oid, do_fixed_bs in CASES:
    print(f"\n================ 诊断 {oid} ================", flush=True)
    sd = f"{OUT}/{oid}/sample"
    man = json.load(open(f"{sd}/manifest.json"))
    z = dict(np.load(f"{sd}/arrays.npz"))
    texA = np.asarray(Image.open(f"{sd}/reference_basecolor.png"),
                      float)[:, :, :3] / 255.0
    V, F = z["vertices"], z["faces"]
    face_refuv, valid = z["source_uv"], z["source_uv_valid"]
    charts = charts_from_sample(z)
    C = len(charts)
    tris = V[F]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    A3 = np.array([fa3[c["gidx"]].sum() for c in charts])
    a2 = np.array([c["a2"] for c in charts])
    scales_td = z["chart_target_scale"].astype(float)
    scales_uni = np.sqrt(A3 / a2)
    demand = z["chart_demand_normalized"].astype(float)
    area_norm = A3 / A3.sum()
    pu_like = dict(charts=charts, F=F, area=fa3)
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    B_source = texA.shape[0] * texA.shape[1]
    R = max(int(round(np.sqrt(0.50 * B_source))), 64)

    s = surface_samples(pu_like, face_refuv, valid, texA, N_SAMPLES, seed=SEED_EVAL)
    fid, bary = s["fid"], s["bary"]
    ref_lin = srgb2lin(np.asarray(s["ref_color"]))
    seam = bary.min(1) < SEAM_BARY
    f2c = z["face_to_chart"]
    samp_chart = f2c[fid]

    packs = {m: build_pack(charts, sc, R, pu_like, face_refuv, valid, texA, ch_masks)
             for m, sc in [("Uniform", scales_uni), ("TD", scales_td)]}
    case = dict(n_charts=C, R50=R, signal_dist=round(
        float(0.5 * np.abs(demand - area_norm).sum()), 4))

    for m, p in packs.items():
        N_share = p["N_c"] / max(p["N_c"].sum(), 1)
        D_share = demand if m == "TD" else area_norm
        e_alloc = float(0.5 * np.abs(N_share - D_share).sum())
        # subpixel / padding 开销(packed bbox, px)
        spans = np.array([[np.ptp(uv[:, 0]), np.ptp(uv[:, 1])]
                          for uv in p["uvs"]]) * R
        subpix = float((spans.min(1) < 1.0).mean())
        wh = np.maximum(spans, 1.0)
        pad_ratio = float((((wh[:, 0] + 8) * (wh[:, 1] + 8)) - wh[:, 0] * wh[:, 1]
                           ).sum() / max(p["b_signal"], 1))
        d = sample_mse(p, fid, bary, ref_lin)
        case[m] = dict(
            B_signal=p["b_signal"], fill=round(p["b_signal"] / R / R, 4),
            overlap=p["overlap"], E_alloc_vs_own_demand=round(e_alloc, 4),
            subpixel_chart_ratio=round(subpix, 4),
            padding_over_signal=round(pad_ratio, 4),
            mse=float(d.mean()), mse_seam=float(d[seam].mean()),
            mse_interior=float(d[~seam].mean()),
            seam_error_share=round(float(d[seam].sum() / max(d.sum(), 1e-20)), 4))

    # top 偏差 charts(TD: 需求份额 vs 实际纹素份额)
    N_share_td = packs["TD"]["N_c"] / max(packs["TD"]["N_c"].sum(), 1)
    dev = N_share_td - demand
    top_dev = np.argsort(-np.abs(dev))[:5]
    case["top_alloc_deviation_charts"] = [
        dict(chart=int(ci), demand_share=round(float(demand[ci]), 4),
             texel_share=round(float(N_share_td[ci]), 4),
             dev=round(float(dev[ci]), 4)) for ci in top_dev]

    # 需求最高/最低 chart: 内容裁剪 + 误差贡献
    d_td = sample_mse(packs["TD"], fid, bary, ref_lin)
    err_by_chart = np.bincount(samp_chart, weights=d_td, minlength=C)
    err_share = err_by_chart / max(err_by_chart.sum(), 1e-20)
    hi_c, lo_c = int(np.argmax(demand)), int(np.argmin(demand))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
    for ax, ci, tag in [(axs[0], hi_c, "最高需求"), (axs[1], lo_c, "最低需求")]:
        fids = charts[ci]["gidx"]
        uvv = face_refuv[fids][valid[fids].astype(bool)]
        Ht, Wt = texA.shape[:2]
        if len(uvv):
            u0, v0 = uvv.reshape(-1, 2).min(0)
            u1, v1 = uvv.reshape(-1, 2).max(0)
            x0, x1 = int(u0 * Wt), max(int(u1 * Wt), int(u0 * Wt) + 4)
            y0, y1 = int((1 - v1) * Ht), max(int((1 - v0) * Ht), int((1 - v1) * Ht) + 4)
            ax.imshow(texA[max(y0, 0):y1, max(x0, 0):x1])
        ax.set_axis_off()
        ax.set_title(f"chart{ci} {tag}\ndemand={demand[ci]:.4f} "
                     f"err_share={err_share[ci]:.4f}", fontsize=9)
        case[f"{'top' if ci == hi_c else 'bottom'}_demand_chart"] = dict(
            chart=ci, demand_share=round(float(demand[ci]), 4),
            area_share=round(float(area_norm[ci]), 4),
            texel_share=round(float(N_share_td[ci]), 4),
            error_share=round(float(err_share[ci]), 4))
    os.makedirs(f"{OUT11}/{oid}", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{OUT11}/{oid}/demand_extremes.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # fixed-B_signal 归因(仅 fill 差 >5pp 的案例)
    if do_fixed_bs:
        S_target = packs["Uniform"]["b_signal"]
        lo_r, hi_r = int(R * 0.7), int(R * 1.6)
        best = None
        for _ in range(8):
            mid = (lo_r + hi_r) // 2
            try:
                p = build_pack(charts, scales_td, mid, pu_like, face_refuv,
                               valid, texA, ch_masks)
            except PackingFailedError:
                lo_r = mid + 1
                continue
            if best is None or abs(p["b_signal"] - S_target) < abs(best[1] - S_target):
                best = (mid, p["b_signal"], p)
            if p["b_signal"] < S_target:
                lo_r = mid + 1
            else:
                hi_r = mid - 1
        if best is None:
            case["fixed_B_signal"] = dict(verdict="unresolved: 全部候选分辨率打包失败")
            report["cases"][oid] = case
            continue
        R_td, S_td, p_td = best
        mse_td_eq = sample_mse(p_td, fid, bary, ref_lin, sel=slice(None))
        mse_u = case["Uniform"]["mse"]
        mse_td_eq_m = float(np.asarray(mse_td_eq).mean())
        still_worse = mse_td_eq_m > mse_u * 1.05
        case["fixed_B_signal"] = dict(
            note="诊断专用; fixed-B_raw 仍是最终产品指标",
            S_target=int(S_target), R_td=int(R_td), S_td=int(S_td),
            bsignal_match=round(S_td / max(S_target, 1), 4),
            mse_uniform=mse_u, mse_td_equal_bsignal=mse_td_eq_m,
            ratio=round(mse_td_eq_m / max(mse_u, 1e-20), 4),
            verdict=("content_ranking_error" if still_worse
                     else "packing_delivery"))
        ranking_error_confirmed |= still_worse
        print(f"  fixed-B_signal: TD/Uni mse ratio="
              f"{case['fixed_B_signal']['ratio']} -> "
              f"{case['fixed_B_signal']['verdict']}", flush=True)

    report["cases"][oid] = case
    print(json.dumps({k: v for k, v in case.items()
                      if k in ("Uniform", "TD", "signal_dist")},
                     ensure_ascii=False, indent=1), flush=True)

# ---- Signal V2 候选排序验证(仅当内容排序错误被确认; 只测这一个候选) ----
if ranking_error_confirmed:
    n = 256
    xs = np.linspace(0, 1, n)
    constant = np.full((n, n), 0.5)
    ramp = np.tile(xs, (n, 1))                         # 平滑线性渐变
    checker = ((np.add.outer(np.arange(n) // 8, np.arange(n) // 8)) % 2).astype(float)
    text = np.zeros((n, n))                            # 细文字近似: 稀疏细线
    rng = np.random.RandomState(0)
    for _ in range(40):
        r, c0 = rng.randint(8, n - 8), rng.randint(8, n - 120)
        text[r:r + 2, c0:c0 + rng.randint(30, 110)] = 1.0
    scores = {k: round(multiscale_residual(v), 5) for k, v in
              [("constant", constant), ("smooth_linear_ramp", ramp),
               ("checkerboard", checker), ("thin_text", text)]}
    lo = max(scores["constant"], scores["smooth_linear_ramp"])
    hi = min(scores["checkerboard"], scores["thin_text"])
    ok = (abs(scores["constant"] - scores["smooth_linear_ramp"]) < 0.02
          and hi > 5 * max(lo, 1e-6))
    report["signal_v2"] = dict(
        candidate="surface-domain multiscale downsample -> reconstruction residual",
        status="排序验证" + ("通过" if ok else "未通过") + "(仅合成控制, 未集成/未替换 teacher)",
        expected="constant ≈ smooth_linear_ramp < checkerboard / thin_text",
        scores=scores, ordering_ok=bool(ok))
    print("\nSignal V2 排序验证:", json.dumps(report["signal_v2"],
                                          ensure_ascii=False), flush=True)
else:
    report["signal_v2"] = dict(status="未执行: fixed-B_signal 未确认 content ranking 错误")

with open(f"{OUT11}/diagnosis.json", "w") as fp:
    json.dump(report, fp, indent=1, ensure_ascii=False)
print("\nDIAGNOSE: DONE")
