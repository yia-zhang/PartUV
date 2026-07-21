# -*- coding: utf-8 -*-
"""Pseudo-GT TD label quality gate —— 只消费已 ACCEPTED 的样本目录(自包含验证).

quality_gate(sample_dir, out_root, ...) 可复用: 对比 Reference / PartUV-Uniform /
PseudoGT-TD, 同 chart hash / local UV / packer / padding / baker / 相机 / atlas
分辨率; 主公平轴 = 相同 B_raw(两档: 源纹素的 50%/25%); 全部布局直接从全分辨率
Reference bake(4x 超采样, 两方法一致); LPIPS 不可用 -> foreground-masked SSIM。
作用域字段(冻结语义): quality_scope=td_allocation_only;
training_eligible = {td_allocation, artist_local_refinement=false,
final_packed_uv_regression=false}。
__main__ = 鞋 development case(沿用其冻结 ROI 常量)。
"""
import hashlib
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
from skimage.metrics import structural_similarity as ssim

from tdlib.budget import rasterize_masks
from tdlib.layout import xatlas_pack
from tdlib.rd import (bake_atlas_masks, bake_atlas_ss, bilinear,
                      ref_gradient_at_samples, surface_samples)

# ---- 冻结协议常量(pilot 与 development case 共用) ----
SS = 4
SEED_EVAL = 2
N_SAMPLES = 150_000
SSIM_VIEWS = [(15, a) for a in range(0, 360, 45)]
SEAM_BARY = 0.08
LOW_SIGNAL_DIST = 0.05
GATE = dict(braw_dev=0.01, global_ratio=1.02, hf_gain=0.05,
            ssim_tol=0.002, fill_drop_pp=2.0)
# V1.1: SSIM paired per-view delta 分级(轻微降不再硬失败, 明显降=负向证据)
SSIM_SLIGHT = -0.002   # Δmean >= 该值: ok
SSIM_CLEAR = -0.010    # Δmean < 该值: 明显下降(负向证据); 介于两者=轻微(BORDERLINE)
TIER_FRACS = [("50pct", 0.50), ("25pct", 0.25)]


def srgb2lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def charts_from_sample(z):
    """从样本精确重建 charts(exact-byte 顶点焊接, 无重算)."""
    f2c = z["face_to_chart"]
    luv = z["local_uv_before_td"]
    charts = []
    for ci in range(int(z["chart_ids"].max()) + 1):
        fids = np.where(f2c == ci)[0]
        corners = luv[fids].reshape(-1, 2)
        uniq, inv = np.unique(corners, axis=0, return_inverse=True)
        t = luv[fids]
        a2 = float(np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum() / 2)
        charts.append(dict(UV=uniq, F=inv.reshape(-1, 3).astype(np.int64),
                           gidx=fids, a2=max(a2, 1e-12)))
    return charts


def fg_mask(img):
    return (img < 0.995).any(axis=2)


def masked_ssim(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    _, smap = ssim(a, b, channel_axis=2, data_range=1.0, full=True)
    m = fg_mask(a) | fg_mask(b)
    if not m.any():
        return 1.0
    return float(smap.mean(axis=2)[m].mean())


def quality_gate(sample_dir, out_root, views_show=((15, 45), (15, 225)),
                 crop=(0.25, 0.75, 0.25, 0.75), make_figs=True):
    """对一个已导出样本执行质量门. 返回 quality_report dict(并写盘)."""
    from tdlib import gpu as tdgpu

    protocol = dict(tier_fracs=TIER_FRACS, ss=SS, seed=SEED_EVAL,
                    reduce="coverage_center_v1",  # Baker Convergence Audit 修复
                    sampler="texel_center_v1",    # Coordinate Rebaseline 约定

                    n_samples=N_SAMPLES, ssim_views=SSIM_VIEWS,
                    seam_bary=SEAM_BARY, low_signal_dist=LOW_SIGNAL_DIST,
                    gate=GATE, views_show=list(views_show), crop=list(crop),
                    lpips="不可用(无 lpips 包) -> foreground-masked SSIM(已标注)")
    protocol_hash = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    manifest = json.load(open(f"{sample_dir}/manifest.json"))
    out = f"{out_root}/{manifest['sample_id']}"
    os.makedirs(out, exist_ok=True)
    z = dict(np.load(f"{sample_dir}/arrays.npz"))
    texA = np.asarray(Image.open(f"{sample_dir}/reference_basecolor.png"),
                      float)[:, :, :3] / 255.0
    V, F = z["vertices"], z["faces"]
    face_refuv, valid = z["source_uv"], z["source_uv_valid"]
    tm = z["train_face_mask"]

    chart_hash = hashlib.sha256(
        np.ascontiguousarray(z["face_to_chart"]).tobytes()
        + np.ascontiguousarray(z["local_uv_before_td"]).tobytes()).hexdigest()
    assert chart_hash == manifest["teacher"]["chart_hash"], "chart hash 不一致"
    charts = charts_from_sample(z)
    C = len(charts)

    scales_td = z["chart_target_scale"].astype(float)
    tris = V[F]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    A3 = np.array([fa3[c["gidx"]].sum() for c in charts])
    a2 = np.array([c["a2"] for c in charts])
    scales_uni = np.sqrt(A3 / a2)
    area_norm = A3 / A3.sum()
    signal_dist = float(0.5 * np.abs(z["chart_demand_normalized"] - area_norm).sum())

    pu_like = dict(charts=charts, F=F, area=fa3)
    s = surface_samples(pu_like, face_refuv, valid, texA, N_SAMPLES, seed=SEED_EVAL)
    g = ref_gradient_at_samples(texA, face_refuv, s)
    hi = g >= np.quantile(g, 0.9)
    fid, bary = s["fid"], s["bary"]
    seam = bary.min(1) < SEAM_BARY
    ref_lin = srgb2lin(np.asarray(s["ref_color"]))
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    B_source = texA.shape[0] * texA.shape[1]

    def build(scales, R):
        uvs = xatlas_pack(charts, scales, resolution=R, padding_px=4)
        owner, ov, _ = rasterize_masks(ch_masks, uvs, R, R)
        # Baker Convergence Audit 修复: 覆盖感知降采样(未覆盖子纹素不参与平均)
        tex, _, _ = bake_atlas_ss(pu_like, uvs, R, SS, face_refuv, valid, texA)
        nuv = np.zeros((len(F), 3, 2))
        okm = np.zeros(len(F), bool)
        for ci, c in enumerate(charts):
            nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
            okm[c["gidx"]] = True
        return dict(uvs=uvs, tex=tex, nuv=nuv, ok=okm,
                    b_signal=int((owner >= 0).sum()), overlap=int(ov))

    metrics = dict(sample_id=manifest["sample_id"], protocol=protocol,
                   protocol_hash=protocol_hash, chart_hash=chart_hash,
                   n_charts=C, B_source_raw=int(B_source),
                   signal_dist=signal_dist, tiers={})
    layouts = {}
    for tname, frac in TIER_FRACS:
        R = max(int(round(np.sqrt(frac * B_source))), 64)
        braw_dev = abs(R * R / (frac * B_source) - 1)
        row = dict(B_raw=R * R, R=R, braw_dev=round(float(braw_dev), 5), methods={})
        packs = {"PartUV-Uniform": build(scales_uni, R),
                 "PseudoGT-TD": build(scales_td, R)}
        for m, p in packs.items():
            uvq = np.einsum("nk,nkd->nd", bary, p["nuv"][fid])
            c = srgb2lin(bilinear(p["tex"], uvq))
            d = ((c - ref_lin) ** 2).mean(1)
            mse = float(d.mean())
            row["methods"][m] = dict(
                mse_linear=mse, psnr_db=round(float(10 * np.log10(1 / max(mse, 1e-12))), 2),
                mse_hf=float(d[hi].mean()), mse_seam=float(d[seam].mean()),
                mse_interior=float(d[~seam].mean()),
                B_signal=p["b_signal"], packing_fill=round(p["b_signal"] / R / R, 4),
                overlap=p["overlap"])
            vals = []
            for vw in SSIM_VIEWS:
                a = tdgpu.textured_render(V, F, face_refuv, valid, texA, view=vw)
                b = tdgpu.textured_render(V, F, p["nuv"], p["ok"], p["tex"], view=vw)
                vals.append(masked_ssim(a, b))
            row["methods"][m]["masked_ssim_mean"] = round(float(np.mean(vals)), 4)
            row["methods"][m]["masked_ssim_min"] = round(float(np.min(vals)), 4)
            row["methods"][m]["masked_ssim_views"] = [round(v, 5) for v in vals]
        eu, et = row["methods"]["PartUV-Uniform"], row["methods"]["PseudoGT-TD"]
        row["G_global"] = round(1 - et["mse_linear"] / max(eu["mse_linear"], 1e-12), 4)
        row["G_HF"] = round(1 - et["mse_hf"] / max(eu["mse_hf"], 1e-12), 4)
        metrics["tiers"][tname] = row
        layouts[tname] = packs

    if make_figs:
        _figures(out, z, charts, layouts, metrics, V, F, face_refuv, valid,
                 texA, tm, fid, bary, ref_lin, views_show, crop)

    # ---- V1.1 分类: processing_status 与 label_quality 分离 ----
    # 硬失败(有效性): overlap / OOB(打包器保证, 由 overlap+B_signal<=R^2 隐含) /
    # 预算无法满足(braw_dev). fill 降级为 diagnostic warning, 不再单独 FAIL.
    # SSIM: 轻微降(SSIM_SLIGHT..SSIM_CLEAR)=BORDERLINE 标记; 明显降(<SSIM_CLEAR)=负向证据.
    low_signal = signal_dist < LOW_SIGNAL_DIST
    gates, evidence, warnings = {}, {}, []
    hard_fail = pos_any = neg_any = borderline = False
    for tname in metrics["tiers"]:
        row = metrics["tiers"][tname]
        eu, et = row["methods"]["PartUV-Uniform"], row["methods"]["PseudoGT-TD"]
        dv = [round(tv - uv_, 5) for tv, uv_ in
              zip(et["masked_ssim_views"], eu["masked_ssim_views"])]
        d_mean = float(np.mean(dv))
        row["ssim_delta_views"] = dv
        row["ssim_delta_mean"] = round(d_mean, 5)
        hard = {
            "braw_dev<=1%": row["braw_dev"] <= GATE["braw_dev"],
            "no_overlap": et["overlap"] == 0 and eu["overlap"] == 0,
        }
        gates[tname] = hard
        hard_fail |= not all(hard.values())
        glob_neg = et["mse_linear"] > eu["mse_linear"] * GATE["global_ratio"]
        ssim_state = ("ok" if d_mean >= SSIM_SLIGHT else
                      "slight_drop" if d_mean >= SSIM_CLEAR else "clear_drop")
        ev = dict(global_positive=row["G_global"] >= GATE["hf_gain"],
                  global_acceptable=not glob_neg,
                  hf_positive=row["G_HF"] >= GATE["hf_gain"],
                  hf_negative=row["G_HF"] <= -GATE["hf_gain"],
                  ssim=ssim_state, ssim_delta_mean=round(d_mean, 5))
        evidence[tname] = ev
        pos_any |= ev["global_positive"] or ev["hf_positive"]
        neg_any |= glob_neg or ev["hf_negative"] or ssim_state == "clear_drop"
        borderline |= ssim_state == "slight_drop"
        fill_drop = eu["packing_fill"] - et["packing_fill"]
        if fill_drop > GATE["fill_drop_pp"] / 100:
            warnings.append(f"{tname}: TD fill 低于 Uniform "
                            f"{fill_drop * 100:.1f}pp (diagnostic, 不再单独 FAIL)")
    if hard_fail:
        label = "NEGATIVE"
        fail_reason = "hard: " + "; ".join(
            f"{t}:{k}" for t, gg in gates.items() for k, v in gg.items() if not v)
    elif low_signal:
        label = "NEUTRAL"          # 即 LOW_TD_CONTRAST: chart 间无明显重分配
        fail_reason = ""
    elif pos_any and not neg_any and not borderline:
        label, fail_reason = "POSITIVE", ""
    elif neg_any and not pos_any:
        label = "NEGATIVE"
        parts = []
        for t, ev in evidence.items():
            neg = [s for s, hit in [("global>1.02x", not ev["global_acceptable"]),
                                    ("hf_negative", ev["hf_negative"]),
                                    ("ssim_clear_drop", ev["ssim"] == "clear_drop")]
                   if hit]
            if neg:
                parts.append(f"{t}: " + ", ".join(neg))
        fail_reason = "; ".join(parts)
    else:
        label = "MIXED"
        fail_reason = "; ".join(
            f"{t}: ssim={evidence[t]['ssim']}, G_g={metrics['tiers'][t]['G_global']:+}"
            f", G_hf={metrics['tiers'][t]['G_HF']:+}" for t in evidence)
    quality = {"POSITIVE": "PASS", "NEUTRAL": "LOW_SIGNAL"}.get(label, "FAIL")

    report = dict(
        sample_id=manifest["sample_id"], sample_dir=sample_dir,
        processing_status="OK",
        structural_status=manifest["status"],
        label_quality=label,
        label_quality_alias=("LOW_TD_CONTRAST" if label == "NEUTRAL" else ""),
        label_quality_borderline=bool(borderline and label == "MIXED"),
        quality_status=quality,   # 兼容旧字段(POSITIVE->PASS 等映射)
        evidence=evidence, warnings=warnings,
        quality_scope="td_allocation_only",
        training_eligible=dict(
            td_allocation=bool(manifest["status"] == "ACCEPTED"
                               and label in ("POSITIVE", "NEUTRAL")),
            artist_local_refinement=False,
            final_packed_uv_regression=False),
        signal_dist=signal_dist, low_signal=low_signal,
        failure_reason=fail_reason,
        protocol_hash=protocol_hash, chart_hash=chart_hash,
        gates=gates,
        gains={t: dict(G_global=metrics["tiers"][t]["G_global"],
                       G_HF=metrics["tiers"][t]["G_HF"])
               for t in metrics["tiers"]},
        notes=("对比仅 Reference/PartUV-Uniform/PseudoGT-TD; 同 chart hash/local "
               "UV/packer/padding/baker/相机/atlas 分辨率; 主公平轴=相同 B_raw; "
               "LPIPS 不可用->masked SSIM(已标注); 线性 RGB 域; teacher-generated "
               "pseudo-GT, 非 artist GT。"))
    with open(f"{out}/metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=1, ensure_ascii=False)
    with open(f"{out}/quality_report.json", "w") as fp:
        json.dump(report, fp, indent=1, ensure_ascii=False)
    return report, metrics


def _figures(out, z, charts, layouts, metrics, V, F, face_refuv, valid, texA,
             tm, fid, bary, ref_lin, views_show, crop):
    from matplotlib.collections import PolyCollection
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from tdlib import gpu as tdgpu
    from gen_dashboard_assets import render_img

    lc = np.log(np.maximum(z["chart_content_score"], 1e-6))
    heat_c = plt.cm.magma(np.clip((lc - lc.min()) / max(np.ptp(lc), 1e-9), 0, 1))[:, :3]
    tiers = list(metrics["tiers"])
    fig, axs = plt.subplots(len(tiers), 2, figsize=(11, 5.5 * len(tiers)),
                            squeeze=False)
    for r_i, tname in enumerate(tiers):
        for c_i, m in enumerate(["PartUV-Uniform", "PseudoGT-TD"]):
            ax = axs[r_i][c_i]
            for c, uv, col in zip(charts, layouts[tname][m]["uvs"], heat_c):
                ax.add_collection(PolyCollection(uv[np.asarray(c["F"])],
                                                 facecolors=col, edgecolors="none"))
            ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_axis_off()
            ax.set_title(f"{m} @{tname} (fill "
                         f"{metrics['tiers'][tname]['methods'][m]['packing_fill']*100:.0f}%)",
                         fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out}/layout_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    t0 = layouts[tiers[0]]
    fig, axs = plt.subplots(len(views_show), 3,
                            figsize=(13.5, 4.6 * len(views_show)), squeeze=False)
    for r_i, vw in enumerate(views_show):
        ims = [tdgpu.textured_render(V, F, face_refuv, valid, texA, view=vw),
               tdgpu.textured_render(V, F, t0["PartUV-Uniform"]["nuv"],
                                     t0["PartUV-Uniform"]["ok"],
                                     t0["PartUV-Uniform"]["tex"], view=vw),
               tdgpu.textured_render(V, F, t0["PseudoGT-TD"]["nuv"],
                                     t0["PseudoGT-TD"]["ok"],
                                     t0["PseudoGT-TD"]["tex"], view=vw)]
        for ax, im, t in zip(axs[r_i], ims,
                             ["Reference", f"PartUV-Uniform @{tiers[0]}",
                              f"PseudoGT-TD @{tiers[0]}"]):
            ax.imshow(im); ax.set_axis_off(); ax.set_title(t, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out}/render_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
    y0, y1, x0, x1 = crop
    for ax, (m, p) in zip(axs, [("Reference", None)] + list(t0.items())):
        im = (tdgpu.textured_render(V, F, face_refuv, valid, texA,
                                    view=views_show[0]) if p is None else
              tdgpu.textured_render(V, F, p["nuv"], p["ok"], p["tex"],
                                    view=views_show[0]))
        H, W = im.shape[:2]
        ax.imshow(im[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)])
        ax.set_axis_off()
        ax.set_title(m + ("" if m == "Reference" else f" @{tiers[0]}"), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{out}/detail_crops.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    errf = {}
    for m, p in t0.items():
        uvq = np.einsum("nk,nkd->nd", bary, p["nuv"][fid])
        e = np.abs(srgb2lin(bilinear(p["tex"], uvq)) - ref_lin).mean(1)
        acc = np.zeros(len(F)); cnt = np.zeros(len(F))
        np.add.at(acc, fid, e); np.add.at(cnt, fid, 1)
        errf[m] = np.divide(acc, np.maximum(cnt, 1))
    vmax = max(float(np.percentile(errf[m][tm], 99)) for m in errf)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, m in zip(axs, errf):
        col = plt.cm.inferno(np.clip(errf[m] / max(vmax, 1e-9), 0, 1))[:, :3]
        ax.imshow(render_img(V, F, col, view=views_show[0]))
        ax.set_axis_off()
        ax.set_title(f"{m} |err| vs Reference @{tiers[0]}", fontsize=10)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="inferno"),
                      ax=axs, fraction=0.025, pad=0.01)
    cb.set_label("per-face mean |dRGB| (linear)", fontsize=9)
    plt.savefig(f"{out}/error_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from tdlib import gpu as tdgpu
    tdgpu.pick_free_gpu()
    # development case: 鞋(沿用其人工冻结 ROI)
    report, _ = quality_gate(
        "/root/youjiaZhang/PartUV/code/notebook/outputs/pseudo_gt/shoe_22b822_v1",
        "/root/youjiaZhang/PartUV/code/notebook/outputs/pseudo_gt_quality",
        views_show=((15, 45), (-10, -175)),
        crop=(0.70, 0.97, 0.03, 0.33))
    print(f"quality_status={report['quality_status']} "
          f"training_eligible={report['training_eligible']}")
    print("QUALITY_GATE: DONE")
