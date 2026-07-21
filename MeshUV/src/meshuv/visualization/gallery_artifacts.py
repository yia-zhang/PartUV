# -*- coding: utf-8 -*-
"""Gallery/QA 专用的真实 packing/rebake 产物生成(不属于 frozen teacher 面).

frozen teacher hash 只覆盖 teacher_adapter.py —— 本模块刻意独立于它:
可视化需求(返回 layout/纹理)不得污染冻结 canonical adapter。
打包/烘焙调用与 adapter.quality_check_medium 相同的 tdlib 基础设施
(xatlas + texel-center coverage baker), 语义: 等 B_signal(±1%)公平轴,
即 equal occupied signal texels —— **非**严格相同 raw atlas 分辨率/显存。
仅供 gallery/QA; 不用于数据集构建, 不入训练 schema。
"""
import os

import numpy as np

from meshuv.teacher_adapter import PARTUV_ROOT, _ensure_path  # noqa: F401


def textured_render(V, F, nuv, ok, tex, view=(15, 45), px=700):
    """tdlib GPU 纹理渲染直通(QA 用)."""
    _ensure_path()
    from tdlib import gpu as tdgpu
    return tdgpu.textured_render(np.asarray(V, float), np.asarray(F),
                                 np.asarray(nuv, float), np.asarray(ok, bool),
                                 np.asarray(tex, float), view=view, px=px)


def tc_from_sample(sample_dir):
    """从已导出样本重建打包上下文(chart 分解/标签来自冻结样本, 不重跑 teacher)."""
    from PIL import Image
    _ensure_path()
    from run_pseudo_gt_quality_gate import charts_from_sample
    z = dict(np.load(os.path.join(sample_dir, "arrays.npz")))
    texA = np.asarray(Image.open(os.path.join(
        sample_dir, "reference_basecolor.png")), float)[:, :, :3] / 255.0
    charts = charts_from_sample(z)
    V, F = z["vertices"].astype(float), z["faces"]
    tris = V[F]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    return dict(charts=charts, F=F, V=V, fa3=fa3,
                face_refuv=z["source_uv"].astype(float),
                valid=z["source_uv_valid"], texA=texA), z


def pack_and_rebake_pair(tc, scales_td, frac=0.5, r_cap=2048):
    """uniform 与 TD 的真实 xatlas packing + texel-center rebake(等 B_signal).
    返回 dict(status, G 指标, uniform/td 各含 tex/nuv/uvs/R/B_signal) 或失败原因."""
    _ensure_path()
    from tdlib.budget import rasterize_masks
    from tdlib.layout import xatlas_pack, PackingFailedError
    from tdlib.rd import (bake_atlas_ss, bilinear, ref_gradient_at_samples,
                          surface_samples)
    charts, F, fa3, texA = tc["charts"], tc["F"], tc["fa3"], tc["texA"]
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    pu_like = dict(charts=charts, F=F, area=fa3)

    def srgb2lin(x):
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def build(scales, R):
        uvs = xatlas_pack(charts, scales, resolution=R, padding_px=4)
        owner, ov, _ = rasterize_masks(ch_masks, uvs, R, R)
        tex, _, _ = bake_atlas_ss(pu_like, uvs, R, 4, tc["face_refuv"],
                                  tc["valid"], texA)
        nuv = np.zeros((len(F), 3, 2))
        for ci, c in enumerate(charts):
            nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
        return dict(S=int((owner >= 0).sum()), ov=int(ov), tex=tex, nuv=nuv,
                    uvs=uvs, R=R)

    A3 = np.array([float(fa3[c["gidx"]].sum()) for c in charts])
    a2 = np.array([float(c["a2"]) for c in charts])
    scales_u = np.sqrt(np.maximum(A3, 0) / np.maximum(a2, 1e-12))
    B = texA.shape[0] * texA.shape[1]
    R = min(max(int(round(np.sqrt(frac * B))), 64), r_cap)
    try:
        pu_b = build(scales_u, R)
    except PackingFailedError as e:
        return dict(status="PACKING_FAILED", reason=str(e)[:160])
    lo, hi_r, best = int(R * 0.6), int(R * 1.8), None
    for _ in range(9):
        mid = (lo + hi_r) // 2
        try:
            pt = build(scales_td, mid)
        except PackingFailedError:
            lo = mid + 1
            continue
        if best is None or abs(pt["S"] - pu_b["S"]) < abs(best["S"] - pu_b["S"]):
            best = pt
        if pt["S"] < pu_b["S"]:
            lo = mid + 1
        else:
            hi_r = mid - 1
    if best is None:
        return dict(status="TEACHER_PACKING_FAILED")
    match = best["S"] / max(pu_b["S"], 1)
    if abs(match - 1) > 0.01:
        return dict(status="BSIGNAL_DEV_FAIL", bsignal_match=round(match, 4))
    s = surface_samples(pu_like, tc["face_refuv"], tc["valid"], texA,
                        150_000, seed=2)
    g = ref_gradient_at_samples(texA, tc["face_refuv"], s)
    hi = g >= np.quantile(g, 0.9)
    ref_lin = srgb2lin(np.asarray(s["ref_color"]))

    def err(p):
        uvq = np.einsum("nk,nkd->nd", s["bary"], p["nuv"][s["fid"]])
        return ((srgb2lin(bilinear(p["tex"], uvq)) - ref_lin) ** 2).mean(1)

    d_u, d_t = err(pu_b), err(best)
    return dict(
        status="OK", bsignal_match=round(match, 4),
        G_global_eq=round(1 - float(d_t.mean()) / max(float(d_u.mean()), 1e-20), 4),
        G_HF_eq=round(1 - float(d_t[hi].mean()) / max(float(d_u[hi].mean()), 1e-20), 4),
        uniform={k: pu_b[k] for k in ("tex", "nuv", "uvs", "R", "S")},
        td={k: best[k] for k in ("tex", "nuv", "uvs", "R", "S")})
