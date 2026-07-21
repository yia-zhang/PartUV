# -*- coding: utf-8 -*-
"""V1.1+ 诊断脚本公共工具(只读样本; 打包/烘焙用既有 tdlib 基础设施).

被 run_dual_axis_split.py / packing_ab_min.py / seam_diagnosis_gradient.py 共用。
不修改 teacher/β/content signal/PartUV。
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image

from tdlib.budget import rasterize_masks
from tdlib.layout import xatlas_pack
from tdlib.rd import (bake_atlas_masks, bilinear, ref_gradient_at_samples,
                      surface_samples)
from run_pseudo_gt_quality_gate import (charts_from_sample, srgb2lin, lin2srgb,
                                        SS, SEED_EVAL, N_SAMPLES, SEAM_BARY)


def load_sample(sample_dir):
    """加载已导出样本并重建评价上下文(chart hash 校验)."""
    man = json.load(open(f"{sample_dir}/manifest.json"))
    z = dict(np.load(f"{sample_dir}/arrays.npz"))
    chart_hash = hashlib.sha256(
        np.ascontiguousarray(z["face_to_chart"]).tobytes()
        + np.ascontiguousarray(z["local_uv_before_td"]).tobytes()).hexdigest()
    assert chart_hash == man["teacher"]["chart_hash"], "chart hash 不一致"
    texA = np.asarray(Image.open(f"{sample_dir}/reference_basecolor.png"),
                      float)[:, :, :3] / 255.0
    V, F = z["vertices"], z["faces"]
    charts = charts_from_sample(z)
    tris = V[F]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    A3 = np.array([fa3[c["gidx"]].sum() for c in charts])
    a2 = np.array([c["a2"] for c in charts])
    ctx = dict(
        manifest=man, z=z, texA=texA, V=V, F=F,
        face_refuv=z["source_uv"], valid=z["source_uv_valid"],
        charts=charts, fa3=fa3, A3=A3, a2=a2,
        scales_td=z["chart_target_scale"].astype(float),
        scales_uni=np.sqrt(A3 / a2),
        demand=z["chart_demand_normalized"].astype(float),
        area_norm=A3 / A3.sum(),
        pu_like=dict(charts=charts, F=F, area=fa3),
        ch_masks=[dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts],
        B_source=texA.shape[0] * texA.shape[1])
    ctx["signal_dist"] = float(0.5 * np.abs(ctx["demand"] - ctx["area_norm"]).sum())
    return ctx


def eval_samples(ctx):
    """冻结评价采样(150k, seed=2) + HF/seam 掩码 + 线性 ref."""
    s = surface_samples(ctx["pu_like"], ctx["face_refuv"], ctx["valid"],
                        ctx["texA"], N_SAMPLES, seed=SEED_EVAL)
    g = ref_gradient_at_samples(ctx["texA"], ctx["face_refuv"], s)
    return dict(fid=s["fid"], bary=s["bary"],
                ref_lin=srgb2lin(np.asarray(s["ref_color"])),
                hi=g >= np.quantile(g, 0.9),
                seam=s["bary"].min(1) < SEAM_BARY)


def pack_only(ctx, scales, R, rotate=True, order=None, padding_px=4):
    """仅打包+度量 B_signal(不烘焙). order=chart 排列(None=原序). 失败抛异常."""
    charts, n = ctx["charts"], len(ctx["charts"])
    perm = np.arange(n) if order is None else np.asarray(order)
    uvs_p = xatlas_pack([charts[i] for i in perm],
                        np.asarray(scales)[perm], resolution=R,
                        padding_px=padding_px, rotate=rotate)
    uvs = [None] * n
    for k, i in enumerate(perm):
        uvs[i] = uvs_p[k]
    owner, ov, _ = rasterize_masks(ctx["ch_masks"], uvs, R, R)
    N_c = np.bincount(owner[owner >= 0], minlength=n).astype(float)
    return dict(uvs=uvs, b_signal=int((owner >= 0).sum()), overlap=int(ov),
                N_c=N_c, fill=float((owner >= 0).sum() / R / R))


def bake_layout(ctx, uvs, R, ss=SS, tex_src=None):
    """在冻结 layout 上烘焙(可换源纹理/超采样倍率), 返回 (tex, nuv).
    Baker Convergence Audit 后: 覆盖感知降采样(rd.bake_atlas_ss),
    未覆盖子纹素不再被当黑色平均。"""
    from tdlib.rd import bake_atlas_ss
    src = ctx["texA"] if tex_src is None else tex_src
    tex, _, _ = bake_atlas_ss(ctx["pu_like"], uvs, R, ss,
                              ctx["face_refuv"], ctx["valid"], src)
    nuv = np.zeros((len(ctx["F"]), 3, 2))
    for ci, c in enumerate(ctx["charts"]):
        nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
    return tex, nuv


def surface_err(tex, nuv, ev):
    """逐采样点线性 RGB 平方误差."""
    uvq = np.einsum("nk,nkd->nd", ev["bary"], nuv[ev["fid"]])
    return ((srgb2lin(bilinear(tex, uvq)) - ev["ref_lin"]) ** 2).mean(1)
