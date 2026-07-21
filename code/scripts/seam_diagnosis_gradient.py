# -*- coding: utf-8 -*-
"""Seam 成因诊断 —— 仅 synth_gradient, 冻结同一 TD@50pct layout, 逐因素单变量.

报告:
  1) seam band 表面积份额(采样点 min bary<0.08 占比)
  2) seam error share(标准 SS4 bake)
  3) seam error enrichment = error share / area share
  4) full-resolution seam_mse_floor(同 layout, R=sqrt(B_source))
  5) 低预算 seam_excess = seam_mse(50pct) - floor
  6) SS4 vs SS8 单变量(同 layout 同分辨率, 只改超采样)
  7) analytic smooth-gradient 直接 bake: 用 identity-UV 纹理经同一 baker 得到
     每 atlas 纹素的插值源 UV, 再代入解析线性模型 c(u,v)=a+b·u+c·v
     (对源纹理最小二乘拟合, 报告拟合残差) —— 隔离「源纹理重采样」与
     「chart 拓扑/有限分辨率/评价定义」两类 seam 误差来源。
决策规则(预注册): 仅当 SS8(或 chart-aware bake, 未实现)使 seam error density
降低 >=25% 才进入 baker 修正; 否则归为 PartUV chart topology /
finite-resolution floor / metric-definition, 不在 TD 模块加 seam 算法。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.rd import bilinear
from diag_common import (load_sample, eval_samples, pack_only, bake_layout,
                         surface_err, srgb2lin)

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"

ctx = load_sample(f"{OUT}/synth_gradient/sample")
ev = eval_samples(ctx)
seam, hi = ev["seam"], ev["hi"]
R50 = max(int(round(np.sqrt(0.50 * ctx["B_source"]))), 64)
R_full = int(round(np.sqrt(ctx["B_source"])))

# 冻结 layout: TD@50pct 一次打包, 全部变量共用(full-res 仅换光栅化分辨率)
pack = pack_only(ctx, ctx["scales_td"], R50)
uvs = pack["uvs"]
print(f"layout 冻结: R50={R50} fill={pack['fill']:.3f} overlap={pack['overlap']}",
      flush=True)


def seam_stats(tex, nuv):
    d = surface_err(tex, nuv, ev)
    return dict(mse=float(d.mean()), mse_seam=float(d[seam].mean()),
                mse_interior=float(d[~seam].mean()),
                seam_error_share=float(d[seam].sum() / max(d.sum(), 1e-20)))


rep = dict(asset="synth_gradient", layout="TD@50pct(冻结)", R50=R50, R_full=R_full,
           seam_bary=0.08,
           seam_area_share=float(seam.mean()))

# (2)(3) 标准 SS4
tex4, nuv = bake_layout(ctx, uvs, R50, ss=4)
s4 = seam_stats(tex4, nuv)
rep["standard_SS4_at_50pct"] = s4
rep["seam_error_enrichment"] = round(s4["seam_error_share"] / max(rep["seam_area_share"], 1e-9), 2)

# (4)(5) full-res floor(同 layout)
texF, nuvF = bake_layout(ctx, uvs, R_full, ss=4)
sF = seam_stats(texF, nuvF)
rep["fullres_SS4_floor"] = sF
rep["seam_mse_floor"] = sF["mse_seam"]
rep["seam_excess_50pct"] = s4["mse_seam"] - sF["mse_seam"]

# (6) SS8 单变量
tex8, _ = bake_layout(ctx, uvs, R50, ss=8)
s8 = seam_stats(tex8, nuv)
rep["SS8_at_50pct"] = s8
rep["ss8_seam_density_reduction"] = round(1 - s8["mse_seam"] / max(s4["mse_seam"], 1e-20), 4)

# (7) analytic bake: 拟合线性模型(线性 RGB 域) + identity-UV 经同一 baker
texA_lin = srgb2lin(ctx["texA"])
Ht, Wt = texA_lin.shape[:2]
uu, vv = np.meshgrid((np.arange(Wt) + .5) / Wt, 1 - (np.arange(Ht) + .5) / Ht)
Amat = np.stack([np.ones_like(uu).ravel(), uu.ravel(), vv.ravel()], 1)
coef, *_ = np.linalg.lstsq(Amat, texA_lin.reshape(-1, 3), rcond=None)
fit_rms = float(np.sqrt(((Amat @ coef - texA_lin.reshape(-1, 3)) ** 2).mean()))
rep["analytic_fit_rms_linear"] = fit_rms

analytic = lambda uv: np.clip(
    np.stack([np.ones(len(uv)), uv[:, 0], uv[:, 1]], 1) @ coef, 0, 1)

# identity-UV 纹理(线性函数, bilinear 精确重建); 先数值验证采样约定
texUV = np.zeros((Ht, Wt, 3))
texUV[:, :, 0], texUV[:, :, 1] = uu, vv
probe = np.random.RandomState(0).rand(2000, 2) * 0.9 + 0.05
err_conv = float(np.abs(bilinear(texUV, probe)[:, :2] - probe).max())
assert err_conv < 2e-3, f"identity-UV 采样约定校验失败({err_conv})"
rep["identity_uv_convention_err"] = err_conv

# 经同一 baker 烘 identity-UV -> 每 atlas 纹素的插值源 UV -> 解析上色
# (bake_layout 会做 srgb2lin/lin2srgb, 对 UV 数据是非线性失真 -> 这里直接
#  用原始 bake 的高分结果做 box 降采样, 全程线性)
from tdlib.rd import bake_atlas_masks
uv_hi, _, _ = bake_atlas_masks(ctx["pu_like"], uvs, R50 * 4,
                               ctx["face_refuv"], ctx["valid"], texUV)
uv_atlas = uv_hi.reshape(R50, 4, R50, 4, 3).mean(axis=(1, 3))
tex_an_lin = analytic(uv_atlas[:, :, :2].reshape(-1, 2)).reshape(R50, R50, 3)

# 评价: ref 也解析(隔离评价端源采样), 误差在线性域
uvq = np.einsum("nk,nkd->nd", ev["bary"], nuv[ev["fid"]])
src_uv = np.einsum("nk,nkd->nd", ev["bary"],
                   ctx["face_refuv"][ev["fid"]].astype(float))
ref_an = analytic(src_uv)
# 对 atlas 的读取与标准管线一致(bilinear); tex_an_lin 已是线性域
rec_an = bilinear(tex_an_lin, uvq)
d_an = ((rec_an - ref_an) ** 2).mean(1)
rep["analytic_SS4_at_50pct"] = dict(
    mse=float(d_an.mean()), mse_seam=float(d_an[seam].mean()),
    mse_interior=float(d_an[~seam].mean()),
    seam_error_share=float(d_an[seam].sum() / max(d_an.sum(), 1e-20)))
rep["analytic_seam_vs_standard"] = round(
    rep["analytic_SS4_at_50pct"]["mse_seam"] / max(s4["mse_seam"], 1e-20), 4)

# 预注册决策
if rep["ss8_seam_density_reduction"] >= 0.25:
    rep["decision"] = "SS8 使 seam error density 降低>=25% -> 可进入 baker 修正评审"
else:
    rep["decision"] = ("SS8 降幅不足 25% -> 归为 PartUV chart topology / "
                       "finite-resolution floor / metric-definition, "
                       "不在 TD 模块加入 seam 算法")

with open(f"{OUT11}/seam_diagnosis_gradient.json", "w") as fp:
    json.dump(rep, fp, indent=1, ensure_ascii=False)
print(json.dumps(rep, ensure_ascii=False, indent=1))
print("SEAM_DIAG: DONE")
