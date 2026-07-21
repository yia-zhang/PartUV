# -*- coding: utf-8 -*-
"""固定存储预算(B_raw)视觉质量实验 —— 鞋, 两档预算(50%/25% raw texels).

核心问题: 相同 base-color raw texel storage 下, PartUV-TD 是否比
SourceUV-Downsample 与 PartUV-Uniform 都更接近 Reference?
(不预设结论; PartUV-Uniform 用于隔离 TD 分配本身的贡献)

四组方法(同 renderer/视角/滤波/颜色空间):
  Reference           原 UV + 原生 1024² 纹理
  SourceUV-Downsample 原 UV + 线性 RGB BOX(area) 低通降采样到 R²
  PartUV-Uniform      缓存 charts + xatlas + 4x 超采样 rebake, β=0, atlas=R²
  PartUV-TD           同上, β=0.75

公平轴: B_raw = Σ W_k·H_k (存储纹素)。R=724(50.0%-0.02%) / R=512(恰 25%)。
指标: 线性 RGB surface MSE/PSNR、固定 8 视角 render SSIM(LPIPS 不可用)、
高频区(Reference 自身梯度 top-10% 采样点)误差。
ROI 协议常量: 视角/裁剪在脚本头部冻结, 不由 top_chart/luminance-std 选择。
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from tdlib.layout import layout_with_scales
from tdlib.pipeline import load_reference
from tdlib.rd import (bake_atlas_masks, bilinear, ref_gradient_at_samples,
                      surface_samples)
from tdlib.signal import demand_weights, luminance_std_heuristic
from gen_dashboard_assets import render_img

# ================= 协议常量(冻结) =================
ASSET = "/root/youjiaZhang/PartUV/code/data/objaverse_22b822c6520d4d49.glb"
CACHE = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1b/shoe_22b822/charts_cache.pkl"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/storage_ab/shoe"
TIERS = [("50pct", 724), ("25pct", 512)]     # R² / 1024² = 49.98% / 25.00%
SS = 4                                       # rebake 超采样(线性域 box, 与基线低通等价)
SEED_EVAL = 2
N_SAMPLES = 150_000
VIEW_MAIN = (15, 45)
VIEW_LOGO = (-10, -175)                      # 人工冻结: 正对鞋标一侧
CROP_LOGO = (0.70, 0.97, 0.03, 0.33)         # 人工冻结: logo 视角渲染的相对裁剪框(y0,y1,x0,x1)
SSIM_VIEWS = [(15, a) for a in range(0, 360, 45)]   # 固定 8 视角
BETA_TD = 0.75
# ==================================================


def srgb2lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def downsample_linear(tex_srgb, R):
    """线性 RGB 域 BOX(area) 低通降采样(PIL F 模式逐通道)."""
    lin = srgb2lin(tex_srgb)
    chans = [np.asarray(Image.fromarray(lin[:, :, c].astype(np.float32), "F")
                        .resize((R, R), Image.BOX)) for c in range(3)]
    return lin2srgb(np.stack(chans, -1).astype(float))


def bake_ss(pu, uvs, R, face_refuv, valid, texA):
    """4x 超采样 rebake -> 线性域 box 降回 R(与基线 area 低通等价的抗锯齿)."""
    tex_hi, _, _ = bake_atlas_masks(pu, uvs, R * SS, face_refuv, valid, texA)
    lin = srgb2lin(tex_hi).reshape(R, SS, R, SS, 3).mean(axis=(1, 3))
    return lin2srgb(lin)


def newuv_faces(charts, uvs, n_faces):
    out = np.zeros((n_faces, 3, 2))
    ok = np.zeros(n_faces, bool)
    for ci, c in enumerate(charts):
        cF = np.asarray(c["F"])
        out[c["gidx"]] = uvs[ci][cF]
        ok[c["gidx"]] = True
    return out, ok


def main():
    os.makedirs(OUT, exist_ok=True)
    pu = pickle.load(open(CACHE, "rb"))
    charts, F, V = pu["charts"], pu["F"], pu["V"]
    area, covered = pu["area"], pu["covered"]
    ref = load_reference(ASSET, V, F, pu["mesh_scale"])
    texA = ref["texA"]
    from tdlib.rd import prepare_face_ref_uv
    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)
    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"],
                                 ref["ok_map"])
    sel = covered & ref["ok_map"]
    _, w_td = demand_weights(cw, sel, area, beta=BETA_TD)
    w_uni = np.ones(len(F))

    # 固定评价采样(全方法共用) + 高频子集(Reference 自身梯度 top-10%)
    s = surface_samples(pu, face_refuv, valid, texA, N_SAMPLES, seed=SEED_EVAL)
    g = ref_gradient_at_samples(texA, face_refuv, s)
    hi = g >= np.quantile(g, 0.9)
    fid, bary = s["fid"], s["bary"]
    uv0_s = np.einsum("nk,nkd->nd", bary, face_refuv[fid])   # 采样点的原 UV
    ref_lin = srgb2lin(np.asarray(s["ref_color"]))

    B_source = texA.shape[0] * texA.shape[1]
    summary = dict(asset="shoe_22b822", B_source_raw=int(B_source),
                   ss=SS, n_samples=N_SAMPLES, seed=SEED_EVAL,
                   views_ssim=SSIM_VIEWS, view_logo=VIEW_LOGO,
                   crop_logo=CROP_LOGO, lpips="不可用(环境无 lpips 包), 用 SSIM",
                   tiers={})

    for tname, R in TIERS:
        print(f"===== tier {tname} (R={R}, B_raw={R*R:,} = "
              f"{R*R/B_source*100:.2f}% of source) =====", flush=True)
        # --- 三个压缩方法(B_raw 全部恰为 R²) ---
        tex_ds = downsample_linear(texA, R)
        uvs_uni, _ = layout_with_scales(charts, w_uni, packer="xatlas", resolution=R)
        uvs_td, _ = layout_with_scales(charts, w_td, packer="xatlas", resolution=R)
        tex_uni = bake_ss(pu, uvs_uni, R, face_refuv, valid, texA)
        tex_td = bake_ss(pu, uvs_td, R, face_refuv, valid, texA)
        nuv_uni, ok_uni = newuv_faces(charts, uvs_uni, len(F))
        nuv_td, ok_td = newuv_faces(charts, uvs_td, len(F))

        # --- 采样域指标(线性 RGB) ---
        def sample_colors(tex, uv_per_face_corner):
            uvq = np.einsum("nk,nkd->nd", bary, uv_per_face_corner[fid])
            return srgb2lin(bilinear(tex, uvq))

        cols = {
            "SourceUV-Downsample": sample_colors(tex_ds, face_refuv),
            "PartUV-Uniform": sample_colors(tex_uni, nuv_uni),
            "PartUV-TD": sample_colors(tex_td, nuv_td),
        }
        res_t = {"B_raw": int(R * R), "B_raw_pct_of_source": round(R*R/B_source*100, 2),
                 "methods": {}}
        err_faces = {}
        for m, c in cols.items():
            d2 = ((c - ref_lin) ** 2).mean()
            mse_hi = float(((c - ref_lin)[hi] ** 2).mean())
            psnr = float(10 * np.log10(1.0 / max(d2, 1e-12)))
            res_t["methods"][m] = dict(mse_linear=float(d2), psnr_db=round(psnr, 2),
                                       mse_hifreq_linear=mse_hi)
            e = np.abs(c - ref_lin).mean(1)
            acc = np.zeros(len(F)); cnt = np.zeros(len(F))
            np.add.at(acc, fid, e); np.add.at(cnt, fid, 1)
            err_faces[m] = np.divide(acc, np.maximum(cnt, 1))

        # --- 固定 8 视角 render SSIM(同 renderer/滤波/颜色空间) ---
        rends = {m: [] for m in ["Reference"] + list(cols)}
        for vw in SSIM_VIEWS:
            rr = tdgpu.textured_render(V, F, face_refuv, valid, texA, view=vw)
            rends["Reference"].append(rr)
            rends["SourceUV-Downsample"].append(
                tdgpu.textured_render(V, F, face_refuv, valid, tex_ds, view=vw))
            rends["PartUV-Uniform"].append(
                tdgpu.textured_render(V, F, nuv_uni, ok_uni, tex_uni, view=vw))
            rends["PartUV-TD"].append(
                tdgpu.textured_render(V, F, nuv_td, ok_td, tex_td, view=vw))
        for m in cols:
            vals = []
            for a, b in zip(rends["Reference"], rends[m]):
                h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
                vals.append(ssim(a[:h, :w], b[:h, :w], channel_axis=2, data_range=1.0))
            res_t["methods"][m]["ssim_8view_mean"] = round(float(np.mean(vals)), 4)
            res_t["methods"][m]["ssim_8view_min"] = round(float(np.min(vals)), 4)

        # --- 四联图(主视角 + logo 视角) + 固定裁剪 ---
        order = ["Reference", "SourceUV-Downsample", "PartUV-Uniform", "PartUV-TD"]
        texmap = {"Reference": (face_refuv, valid, texA),
                  "SourceUV-Downsample": (face_refuv, valid, tex_ds),
                  "PartUV-Uniform": (nuv_uni, ok_uni, tex_uni),
                  "PartUV-TD": (nuv_td, ok_td, tex_td)}
        for tag, vw in [("main", VIEW_MAIN), ("logo", VIEW_LOGO)]:
            fig, axs = plt.subplots(1, 4, figsize=(18, 5.2))
            for ax, m in zip(axs, order):
                uvf, okm, tx = texmap[m]
                im = tdgpu.textured_render(V, F, uvf, okm, tx, view=vw)
                ax.imshow(im); ax.set_axis_off()
                ax.set_title(f"{m}" + ("" if m == "Reference"
                                       else f"  (B_raw={R}x{R})"), fontsize=10)
                if tag == "logo":
                    y0, y1, x0, x1 = CROP_LOGO
                    H, W = im.shape[:2]
                    ax.add_patch(plt.Rectangle((x0*W, y0*H), (x1-x0)*W, (y1-y0)*H,
                                               fill=False, ec="lime", lw=1.5))
            plt.tight_layout()
            plt.savefig(f"{OUT}/{tname}_quad_{tag}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
        # 固定 crop 条(logo 视角)
        fig, axs = plt.subplots(1, 4, figsize=(16, 4.6))
        for ax, m in zip(axs, order):
            uvf, okm, tx = texmap[m]
            im = tdgpu.textured_render(V, F, uvf, okm, tx, view=VIEW_LOGO)
            y0, y1, x0, x1 = CROP_LOGO
            H, W = im.shape[:2]
            ax.imshow(im[int(y0*H):int(y1*H), int(x0*W):int(x1*W)])
            ax.set_axis_off(); ax.set_title(m, fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{OUT}/{tname}_crop_logo.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # --- absolute-error heatmap(线性域, 共享色标) ---
        vmax = max(float(np.percentile(err_faces[m][covered & valid], 99))
                   for m in cols)
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        for ax, m in zip(axs, cols):
            colh = plt.cm.inferno(np.clip(err_faces[m] / max(vmax, 1e-9), 0, 1))[:, :3]
            ax.imshow(render_img(V, F, colh, view=VIEW_MAIN))
            ax.set_axis_off(); ax.set_title(f"{m} |err| vs Reference", fontsize=10)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap="inferno"),
                          ax=axs, fraction=0.02, pad=0.01)
        cb.set_label("per-face mean |dRGB| (linear)", fontsize=9)
        plt.savefig(f"{OUT}/{tname}_err_heatmap.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        summary["tiers"][tname] = res_t
        for m in order[1:]:
            r = res_t["methods"][m]
            print(f"  {m:22s} MSE(lin)={r['mse_linear']:.3e} PSNR={r['psnr_db']}dB "
                  f"SSIM={r['ssim_8view_mean']} hiMSE={r['mse_hifreq_linear']:.3e}",
                  flush=True)

    with open(f"{OUT}/summary.json", "w") as fp:
        json.dump(summary, fp, indent=1, ensure_ascii=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
