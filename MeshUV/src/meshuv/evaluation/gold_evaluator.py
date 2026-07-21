# -*- coding: utf-8 -*-
"""Gold QA(可视化/评测专用, 不影响 dataset acceptance):
uniform vs TD 在**相同 raw atlas budget(同 R×R)**下的真实 packing+rebake。
依赖 PARTUV_ROOT teacher checkout 的 xatlas/texel-center baker。"""
import os
import sys

import numpy as np

PARTUV_ROOT = os.environ.get("PARTUV_ROOT", "/root/youjiaZhang/PartUV/code")


def _wire():
    for p in (PARTUV_ROOT, os.path.join(PARTUV_ROOT, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def compare_methods(root, item, R=724, student_fraction=None):
    """Uniform/Teacher(/Student) 相同 raw atlas budget 下 packing+rebake,
    对 source reference 表面采样报告 MSE/PSNR/HF error。
    student_fraction: Student 预测的 chart_target_area_fraction(可选)。"""
    _wire()
    from tdlib.budget import rasterize_masks
    from tdlib.layout import xatlas_pack, PackingFailedError
    from tdlib.rd import bake_atlas_ss
    from tdlib import gpu as tdgpu
    from PIL import Image
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    z, tt = item["inputs"], item["targets"]
    f2c, luv = z["face_to_chart"], z["local_uv"].astype(float)
    nC = len(tt["chart_surface_area"])
    charts = []
    for ci in range(nC):
        fids = np.where(f2c == ci)[0]
        cor = luv[fids].reshape(-1, 2)
        uq, inv = np.unique(cor, axis=0, return_inverse=True)
        a2 = float(np.abs(np.cross(cor[1::3] - cor[0::3],
                                   cor[2::3] - cor[0::3])).sum() / 2)
        charts.append(dict(UV=uq, F=inv.reshape(-1, 3), gidx=fids,
                           a2=max(a2, 1e-12)))
    A3 = tt["chart_surface_area"].astype(float)
    a2 = np.array([c["a2"] for c in charts])
    s_uni = np.sqrt(np.maximum(A3, 1e-12) / a2)
    frac = tt["chart_target_area_fraction"].astype(float)
    s_td = np.sqrt(np.maximum(frac, 1e-12) / a2)
    methods = [("uniform", s_uni), ("teacher", s_td)]
    if student_fraction is not None:
        sf = np.maximum(np.asarray(student_fraction, float), 1e-12)
        methods.append(("student", np.sqrt(sf / a2)))
    tex_src = np.asarray(Image.open(item["basecolor"]), float)[:, :, :3] / 255
    pu_like = dict(charts=charts, F=z["faces"], area=z["face_area"].astype(float))
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    out = {}
    for name, sc in methods:
        try:
            uvs = xatlas_pack(charts, sc, resolution=R, padding_px=4)
        except PackingFailedError as e:
            return dict(status="PACKING_FAILED", which=name, reason=str(e)[:120])
        owner, ov, _ = rasterize_masks(ch_masks, uvs, R, R)
        tex, _, _ = bake_atlas_ss(pu_like, uvs, R, 4, z["source_uv"].astype(float),
                                  z["source_uv_valid"], tex_src)
        nuv = np.zeros((len(z["faces"]), 3, 2))
        for ci, c in enumerate(charts):
            nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
        out[name] = dict(uvs=uvs, tex=tex, nuv=nuv,
                         occ=float((owner >= 0).sum()) / R / R, ov=int(ov))
    # 表面采样 vs source reference(线性域 MSE/PSNR/HF)
    from tdlib.rd import surface_samples, ref_gradient_at_samples, bilinear
    smp = surface_samples(pu_like, z["source_uv"].astype(float),
                          z["source_uv_valid"], tex_src, 120_000, seed=2)
    g = ref_gradient_at_samples(tex_src, z["source_uv"].astype(float), smp)
    hi = g >= np.quantile(g, 0.9)

    def s2l(x):
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    ref_lin = s2l(np.asarray(smp["ref_color"]))
    for name in out:
        uvq = np.einsum("nk,nkd->nd", smp["bary"], out[name]["nuv"][smp["fid"]])
        d = ((s2l(bilinear(out[name]["tex"], uvq)) - ref_lin) ** 2).mean(1)
        out[name]["mse"] = float(d.mean())
        out[name]["psnr"] = round(10 * np.log10(1 / max(d.mean(), 1e-12)), 2)
        out[name]["hf_mse"] = float(d[hi].mean())
    ok = np.ones(len(z["faces"]), bool)
    renders = {n: tdgpu.textured_render(
        z["vertices"].astype(float), z["faces"], out[n]["nuv"], ok,
        out[n]["tex"], view=(18, 40)) for n in out}
    dif = np.abs(renders["uniform"].astype(float)
                 - renders[list(out)[-1]].astype(float)).mean(-1)

    def draw_uv(which):
        def f(ax):
            for ci, uv in enumerate(out[which]["uvs"]):
                ax.add_collection(PolyCollection(
                    uv[np.asarray(charts[ci]["F"])],
                    facecolors=plt.cm.tab20(ci % 20), edgecolors="none"))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal"); ax.set_axis_off()
            ax.set_title(f"{which} packed UV (same R={R})", fontsize=9)
        return f

    def draw_tex(which):
        def f(ax):
            ax.imshow(np.clip(out[which]["tex"], 0, 1))
            ax.set_axis_off()
            ax.set_title(f"{which} rebake (same raw budget)", fontsize=9)
        return f

    def draw_diff(ax):
        im = ax.imshow(dif, cmap="inferno")
        ax.set_axis_off(); ax.set_title("render |uniform-TD|", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)

    lines = [f"same raw atlas budget R={R} ({R*R:,} px)"]
    for n in out:
        lines.append(f"{n:8s} PSNR={out[n]['psnr']:6.2f}dB "
                     f"hf_mse={out[n]['hf_mse']:.2e} occ={out[n]['occ']*100:.0f}%")
    lines.append(f"charts={nC} coverage={item['manifest']['coverage_area']*100:.1f}%")
    draw = dict(diff=draw_diff)
    for n in out:
        draw[f"{n}_uv"] = draw_uv(n)
        draw[f"{n}_tex"] = draw_tex(n)
    # 兼容旧 notebook 键名
    draw["td_uv"] = draw.get("teacher_uv", draw.get("td_uv"))
    draw["td_tex"] = draw.get("teacher_tex", draw.get("td_tex"))
    return dict(status="OK", metrics_text="\n".join(lines), metrics=out,
                renders=renders, draw=draw)


compare_uniform_td = compare_methods   # 兼容别名
