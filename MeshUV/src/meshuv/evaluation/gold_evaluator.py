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


def compare_uniform_td(root, item, R=724):
    """item: CleanDataset 样本. 返回 dict(status, draw{...}, metrics_text)."""
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
    tex_src = np.asarray(Image.open(item["basecolor"]), float)[:, :, :3] / 255
    pu_like = dict(charts=charts, F=z["faces"], area=z["face_area"].astype(float))
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    out = {}
    for name, sc in (("uniform", s_uni), ("td", s_td)):
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
    ok = np.ones(len(z["faces"]), bool)
    r_u = tdgpu.textured_render(z["vertices"].astype(float), z["faces"],
                                out["uniform"]["nuv"], ok,
                                out["uniform"]["tex"], view=(18, 40))
    r_t = tdgpu.textured_render(z["vertices"].astype(float), z["faces"],
                                out["td"]["nuv"], ok, out["td"]["tex"],
                                view=(18, 40))
    dif = np.abs(r_u.astype(float) - r_t.astype(float)).mean(-1)

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

    txt = (f"same raw atlas budget R={R} ({R*R:,} px)\n"
           f"occupancy: uniform {out['uniform']['occ']*100:.1f}% / "
           f"td {out['td']['occ']*100:.1f}%\n"
           f"charts={nC}  coverage={item['manifest']['coverage_area']*100:.2f}%\n"
           f"overlap: {out['uniform']['ov']}/{out['td']['ov']}")
    return dict(status="OK", metrics_text=txt,
                draw=dict(uniform_uv=draw_uv("uniform"), td_uv=draw_uv("td"),
                          uniform_tex=draw_tex("uniform"),
                          td_tex=draw_tex("td"), diff=draw_diff))
