# -*- coding: utf-8 -*-
"""为 Dashboard v1.1 一次性生成可视化素材(调用 tdlib, 结果存 outputs/dashboard/).
Dashboard notebook 本身只加载这些文件, 不重跑 PartUV/长实验.

每资产(鞋/车轮), 参考预算 B=1M (R=1000), 方法 L1 / L2_heuristic / RD_hull_global_mse:
  zoom_compare.png     最高内容 chart(鞋=logo)局部放大(标注光栅化 interior texel 数)
  zoom_location.png    zoom chart 在 3D 模型上的位置标记
  layout_compare.png   三方法 UV 布局 + 共享 colorbar + zoom chart 圈注
  renders_compare.png  reference + 三方法同视角紧裁渲染
  error_heatmap.png    三方法逐面 |Δ sRGB| 热图(共享色标)
  rebake_<m>.png       烘焙 atlas 全图(附录用)
  rd_points.json       凸包修正最大的 3 个 chart 的原始/hull 后 R-D 点
  dashboard_data.json  interior texel 数/q²统计/hull 诊断等
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tdlib.budget import rasterize_masks
from tdlib.geometry import tri_area_2d
from tdlib.layout import layout_with_scales
from tdlib.pipeline import load_reference, run_partuv
from tdlib.rd import (bake_atlas_masks, bilinear, chart_rd_curves, hull_curves,
                      oracle_allocate, prepare_face_ref_uv, surface_samples)
from tdlib.signal import demand_weights, luminance_std_heuristic

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT_ROOT = "/root/youjiaZhang/PartUV/code/notebook/outputs/dashboard"
B_REF = 1_000_000
R_REF = 1000
BETA, Z_MAX, Q_MIN, Q_MAX = 0.4, 2.5, 0.5, 2.83

ASSETS = [
    ("shoe_22b822", f"{DATA}/objaverse_22b822c6520d4d49.glb"),
    ("wheel_92ff6", f"{DATA}/objaverse_92ff65712c62408d.glb"),
]
# 注: teacher candidate v0 — raw R-D 校准与 fixed-B_signal 验证通过前不作为正式 GT
METHOD_LABEL = {"L1": "L1 uniform", "L2_heuristic": "L2 luminance-std",
                "RD_hull_global_mse": "RD_hull_global_mse (teacher cand. v0)"}


def render3d(ax, V, F, col, title, view=(15, 45), cull=True):
    """同视角渲染: 逐轴真实包围盒 + 等比例 box aspect, 无线框.
    cull: 背面剔除 —— 双面重合面片在 matplotlib 深度排序下会互相穿插
    (黑色三角斑纹), 剔除背向相机的面即可消除."""
    tris = V[F]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    if cull:
        e, a = np.radians(view[0]), np.radians(view[1])
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        keep = (n @ cam) > 1e-9
        if keep.sum() > len(tris) * 0.1:      # 防止绕向不一致的 mesh 被剔空
            tris, n, col = tris[keep], n[keep], np.asarray(col)[keep]
    shade = (0.72 + 0.28 * np.abs(n @ np.array([0.4, 0.5, 0.77])))[:, None]
    fc = np.clip(col * shade, 0, 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc,
                                         linewidths=0.05))
    mn, mx = V.min(0), V.max(0)
    ext = np.maximum(mx - mn, 1e-9)
    pad = 0.02 * ext.max()
    ax.set_xlim(mn[0] - pad, mx[0] + pad)
    ax.set_ylim(mn[1] - pad, mx[1] + pad)
    ax.set_zlim(mn[2] - pad, mx[2] + pad)
    ax.set_box_aspect(tuple(ext + 2 * pad))
    ax.set_axis_off()
    ax.view_init(elev=view[0], azim=view[1])
    if title:
        ax.set_title(title, fontsize=10)


def render_img(V, F, col, view=(15, 45), px=900):
    """离屏渲染 -> 按内容紧裁的 RGB 数组(白底), 保证物体充满画面."""
    import io
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    render3d(fig.add_subplot(111, projection="3d"), V, F, col, None, view=view)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = plt.imread(buf)[:, :, :3]
    fg = (img < 0.995).any(axis=2)
    ys, xs = np.where(fg)
    if len(ys) == 0:
        return img
    p = 6
    y0, y1 = max(ys.min() - p, 0), min(ys.max() + p, img.shape[0])
    x0, x1 = max(xs.min() - p, 0), min(xs.max() + p, img.shape[1])
    return img[y0:y1, x0:x1]


def facing_view(V, F, gidx, w=None):
    """选一个能看到指定面集合的相机视角(可选 w: 每面权重, 如内容分数).
    有 GPU 时在 15° 候选网格上按 z-buffer 实测"可见加权面积"取最优,
    遮挡被真实处理(如内赤道 chart 自动选到能从洞口看进去的方向);
    无 GPU 回退: 面积加权平均法线, 法线抵消退化时改用不含遮挡的
    可见投影面积 Σ max(0, n·d)·A·w 启发式."""
    tris = V[F[gidx]]
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])   # 模 = 2A
    if w is not None and np.max(w) > 0:
        n = n * (np.asarray(w, float) / np.max(w))[:, None]
    try:
        from tdlib import gpu as _tdgpu
        gpu_ok = _tdgpu.available()
    except Exception:
        gpu_ok = False
    if gpu_ok:
        wf = np.zeros(len(F))
        wf[np.asarray(gidx)] = (np.asarray(w, float) / np.max(w)
                                if w is not None and np.max(w) > 0 else 1.0)
        best, best_v = (30.0, 45.0), -1.0
        for e in range(-60, 61, 15):
            for a in range(-180, 180, 15):
                v = _tdgpu.visible_weight(V, F, wf, (e, a))
                if v > best_v:
                    best_v, best = v, (float(e), float(a))
        return best
    m = n.sum(axis=0)
    if np.linalg.norm(m) < 0.3 * max(np.linalg.norm(n, axis=1).sum(), 1e-12):
        el = np.radians(np.arange(-60, 61, 15))
        az = np.radians(np.arange(-180, 180, 15))
        d = np.stack([np.outer(np.cos(el), np.cos(az)),
                      np.outer(np.cos(el), np.sin(az)),
                      np.repeat(np.sin(el)[:, None], len(az), 1)], -1).reshape(-1, 3)
        m = d[int(np.maximum(d @ n.T, 0).sum(axis=1).argmax())]
    m = m / max(np.linalg.norm(m), 1e-12)
    elev = float(np.degrees(np.arcsin(np.clip(m[2], -1, 1))))
    azim = float(np.degrees(np.arctan2(m[1], m[0])))
    return (np.clip(elev, -60, 60), azim)


def main():
    for name, path in ASSETS:
        out = f"{OUT_ROOT}/{name}/"
        os.makedirs(out, exist_ok=True)
        pu = run_partuv(path, out)
        F, area, covered = pu["F"], pu["area"], pu["covered"]
        charts = pu["charts"]
        ref = load_reference(path, pu["V"], F, pu["mesh_scale"])
        texA = ref["texA"]
        face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)
        s_curve = surface_samples(pu, face_refuv, valid, texA, 150_000, seed=1)

        cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"], ref["ok_map"])
        sel = covered & ref["ok_map"]
        _, w_l2 = demand_weights(cw, sel, area, BETA, Z_MAX, Q_MIN, Q_MAX)

        print(f"[{name}] R-D 曲线 ...", flush=True)
        curves_raw = chart_rd_curves(pu, face_refuv, valid, texA, s_curve, face2chart)
        curves_hull, diag = hull_curves(curves_raw)
        w_or, _, _ = oracle_allocate(curves_hull, pu, B_REF)

        # zoom chart = 面积加权 luminance-std 最高的 >=100 面 chart(鞋上即 logo)
        cw_n = cw / max(np.median(cw[sel]), 1e-9)
        mean_cw = [float(np.average(cw_n[c["gidx"]], weights=area[c["gidx"]]))
                   for c in charts]
        big = [i for i, c in enumerate(charts) if len(c["F"]) >= 100]
        kk = max(big, key=lambda i: mean_cw[i]) if big else int(np.argmax(mean_cw))

        # reference 逐面颜色(原贴图在面质心处采样; 未匹配面置灰)
        col_ref = np.full((len(F), 3), 0.6)
        col_ref[valid] = bilinear(texA, face_refuv[valid].mean(axis=1))

        weights = {"L1": np.ones(len(F)), "L2_heuristic": w_l2,
                   "RD_hull_global_mse": w_or}
        dash = dict(asset=name, B_ref=B_REF, R_ref=R_REF,
                    hull_diag=diag, zoom_chart=int(kk), methods={})
        texs, uvss, col_recs, err_faces = {}, {}, {}, {}
        pa_l1 = None
        for tag, w in weights.items():
            uvs, _ = layout_with_scales(charts, w)
            uvss[tag] = uvs
            tex, sig, filled = bake_atlas_masks(pu, uvs, R_REF, face_refuv, valid, texA)
            texs[tag] = tex
            _, _, per_chart = rasterize_masks(charts, uvs, R_REF, R_REF)
            # chart 面积倍率(相对 L1): 打包后实测 UV 面积比(含全局缩放差异)
            pa = np.array([tri_area_2d(uv[np.asarray(c["F"])]).sum()
                           for c, uv in zip(charts, uvs)])
            if tag == "L1":
                pa_l1 = pa
            q2 = pa / np.maximum(pa_l1, 1e-12)
            dash["methods"][tag] = dict(
                B_raw=R_REF * R_REF, B_signal=int(sig.sum()),
                B_pad=int(filled.sum()) - int(sig.sum()),
                zoom_interior_texels=int(per_chart[kk]),
                q2_min=float(q2.min()), q2_med=float(np.median(q2)),
                q2_max=float(q2.max()))
            dash["methods"][tag]["q2"] = q2.tolist()
            # 逐面重建颜色(新 atlas 面质心) + 逐面绝对误差
            col = np.full((len(F), 3), 0.6)
            for ci, c in enumerate(charts):
                cF = np.asarray(c["F"])
                cent = uvs[ci][cF].mean(axis=1)
                col[c["gidx"]] = bilinear(tex, cent)
            col_recs[tag] = col
            e = np.abs(col - col_ref).mean(axis=1)
            e[~valid] = 0.0
            err_faces[tag] = e
            # ---- rebake atlas 全图(附录) ----
            fig, ax = plt.subplots(figsize=(5.6, 5.6))
            ax.imshow(tex); ax.set_axis_off()
            ax.set_title(f"{METHOD_LABEL[tag]}  rebake @ {R_REF}^2", fontsize=9)
            plt.tight_layout(); plt.savefig(f"{out}/rebake_{tag}.png", dpi=110,
                                            bbox_inches="tight"); plt.close(fig)

        # ---- layout_compare: 三方法 + 共享 colorbar + zoom chart 圈注 ----
        cmap = plt.cm.coolwarm
        norm = Normalize(vmin=-3, vmax=3)
        fig, axs = plt.subplots(1, 3, figsize=(15.6, 5.6))
        for ax, tag in zip(axs, weights):
            uvs = uvss[tag]
            q2 = np.asarray(dash["methods"][tag]["q2"])
            lg = np.clip(np.log2(np.maximum(q2, 1e-9)), -3, 3)
            for c, uv, v in zip(charts, uvs, lg):
                ax.add_collection(PolyCollection(uv[np.asarray(c["F"])],
                                  facecolors=cmap(norm(v)), edgecolors="none"))
            uvk = uvs[kk]
            ctr = (uvk.min(0) + uvk.max(0)) / 2
            rad = float(np.linalg.norm(uvk.max(0) - uvk.min(0)) / 2) * 1.25 + 0.01
            ax.add_patch(Circle(ctr, rad, fill=False, ec="lime", lw=2.2))
            ax.annotate("zoom chart", ctr + [0, rad + 0.012], color="green",
                        ha="center", fontsize=9, weight="bold")
            ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_axis_off(); ax.set_title(METHOD_LABEL[tag], fontsize=10)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                          fraction=0.025, pad=0.01)
        cb.set_label("log2 (packed chart area / L1)", fontsize=9)
        plt.savefig(f"{out}/layout_compare.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

        # ---- renders_compare: reference + 三方法, 同视角+离屏紧裁 ----
        panels = [("Reference (original texture)", col_ref)] + \
                 [(METHOD_LABEL[t] + f" rebake @ {R_REF}^2", col_recs[t]) for t in weights]
        fig, axs = plt.subplots(1, 4, figsize=(19, 5.6))
        for ax, (ttl, col) in zip(axs, panels):
            ax.imshow(render_img(pu["V"], F, col))
            ax.set_axis_off(); ax.set_title(ttl, fontsize=10)
        plt.tight_layout(); plt.savefig(f"{out}/renders_compare.png", dpi=115,
                                        bbox_inches="tight"); plt.close(fig)

        # ---- error_heatmap: 逐面 |Δ sRGB| (共享色标) ----
        vmax = max(float(np.percentile(err_faces[t][valid], 99)) for t in weights)
        vmax = max(vmax, 1e-6)
        enorm = Normalize(vmin=0, vmax=vmax)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5.4))
        for ax, tag in zip(axs, weights):
            col = plt.cm.inferno(enorm(err_faces[tag]))[:, :3]
            col[~valid] = 0.35
            ax.imshow(render_img(pu["V"], F, col))
            ax.set_axis_off()
            ax.set_title(f"{METHOD_LABEL[tag]}  abs error vs reference", fontsize=10)
        cb = fig.colorbar(ScalarMappable(norm=enorm, cmap="inferno"), ax=axs,
                          fraction=0.02, pad=0.01)
        cb.set_label("per-face mean |dRGB| (sRGB)", fontsize=9)
        plt.savefig(f"{out}/error_heatmap.png", dpi=115, bbox_inches="tight")
        plt.close(fig)

        # ---- zoom_compare(主视觉): 标注光栅化 interior texel 数 ----
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 5.0))
        for ax, tag in zip(axs, weights):
            uv = uvss[tag][kk]
            mn = np.maximum((np.array([uv[:, 0].min(), 1 - uv[:, 1].max()]) * R_REF
                             ).astype(int) - 2, 0)
            mx = np.minimum((np.array([uv[:, 0].max(), 1 - uv[:, 1].min()]) * R_REF
                             ).astype(int) + 2, R_REF)
            crop = texs[tag][mn[1]:mx[1], mn[0]:mx[0]]
            n_int = dash["methods"][tag]["zoom_interior_texels"]
            ax.imshow(crop); ax.set_axis_off()
            ax.set_title(f"{METHOD_LABEL[tag]}\ninterior {n_int:,} texels | "
                         f"crop {crop.shape[1]}x{crop.shape[0]} px "
                         f"(crop size = visual scale only)", fontsize=9)
        plt.tight_layout(); plt.savefig(f"{out}/zoom_compare.png", dpi=110,
                                        bbox_inches="tight"); plt.close(fig)

        # ---- zoom_location: zoom chart 在模型上的位置 ----
        colL = np.full((len(F), 3), 0.72)
        colL[charts[kk]["gidx"]] = [0.85, 0.08, 0.08]
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.imshow(render_img(pu["V"], F, colL,
                             view=facing_view(pu["V"], F, charts[kk]["gidx"])))
        ax.set_axis_off()
        ax.set_title(f"zoom chart #{kk} location (red)", fontsize=10)
        plt.tight_layout(); plt.savefig(f"{out}/zoom_location.png", dpi=115,
                                        bbox_inches="tight"); plt.close(fig)

        # ---- 原始 vs hull 后 R-D 点: 取凸包修正最大的 3 个 chart ----
        def corr_size(i):
            raw, hl = curves_raw[i], curves_hull[i]
            if not raw["P"] or len(raw["P"]) == len(hl["P"]):
                em = np.minimum.accumulate(raw["E"])
                return float(np.abs(np.array(raw["E"]) - em).sum())
            return 1e9 + len(raw["P"]) - len(hl["P"])
        cand = [i for i in range(len(curves_raw)) if curves_raw[i]["P"]]
        top3 = sorted(cand, key=corr_size, reverse=True)[:3]
        rd_pts = [dict(chart=int(i),
                       raw=dict(P=curves_raw[i]["P"], E=curves_raw[i]["E"]),
                       hull=dict(P=curves_hull[i]["P"], E=curves_hull[i]["E"]))
                  for i in top3]
        for tag in weights:                       # q2 明细不入 json(体积)
            dash["methods"][tag].pop("q2")
        with open(f"{out}/rd_points.json", "w") as fp:
            json.dump(rd_pts, fp, indent=1)
        with open(f"{out}/dashboard_data.json", "w") as fp:
            json.dump(dash, fp, indent=1)
        print(f"[{name}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
