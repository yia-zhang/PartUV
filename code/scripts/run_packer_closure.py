# -*- coding: utf-8 -*-
"""Packer–Budget Closure V1 验证: shelf vs xatlas 生产合同 + 预算平价 + 浪费分解.
资产: shoe / wheel / Corset。相同 charts、相同 TD 目标尺度、相同 padding 语义。
输出: notebook/outputs/closure/{summary.json, <asset>/renders.png}
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

from tdlib.api import map_partuv_td
from tdlib.budget import rasterize_masks
from tdlib.layout import chart_scales, layout_with_scales, xatlas_pack
from tdlib.pipeline import load_reference, run_partuv
from tdlib.rd import (bake_atlas_masks, bilinear, eval_surface_error,
                      prepare_face_ref_uv, surface_samples)
from tdlib.signal import demand_weights, luminance_std_heuristic

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/closure"
CACHE = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1b"

ASSETS = [
    ("shoe", f"{DATA}/objaverse_22b822c6520d4d49.glb", 2048),
    ("wheel", f"{DATA}/objaverse_92ff65712c62408d.glb", 4096),
    ("corset", f"{DATA}/sample_Corset.glb", 8192),
]
fails, summary = [], {}


def tri_area(uv, F):
    t = np.asarray(uv)[np.asarray(F)]
    return float(np.abs(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum() / 2)


def get_pu(tag, path):
    c = {"shoe": f"{CACHE}/shoe_22b822/charts_cache.pkl",
         "wheel": f"{CACHE}/wheel_92ff6/charts_cache.pkl",
         "corset": f"{OUT}/corset_cache.pkl"}[tag]
    if os.path.exists(c):
        return pickle.load(open(c, "rb"))
    pu = run_partuv(path, f"{OUT}/{tag}_pu/")
    pickle.dump(pu, open(c, "wb"))
    return pu


def metrics(charts, uvs, R, D):
    ch = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    owner, overlap, per = rasterize_masks(ch, uvs, R, R)
    fill = float((owner >= 0).mean())
    N = per.astype(float)
    e_alloc = float(0.5 * np.abs(N / max(N.sum(), 1) - D / D.sum()).sum())
    allv = np.concatenate(list(uvs))
    oob = bool(allv.min() < -1e-6 or allv.max() > 1 + 1e-6
               or not np.isfinite(allv).all())
    ext = max(allv[:, 0].max() - allv[:, 0].min(),
              allv[:, 1].max() - allv[:, 1].min())
    return dict(fill=round(fill, 4), overlap=int(overlap), oob=oob,
                e_alloc=round(e_alloc, 5), min_bound_side_frac=round(float(ext), 4))


def parity_layout(charts, w, B_target, ch_masks, packer):
    """在**同一份缓存 charts** 上按平价规则选 R 并给出布局(外观公平对比用).
    shelf 布局与 R 无关(只调一次); xatlas 布局依赖 R(逐迭代重打包)。"""
    uvs_shelf = (layout_with_scales(charts, w, packer="shelf")[0]
                 if packer == "shelf" else None)

    def lay(R):
        if packer == "shelf":
            return uvs_shelf
        return layout_with_scales(charts, w, packer="xatlas", resolution=R)[0]

    uvs = lay(1024)
    owner, _, _ = rasterize_masks(ch_masks, uvs, 1024, 1024)
    fill = max(float((owner >= 0).mean()), 1e-6)
    R = int(np.ceil(np.sqrt(B_target * 1.02 / fill) / 16) * 16)
    for _ in range(5):
        uvs = lay(R)
        owner, _, _ = rasterize_masks(ch_masks, uvs, R, R)
        ratio = int((owner >= 0).sum()) / B_target
        if 1.0 <= ratio <= 1.05:
            break
        R = int(np.ceil(R * np.sqrt(1.02 / ratio) / 16) * 16)
    return R, uvs, ratio


os.makedirs(OUT, exist_ok=True)
for tag, path, R_ab in ASSETS:
    print(f"\n================ {tag} ================", flush=True)
    pu = get_pu(tag, path)
    charts, F, area, covered = pu["charts"], pu["F"], pu["area"], pu["covered"]
    ref = load_reference(path, pu["V"], F, pu["mesh_scale"])
    texA = ref["texA"]
    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)
    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"],
                                 ref["ok_map"])
    sel = covered & ref["ok_map"]
    _, w_td = demand_weights(cw, sel, area, beta=0.75)
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    rows = {}

    # ---- A/B(同 R, 同 charts, 同 TD 尺度) ----
    for lname, w in [("uniform", np.ones(len(F))), ("TD", w_td)]:
        scales = chart_scales(charts, w)
        D = np.array([(f ** 2) * c["a2"] for c, f in zip(charts, scales)])
        t0 = time.time()
        uvs_s, _ = layout_with_scales(charts, w, packer="shelf")
        t_s = time.time() - t0
        m_s = metrics(charts, uvs_s, R_ab, D)
        t0 = time.time()
        uvs_x, _ = layout_with_scales(charts, w, packer="xatlas", resolution=R_ab)
        t_x = time.time() - t0
        uvs_x2, _ = layout_with_scales(charts, w, packer="xatlas", resolution=R_ab)
        det = all(np.array_equal(a, b) for a, b in zip(uvs_x, uvs_x2))
        m_x = metrics(charts, uvs_x, R_ab, D)
        rows[lname] = dict(shelf=m_s, xatlas=m_x, t_shelf=round(t_s, 2),
                           t_xatlas=round(t_x, 2), deterministic=det)
        print(f"  [{lname}] shelf fill={m_s['fill']*100:.1f}% -> xatlas "
              f"{m_x['fill']*100:.1f}% (+{(m_x['fill']-m_s['fill'])*100:.1f}pp) | "
              f"E_alloc {m_s['e_alloc']*100:.2f}% -> {m_x['e_alloc']*100:.2f}% | "
              f"overlap={m_x['overlap']} oob={m_x['oob']} det={det} "
              f"t={t_s:.2f}s/{t_x:.2f}s", flush=True)
        if m_x["fill"] < m_s["fill"] or m_x["e_alloc"] > 0.01 or \
                m_x["overlap"] or m_x["oob"] or not det:
            fails.append(f"{tag}:{lname}:contract")

    # ---- 浪费来源分解 ----
    scales = chart_scales(charts, w_td)
    D = np.array([(f ** 2) * c["a2"] for c, f in zip(charts, scales)])
    uvs_nr = xatlas_pack(charts, scales, resolution=R_ab, rotate=False)
    fill_nr = metrics(charts, uvs_nr, R_ab, D)["fill"]
    uvs_p0 = xatlas_pack(charts, scales, resolution=R_ab, padding_px=0)
    fill_p0 = metrics(charts, uvs_p0, R_ab, D)["fill"]
    # shelf 侧解析分解(bbox 凹形/行级浪费/padding)
    uvs_s, _ = layout_with_scales(charts, w_td, packer="shelf")
    S_poly = sum(tri_area(uv, c["F"]) for c, uv in zip(charts, uvs_s))
    S_bbox = sum(float(np.prod(uv.max(0) - uv.min(0))) for uv in uvs_s)
    fill_s = rows["TD"]["shelf"]["fill"]
    # 环形/框形内嵌套证据: xatlas 输出中 chart bbox 完全落入另一 chart bbox 的对数
    bb = np.array([[uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]
                   for uv in uvs_x])          # TD 布局的 xatlas 输出(上轮循环末值)
    nest = 0
    order = np.argsort([-(b[2]-b[0])*(b[3]-b[1]) for b in bb])
    for i in order[:30]:
        for j in range(len(bb)):
            if i == j:
                continue
            if (bb[j][0] >= bb[i][0] and bb[j][1] >= bb[i][1]
                    and bb[j][2] <= bb[i][2] and bb[j][3] <= bb[i][3]):
                nest += 1
    waste = dict(
        rotation_gain_pp=round((rows["TD"]["xatlas"]["fill"] - fill_nr) * 100, 2),
        padding_cost_pp=round((fill_p0 - rows["TD"]["xatlas"]["fill"]) * 100, 2),
        shelf_bbox_concavity_loss=round(1 - S_poly / max(S_bbox, 1e-12), 4),
        shelf_row_waste=round(1 - S_bbox / max(S_poly / max(fill_s, 1e-6), 1e-12), 4),
        xatlas_nested_bbox_pairs=int(nest))
    print(f"  [waste] 旋转贡献={waste['rotation_gain_pp']}pp  "
          f"padding开销={waste['padding_cost_pp']}pp  "
          f"shelf bbox凹形损失={waste['shelf_bbox_concavity_loss']*100:.1f}%  "
          f"shelf 行级浪费={waste['shelf_row_waste']*100:.1f}%  "
          f"xatlas 嵌套bbox对={nest}", flush=True)

    # ---- 生产 e2e(auto 平价, xatlas) + shelf 平价外观对比 ----
    res = map_partuv_td(path, f"{OUT}/{tag}/")
    b = res["budget"]
    print(f"  [auto] R={b['selected_atlas_size']} ratio={b['budget_ratio']} "
          f"fill={b['output_packing_fill']*100:.0f}% E_alloc={b['E_alloc']*100:.2f}% "
          f"met={b['budget_met']} reload={res['integrity']['reload_ok']}", flush=True)
    if not (1.0 <= b["budget_ratio"] <= 1.06 and b["budget_met"]
            and res["integrity"]["reload_ok"]):
        fails.append(f"{tag}:auto_parity")
    # 注意: 外观对比必须在**同一份缓存 charts** 上做(map 内部重跑 PartUV 有
    # 跨运行 chart 漂移, face2chart/samples 不可混用) —— 两 packer 都在缓存
    # pu 上按平价规则重建布局与烘焙, 共用同一评价采样集。
    B_target = b["B_target"]
    R_sh, uvs_sh, ratio_sh = parity_layout(charts, w_td, B_target, ch_masks, "shelf")
    R_xa, uvs_xa, ratio_xa = parity_layout(charts, w_td, B_target, ch_masks, "xatlas")
    tex_sh, sig_sh, _ = bake_atlas_masks(pu, uvs_sh, R_sh, face_refuv, valid, texA)
    tex_xa, sig_xa, _ = bake_atlas_masks(pu, uvs_xa, R_xa, face_refuv, valid, texA)
    s_eval = surface_samples(pu, face_refuv, valid, texA, 150_000, seed=2)
    mse_sh = eval_surface_error(tex_sh, pu, uvs_sh, s_eval, face2chart)
    mse_x = eval_surface_error(tex_xa, pu, uvs_xa, s_eval, face2chart)
    print(f"  [外观@平价预算] shelf(R={R_sh}, sig={int(sig_sh.sum())/1e6:.2f}M, "
          f"ratio={ratio_sh:.3f}) MSE={mse_sh:.6f}  vs  xatlas(R={R_xa}, "
          f"sig={int(sig_xa.sum())/1e6:.2f}M, ratio={ratio_xa:.3f}) MSE={mse_x:.6f} "
          f"(xatlas/shelf={mse_x/mse_sh:.3f})", flush=True)
    if mse_x > mse_sh * 1.10:
        fails.append(f"{tag}:appearance")

    # ---- 同视角渲染: source / shelf / xatlas ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gen_dashboard_assets import render_img
    V = pu["V"]
    col_src = np.full((len(F), 3), 0.6)
    col_src[valid] = bilinear(texA, face_refuv[valid].mean(axis=1))
    def face_colors(tex, uvs):
        col = np.full((len(F), 3), 0.6)
        for ci, c in enumerate(charts):
            col[c["gidx"]] = bilinear(tex, uvs[ci][np.asarray(c["F"])].mean(axis=1))
        return col
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 5))
    for ax, im, t in zip(axs,
                         [render_img(V, F, col_src),
                          render_img(V, F, face_colors(tex_sh, uvs_sh)),
                          render_img(V, F, face_colors(tex_xa, uvs_xa))],
                         ["source", f"shelf parity R={R_sh}",
                          f"xatlas parity R={R_xa}"]):
        ax.imshow(im); ax.set_axis_off(); ax.set_title(t, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{tag}/renders.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    summary[tag] = dict(ab=rows, waste=waste, budget=b,
                        appearance=dict(R_shelf=R_sh, mse_shelf=mse_sh,
                                        mse_xatlas=mse_x,
                                        ratio=round(mse_x / mse_sh, 4)),
                        reload_ok=res["integrity"]["reload_ok"])

# ---- 合同门: 中位 fill 提升 >= 5pp ----
gains = [summary[t]["ab"]["TD"]["xatlas"]["fill"]
         - summary[t]["ab"]["TD"]["shelf"]["fill"] for t, _, _ in ASSETS]
med_gain = float(np.median(gains)) * 100
print(f"\n中位 fill 提升(TD 布局) = {med_gain:.1f}pp (要求 >= 5pp)")
if med_gain < 5:
    fails.append("median_gain<5pp")

with open(f"{OUT}/summary.json", "w") as fp:
    json.dump(dict(summary=summary, fails=fails, median_gain_pp=med_gain),
              fp, indent=1, ensure_ascii=False, default=float)
print("\nCLOSURE:", "ALL PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
