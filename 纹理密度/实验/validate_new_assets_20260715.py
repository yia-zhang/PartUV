# -*- coding: utf-8 -*-
"""验证 2026-07-15 新增资产(polyhaven_* 15 个 + Khronos sample_* 5 个):
check_asset_support + measure_source_budget + build_catalog 同款
orig TD CVw / content contrast。输出 markdown 表追加进 CATALOG.md。

运行: cd code && /root/miniconda3/envs/tdf_render/bin/python \
    ../纹理密度/实验/validate_new_assets_20260715.py
"""
import glob
import os
import sys

sys.path.insert(0, "/root/youjiaZhang/PartUV/code")

from tdlib.gpu import pick_free_gpu
pick_free_gpu()  # GPU0-6 常年被占, 必须在 torch 初始化前选卡

import numpy as np
import trimesh

from tdlib.api import check_asset_support, measure_source_budget

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "validate_new_assets_20260715_result.md")
NEW = sorted(glob.glob(f"{DATA}/polyhaven_*.glb")) + [
    f"{DATA}/sample_{n}.glb" for n in
    ["AntiqueCamera", "ToyCar", "FlightHelmet", "SciFiHelmet",
     "ABeautifulGame"]]


def tri_a2(uv):
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def tri_a3(v):
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def catalog_metrics(m):
    """orig TD CVw + content contrast, 与 build_catalog.py 完全一致."""
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    uv = getattr(m.visual, "uv", None)
    tex = None
    try:
        mat = m.visual.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None:
            tex = np.asarray(img.convert("RGB"), float) / 255.0
    except Exception:
        pass
    a3 = tri_a3(V[F]); ok3 = a3 > 1e-14
    cvw_td = contrast = None
    if uv is not None:
        uv = np.asarray(uv, float)
        a2 = tri_a2(uv[F])
        sel = ok3 & (a2 > 1e-16)
        td = np.sqrt(a2[sel] / a3[sel]); wgt = a3[sel]
        mu = np.average(td, weights=wgt)
        cvw_td = float(np.sqrt(np.average((td - mu) ** 2, weights=wgt)) / mu)
    if tex is not None and uv is not None:
        rng = np.random.RandomState(0)
        bar = rng.dirichlet((1.2, 1.2, 1.2), 8)
        lum = tex @ np.array([0.299, 0.587, 0.114])
        UV3 = uv[F]
        acc = acc2 = None
        for s in range(len(bar)):
            pt = np.einsum("k,fkd->fd", bar[s], UV3)
            x = np.clip(pt[:, 0], 0, 1) * (lum.shape[1] - 1)
            y = np.clip(1 - pt[:, 1], 0, 1) * (lum.shape[0] - 1)
            v = lum[y.astype(int), x.astype(int)]
            acc = v.copy() if acc is None else acc + v
            acc2 = v ** 2 if acc2 is None else acc2 + v ** 2
        cw = np.sqrt(np.maximum(acc2 / 8 - (acc / 8) ** 2, 0))[ok3]
        p90 = float(np.quantile(cw, 0.9))
        med = max(float(np.median(cw)), 1e-3 * max(p90, 1e-9))
        contrast = min(p90 / max(med, 1e-9), 99.0)
    tex_str = f"{tex.shape[1]}x{tex.shape[0]}" if tex is not None else "-"
    return cvw_td, contrast, tex_str


rows, fails = [], []
for p in NEW:
    name = os.path.basename(p)
    try:
        s = check_asset_support(p)
        if not s["supported"]:
            rows.append(dict(name=name, status=f"UNSUPPORTED: {s['reason'][:60]}"))
            fails.append(name)
            continue
        m = trimesh.load(p, force="mesh")
        b = measure_source_budget(m)
        cvw, contrast, tex_str = catalog_metrics(m)
        rows.append(dict(
            name=name, status="OK", faces=len(m.faces),
            tex=tex_str, islands=s.get("n_uv_islands", -1),
            b_surface=b["source_B_surface"], reuse=b["source_reuse_factor"],
            cvw=cvw, contrast=contrast))
    except Exception as e:
        rows.append(dict(name=name, status=f"CRASH: {type(e).__name__}: {str(e)[:60]}"))
        fails.append(name)
    print(f"progress: {name} done", flush=True)

lines = ["| 资产 | 面数 | 贴图 | UV岛 | B_surface | reuse | orig TD CVw | content contrast |",
         "|---|---|---|---|---|---|---|---|"]
for r in rows:
    if r["status"] != "OK":
        lines.append(f"| {r['name']} | — | — | — | — | — | — | {r['status']} |")
        continue
    lines.append(
        f"| {r['name']} | {r['faces']:,} | {r['tex']} | {r['islands']} "
        f"| {r['b_surface']/1e6:.2f}M | {r['reuse']:.2f}× "
        f"| {r['cvw']:.3f} | {r['contrast']:.1f} |")
open(OUT, "w").write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\n{len(rows)-len(fails)}/{len(rows)} OK; fails: {fails if fails else 'none'}")
print(f"result written to {OUT}")
