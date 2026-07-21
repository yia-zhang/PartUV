# -*- coding: utf-8 -*-
"""Basecolor-only canonicalizer: 多 geometry/多贴图 GLB -> 单 mesh + 单 RGB atlas.

步骤: 逐 geometry 将 baseColorFactor 烘入 RGB(无贴图者生成纯色块) ->
所有源图按 shelf 布局原样(texel 级拷贝, 不重采样)放进 canonical atlas ->
UV 先按 REPEAT 语义取小数部分再线性映射到各自 cell -> 合并单一 mesh。
multi-geometry 永远不是拒绝原因; 无任何可用 RGB+UV 的 geometry 记 warning
并剔除, 全部不可用才拒绝。face_source 记录每面的来源 geometry。"""
import numpy as np
from PIL import Image

from .loader import load_glb

SOLID_RES = 8            # 无贴图 geometry 的纯色块尺寸
ADAPTER_VERSION = "canonicalizer_rgb_v1"


def _bake_factor(img, factor):
    a = np.asarray(img, float) / 255.0
    return np.clip(a * np.asarray(factor)[None, None, :], 0, 1)


def canonicalize(path):
    """返回 dict(V,F,uv,atlas(H,W,3 float),face_source,warnings,adapter_version).
    抛 ValueError('NO_USABLE_RGB_UV') 当无任何可用 geometry。"""
    geoms = load_glb(path)
    usable, warnings = [], []
    for g in geoms:
        if g["uv"] is None or len(np.atleast_1d(g["uv"])) == 0:
            warnings.append(f"{g['name']}: 无 UV, 剔除")
            continue
        if g["image"] is None:
            g = dict(g, image=Image.fromarray(
                np.full((SOLID_RES, SOLID_RES, 3), 255, np.uint8)))
            warnings.append(f"{g['name']}: 无贴图, 以 baseColorFactor 纯色块代替")
        usable.append(g)
    if not usable:
        raise ValueError("NO_USABLE_RGB_UV")

    # 相同贴图对象共享 cell(多 geometry 共用一张图的常见情形)
    imgs, img_of = [], []
    for g in usable:
        baked = _bake_factor(g["image"], g["factor"])
        key = None
        for k, (arr, _) in enumerate(imgs):
            if arr.shape == baked.shape and np.array_equal(arr, baked):
                key = k
                break
        if key is None:
            imgs.append((baked, g["name"]))
            key = len(imgs) - 1
        img_of.append(key)

    # shelf 布局(texel 原样拷贝): 行内横排, 行高=本行最高图
    PAD = 2
    W_cap = max(max(a.shape[1] for a, _ in imgs) + 2 * PAD, 1024)
    cells, x, y, row_h = [], PAD, PAD, 0
    for a, _ in imgs:
        h, w = a.shape[:2]
        if x + w + PAD > W_cap:
            x, y = PAD, y + row_h + PAD
            row_h = 0
        cells.append((x, y, w, h))
        x += w + PAD
        row_h = max(row_h, h)
    H_tot, W_tot = y + row_h + PAD, W_cap
    atlas = np.zeros((H_tot, W_tot, 3))
    for (a, _), (cx, cy, w, h) in zip(imgs, cells):
        atlas[cy:cy + h, cx:cx + w] = a

    # 合并 mesh + UV 重映射(REPEAT 取小数, 跨 tile 面记 warning)
    Vs, Fs, UVs, src = [], [], [], []
    off = 0
    for gi, g in enumerate(usable):
        uv = np.asarray(g["uv"], float).copy()
        span = uv[g["F"]].max(1) - uv[g["F"]].min(1)
        w = uv - np.floor(uv)                        # REPEAT 语义
        w[(w == 0) & (uv != 0)] = 1.0                # 上边界 1.0 不折返到 0
        uv = w
        if (span > 0.5).any():
            warnings.append(f"{g['name']}: {(span > 0.5).any(1).sum()} 面跨 UV tile, "
                            f"wrap 后可能失真(V1 边界)")
        cx, cy, w, h = cells[img_of[gi]]
        u = (cx + uv[:, 0] * w) / W_tot
        v = 1 - (cy + (1 - uv[:, 1]) * h) / H_tot    # 顶行 v=1 约定
        Vs.append(g["V"])
        UVs.append(np.stack([u, v], 1))
        Fs.append(g["F"] + off)
        src.append(np.full(len(g["F"]), gi))
        off += len(g["V"])
    return dict(V=np.concatenate(Vs), F=np.concatenate(Fs),
                uv=np.concatenate(UVs), atlas=atlas,
                face_source=np.concatenate(src),
                geometry_names=[g["name"] for g in usable],
                warnings=warnings, adapter_version=ADAPTER_VERSION)


def export_canonical_glb(canon, out_path):
    """canonical mesh + atlas 导出 GLB(供 baseline chart generator 消费)."""
    import trimesh
    img = Image.fromarray((np.clip(canon["atlas"], 0, 1) * 255).astype(np.uint8))
    mesh = trimesh.Trimesh(vertices=canon["V"], faces=canon["F"], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=canon["uv"], image=img)
    mesh.export(out_path)
    return out_path
