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
ADAPTER_VERSION = "canonicalizer_rgb_v2"


def _srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin2srgb(x):
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.clip(x, 0, 1) ** (1 / 2.4) - 0.055)


def _bake_factor(img, factor):
    """glTF 语义: sRGB 贴图 -> linear, 乘 linear factor, 转回 sRGB."""
    a = _srgb2lin(np.asarray(img, float) / 255.0)
    return _lin2srgb(np.clip(a * np.asarray(factor)[None, None, :], 0, 1))


def canonicalize(path):
    """返回 dict(V,F,uv,atlas(H,W,3 float),face_source,warnings,adapter_version).
    抛 ValueError('NO_USABLE_RGB_UV') 当无任何可用 geometry。"""
    geoms = load_glb(path)
    usable, warnings = [], []
    orig_area = orig_faces = 0.0
    for g in geoms:
        t = g["V"][g["F"]]
        orig_area += float(np.linalg.norm(np.cross(
            t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)
        orig_faces += len(g["F"])
        no_uv = g["uv"] is None or len(np.atleast_1d(g["uv"])) == 0
        if no_uv and g["image"] is not None:
            # 有纹理但无 UV: V1 明确拒绝(不得静默删除)
            raise ValueError("TEXTURED_NO_UV_UNSUPPORTED")
        if no_uv:
            # 纯色材质无 UV: 保留 —— 常量 UV 指向纯色 cell 中心(合法常量采样)
            g = dict(g, uv=np.full((len(g["V"]), 2), 0.5),
                     image=Image.fromarray(
                         np.full((SOLID_RES, SOLID_RES, 3), 255, np.uint8)))
            warnings.append(f"{g['name']}: 纯色材质无 UV, 常量采样映射保留")
        elif g["image"] is None:
            g = dict(g, image=Image.fromarray(
                np.full((SOLID_RES, SOLID_RES, 3), 255, np.uint8)))
            warnings.append(f"{g['name']}: 无贴图, baseColorFactor 纯色块")
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
        # gutter: 边缘 texel 复制进 padding 环(等价 clamp 语义, 消除边界渗黑)
        atlas[cy - 1, cx:cx + w] = a[0]
        atlas[cy + h, cx:cx + w] = a[-1]
        atlas[cy:cy + h, cx - 1] = a[:, 0]
        atlas[cy:cy + h, cx + w] = a[:, -1]
        atlas[cy - 1, cx - 1] = a[0, 0]
        atlas[cy - 1, cx + w] = a[0, -1]
        atlas[cy + h, cx - 1] = a[-1, 0]
        atlas[cy + h, cx + w] = a[-1, -1]

    # 合并 mesh + UV 重映射: 仅允许 [0,1] 或整套统一整数 tile 平移;
    # 单面跨 tile => TILED_UV_UNSUPPORTED(拒绝, 不产生错误 RGB)
    Vs, Fs, UVs, src = [], [], [], []
    off = 0
    for gi, g in enumerate(usable):
        uv = np.asarray(g["uv"], float).copy()
        used = np.unique(g["F"])
        if len(used):
            shift = np.floor(uv[used].min(0) + 1e-9)   # 整套统一整数平移
            uv = uv - shift
        crosses = (np.floor(uv[g["F"]].max(1) - 1e-9)
                   != np.floor(uv[g["F"]].min(1) + 1e-9)).any(1)
        if crosses.any():
            raise ValueError(
                f"TILED_UV_UNSUPPORTED: {int(crosses.sum())} 面跨 UV tile")
        w = uv - np.floor(uv)
        w[(w == 0) & (uv != 0)] = 1.0
        uv = np.clip(w, 0, 1)
        cx, cy, w, h = cells[img_of[gi]]
        u = (cx + uv[:, 0] * w) / W_tot
        v = 1 - (cy + (1 - uv[:, 1]) * h) / H_tot    # 顶行 v=1 约定
        Vs.append(g["V"])
        UVs.append(np.stack([u, v], 1))
        Fs.append(g["F"] + off)
        src.append(np.full(len(g["F"]), gi))
        off += len(g["V"])
    V, F = np.concatenate(Vs), np.concatenate(Fs)
    t = V[F]
    ret_area = float(np.linalg.norm(np.cross(
        t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)
    return dict(V=V, F=F, uv=np.concatenate(UVs), atlas=atlas,
                face_source=np.concatenate(src),
                geometry_names=[g["name"] for g in usable],
                original=dict(n_geometries=len(geoms),
                              n_faces=int(orig_faces),
                              surface_area=orig_area),
                retained=dict(n_geometries=len(usable),
                              n_faces=int(len(F)), surface_area=ret_area),
                retained_area_ratio=ret_area / max(orig_area, 1e-20),
                warnings=warnings, adapter_version=ADAPTER_VERSION)


def export_canonical_glb(canon, out_path):
    """canonical mesh + atlas 导出 GLB(供 baseline chart generator 消费)."""
    import trimesh
    img = Image.fromarray((np.clip(canon["atlas"], 0, 1) * 255).astype(np.uint8))
    mesh = trimesh.Trimesh(vertices=canon["V"], faces=canon["F"], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=canon["uv"], image=img)
    mesh.export(out_path)
    return out_path
