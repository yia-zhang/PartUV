# -*- coding: utf-8 -*-
"""快速预检(PartUV/PartField 之前, 纯 trimesh+PIL, 秒级).

只接受当前支持范围: 单一可用 basecolor 的 GLB。多材质/多 atlas/UDIM/无 UV/
无贴图/超大 mesh 一律快速拒绝并给出原因, 不尝试修复。"""
import numpy as np
import trimesh

MAX_FACES = 300_000          # 明显超大 mesh 前置拒绝(calibration 实测超时线)
MIN_FACES = 8


def quick_preflight(glb_path, max_faces=MAX_FACES):
    """返回 dict(ok, reason, n_faces, n_verts, n_geoms, tex_size, tex_mode)."""
    out = dict(ok=False, reason="", n_faces=0, n_verts=0, n_geoms=0,
               tex_size=None, tex_mode=None)
    try:
        scene = trimesh.load(glb_path, process=False)
    except Exception as e:
        out["reason"] = f"UNPARSABLE: {type(e).__name__}: {str(e)[:80]}"
        return out
    geoms = (list(scene.geometry.values())
             if isinstance(scene, trimesh.Scene) else [scene])
    out["n_geoms"] = len(geoms)
    if len(geoms) == 0:
        out["reason"] = "EMPTY_MESH"
        return out
    if len(geoms) > 1:
        out["reason"] = f"MULTI_MATERIAL: {len(geoms)} geometries(暂不支持)"
        return out
    m = geoms[0]
    if not hasattr(m, "faces") or len(m.faces) == 0:
        out["reason"] = "EMPTY_MESH"
        return out
    out["n_faces"], out["n_verts"] = int(len(m.faces)), int(len(m.vertices))
    if out["n_faces"] < MIN_FACES:
        out["reason"] = f"TOO_FEW_FACES: {out['n_faces']}"
        return out
    if out["n_faces"] > max_faces:
        out["reason"] = f"MESH_TOO_LARGE: {out['n_faces']} faces > {max_faces}"
        return out
    uv = getattr(m.visual, "uv", None)
    if uv is None or len(np.atleast_1d(uv)) == 0:
        out["reason"] = "NO_UV"
        return out
    img = None
    try:
        mat = m.visual.material
        img = (getattr(mat, "baseColorTexture", None)
               or getattr(mat, "image", None))
    except Exception:
        pass
    if img is None:
        out["reason"] = "NO_BASECOLOR"
        return out
    out["tex_size"] = tuple(img.size)
    out["tex_mode"] = img.mode
    if min(img.size) < 32:
        out["reason"] = f"TEXTURE_TOO_SMALL: {img.size}"
        return out
    out["ok"] = True
    return out
