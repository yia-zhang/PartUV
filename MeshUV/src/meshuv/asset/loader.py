# -*- coding: utf-8 -*-
"""GLB 加载: 遍历 scene mesh nodes, 应用 node transform, 收集 RGB basecolor.

输出 geometry 记录列表(基本事实, 不做合并): 每条含世界坐标 V/F/UV/贴图/
baseColorFactor/来源名。仅 RGB basecolor; 其他 PBR 通道忽略(V1 边界)。"""
import numpy as np
import trimesh
from PIL import Image


def load_glb(path):
    """返回 list[dict(V,F,uv,image,factor,name)]; 不可解析/空 scene 抛 ValueError."""
    scene = trimesh.load(path, process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) == 0:
        raise ValueError("EMPTY_SCENE")
    out = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        V = np.asarray(g.vertices, float)
        V = V @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]   # node transform
        uv = getattr(g.visual, "uv", None)
        img, factor = None, np.ones(3)
        mat = getattr(g.visual, "material", None)
        if mat is not None:
            img = getattr(mat, "baseColorTexture", None) \
                or getattr(mat, "image", None)
            f = getattr(mat, "baseColorFactor", None)
            if f is not None:
                factor = np.asarray(f, float)[:3]
                if factor.max() > 1.001:                          # 0-255 编码
                    factor = factor / 255.0
        out.append(dict(V=V, F=np.asarray(g.faces),
                        uv=None if uv is None else np.asarray(uv, float),
                        image=None if img is None else img.convert("RGB"),
                        factor=factor, name=f"{node}/{gname}"))
    if not out:
        raise ValueError("EMPTY_SCENE")
    return out
