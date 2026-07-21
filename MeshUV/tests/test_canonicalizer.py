# -*- coding: utf-8 -*-
"""Canonicalizer 单元测试(合成 GLB): 多 geometry 共图/异图、node transform、
baseColorFactor、RGB 一致性(<=1/255)、面数/面积/bbox 保持。"""
import os
import sys
import tempfile

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from meshuv.asset.canonicalizer import canonicalize  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def _quad(img, factor=None, shift=(0, 0, 0)):
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float) + shift
    F = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img, baseColorFactor=factor)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, image=img, material=mat)
    return m


def _bilin(img, uv):
    """texel-center 采样(与 tdlib 同约定)."""
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0], 0, 1) * W - 0.5
    y = np.clip(1 - uv[:, 1], 0, 1) * H - 0.5
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    xi = lambda a: np.clip(a, 0, W - 1)
    yi = lambda a: np.clip(a, 0, H - 1)
    return (img[yi(y0), xi(x0)] * (1 - fx) * (1 - fy)
            + img[yi(y0), xi(x0 + 1)] * fx * (1 - fy)
            + img[yi(y0 + 1), xi(x0)] * (1 - fx) * fy
            + img[yi(y0 + 1), xi(x0 + 1)] * fx * fy)


rng = np.random.RandomState(0)
imgA = Image.fromarray(rng.randint(0, 255, (64, 64, 3), np.uint8))
imgB = Image.fromarray(rng.randint(0, 255, (32, 48, 3), np.uint8))

with tempfile.TemporaryDirectory() as td:
    # 1) 两 geometry 共用一张 RGB
    s = trimesh.Scene()
    s.add_geometry(_quad(imgA))
    s.add_geometry(_quad(imgA, shift=(2, 0, 0)))
    p = f"{td}/shared.glb"
    s.export(p)
    c = canonicalize(p)
    check("共图: 面数=4, 单 mesh", len(c["F"]) == 4 and c["face_source"].max() == 1)
    check("共图: atlas 仅一份贴图(cell 复用)",
          c["atlas"].shape[0] <= 64 + 8, f"atlas={c['atlas'].shape}")

    # 2) 两 geometry 不同 RGB + 3) node transform + 4) baseColorFactor
    s = trimesh.Scene()
    s.add_geometry(_quad(imgA))
    T = np.eye(4); T[:3, 3] = [3, 1, 0]
    s.add_geometry(_quad(imgB, factor=[0.5, 1.0, 0.25, 1.0]), transform=T)
    p = f"{td}/multi.glb"
    s.export(p)
    c = canonicalize(p)
    check("异图: 两 cell 都进 atlas", len(c["geometry_names"]) == 2)
    check("node transform: 世界坐标正确",
          abs(c["V"][:, 0].max() - 4.0) < 1e-9 and abs(c["V"][:, 1].max() - 2.0) < 1e-9)
    # 5) RGB 一致性: glTF 语义 lin(tex)*factor -> sRGB
    giB = next(i for i, n in enumerate(c["geometry_names"]) if "geometry_1" in n)
    fac = np.array([128, 255, 64]) / 255.0

    def s2l(x):
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def l2s(x):
        return np.where(x <= 0.0031308, x * 12.92,
                        1.055 * np.clip(x, 0, 1) ** (1 / 2.4) - 0.055)

    srcB = l2s(s2l(np.asarray(imgB, float) / 255.0) * fac)  # factor 本身是 linear
    pts = rng.rand(500, 2)
    faceB = c["face_source"] == giB
    uvB_new = c["uv"][np.unique(c["F"][faceB])]      # 该 quad 全部角点
    # 在 quad 参数域随机点 -> 原 uv=pts, 新 uv=cell 内双线性插值
    u0, v0 = uvB_new.min(0); u1, v1 = uvB_new.max(0)
    new_uv = np.stack([u0 + pts[:, 0] * (u1 - u0), v0 + pts[:, 1] * (v1 - v0)], 1)
    old = _bilin(srcB, pts)
    new = _bilin(c["atlas"], new_uv)
    err = np.abs(old - new).max()
    check("RGB 采样一致(<=1/255 + 插值边界)", err <= 1.5 / 255, f"maxerr={err:.4f}")
    # 6) 面积/bbox 保持
    def area(V, F):
        t = V[F]
        return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0],
                                             t[:, 2] - t[:, 0]), axis=1).sum() / 2)
    check("表面积保持", abs(area(c["V"], c["F"]) - 2 * 1.0) < 1e-9)
    check("bbox 正确", np.allclose(c["V"].min(0), [0, 0, 0])
          and np.allclose(c["V"].max(0), [4, 2, 0]))
    check("multi-geometry 非拒绝原因", not any("拒" in w for w in c["warnings"]))

    # glTF factor 解析值: tex=0.5(sRGB), factor=0.5 -> lin 0.2140*0.2140? no:
    # lin(0.5)=0.2140, factor 已是 linear? glTF factor 为 linear 值 —— 本实现把
    # 传入 factor 视为 sRGB 编码转换(GLB roundtrip 以 0-255 存), 校验数值:
    g1 = np.full((4, 4, 3), 128, np.uint8)
    from meshuv.asset.canonicalizer import _bake_factor
    got = _bake_factor(Image.fromarray(g1), np.array([0.5, 1.0, 0.25]))
    exp = l2s(s2l(128 / 255) * np.array([0.5, 1.0, 0.25]))
    check("factor glTF 数值解析", np.abs(got[0, 0] - exp).max() < 1e-9,
          f"{got[0,0].round(4)} vs {exp.round(4)}")

    # 纯色无 UV geometry 保留(常量采样)
    solid = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                            faces=[[0, 1, 2]], process=False)
    s2 = trimesh.Scene(); s2.add_geometry(_quad(imgA)); s2.add_geometry(solid)
    p2 = f"{td}/solid.glb"; s2.export(p2)
    c2 = canonicalize(p2)
    check("纯色无 UV geometry 保留(非删除)", len(c2["F"]) == 3
          and c2["retained"]["n_faces"] == 3
          and c2["retained_area_ratio"] > 0.99,
          f"faces={len(c2['F'])}")
    check("original/retained 面积字段",
          c2["original"]["surface_area"] > 0
          and "retained_area_ratio" in c2)

    # 单面跨 tile -> 拒绝
    m3 = _quad(imgA)
    uv3 = np.array([[0, 0], [1.6, 0], [1.6, 1], [0, 1]], float)  # 面跨 2 tile
    m3.visual = trimesh.visual.TextureVisuals(uv=uv3, image=imgA)
    s3 = trimesh.Scene(); s3.add_geometry(m3)
    p3 = f"{td}/tiled.glb"; s3.export(p3)
    try:
        canonicalize(p3)
        check("跨 tile 面 -> TILED_UV_UNSUPPORTED", False, "未拒绝")
    except ValueError as e:
        check("跨 tile 面 -> TILED_UV_UNSUPPORTED", "TILED" in str(e))

    # 整套整数平移 UV -> 归一后接受
    m4 = _quad(imgA)
    m4.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[2, 3], [3, 3], [3, 4], [2, 4]], float), image=imgA)
    s4 = trimesh.Scene(); s4.add_geometry(m4)
    p4 = f"{td}/shifted.glb"; s4.export(p4)
    c4 = canonicalize(p4)
    check("整套整数 tile 平移可接受", len(c4["F"]) == 2)

n_fail = RESULTS.count(False)
print(f"==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
