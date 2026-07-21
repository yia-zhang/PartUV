# -*- coding: utf-8 -*-
"""生成系统化覆盖的合成测试资产(带纹理 GLB) -> code/data/synth_*.glb.

覆盖维度:
- 纹理内容: 纯色/渐变/密集文字/半平坦半细节/多尺度噪声/4K 高细节/超小贴图/非方形
- UV 拓扑: 单岛/大量小岛/trim-sheet 重用(reuse~6)/镜像重用(~2)/tiled(应 UNSUPPORTED)
  /UDIM 式(应 UNSUPPORTED)/退化 UV
- 几何: 闭合/开放/双面重合壳(孪生面)/高面数/多组件
- 材质: 双材质双贴图 / 无 UV(应 UNSUPPORTED) / 仅顶点色(应 UNSUPPORTED)
全部确定性(固定种子)。
"""
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw

DATA = "/root/youjiaZhang/PartUV/code/data"
rng = np.random.default_rng(7)


# ---------------- 纹理生成 ----------------
def tex_solid(n=512, color=(178, 96, 60)):
    return Image.new("RGB", (n, n), color)


def tex_gradient(n=1024):
    y, x = np.mgrid[0:n, 0:n] / n
    img = np.stack([120 + 100 * x, 80 + 120 * y, 200 - 120 * x * y], -1)
    return Image.fromarray(img.astype(np.uint8))


def tex_text(n=1024, lines=28):
    img = Image.new("RGB", (n, n), (235, 230, 220))
    d = ImageDraw.Draw(img)
    for i in range(lines):
        y = int(n * (i + 0.5) / lines)
        d.text((8, y - 6), f"SPEC-{i:02d} PartUV texel density test 0123456789 "
                           f"ABCDEFGHIJKLMNOPQRSTUVWXYZ", fill=(30, 30, 40))
        d.line([(0, y + 8), (n, y + 8)], fill=(180, 60, 50), width=1)
    return img


def tex_half_flat_detail(n=1024):
    a = np.full((n, n, 3), (90, 110, 160), np.uint8)          # 左半: 平坦
    c = ((np.indices((n, n // 2)).sum(0) // 16) % 2 * 255).astype(np.uint8)
    a[:, n // 2:] = np.stack([c, c, c], -1)                   # 右半: 细棋盘
    img = Image.fromarray(a)
    d = ImageDraw.Draw(img)
    for i in range(10):                                       # 右半叠文字
        d.text((n // 2 + 10, 40 + i * n // 11), f"DETAIL-{i}", fill=(200, 40, 40))
    return img


def tex_multiscale_noise(n=2048):
    img = np.zeros((n, n))
    for k in (8, 32, 128, 512):
        g = rng.normal(0, 1, (k, k))
        img += np.asarray(Image.fromarray((g - g.min()) / np.ptp(g) * 255)
                          .resize((n, n), Image.BILINEAR), float) / 4
    img = (img - img.min()) / np.ptp(img)
    return Image.fromarray((np.stack([img, img * 0.8 + 0.1, 1 - img], -1)
                            * 255).astype(np.uint8))


def tex_4k_detail(n=4096):
    y, x = np.mgrid[0:n, 0:n]
    a = ((x // 8 + y // 8) % 2) * 90 + 80
    b = (np.sin(x / 23.0) * np.cos(y / 31.0) * 60 + 60)
    img = np.stack([a, (a + b) / 2, b + 100], -1).clip(0, 255)
    return Image.fromarray(img.astype(np.uint8))


def tex_label_2to1(w=2048, h=1024):
    img = Image.new("RGB", (w, h), (250, 245, 235))
    d = ImageDraw.Draw(img)
    for i in range(8):
        d.text((30, 40 + i * h // 9), f"NON-SQUARE LABEL ROW {i} — texel density",
               fill=(20, 60, 120))
    d.rectangle([w // 2, 0, w, h], outline=(200, 50, 40), width=6)
    return img


# ---------------- 几何/UV 构造 ----------------
def uv_sphere_mesh(subdiv=4):
    m = trimesh.creation.icosphere(subdivisions=subdiv)
    V = m.vertices
    u = 0.5 + np.arctan2(V[:, 1], V[:, 0]) / (2 * np.pi)
    v = 0.5 + np.arcsin(np.clip(V[:, 2] / np.linalg.norm(V, axis=1), -1, 1)) / np.pi
    return m, np.stack([u, v], 1)


def make_glb(path, mesh, uv, img):
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False,
                           visual=trimesh.visual.TextureVisuals(
                               uv=uv, material=trimesh.visual.material.PBRMaterial(
                                   baseColorTexture=img)))
    mesh.export(path)
    print(f"  {os.path.basename(path):32s} {len(mesh.faces):>6,} 面  "
          f"tex {img.size[0]}x{img.size[1]}")


def quad_grid(nx, ny, cell, gap):
    """nx*ny 个独立小方块(不共享顶点 => 每块一个 UV 岛/一个组件)."""
    V, F, UV = [], [], []
    off = 0
    for j in range(ny):
        for i in range(nx):
            x0, y0 = i * (cell + gap), j * (cell + gap)
            V += [[x0, y0, 0], [x0 + cell, y0, 0],
                  [x0 + cell, y0 + cell, 0], [x0, y0 + cell, 0]]
            u0, v0 = i / nx, j / ny
            u1, v1 = (i + 0.92) / nx, (j + 0.92) / ny
            UV += [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
            F += [[off, off + 1, off + 2], [off, off + 2, off + 3]]
            off += 4
    return (trimesh.Trimesh(np.array(V, float), np.array(F), process=False),
            np.array(UV, float))


def main():
    os.makedirs(DATA, exist_ok=True)
    print("生成合成测试资产:")

    # 1-5: 纹理内容谱系(同一球体几何, 语义清晰)
    sp, sp_uv = uv_sphere_mesh(4)
    make_glb(f"{DATA}/synth_flat_solid.glb", sp, sp_uv, tex_solid())
    make_glb(f"{DATA}/synth_gradient.glb", sp, sp_uv, tex_gradient())
    make_glb(f"{DATA}/synth_text_dense.glb", sp, sp_uv, tex_text())
    make_glb(f"{DATA}/synth_multiscale_noise.glb", sp, sp_uv, tex_multiscale_noise())
    make_glb(f"{DATA}/synth_tiny_tex_128.glb", sp, sp_uv,
             tex_text(128, lines=6))

    # 6: 半平坦半细节(torus, 密度重分配的展示例)
    to = trimesh.creation.torus(major_radius=1.0, minor_radius=0.4,
                                major_sections=96, minor_sections=48)
    Vt = to.vertices
    u = 0.5 + np.arctan2(Vt[:, 1], Vt[:, 0]) / (2 * np.pi)
    r = np.linalg.norm(Vt[:, :2], axis=1) - 1.0
    v = 0.5 + np.arctan2(Vt[:, 2], r) / (2 * np.pi)
    make_glb(f"{DATA}/synth_half_flat_detail.glb", to,
             np.stack([u, v], 1), tex_half_flat_detail())

    # 7: 4K 高细节 + 高面数(预算/规模压力)
    sp6, sp6_uv = uv_sphere_mesh(6)
    make_glb(f"{DATA}/synth_highpoly_4ktex.glb", sp6, sp6_uv, tex_4k_detail())

    # 8: 大量小岛(144 岛, packing 压力)
    g, g_uv = quad_grid(12, 12, 1.0, 0.15)
    make_glb(f"{DATA}/synth_many_islands_144.glb", g, g_uv, tex_text(1024, 40))

    # 9: trim-sheet 重用(6 个方块共享同一 UV 区域, reuse≈6)
    V, F, UV = [], [], []
    off = 0
    for k in range(6):
        x0 = k * 1.3
        V += [[x0, 0, 0], [x0 + 1, 0, 0], [x0 + 1, 1, 0], [x0, 1, 0]]
        UV += [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]]
        F += [[off, off + 1, off + 2], [off, off + 2, off + 3]]
        off += 4
    make_glb(f"{DATA}/synth_trimsheet_reuse6.glb",
             trimesh.Trimesh(np.array(V, float), np.array(F), process=False),
             np.array(UV, float), tex_text(512, 12))

    # 10: 镜像重用(左右半球映射到同一半张贴图, reuse≈2)
    mu = sp_uv.copy()
    mu[:, 0] = np.abs(mu[:, 0] - 0.5) * 2          # 镜像折叠到 [0,1]
    make_glb(f"{DATA}/synth_mirrored_uv.glb", sp, mu, tex_gradient())

    # 11: tiled UV(uv∈[0,4], 应 UNSUPPORTED)
    pl = trimesh.Trimesh(np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], float),
                         np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    make_glb(f"{DATA}/synth_tiled_uv.glb", pl,
             np.array([[0, 0], [4, 0], [4, 4], [0, 4]], float), tex_text(256, 8))

    # 12: UDIM 式(两块分别位于 tile0/tile1, 应 UNSUPPORTED)
    V2 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                   [2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]], float)
    F2 = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    UV2 = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                    [1, 0], [2, 0], [2, 1], [1, 1]], float)
    make_glb(f"{DATA}/synth_udim_like.glb",
             trimesh.Trimesh(V2, F2, process=False), UV2, tex_text(256, 8))

    # 13: 非方形贴图(圆柱标签, 2:1)
    cy = trimesh.creation.cylinder(radius=0.5, height=2.0, sections=64)
    Vc = cy.vertices
    uc = 0.5 + np.arctan2(Vc[:, 1], Vc[:, 0]) / (2 * np.pi)
    vc = (Vc[:, 2] - Vc[:, 2].min()) / np.ptp(Vc[:, 2])
    make_glb(f"{DATA}/synth_nonsquare_label.glb", cy,
             np.stack([uc, vc], 1), tex_label_2to1())

    # 14: 双面重合壳(孪生面, FaceMatcher 压力)
    sh_in = trimesh.Trimesh(sp.vertices * 0.999, sp.faces[:, ::-1], process=False)
    both = trimesh.util.concatenate(
        [trimesh.Trimesh(sp.vertices, sp.faces, process=False), sh_in])
    make_glb(f"{DATA}/synth_twin_shell.glb", both,
             np.concatenate([sp_uv, sp_uv]), tex_gradient())

    # 15: 开放面片(非闭合, 地形式)
    xg, yg = np.meshgrid(np.linspace(0, 4, 40), np.linspace(0, 4, 40))
    zg = 0.3 * np.sin(xg * 2) * np.cos(yg * 2)
    Vg = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], 1)
    Fg = []
    for j in range(39):
        for i in range(39):
            a = j * 40 + i
            Fg += [[a, a + 1, a + 41], [a, a + 41, a + 40]]
    UVg = np.stack([xg.ravel() / 4, yg.ravel() / 4], 1)
    make_glb(f"{DATA}/synth_open_terrain.glb",
             trimesh.Trimesh(Vg, np.array(Fg), process=False), UVg,
             tex_multiscale_noise(1024))

    # 16: 退化 UV(含零面积 UV 面 + 游离顶点, 鲁棒性)
    dg, dg_uv = quad_grid(3, 3, 1.0, 0.2)
    dg_uv[0:4] = [[0.5, 0.5]] * 4                  # 第一块 UV 退化为一点
    make_glb(f"{DATA}/synth_degenerate_uv.glb", dg, dg_uv, tex_text(512, 12))

    # 17: 双材质双贴图(合并读取路径)
    b1 = trimesh.creation.box(extents=(1, 1, 1))
    b2 = trimesh.creation.box(extents=(1, 1, 1))
    b2.apply_translation([1.6, 0, 0])
    def _box_uv(m):
        # 简单平面投影(每面一致即可)
        v = m.vertices
        return np.stack([(v[:, 0] - v[:, 0].min()) / max(np.ptp(v[:, 0]), 1e-9),
                         (v[:, 1] - v[:, 1].min()) / max(np.ptp(v[:, 1]), 1e-9)], 1)
    m1 = trimesh.Trimesh(b1.vertices, b1.faces, process=False,
                         visual=trimesh.visual.TextureVisuals(
                             uv=_box_uv(b1), material=trimesh.visual.material.
                             PBRMaterial(baseColorTexture=tex_text(512, 10))))
    m2 = trimesh.Trimesh(b2.vertices, b2.faces, process=False,
                         visual=trimesh.visual.TextureVisuals(
                             uv=_box_uv(b2), material=trimesh.visual.material.
                             PBRMaterial(baseColorTexture=tex_gradient(512))))
    trimesh.Scene([m1, m2]).export(f"{DATA}/synth_two_materials.glb")
    print(f"  synth_two_materials.glb          scene(2 材质 2 贴图)")

    # 18: 无 UV(应 UNSUPPORTED)
    trimesh.Trimesh(sp.vertices, sp.faces, process=False).export(
        f"{DATA}/synth_no_uv.glb")
    print(f"  synth_no_uv.glb                  无 UV")

    # 19: 仅顶点色(应 UNSUPPORTED)
    vc_mesh = trimesh.Trimesh(sp.vertices, sp.faces, process=False)
    vc_mesh.visual = trimesh.visual.ColorVisuals(
        vc_mesh, vertex_colors=(np.abs(sp.vertices) * 255).astype(np.uint8))
    vc_mesh.export(f"{DATA}/synth_vertex_color.glb")
    print(f"  synth_vertex_color.glb           仅顶点色")

    print("完成。")


if __name__ == "__main__":
    main()
