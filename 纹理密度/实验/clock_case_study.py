# -*- coding: utf-8 -*-
"""Case study: clock_2mat.glb — 多材质 + 低贴图利用率 在我们管线下的处理.

Part A: 逐材质检查 — 贴图分辨率 / UV 范围(是否平铺) / 实际被引用的纹素占比(浪费量化)
Part B: 完整管线 — PartUV 分解 + L2 布局 + rebake 单图集, 报告回收后的利用率
输出: code/notebook/outputs/clock_2mat_case/{original_waste.png, ours_atlas.png} + stats
"""
import os
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

MESH = "/root/youjiaZhang/PartUV/code/data/clock_2mat.glb"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/clock_2mat_case"
os.makedirs(OUT, exist_ok=True)

def rasterize_used(uv, F, H, W):
    """标记贴图上被 UV 三角形覆盖的纹素."""
    used = np.zeros((H, W), bool)
    P_all = np.stack([np.clip(uv[:, 0], 0, 1) * (W - 1),
                      np.clip(1 - uv[:, 1], 0, 1) * (H - 1)], 1)
    for f in F:
        P = P_all[f]
        mn = np.maximum(np.floor(P.min(0)).astype(int), 0)
        mx = np.minimum(np.ceil(P.max(0)).astype(int), [W - 1, H - 1])
        if (mx < mn).any():
            continue
        xs, ys = np.meshgrid(np.arange(mn[0], mx[0] + 1), np.arange(mn[1], mx[1] + 1))
        pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
        T = np.stack([P[1] - P[0], P[2] - P[0]], 1)
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        if abs(det) < 1e-12:
            continue
        invT = np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]]) / det
        w12 = (pts - P[0]) @ invT.T
        w0 = 1 - w12.sum(1)
        m = (w12[:, 0] >= -0.02) & (w12[:, 1] >= -0.02) & (w0 >= -0.02)
        used[ys.ravel()[m], xs.ravel()[m]] = True
    return used

# ================= Part A: 逐材质浪费量化 =================
scene = trimesh.load(MESH)
print("=== Part A: 原始资产结构 ===")
geoms = list(scene.geometry.items()) if isinstance(scene, trimesh.Scene) else [("mesh", scene)]
fig, axs = plt.subplots(1, len(geoms), figsize=(6.2 * len(geoms), 6.2))
axs = np.atleast_1d(axs)
per_mat = []
for ax, (gname, g) in zip(axs, geoms):
    uv = np.asarray(g.visual.uv, float)
    F = np.asarray(g.faces)
    img = getattr(g.visual.material, "baseColorTexture", None) or getattr(g.visual.material, "image", None)
    tex = np.asarray(img.convert("RGB"), float) / 255.0
    H, W = tex.shape[:2]
    tiled = bool(uv.min() < -0.01 or uv.max() > 1.01)
    used = rasterize_used(uv, F, H, W)
    frac = float(used.mean())
    per_mat.append(dict(name=gname, faces=len(F), res=f"{W}x{H}", used=frac, tiled=tiled))
    print(f"  材质 {gname}: faces={len(F)} tex={W}x{H} UV范围=[{uv.min():.2f},{uv.max():.2f}] "
          f"平铺={tiled} 被引用纹素={frac:.1%}")
    vis = tex.copy()
    vis[~used] *= 0.22                                   # 未引用区域压暗
    ax.imshow(vis); ax.set_axis_off()
    ax.set_title(f"{gname}: {W}x{H}, used = {frac:.1%}\n(dimmed = never referenced by any face)",
                 fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/original_waste.png", dpi=110, bbox_inches="tight")
total_texels = sum(int(np.prod([int(x) for x in m["res"].split("x")])) for m in per_mat)
used_texels = sum(m["used"] * np.prod([int(x) for x in m["res"].split("x")]) for m in per_mat)
print(f"  合计: {len(per_mat)} 张贴图 {total_texels/1e6:.2f}M 纹素, "
      f"实际被引用 {used_texels/total_texels:.1%}")

# ================= Part B: 我们的管线(合并加载 -> PartUV -> L2 -> rebake 单图集) =================
print("=== Part B: 我们的管线 ===")
import partuv
from partuv.preprocess_utils.partfield_official.run_PF import PFInferenceModel

pf = PFInferenceModel(device="cuda",
                      checkpoint_path="/root/zhaotianhao/PartField/model/model_objaverse.ckpt")
mesh, tf, tree, _ = partuv.preprocess(MESH, pf, OUT + "/", merge_vertices_epsilon=None)
V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, np.int32)
final, parts = partuv.pipeline_numpy(V=V, F=F, tree_dict=tree,
                                     configPath="/root/youjiaZhang/PartUV/code/notebook/partuv_config.yaml",
                                     threshold=1.25)

kd = cKDTree(V[F].mean(axis=1))
mesh_scale = float(np.linalg.norm(V.max(0) - V.min(0)))

def a2f(uv):
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
def a3f(v):
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

charts = []
face_chart = np.full(len(F), -1)
for pi, p in enumerate(parts):
    for c in p.components:
        cV, cF, cUV = np.asarray(c.V), np.asarray(c.F), np.asarray(c.UV)
        d, g = kd.query(cV[cF].mean(axis=1))
        face_chart[g] = len(charts)
        charts.append(dict(V=cV, F=cF, UV=cUV, gidx=g,
                           a2=float(a2f(cUV[cF]).sum()), a3=float(a3f(cV[cF]).sum())))
covered = face_chart >= 0
print(f"  PartUV: parts={len(parts)} charts={len(charts)} coverage={covered.mean():.3f}")

# 合并加载(trimesh 自动把两张贴图打包成一张 + 重映射 UV)
orig = trimesh.load(MESH, force="mesh")
uv0 = np.asarray(orig.visual.uv, float)
Vo, Fo = np.asarray(orig.vertices, float), np.asarray(orig.faces)
mimg = orig.visual.material.baseColorTexture or orig.visual.material.image
texA = np.asarray(mimg.convert("RGB"), float) / 255.0
print(f"  trimesh 合并加载: {len(geoms)} 材质 -> 单张 {texA.shape[1]}x{texA.shape[0]} 合成图")

co = (Vo.max(0) + Vo.min(0)) / 2; cp = (V.max(0) + V.min(0)) / 2
s_al = np.linalg.norm(V.max(0) - V.min(0)) / max(np.linalg.norm(Vo.max(0) - Vo.min(0)), 1e-12)
Vo_al = (Vo - co) * s_al + cp
kd_o = cKDTree(Vo_al[Fo].mean(axis=1))
d_o, f2o = kd_o.query(V[F].mean(axis=1))
ok_map = d_o < 1e-4 * mesh_scale
print(f"  面映射成功率={ok_map.mean():.2%}")

# 内容权重 + L2 布局(shelf, 够用于利用率统计)
rng = np.random.RandomState(0)
bar = rng.dirichlet((1.2, 1.2, 1.2), 24)
lum = texA @ np.array([0.299, 0.587, 0.114])
OUV3 = uv0[Fo[f2o]]
samp = np.einsum("sk,fkd->sfd", bar, OUV3)
acc = acc2 = None
for s in range(len(bar)):
    x = np.clip(samp[s, :, 0], 0, 1) * (lum.shape[1] - 1)
    y = np.clip(1 - samp[s, :, 1], 0, 1) * (lum.shape[0] - 1)
    v = lum[y.astype(int), x.astype(int)]
    acc = v.copy() if acc is None else acc + v
    acc2 = v ** 2 if acc2 is None else acc2 + v ** 2
cw = np.sqrt(np.maximum(acc2 / 24 - (acc / 24) ** 2, 0)); cw[~ok_map] = 0
cw_n = cw / max(np.median(cw[covered & ok_map]), 1e-9)
w = np.clip((1 + 3.0 * cw_n) ** 1.5, 1, 8.0)

def shelf(rects, pad=0.004):
    order = sorted(range(len(rects)), key=lambda i: -rects[i][1])
    Wd = max(np.sqrt(sum(a * b for a, b in rects)) * 1.15, max(a for a, _ in rects) + 2 * pad)
    x = y = rh = 0.0; pos = [None] * len(rects)
    for i in order:
        a, b = rects[i]
        if x + a + pad > Wd: x = 0.0; y += rh + pad; rh = 0.0
        pos[i] = (x + pad / 2, y + pad / 2); x += a + pad; rh = max(rh, b)
    side = max(Wd, y + rh + pad)
    return [(px / side, py / side) for px, py in pos], side

uvs, rects = [], []
for c in charts:
    cF = np.asarray(c["F"]); g = c["gidx"]
    dem = float((a3f(np.asarray(c["V"])[cF]) * w[g]).sum())
    u = c["UV"] * np.sqrt(dem / max(c["a2"], 1e-12)); u = u - u.min(0)
    uvs.append(u); rects.append(tuple(u.max(0)))
off, side = shelf(rects)
uv_l2 = [u / side + np.array(o) for u, o in zip(uvs, off)]

# rebake 到单张 1024
RES = 1024
def bil(img, uvq):
    x = np.clip(uvq[:, 0], 0, 1) * (img.shape[1] - 1); y = np.clip(1 - uvq[:, 1], 0, 1) * (img.shape[0] - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, img.shape[1] - 1); y1 = np.minimum(y0 + 1, img.shape[0] - 1)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)

texB = np.zeros((RES, RES, 3)); fil = np.zeros((RES, RES), bool)
for c, uvc in zip(charts, uv_l2):
    cF = np.asarray(c["F"]); cV = np.asarray(c["V"]); g = c["gidx"]
    for i in range(len(cF)):
        gi = int(g[i])
        if not ok_map[gi]:
            continue
        og = int(f2o[gi]); uvP = uvc[cF[i]]
        P = np.stack([uvP[:, 0], 1 - uvP[:, 1]], 1) * RES
        C3 = cV[cF[i]]; O3 = Vo_al[Fo[og]]
        perm = [int(np.argmin(((O3 - C3[k]) ** 2).sum(1))) for k in range(3)]
        OUVp = uv0[Fo[og]][perm]
        mn = np.maximum(np.floor(P.min(0)).astype(int), 0)
        mx = np.minimum(np.ceil(P.max(0)).astype(int), RES - 1)
        if (mx < mn).any():
            continue
        xs, ys = np.meshgrid(np.arange(mn[0], mx[0] + 1), np.arange(mn[1], mx[1] + 1))
        pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
        T = np.stack([P[1] - P[0], P[2] - P[0]], 1)
        det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        if abs(det) < 1e-12:
            continue
        invT = np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]]) / det
        w12 = (pts - P[0]) @ invT.T; w0 = 1 - w12.sum(1)
        m = (w12[:, 0] >= -1e-4) & (w12[:, 1] >= -1e-4) & (w0 >= -1e-4)
        if not m.any():
            continue
        bry = np.stack([w0[m], w12[m, 0], w12[m, 1]], 1)
        texB[ys.ravel()[m], xs.ravel()[m]] = bil(texA, bry @ OUVp)
        fil[ys.ravel()[m], xs.ravel()[m]] = True

fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(texB); ax.set_axis_off()
ax.set_title(f"Ours: single {RES}x{RES} atlas (L2, shelf-packed)\n"
             f"fill = {fil.mean():.1%}, all referenced content consolidated", fontsize=10)
plt.tight_layout(); plt.savefig(f"{OUT}/ours_atlas.png", dpi=110, bbox_inches="tight")
print(f"  Ours: 单张 {RES}x{RES} 图集, 有效纹素 {fil.mean():.1%} "
      f"(原始 {len(geoms)} 张贴图合计仅 {used_texels/total_texels:.1%} 被引用)")
print("DONE")
