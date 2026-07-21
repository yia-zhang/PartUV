# -*- coding: utf-8 -*-
"""P1: 表面重建误差 + chart-level R-D 曲线 + oracle 预算分配.

协议 A(最小范围): 固定 PartUV charts / 单 atlas / 固定 shelf packer / base-color
reference / 同名义预算. 评价在相同 3D 表面采样点上进行(不比较 atlas 图片本身):
  E_surface = mean_x || F_ref(x) - F̂_U(x) ||²   (线性插值重建, sRGB 域, 全方法一致)
"""
import numpy as np

from .geometry import best_corner_perm, tri_area_2d, tri_area_3d


def bilinear(img, uv):
    """uv ∈ [0,1]²(clamp-to-edge), 图像约定 v=1 在顶行. 返回 (n,3).

    项目唯一坐标约定(Coordinate Rebaseline): texel-center —— 第 i 个纹素中心
    在 u=(i+0.5)/W, 即 x = u*W - 0.5。与光栅化(u*W, 像素 i 覆盖 [i,i+1),
    中心 i+0.5)一致。旧 (W-1) 角点约定有系统性亚纹素偏移, 已废弃。
    边界语义: clamp-to-edge(u=0/1 落在首/末纹素中心外半纹素, 取边缘值)。"""
    H, W = img.shape[:2]
    x = np.clip(uv[:, 0], 0, 1) * W - 0.5
    y = np.clip(1 - uv[:, 1], 0, 1) * H - 0.5
    x0f = np.floor(x); y0f = np.floor(y)
    fx = (x - x0f)[:, None]; fy = (y - y0f)[:, None]
    x0 = np.clip(x0f.astype(int), 0, W - 1)
    y0 = np.clip(y0f.astype(int), 0, H - 1)
    x1 = np.clip(x0f.astype(int) + 1, 0, W - 1)
    y1 = np.clip(y0f.astype(int) + 1, 0, H - 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def prepare_face_ref_uv(pu, ref):
    """每个 covered∩ok 面: 与 processed 角序对齐的原 UV 角点 (P0.5: 3! 双射).
    返回 (face_refuv (nF,3,2), valid mask, face2chart (nF,2) [chart_id,row])."""
    V, F, charts = pu["V"], pu["F"], pu["charts"]
    Vo_al = (ref["Vo"] - (ref["Vo"].max(0) + ref["Vo"].min(0)) / 2) * ref["s_align"] \
        + (V.max(0) + V.min(0)) / 2
    Fo, uv0, f2o, ok = ref["Fo"], ref["uv0"], ref["f2o"], ref["ok_map"]
    nF = len(F)
    face_refuv = np.zeros((nF, 3, 2))
    face2chart = np.full((nF, 2), -1, int)
    for ci, c in enumerate(charts):
        for r, g in enumerate(c["gidx"]):
            face2chart[g] = (ci, r)
    valid = (face2chart[:, 0] >= 0) & ok
    idx = np.where(valid)[0]
    for f in idx:
        o = int(f2o[f])
        perm, _ = best_corner_perm(Vo_al[Fo[o]], V[F[f]])
        face_refuv[f] = uv0[Fo[o]][list(perm)]
    return face_refuv, valid, face2chart


def surface_samples(pu, face_refuv, valid, texA, n_total=150_000, seed=0):
    """面积比例采样. 返回 dict(fid, bary, ref_color)."""
    area = pu["area"]
    rng = np.random.RandomState(seed)
    idx = np.where(valid)[0]
    p = area[idx] / area[idx].sum()
    fid = idx[rng.choice(len(idx), size=n_total, p=p)]
    bary = rng.dirichlet((1.0, 1.0, 1.0), n_total)
    uv_ref = np.einsum("nk,nkd->nd", bary, face_refuv[fid])
    return dict(fid=fid, bary=bary, ref_color=bilinear(texA, uv_ref))


def _raster_faces(tex, filled, tris_px, colors_fn):
    """把一组三角形(像素坐标)光栅进 tex; colors_fn(bary, i)->颜色."""
    H, W = tex.shape[:2]
    for i, P in enumerate(tris_px):
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
        m = (w12[:, 0] >= -1e-4) & (w12[:, 1] >= -1e-4) & (w0 >= -1e-4)
        if not m.any():
            continue
        bary = np.stack([w0[m], w12[m, 0], w12[m, 1]], 1)
        tex[ys.ravel()[m], xs.ravel()[m]] = colors_fn(bary, i)
        filled[ys.ravel()[m], xs.ravel()[m]] = True


def _dilate_colors(tex, filled, iters=2):
    for _ in range(iters):
        empty = ~filled
        acc = np.zeros_like(tex)
        cnt = np.zeros(filled.shape)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sf = np.roll(filled, (dy, dx), (0, 1))
            st = np.roll(tex, (dy, dx), (0, 1))
            m = (sf & empty).astype(float)
            acc += st * m[..., None]
            cnt += m
        upd = cnt > 0
        tex[upd] = acc[upd] / cnt[upd][:, None]
        filled |= upd
    return tex, filled


def bake_atlas_masks(pu, uvs, R, face_refuv, valid, texA, dilate_iters=2):
    """全图集 rebake. 返回 (tex, signal_mask(膨胀前), filled(膨胀后)).
    P1a-1: B_signal 用膨胀前 mask, B_pad = 膨胀后 - 膨胀前."""
    from . import gpu
    if gpu.available():
        return gpu.bake_atlas_masks(pu, uvs, R, face_refuv, valid, texA,
                                    dilate_iters)
    charts = pu["charts"]
    tex = np.zeros((R, R, 3))
    filled = np.zeros((R, R), bool)
    for c, uvc in zip(charts, uvs):
        cF = np.asarray(c["F"])
        g = c["gidx"]
        keep = valid[g]
        if not keep.any():
            continue
        uvp = uvc[cF[keep]]
        tris_px = np.stack([uvp[:, :, 0] * R, (1 - uvp[:, :, 1]) * R], -1)
        refuv = face_refuv[g[keep]]

        def colors(bary, i, refuv=refuv):
            return bilinear(texA, bary @ refuv[i])
        _raster_faces(tex, filled, tris_px, colors)
    signal = filled.copy()
    tex, filled = _dilate_colors(tex, filled, dilate_iters)
    return tex, signal, filled


def bake_atlas(pu, uvs, R, face_refuv, valid, texA):
    """兼容旧签名: 返回 (tex, filled)."""
    tex, _, filled = bake_atlas_masks(pu, uvs, R, face_refuv, valid, texA)
    return tex, filled


def _srgb2lin(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin2srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def bake_atlas_ss(pu, uvs, R, ss, face_refuv, valid, texA, dilate_iters=2):
    """超采样 bake + 覆盖加权降采样(Baker Convergence Audit 修复).

    旧评测路径 bug: R*ss 高分 bake 后直接对 ss×ss 块做无权重 mean,
    边界块把膨胀晕(固定 2 高分纹素)之外的未覆盖子纹素当黑色平均进来,
    ss 越大晕圈占比越小 -> seam 随 ss 增加反而恶化(SS8 +223%)。
    修复: 高分 bake 不膨胀; 完全未覆盖的最终纹素由最终分辨率膨胀填充
    (与生产 dilate 语义一致)。返回 (tex, signal, filled)。
    满覆盖块 = 线性域面积平均(抗锯齿语义不变); 部分覆盖块 = 在高分网格上
    以覆盖加权 bilinear 取「最终纹素中心」的值(线性内容下与 SS1 点采样严格
    相等 -> 满足随 SS 非增合同; 质心加权会把有效采样位置偏移达半纹素)。"""
    tex_hi, sig_hi, _ = bake_atlas_masks(pu, uvs, R * ss, face_refuv, valid,
                                         texA, dilate_iters=0)
    lin_hi = _srgb2lin(tex_hi)
    lin = (lin_hi * sig_hi[..., None]).reshape(R, ss, R, ss, 3).sum(axis=(1, 3))
    den = sig_hi.reshape(R, ss, R, ss).sum(axis=(1, 3)).astype(float)
    covered = den > 0
    out = np.zeros((R, R, 3))
    out[covered] = lin[covered] / den[covered, None]
    partial = covered & (den < ss * ss)
    if ss > 1 and partial.any():
        bi, bj = np.nonzero(partial)
        Rh = R * ss
        # 最终纹素中心在高分栅格中心空间的分数索引
        x = np.clip((bj + 0.5) * ss - 0.5, 0, Rh - 1)
        y = np.clip((bi + 0.5) * ss - 0.5, 0, Rh - 1)
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, Rh - 1); y1 = np.minimum(y0 + 1, Rh - 1)
        fx, fy = x - x0, y - y0
        acc = np.zeros((len(bi), 3)); wsum = np.zeros(len(bi))
        ctr = (ss - 1) / 2.0
        di, dj = np.meshgrid(np.arange(ss), np.arange(ss), indexing="ij")
        d2 = ((di - ctr) ** 2 + (dj - ctr) ** 2).reshape(-1)
        for yy, xx, w in [(y0, x0, (1 - fx) * (1 - fy)), (y0, x1, fx * (1 - fy)),
                          (y1, x0, (1 - fx) * fy), (y1, x1, fx * fy)]:
            cw = w * sig_hi[yy, xx]
            acc += lin_hi[yy, xx] * cw[:, None]
            wsum += cw
        ok = wsum > 1e-12
        out[bi[ok], bj[ok]] = acc[ok] / wsum[ok, None]
        if (~ok).any():   # 中心 4 邻均未覆盖: 退回块内离中心最近的已覆盖子采样
            blk_cov = sig_hi.reshape(R, ss, R, ss).transpose(0, 2, 1, 3)[
                bi[~ok], bj[~ok]].reshape(-1, ss * ss)
            blk_lin = lin_hi.reshape(R, ss, R, ss, 3).transpose(0, 2, 1, 3, 4)[
                bi[~ok], bj[~ok]].reshape(-1, ss * ss, 3)
            idx = np.argmin(np.where(blk_cov, d2, 1e18), 1)
            out[bi[~ok], bj[~ok]] = blk_lin[np.arange(len(idx)), idx]
    tex = np.zeros((R, R, 3))
    tex[covered] = _lin2srgb(out[covered])
    signal = covered.copy()
    tex, filled = _dilate_colors(tex, covered, dilate_iters)
    return tex, signal, filled


def eval_surface_error(tex, pu, uvs, samples, face2chart):
    """在固定表面采样点上: 新布局 UV -> bilinear 重建 -> 与 ref MSE."""
    charts = pu["charts"]
    fid, bary = samples["fid"], samples["bary"]
    uv_new = np.zeros((len(fid), 2))
    ci_all, row_all = face2chart[fid, 0], face2chart[fid, 1]
    for ci in np.unique(ci_all):
        m = ci_all == ci
        cF = np.asarray(charts[ci]["F"])
        corners = uvs[ci][cF[row_all[m]]]
        uv_new[m] = np.einsum("nk,nkd->nd", bary[m], corners)
    rec = bilinear(tex, uv_new)
    err = ((rec - samples["ref_color"]) ** 2).mean()
    return float(err)


def chart_rd_curves(pu, face_refuv, valid, texA, samples, face2chart,
                    base_budget=1_000_000, ratios=(0.25, 0.5, 1.0, 2.0, 4.0),
                    min_texels=16):
    """每 chart 的 E_c(P) 离散曲线. P 档 = ratio × (base_budget × a3 份额).
    chart 单独烘进 P texel 的 bbox 矩形 patch, 用该 chart 的表面采样点评误差."""
    charts, area = pu["charts"], pu["area"]
    a3_tot = sum(float(area[c["gidx"]].sum()) for c in charts)
    fid = samples["fid"]
    ci_all = face2chart[fid, 0]
    curves = []
    for ci, c in enumerate(charts):
        cF = np.asarray(c["F"])
        g = c["gidx"]
        keep = valid[g]
        share = float(area[g].sum() / a3_tot)
        P0 = max(base_budget * share, min_texels)
        m = ci_all == ci
        entry = dict(chart=ci, share=share, P=[], E=[])
        if not keep.any() or m.sum() < 8:
            curves.append(entry)
            continue
        uv = np.asarray(c["UV"], float)
        uv = uv - uv.min(0)
        ext = np.maximum(uv.max(0), 1e-9)
        bary = samples["bary"][m]
        rows = face2chart[fid[m], 1]
        refc = samples["ref_color"][m]
        for r in ratios:
            P = max(P0 * r, min_texels)
            wpx = max(int(np.sqrt(P * ext[0] / ext[1])), 2)
            hpx = max(int(P / wpx), 2)
            tex = np.zeros((hpx, wpx, 3))
            filled = np.zeros((hpx, wpx), bool)
            uvn = uv / ext                                    # chart 归一到 patch
            uvp = uvn[cF[keep]]
            tris_px = np.stack([uvp[:, :, 0] * wpx, (1 - uvp[:, :, 1]) * hpx], -1)
            refuv = face_refuv[g[keep]]

            def colors(b, i, refuv=refuv):
                return bilinear(texA, b @ refuv[i])
            _raster_faces(tex, filled, tris_px, colors)
            _dilate_colors(tex, filled, 2)
            corners = uvn[cF[rows]]
            uv_pt = np.einsum("nk,nkd->nd", bary, corners)
            rec = bilinear(tex, uv_pt)
            entry["P"].append(float(wpx * hpx))
            entry["E"].append(float(((rec - refc) ** 2).mean()))
        curves.append(entry)
    return curves


def oracle_allocate(curves, pu, budget):
    """离散贪心: 全 chart 从最低档起, 每次给 ΔE·share/ΔP 最大的 chart 升档.
    返回 face_weight(每面 w=q², q=sqrt(P_c/P0_c(budget)))."""
    charts, area = pu["charts"], pu["area"]
    a3_tot = sum(float(area[c["gidx"]].sum()) for c in charts)
    lvl = []
    total = 0.0
    for cv in curves:
        if cv["P"]:
            lvl.append(0)
            total += cv["P"][0]
        else:
            lvl.append(-1)
    while True:
        best, best_gain = None, 0.0
        for i, cv in enumerate(curves):
            li = lvl[i]
            if li < 0 or li + 1 >= len(cv["P"]):
                continue
            dP = cv["P"][li + 1] - cv["P"][li]
            if total + dP > budget:
                continue
            dE = (cv["E"][li] - cv["E"][li + 1]) * cv["share"]
            gain = dE / max(dP, 1)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        total += curves[best]["P"][lvl[best] + 1] - curves[best]["P"][lvl[best]]
        lvl[best] += 1
    face_weight = np.ones(len(pu["F"]))
    for i, cv in enumerate(curves):
        if lvl[i] < 0:
            continue
        share = cv["share"]
        P0_b = max(budget * share, 1.0)
        q2 = cv["P"][lvl[i]] / P0_b
        face_weight[charts[i]["gidx"]] = q2
    return face_weight, lvl, total


def hull_curves(curves):
    """P1a-4: 单调化 + 下凸包预处理 R-D 曲线, 保证贪心的边际收益递减前提.
    返回 (hulled_curves, diag): diag 统计非单调/非凸 chart 数."""
    n_mono = n_conv = n_eval = 0
    out = []
    for cv in curves:
        P, E = list(cv["P"]), list(cv["E"])
        if len(P) < 2:
            out.append(dict(cv))
            continue
        n_eval += 1
        Em = np.minimum.accumulate(E)                    # 单调非增
        if not np.allclose(Em, E):
            n_mono += 1
        pts = list(zip(P, Em))
        hull = [pts[0]]
        for p, e in pts[1:]:                             # 下凸包(斜率必须递增/变平)
            while len(hull) >= 2:
                (p1, e1), (p2, e2) = hull[-2], hull[-1]
                s1 = (e2 - e1) / max(p2 - p1, 1e-9)
                s2 = (e - e2) / max(p - p2, 1e-9)
                if s2 < s1 - 1e-15:                      # 斜率变得更陡 => 中间点非凸
                    hull.pop()
                else:
                    break
            hull.append((p, e))
        if len(hull) != len(pts):
            n_conv += 1
        out.append(dict(cv, P=[h[0] for h in hull], E=[h[1] for h in hull]))
    return out, dict(n_eval=n_eval, n_nonmonotone=n_mono, n_nonconvex=n_conv)


def ref_gradient_at_samples(texA, face_refuv, samples):
    """P1a-6: 参考纹理自身的亮度梯度模长(在采样点处, 原分辨率有限差分).
    与分配信号无关, 用于定义高频/logo 评价子集."""
    lum = texA @ np.array([0.299, 0.587, 0.114])
    gy, gx = np.gradient(lum)
    gmag = np.sqrt(gx ** 2 + gy ** 2)
    uv_ref = np.einsum("nk,nkd->nd", samples["bary"], face_refuv[samples["fid"]])
    return bilinear(gmag[..., None], uv_ref)[:, 0]
