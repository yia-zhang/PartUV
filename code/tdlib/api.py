# -*- coding: utf-8 -*-
"""Simple V1 统一入口: map_partuv_td(input_mesh, output_dir, atlas_size, beta, max_atlas).

管线(全部复用既有模块): 计算输入 surface texel budget -> 自动选择 atlas size
-> PartUV charts -> luminance-std 内容分数 -> 单一 β -> 预算归一化 ->
chart 缩放 + xatlas packing(利用率一等目标; shelf 仅诊断) -> rebake -> 导出 GLB + atlas。

完整性保证(违反则抛错, 不输出资产):
- 输出顶点恢复原始坐标尺度与位置(逆 preprocess 相似变换);
- 保留全部输入面(含 PartUV 未覆盖面, 未覆盖面给显式 fallback UV 并在报告中列出);
- 只在 UV seam(chart 边界)拆分顶点, 其余顶点共享;
- 输出带光滑顶点法线(seam 两侧按位置分组平均, 无接缝阴影);
- 复制原材质的标量参数(baseColorFactor 等), base-color 贴图替换为新 atlas;
- 面数 / 表面积 / 包围盒 / UV 范围 全部校验。

输入边界:
- 普通 UV overlap/trim-sheet(跨 UV 岛重叠) = **WARNING + 唯一化 rebake**——
  重烘按面读原色, 共享内容唯一化为各 chart 独立纹素副本, 输出正确
  (实测车轮即 33% 重叠资产), 代价是纹素预算被复制内容稀释;
- tiled/UDIM(UV 显著超出 [0,1], clamp 会破坏内容)、无 base-color 贴图、无 UV
  = **UNSUPPORTED**(抛 UnsupportedAssetError, 消息含 "UNSUPPORTED")。
仅输出单 atlas(不做 multi-atlas)。
"""
import os

import numpy as np
import trimesh

from .budget import rasterize_masks
from .layout import chart_scales, layout_with_scales
from .pipeline import load_reference, run_partuv, tile_normalized_uv
from .rd import bake_atlas_masks, bilinear, prepare_face_ref_uv
from .signal import demand_weights, luminance_std_heuristic

UV_TILE_TOL = 0.01        # uv 超出 [0,1] 超过该值视为 tiled/UDIM 候选
UV_TILE_FRAC = 0.005      # 超范围 UV 顶点占比超过该值 -> UNSUPPORTED
UV_OVERLAP_FRAC = 0.05    # 光栅化重叠纹素占比超过该值 -> UNSUPPORTED
AREA_TOL = 1e-3           # 输出/输入 表面积相对偏差上限
BBOX_TOL = 1e-3           # 输出/输入 包围盒偏差上限(相对模型尺度)


class UnsupportedAssetError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


class BudgetLimitExceededError(RuntimeError):
    pass


class NeedsMultiAtlasError(RuntimeError):
    pass


class PartUVFailedError(RuntimeError):
    pass


def _uv_islands(Fo, n_verts):
    """UV 岛 = 面片经共享顶点的连通分量. 返回每面的岛标签."""
    parent = np.arange(n_verts)

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in Fo:
        r0 = _find(tri[0])
        for v in tri[1:]:
            r = _find(v)
            if r != r0:
                parent[r] = r0
    root = np.array([_find(v) for v in range(n_verts)])
    return root[Fo[:, 0]]


def _island_budget(Fo, uv, W, H, face_lbl):
    """源预算测量(按 UV island 的 mask, 非逐三角 footprint 求和):
    每个岛在 (W,H) 分辨率下光栅化为 mask, 重叠区按岛数累加.
    返回 (B_unique=并集纹素数, B_surface=各岛 mask 纹素数之和)。
    (与 budget.rasterize_masks 相同的光栅化数学; 此处为逐岛计数版)"""
    from . import gpu
    if gpu.available():
        return gpu.island_budget(Fo, uv, W, H, face_lbl)
    uvc = np.clip(np.asarray(uv, float), 0, 1)
    count = np.zeros((H, W), np.uint16)
    P_all = np.stack([uvc[:, 0] * W, (1 - uvc[:, 1]) * H], 1)
    for L in np.unique(face_lbl):
        mask = np.zeros((H, W), bool)
        for tri in Fo[face_lbl == L]:
            P = P_all[tri]
            mn = np.maximum(np.floor(P.min(0)).astype(int), 0)
            mx = np.minimum(np.ceil(P.max(0)).astype(int), [W - 1, H - 1])
            if (mx < mn).any():
                continue
            xs, ys = np.meshgrid(np.arange(mn[0], mx[0] + 1),
                                 np.arange(mn[1], mx[1] + 1))
            pts = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
            T = np.stack([P[1] - P[0], P[2] - P[0]], 1)
            det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
            if abs(det) < 1e-12:
                continue
            invT = np.array([[T[1, 1], -T[0, 1]], [-T[1, 0], T[0, 0]]]) / det
            w12 = (pts - P[0]) @ invT.T
            w0 = 1 - w12.sum(1)
            m = (w12[:, 0] >= -1e-6) & (w12[:, 1] >= -1e-6) & (w0 >= -1e-6)
            if m.any():
                mask[ys.ravel()[m], xs.ravel()[m]] = True
        count += mask
    return int((count > 0).sum()), int(count.astype(np.int64).sum())


def measure_source_budget(orig):
    """输入资产的绝对 surface texel budget(在源贴图原生分辨率下测量).
    当前读取能力将多材质合并为单一贴图 domain, 故为单 domain 统计;
    多张独立贴图时应逐 domain 统计后求和(合并读取已隐式做了这件事)。"""
    uv0 = tile_normalized_uv(np.asarray(orig.visual.uv, float))
    Fo = np.asarray(orig.faces)
    mat = orig.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    Ht, Wt = np.asarray(img).shape[:2]
    face_lbl = _uv_islands(Fo, len(uv0))
    b_unique, b_surface = _island_budget(Fo, uv0, Wt, Ht, face_lbl)
    return dict(source_B_unique=b_unique, source_B_surface=b_surface,
                source_reuse_factor=round(b_surface / max(b_unique, 1), 4))


def check_asset_support(input_mesh):
    """输入读取与支持性检查. 返回 dict(supported, reason, n_faces, tex_shape)."""
    orig = trimesh.load(input_mesh, force="mesh", process=False)
    n_faces = len(orig.faces)
    uv0 = getattr(orig.visual, "uv", None)
    img = None
    try:
        mat = orig.visual.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    except Exception:
        pass
    if uv0 is None or len(np.atleast_1d(uv0)) == 0:
        return dict(supported=False, n_faces=n_faces, tex_shape=None,
                    reason="UNSUPPORTED: 输入无 UV 坐标")
    if img is None:
        return dict(supported=False, n_faces=n_faces, tex_shape=None,
                    reason="UNSUPPORTED: 无法读取 base-color 贴图"
                           "(可能为多材质合并后丢失/顶点色/无贴图资产)")
    uv0 = np.asarray(uv0, float)
    tex_shape = np.asarray(img).shape[:2]
    if not np.isfinite(uv0).all():
        return dict(supported=False, n_faces=n_faces, tex_shape=tex_shape,
                    reason="UNSUPPORTED: UV 含非有限值")
    uv0 = tile_normalized_uv(uv0)   # 整数 tile 平移归一(REPEAT 语义等价, 非重采样)
    Fo = np.asarray(orig.faces)
    # tiled/UDIM: 面积加权的超范围面占比(顶点占比会漏掉"少顶点大平铺面") + 跨度硬阈值
    # 按 3D 面积加权(顶点占比会漏"少顶点大平铺面"; 零面积游离 UV 顶点不应误伤)
    fa = np.asarray(orig.area_faces, float)
    face_out = ((uv0[Fo] < -UV_TILE_TOL) | (uv0[Fo] > 1 + UV_TILE_TOL)).any((1, 2))
    out_frac = float(fa[face_out].sum() / max(fa.sum(), 1e-20))
    if out_frac > UV_TILE_FRAC:
        return dict(supported=False, n_faces=n_faces, tex_shape=tex_shape,
                    reason=f"UNSUPPORTED: tiled/UDIM UV(超范围面积占比 "
                           f"{out_frac * 100:.1f}%)")
    # overlap / trim-sheet: 按 UV 岛(顶点连通分量)分 chart 光栅化, 统计跨岛重叠.
    # rasterize_masks 只计跨 chart 冲突, 故必须逐岛传入; 岛内自重叠不在本检测范围.
    face_lbl = _uv_islands(Fo, len(uv0))
    labels = np.unique(face_lbl)
    uvc = np.clip(uv0, 0, 1)
    island_charts = [dict(F=Fo[face_lbl == L], gidx=np.where(face_lbl == L)[0])
                     for L in labels]
    owner, overlap, _ = rasterize_masks(island_charts,
                                        [uvc] * len(island_charts), 512, 512)
    used = max(int((owner >= 0).sum()), 1)
    ov_frac = overlap / used
    # overlap/trim-sheet 不阻断: 管线按面读原色, 重烘后共享内容在各 chart 中复制,
    # 输出仍正确(代价是纹素被复制内容稀释) —— 显式量化告警而非 UNSUPPORTED.
    # (实测车轮即 33% 跨岛重叠资产; 若需硬拒绝, 把下面的 warning 改回 supported=False)
    overlap_warning = ""
    if ov_frac > UV_OVERLAP_FRAC:
        overlap_warning = (f"输入 UV 跨岛纹理复用开销 {ov_frac * 100:.1f}%(重叠纹素/并集纹素)"
                           f"(trim-sheet 型重用, {len(labels)} 岛): rebake 会将共享"
                           f"内容唯一化为各 chart 独立纹素副本, 输出正确, "
                           f"但纹素预算被复制内容稀释")
    return dict(supported=True, n_faces=n_faces, tex_shape=tex_shape,
                reason="", uv_out_frac=out_frac, uv_overlap_frac=float(ov_frac),
                n_uv_islands=int(len(labels)), overlap_warning=overlap_warning)


def _find_empty_block(filled, k=8):
    """在 atlas 空白区找 k×k 全空块, 返回 (row, col)."""
    f = filled.astype(np.int32)
    ii = np.pad(f.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    pos = np.argwhere(s == 0)
    if len(pos) == 0:
        raise IntegrityError("atlas 无空白块可放置未覆盖面 fallback 颜色")
    return int(pos[0][0]), int(pos[0][1])


def _grouped_vertex_normals(V, F, scale):
    """位置分组的面积加权光滑法线(seam 两侧共享同一法线)."""
    tris = V[F]
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])  # 面积加权
    key = np.round(V / (1e-6 * scale)).astype(np.int64)
    _, grp, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    acc = np.zeros((len(grp), 3))
    for i in range(3):
        np.add.at(acc, inv[F[:, i]], fn)
    n = acc[inv]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20), len(grp)


def map_partuv_td(input_mesh, output_dir, atlas_size="auto", beta=0.75,
                  max_atlas=8192):
    """统一入口. 返回结果 dict(资产路径 + 完整性报告 + 预算报告 + 演示用中间产物).

    atlas_size:
    - "auto"(默认, 预算平价模式): B_target = 源 B_surface(island mask 实测);
      搜索最小方形分辨率(16 步长, 非 POT)使 packing(xatlas) 后**实测**
      1.00 <= B_signal/B_target <= 1.05; max_atlas 内无法达到下限则抛
      NeedsMultiAtlasError(NEEDS_MULTI_ATLAS), 不导出、不静默降质。
    - 显式整数(固定存储预算模式): 严格尊重给定尺寸, 不偷偷放大; 预算不足时
      在 warnings 与 budget 报告中诚实说明(budget_ratio / budget_met)。
    """
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_mesh))[0]
    warnings = []

    # ---- 0. 输入读取与支持性检查 ----
    support = check_asset_support(input_mesh)
    if not support["supported"]:
        raise UnsupportedAssetError(support["reason"])
    if support.get("overlap_warning"):
        warnings.append(support["overlap_warning"])
    orig = trimesh.load(input_mesh, force="mesh")
    n_faces_in = len(orig.faces)
    orig_mat = getattr(orig.visual, "material", None)

    # ---- 0b. 输入的绝对 surface texel budget(源贴图原生分辨率, 按 island mask) ----
    src_budget = measure_source_budget(orig)
    B_target = src_budget["source_B_surface"]

    # ---- 1. PartUV(冻结管线) ----
    pu = run_partuv(input_mesh, output_dir)
    V, F, area, covered, charts = (pu["V"], pu["F"], pu["area"],
                                   pu["covered"], pu["charts"])
    if len(F) != n_faces_in:
        raise IntegrityError(
            f"PartUV preprocess 改变了面数: 输入 {n_faces_in} -> {len(F)}, "
            f"无法保证逐面保留, 不输出资产")

    if len(charts) == 0 or not covered.any():
        raise PartUVFailedError(
            f"PARTUV_FAILED: PartUV 未产生可用 charts"
            f"(charts={len(charts)}, 覆盖面={int(covered.sum())}/{len(F)}); "
            f"极端组件/退化分解, 本轮按明确失败处理(不做 chart merge/split)。")

    ref = load_reference(input_mesh, V, F, pu["mesh_scale"])
    if not ref.get("has_tex"):
        raise UnsupportedAssetError("UNSUPPORTED: 无法读取 base-color 贴图")
    texA = ref["texA"]
    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)

    # 逆 preprocess 相似变换: 世界坐标 = (pu 坐标 - cp)/s + co
    co = (np.asarray(orig.vertices).max(0) + np.asarray(orig.vertices).min(0)) / 2
    cp = (V.max(0) + V.min(0)) / 2
    s_align = ref["s_align"]
    to_world = lambda P: (np.asarray(P, float) - cp) / s_align + co

    # ---- 2. 内容分数 -> β(冻结算法) ----
    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"],
                                 ref["ok_map"])
    sel = covered & ref["ok_map"]
    _, w = demand_weights(cw, sel, area, beta=beta)

    mean_cw = [float(np.average(cw[c["gidx"]], weights=area[c["gidx"]]))
               for c in charts]
    big = [i for i, c in enumerate(charts) if len(c["F"]) >= 100]
    top_chart = (max(big, key=lambda i: mean_cw[i]) if big
                 else int(np.argmax(mean_cw)))

    # ---- 3. 输出分辨率(预算平价) + packing(xatlas) + rebake ----
    # auto = 预算平价模式: 搜索最小方形分辨率(16 步长, 非 POT)使
    # 1.00 <= B_signal/B_target <= 1.05; packing 由 xatlas 完成(利用率一等目标),
    # B_signal 一律按 packing 后实测(光栅/bake mask), 不按 R² 推算。
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]

    def _round16(x):
        return max(int(np.ceil(x / 16) * 16), 64)

    def _pack_measure(R):
        uvs, _ = layout_with_scales(charts, w, packer="xatlas", resolution=R)
        owner, _, per_c = rasterize_masks(ch_masks, uvs, R, R)
        return uvs, int((owner >= 0).sum()), per_c

    if atlas_size == "auto":
        uvs_p, _ = layout_with_scales(charts, w, packer="xatlas", resolution=1024)
        owner_p, _, _ = rasterize_masks(ch_masks, uvs_p, 1024, 1024)
        fill_p = max(float((owner_p >= 0).mean()), 1e-6)
        R = min(_round16(np.sqrt(B_target * 1.02 / fill_p)), int(max_atlas))
        uvs_td = per_chart_N = None
        best_ge = None                           # 迭代中见过的最小 ratio>=1.00 解
        for _ in range(6):
            uvs_c, bsig_r, per_c = _pack_measure(R)
            ratio_r = bsig_r / max(B_target, 1)
            if ratio_r >= 1.00 and (best_ge is None or R < best_ge[1]):
                best_ge = (uvs_c, R, per_c, ratio_r)
            if ratio_r < 1.00:
                if R >= int(max_atlas):
                    if best_ge is not None:      # 曾达标: 用最优达标解, 不降质
                        break
                    raise NeedsMultiAtlasError(
                        f"NEEDS_MULTI_ATLAS: max_atlas={max_atlas} 下实测 "
                        f"B_signal={bsig_r:,} < B_target={B_target:,}"
                        f"(ratio={ratio_r:.3f})。本轮不实现 multi-atlas, "
                        f"不静默降质; 可提高 max_atlas。")
                R = min(_round16(R * np.sqrt(1.02 / ratio_r)), int(max_atlas))
                continue
            if ratio_r > 1.05:
                R2 = _round16(R * np.sqrt(1.02 / ratio_r))
                if R2 >= R:                      # 16 步长离散化极限, 接受
                    uvs_td, R_out, per_chart_N = uvs_c, R, per_c
                    break
                R = R2
                continue
            uvs_td, R_out, per_chart_N = uvs_c, R, per_c
            break
        if uvs_td is None:
            # 未在带内收敛: 从最优达标解(ratio>=1.00)向下按 16 步线性细扫,
            # 取最小的仍达标 R(若 16 步粒度下带内不存在, 则略超 1.05 并如实
            # 报告 budget_ratio); 从未达标则不静默降质
            if best_ge is None:
                raise NeedsMultiAtlasError(
                    f"NEEDS_MULTI_ATLAS: 分辨率搜索未能达到 B_target="
                    f"{B_target:,}(max_atlas={max_atlas}); 不静默降质。")
            uvs_td, R_out, per_chart_N, ratio_best = best_ge
            for _ in range(8):
                R2 = R_out - 16
                if R2 < 64 or ratio_best <= 1.05:
                    break
                uvs_c, bsig_r, per_c = _pack_measure(R2)
                ratio_r = bsig_r / max(B_target, 1)
                if ratio_r < 1.00:
                    break
                uvs_td, R_out, per_chart_N, ratio_best = uvs_c, R2, per_c, ratio_r
    else:
        R_out = int(atlas_size)                  # 固定存储预算模式: 严格尊重
        uvs_td, _, per_chart_N = _pack_measure(R_out)

    tex_td, sig_td, filled = bake_atlas_masks(pu, uvs_td, R_out,
                                              face_refuv, valid, texA)
    uvs_uniform, _ = layout_with_scales(charts, np.ones(len(F)),
                                        packer="xatlas", resolution=R_out)
    tex_uniform, sig_uniform, _ = bake_atlas_masks(pu, uvs_uniform, R_out,
                                                   face_refuv, valid, texA)

    out_sig = int(sig_td.sum())
    # E_alloc: 目标需求份额 vs 实际光栅纹素份额的分布误差(L1/2)
    scales_td = chart_scales(charts, w)          # 已有计算, 命名以便暴露给 exporter
    D_c = np.array([(f_ ** 2) * c["a2"]
                    for c, f_ in zip(charts, scales_td)])
    N_c = per_chart_N.astype(float)
    e_alloc = float(0.5 * np.abs(N_c / max(N_c.sum(), 1)
                                 - D_c / max(D_c.sum(), 1e-20)).sum())
    budget = dict(**src_budget,
                  B_target=int(B_target),
                  packer="xatlas",
                  selected_atlas_size=int(R_out),
                  output_B_signal=out_sig,
                  output_B_signal_uniform=int(sig_uniform.sum()),
                  output_packing_fill=round(out_sig / (R_out * R_out), 4),
                  budget_ratio=round(out_sig / max(B_target, 1), 4),
                  budget_met=bool(out_sig >= B_target),
                  E_alloc=round(e_alloc, 5))
    if e_alloc > 0.01:
        warnings.append(f"WARNING: E_alloc={e_alloc * 100:.2f}% 超出 1% 合同, "
                        f"TD 分配保持度不足")
    if not budget["budget_met"]:
        warnings.append(
            f"WARNING: 输出 surface texel budget 未达输入参考"
            f"(budget_ratio={budget['budget_ratio']:.2f}); 固定 atlas_size={R_out} "
            f"是用户主动选择的存储预算, 可能导致外观降级")

    # ---- 4. 未覆盖面: 不静默删除 -> fallback UV + 显式报告 ----
    unc = np.where(~covered)[0]
    fb_uv = None
    if len(unc):
        # 注意: valid 要求 chart 成员资格, 对未覆盖面恒 False —— 这里必须直接用
        # f2o/ok_map 取它们在原贴图上的参考色(未覆盖面通常仍能匹配到原面)
        m = unc[ref["ok_map"][unc]]
        if len(m):
            refuv_unc = ref["uv0"][ref["Fo"][ref["f2o"][m]]]
            fb_color = bilinear(texA, refuv_unc.mean(axis=1)).mean(0)
        else:
            fb_color = texA.reshape(-1, 3).mean(0)
        r0, c0 = _find_empty_block(filled, k=8)
        tex_td[r0:r0 + 8, c0:c0 + 8] = fb_color
        fb_uv = np.array([(c0 + 4) / R_out, 1 - (r0 + 4) / R_out])
        warnings.append(
            f"全部几何面保留; 未覆盖的 "
            f"{area[unc].sum() / area.sum() * 100:.3f}% 面(共 {len(unc)} 个)"
            f"使用近似均色纹素, 外观可能轻微降级; 面号示例 {unc[:5].tolist()}")

    # ---- 5. 组装输出 mesh: 仅在 chart 边界拆点 ----
    Vp, UVp, Fp, off = [], [], [], 0
    for ci, c in enumerate(charts):
        cV = np.asarray(c["V"], float)
        Vp.append(cV)
        UVp.append(uvs_td[ci])
        Fp.append(np.asarray(c["F"], np.int64) + off)
        off += len(cV)
    if len(unc):
        uvid = np.unique(F[unc].ravel())
        remap = np.full(len(V), -1, np.int64)
        remap[uvid] = np.arange(len(uvid)) + off
        Vp.append(V[uvid])
        UVp.append(np.tile(fb_uv, (len(uvid), 1)))
        Fp.append(remap[F[unc]])
        off += len(uvid)
    Vout = to_world(np.concatenate(Vp))
    UVraw = np.concatenate(UVp)
    if not np.isfinite(UVraw).all():          # 必须在 clip 之前(clip 会吞掉 ±inf)
        raise IntegrityError("输出 UV 含非有限值")
    UVout = np.clip(UVraw, 0.0, 1.0)
    Fout = np.concatenate(Fp)

    # ---- 6. 完整性校验(失败不输出) ----
    Vo = np.asarray(orig.vertices, float)
    scale_w = float(np.linalg.norm(Vo.max(0) - Vo.min(0)))
    tris_o, tris_n = Vo[np.asarray(orig.faces)], Vout[Fout]
    area_o = float(np.linalg.norm(np.cross(tris_o[:, 1] - tris_o[:, 0],
                                           tris_o[:, 2] - tris_o[:, 0]), axis=1).sum())
    area_n = float(np.linalg.norm(np.cross(tris_n[:, 1] - tris_n[:, 0],
                                           tris_n[:, 2] - tris_n[:, 0]), axis=1).sum())
    bbox_dev = float(max(np.abs(Vout.min(0) - Vo.min(0)).max(),
                         np.abs(Vout.max(0) - Vo.max(0)).max()) / scale_w)
    checks = dict(
        n_faces_in=int(n_faces_in), n_faces_out=int(len(Fout)),
        n_uncovered_kept=int(len(unc)),
        area_ratio=float(area_n / max(area_o, 1e-20)), bbox_dev=bbox_dev,
        n_vertices_in=int(len(Vo)), n_vertices_out=int(len(Vout)),
        n_vertices_net_increase=int(len(Vout) - len(Vo)))
    if len(Fout) != n_faces_in:
        raise IntegrityError(f"输出面数 {len(Fout)} != 输入面数 {n_faces_in}")
    if abs(checks["area_ratio"] - 1) > AREA_TOL:
        raise IntegrityError(f"表面积偏差 {abs(checks['area_ratio'] - 1) * 100:.3f}%"
                             f" 超限(尺度/transform 恢复失败)")
    if bbox_dev > BBOX_TOL:
        raise IntegrityError(f"包围盒偏差 {bbox_dev:.5f}(相对尺度) 超限")

    # ---- 7. 法线(位置分组光滑) + 材质 + 导出 ----
    normals, n_pos_groups = _grouped_vertex_normals(Vout, Fout, scale_w)
    # 注意: 原始 mesh 自带的重复位置(预拆 seam/孪生面)使 welded 基数 < 输入顶点数,
    # 故相对 welded 基数的增量 ≠ 相对输入顶点数的净增, 两者分开报告
    checks["n_welded_positions"] = int(n_pos_groups)
    checks["n_extra_vs_welded_positions"] = int(len(Vout) - n_pos_groups)

    from PIL import Image
    atlas_img = Image.fromarray((np.clip(tex_td, 0, 1) * 255).astype(np.uint8))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=atlas_img)
    if orig_mat is not None:
        for k in ("baseColorFactor", "metallicFactor", "roughnessFactor",
                  "emissiveFactor", "doubleSided", "alphaMode", "alphaCutoff"):
            v = getattr(orig_mat, k, None)
            if v is not None:
                try:
                    setattr(material, k, v)
                except Exception:
                    pass
        for k in ("metallicRoughnessTexture", "normalTexture",
                  "emissiveTexture", "occlusionTexture"):
            if getattr(orig_mat, k, None) is not None:
                warnings.append(f"原材质含 {k}, 本版本仅重烘 base-color, "
                                f"该贴图未随新 UV 重烘(未携带到输出)")

    mesh_out = trimesh.Trimesh(
        vertices=Vout, faces=Fout, vertex_normals=normals, process=False,
        visual=trimesh.visual.TextureVisuals(uv=UVout, material=material))
    glb_path = os.path.join(output_dir, f"{name}_td_aware.glb")
    atlas_path = os.path.join(output_dir, f"{name}_td_aware_atlas.png")

    # ---- 8. 导出 + 回读校验(校验失败则删除已写文件, 不留下未验证资产) ----
    try:
        mesh_out.export(glb_path)
        atlas_img.save(atlas_path)
        back = trimesh.load(glb_path, force="mesh", process=False)
        back_tex = np.asarray(back.visual.material.baseColorTexture)
        if len(back.faces) != n_faces_in:
            raise IntegrityError(f"回读面数 {len(back.faces)} != {n_faces_in}")
        if back.visual.uv is None or len(back.visual.uv) != len(Vout):
            raise IntegrityError("回读 UV 缺失或长度不符")
        if tuple(back_tex.shape[:2]) != (R_out, R_out):
            raise IntegrityError(f"回读贴图尺寸 {back_tex.shape[:2]} != {R_out}")
    except BaseException:
        for p in (glb_path, atlas_path):
            if os.path.exists(p):
                os.remove(p)
        raise
    checks["reload_ok"] = True

    return dict(
        ok=True, name=name, glb_path=glb_path, atlas_path=atlas_path,
        atlas_size=int(R_out), beta=float(beta), budget=budget,
        integrity=checks, warnings=warnings, support=support,
        # ---- 演示用中间产物 ----
        pu=pu, ref=ref, cw=cw, sel=sel, top_chart=int(top_chart),
        mean_cw=mean_cw, uvs_uniform=uvs_uniform, uvs_td=uvs_td,
        tex_uniform=tex_uniform, tex_td=tex_td, texA=texA,
        face_refuv=face_refuv, valid=valid, to_world=to_world,
        # ---- pseudo-GT exporter 用(均为管线内既有变量, 非重算) ----
        input_mesh=str(input_mesh), td_chart_demand=D_c, td_chart_scales=scales_td)
