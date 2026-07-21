# -*- coding: utf-8 -*-
"""布局: chart 缩放(密度分配) + shelf packing(确定性, P0 用).

layout_with_scales: demand_c = Σ A_3D·w -> f_c = √(demand/A_UV) -> shelf pack
face_td: 逐面相对 TD (线性)
说明: P0 的 e_face/e_chart/e_irreducible 在"等交付预算归一"(metrics.normalize_td2)下
比较, packing 的统一全局缩放不影响结论; blender pack_islands 留在 notebook 演示层.
"""
import numpy as np

from .geometry import tri_area_2d, tri_area_3d


def shelf_pack(rects, pad=0.004):
    order = sorted(range(len(rects)), key=lambda i: -rects[i][1])
    W = max(np.sqrt(sum(a * b for a, b in rects)) * 1.15,
            max(a for a, _ in rects) + 2 * pad)
    x = y = row_h = 0.0
    pos = [None] * len(rects)
    for i in order:
        a, b = rects[i]
        if x + a + pad > W:
            x = 0.0
            y += row_h + pad
            row_h = 0.0
        pos[i] = (x + pad / 2, y + pad / 2)
        x += a + pad
        row_h = max(row_h, b)
    side = max(W, y + row_h + pad)
    return [(px / side, py / side) for px, py in pos], side


class PackingFailedError(RuntimeError):
    pass


def _hull_diameter(uv):
    """凸包直径(最大点对距离)。xatlas 的取向自由度只有"主轴对齐(+90°)",
    主轴对齐后 bbox 长边 ≈ 直径, 故 chart 能刚体放入 atlas 的必要条件用
    直径判定(保守正确; 连续旋转的最小外接正方形会被 45° 长条骗过——
    xatlas 不会用 45° 放置)。"""
    P = np.asarray(uv, float)
    P = P[np.isfinite(P).all(1)]
    if len(P) < 2:
        return 0.0
    try:
        from scipy.spatial import ConvexHull
        P = P[ConvexHull(P).vertices]
    except Exception:
        pass
    if len(P) > 512:
        P = P[:: len(P) // 512 + 1]
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()))


def xatlas_pack(charts, scales, resolution=1024, padding_px=4, rotate=True):
    """existing-UV repack(生产 packer): TD 目标面积由 scales 决定, xatlas 只做
    chart 旋转+平移(texels_per_unit=1), 外加一次全部 charts 共享的全局缩放
    (由本函数以"最大化利用率"为目标搜索: prefill 阶梯下探 + 二分细化)。

    送入约定(经合同测试确立):
    - 全部 charts 拼**单一** uv mesh(逐 mesh 送入会被 binding 逐 mesh 归一化,
      摧毁相对面积); 各 chart 先平移到不重叠网格(纯平移, 供 xatlas 按索引
      连通性识别 chart);
    - 输入换算到纹素单位, texels_per_unit=1 ⇒ xatlas 内部零缩放;
    - get_mesh 返回的 uv 已按 atlas 尺寸归一。
    已知量化: xatlas 将 chart 尺寸对齐到整数纹素(每维 ~±1px 进入输出 UV),
    逐微小 chart 面积误差不可避免; TD 保持按 E_alloc(分布 L1/2)口径验收。
    失败(多 atlas/chart 数改变/越界)抛 PackingFailedError, 无静默回退。"""
    import xatlas

    scaled = [np.asarray(c["UV"], float) * f for c, f in zip(charts, scales)]
    Fs = [np.asarray(c["F"], np.int64) for c in charts]
    areas = np.array([tri_area_2d(uv[F]).sum() for uv, F in zip(scaled, Fs)])
    tot = max(float(areas.sum()), 1e-12)
    nv = [len(u) for u in scaled]
    # 超尺寸预检: chart 凸包直径超过 atlas 可用边长时, xatlas 会**静默逐 chart
    # 缩小**(违反刚体合同), 必须在送入前判该 prefill 不可行
    diam = np.array([_hull_diameter(uv) for uv in scaled])
    maxdim = np.array([float(np.ptp(uv, axis=0).max()) for uv in scaled])
    gap = max(2.0 * padding_px, 8.0)     # 预分离间隙下限(pad=0 时 chart 会被几何合并)

    def attempt(prefill):
        s = np.sqrt(prefill * resolution * resolution / tot)
        if float((diam * s).max()) > resolution - 2 * padding_px - 2:
            return None                  # 有 chart 刚体放不进 atlas -> 全局降 prefill
        uvs_in, faces, off = [], [], 0
        x = y = rh = 0.0
        Wg = max(np.sqrt(tot) * 1.6 * s, 1.0)
        for uv, F in zip(scaled, Fs):
            u = uv * s
            u = u - u.min(0)
            w, h = u.max(0) + gap
            if x + w > Wg and x > 0:
                x, y, rh = 0.0, y + rh, 0.0
            uvs_in.append(u + [x, y])
            x += w
            rh = max(rh, h)
            faces.append(F + off)
            off += len(uv)
        UV = np.ascontiguousarray(np.concatenate(uvs_in), np.float32)
        FF = np.ascontiguousarray(np.concatenate(faces), np.uint32)
        a = xatlas.Atlas()
        a.add_uv_mesh(UV, FF)
        po = xatlas.PackOptions()
        po.padding = int(padding_px)
        po.resolution = int(resolution)
        po.rotate_charts = bool(rotate)
        po.texels_per_unit = 1.0
        a.generate(pack_options=po, verbose=False)
        if not (a.atlas_count == 1 and a.chart_count == len(charts)
                and max(a.width, a.height) <= resolution):
            return None
        vmap, idx, uvp = a.get_mesh(0)
        if not np.array_equal(vmap[idx], FF):
            return None
        uvn = np.zeros((off, 2))
        uvn[vmap] = uvp                      # binding 已按 atlas 尺寸归一
        if (not np.isfinite(uvn).all()
                or uvn.min() < -1e-6 or uvn.max() > 1 + 1e-6):
            return None
        out, o2 = [], 0
        for n in nv:
            out.append(uvn[o2:o2 + n])
            o2 += n
        # 刚体后验(兜底): xatlas 对放不下的 chart 会静默**缩小**(只会变小不会变大)。
        # 参照 = 解析期望值 (s/resolution)²(纯刚体+全局归一下的精确面积比, 无需
        # 经验参照集——单个大 chart 也能检); 只检量化噪声小的中大 chart(>5%R),
        # 缩小超过 3% 即视为发生了逐 chart 缩放 -> 判不可行。
        r = np.array([tri_area_2d(o[F]).sum() / max(a, 1e-20)
                      for o, F, a in zip(out, Fs, areas)])
        expected = (s / resolution) ** 2
        mid = (maxdim * s) > 0.05 * resolution
        if mid.any() and float((r[mid] / expected).min()) < 0.97:
            return None
        return out

    best, pf_ok, pf_fail = None, None, 0.92
    for pf in np.arange(0.78, 0.17, -0.06):
        r = attempt(float(pf))
        if r is not None:
            best, pf_ok = r, float(pf)
            break
        pf_fail = float(pf)
    if best is None:
        raise PackingFailedError(
            f"PACKING_FAILED: xatlas 在 prefill 0.18-0.78 全部无法装入单一 "
            f"{resolution}² atlas(charts={len(charts)})")
    lo, hi = pf_ok, pf_fail                  # 二分细化最大可行 prefill
    for _ in range(4):
        mid = (lo + hi) / 2
        r = attempt(mid)
        if r is not None:
            lo, best = mid, r
        else:
            hi = mid
    return best


def chart_scales(charts, face_weight):
    """每 chart 的目标线性缩放因子 f_c (相对其当前 UV)."""
    out = []
    for c in charts:
        cF = np.asarray(c["F"])
        a3 = tri_area_3d(np.asarray(c["V"])[cF])
        demand = float((a3 * face_weight[c["gidx"]]).sum())
        out.append(np.sqrt(demand / max(c["a2"], 1e-12)))
    return np.asarray(out)


def layout_with_scales(charts, face_weight, pad=0.004, packer="xatlas",
                       resolution=1024, padding_px=4):
    """chart_scales -> packing. packer="xatlas"(生产默认, 利用率一等目标) /
    "shelf"(仅显式诊断基线; pad 参数只作用于 shelf)。无静默回退。"""
    scales = chart_scales(charts, face_weight)
    if packer == "xatlas":
        return xatlas_pack(charts, scales, resolution=resolution,
                           padding_px=padding_px), scales
    if packer != "shelf":
        raise ValueError(f"未知 packer: {packer!r} (可选 'xatlas' / 'shelf')")
    uvs, rects = [], []
    for c, f in zip(charts, scales):
        uv = np.asarray(c["UV"]) * f
        uv = uv - uv.min(axis=0)
        uvs.append(uv)
        rects.append(tuple(uv.max(axis=0)))
    off, side = shelf_pack(rects, pad)
    return [uv / side + np.array(o) for uv, o in zip(uvs, off)], scales


def face_td(charts, uvs, n_faces):
    td = np.zeros(n_faces)
    for c, uv in zip(charts, uvs):
        cF = np.asarray(c["F"])
        a2 = tri_area_2d(uv[cF])
        a3 = tri_area_3d(np.asarray(c["V"])[cF])
        td[c["gidx"]] = np.sqrt(a2 / np.maximum(a3, 1e-16))
    return td
