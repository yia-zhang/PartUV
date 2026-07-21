# -*- coding: utf-8 -*-
"""可选 GPU 加速后端 (torch): 光栅化 / rebake / 预览渲染.

与 numpy 参考实现语义一致(相同覆盖判定阈值, float64), 无 torch 或无可用
GPU 时调用方回退纯 numpy 路径(通过 available() 判断)。torch 延迟导入,
不增加 tdlib 的导入开销。
"""
import os
import subprocess

import numpy as np

_TORCH = None          # 未探测=None, 不可用=False, 可用=torch 模块


def pick_free_gpu(verbose=True):
    """在导入 torch 之前调用: 选取 (利用率, 显存占用) 最低的 GPU 并设置
    CUDA_VISIBLE_DEVICES(已设置则尊重现有值)。选不出来时静默返回。"""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        rows = [[int(x) for x in ln.split(",")] for ln in out.strip().splitlines()]
        idx = min(rows, key=lambda r: (r[1], r[2]))[0]
    except Exception:
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    if verbose:
        print(f"[tdlib.gpu] 使用空闲 GPU {idx} (CUDA_VISIBLE_DEVICES={idx})")
    return str(idx)


def _torch():
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            _TORCH = torch if torch.cuda.is_available() else False
        except Exception:
            _TORCH = False
    return _TORCH


def available():
    return bool(_torch())


# ---------------------------------------------------------------- 光栅化核心
def _raster_pairs(P, W, H, eps):
    """枚举 (三角形, bbox 内像素) 对并做覆盖判定, 与 CPU 版逐三角循环同数学.
    P: (n,3,2) float64 cuda tensor(像素坐标). 返回 (tri, px, py, bary) 已过滤.
    """
    t = _torch()
    dev = P.device
    n = len(P)
    lim = t.tensor([W - 1, H - 1], dtype=t.float64, device=dev)
    bb_mn = t.clamp(t.floor(P.amin(1)), min=0)
    bb_mx = t.minimum(t.ceil(P.amax(1)), lim)
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    det = e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]
    good = (det.abs() >= 1e-12) & (bb_mx >= bb_mn).all(1)
    wh = (bb_mx - bb_mn + 1).to(t.int64)
    wh[~good] = 0
    counts = (wh[:, 0] * wh[:, 1]).clamp(min=0)
    tot = int(counts.sum())
    if tot == 0:
        z = t.zeros(0, dtype=t.int64, device=dev)
        return z, z, z, t.zeros((0, 3), dtype=t.float64, device=dev)
    tri = t.repeat_interleave(t.arange(n, device=dev), counts)
    off = counts.cumsum(0) - counts
    loc = t.arange(tot, device=dev) - off[tri]
    w_ = wh[tri, 0]
    px = bb_mn[tri, 0].to(t.int64) + loc % w_
    py = bb_mn[tri, 1].to(t.int64) + loc // w_
    dx = (px.to(t.float64) + 0.5) - P[tri, 0, 0]
    dy = (py.to(t.float64) + 0.5) - P[tri, 0, 1]
    dt = det[tri]
    w1 = (dx * e2[tri, 1] - dy * e2[tri, 0]) / dt          # invT 展开, 同 CPU 公式
    w2 = (-dx * e1[tri, 1] + dy * e1[tri, 0]) / dt
    w0 = 1 - w1 - w2
    m = (w1 >= eps) & (w2 >= eps) & (w0 >= eps)
    return tri[m], px[m], py[m], t.stack([w0[m], w1[m], w2[m]], 1)


def _bilinear_t(img, uv):
    """bilinear 采样, 同 rd.bilinear: texel-center 约定 x=u*W-0.5,
    clamp-to-edge, v=1 在顶行(Coordinate Rebaseline, 与光栅化中心一致)."""
    t = _torch()
    H, W = img.shape[:2]
    x = t.clamp(uv[:, 0], 0, 1) * W - 0.5
    y = t.clamp(1 - uv[:, 1], 0, 1) * H - 0.5
    x0f = t.floor(x); y0f = t.floor(y)
    fx = (x - x0f)[:, None]; fy = (y - y0f)[:, None]
    x0 = t.clamp(x0f.to(t.int64), 0, W - 1)
    y0 = t.clamp(y0f.to(t.int64), 0, H - 1)
    x1 = t.clamp(x0f.to(t.int64) + 1, 0, W - 1)
    y1 = t.clamp(y0f.to(t.int64) + 1, 0, H - 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def _last_write_wins(p, order_rank):
    """模拟顺序写入(后写覆盖先写): 返回每个像素最后一次写入的 pair 下标.
    p: 扁平像素 id; order_rank: 写入顺序(越大越晚)."""
    t = _torch()
    idx = t.argsort(order_rank)
    ps = p[idx]
    srt2 = t.argsort(ps, stable=True)
    ps2 = ps[srt2]
    keep = t.ones(len(ps2), dtype=t.bool, device=p.device)
    if len(ps2) > 1:
        keep[:-1] = ps2[:-1] != ps2[1:]
    return idx[srt2[keep]]


# ---------------------------------------------------------------- rebake
def bake_atlas_masks(pu, uvs, R, face_refuv, valid, texA, dilate_iters=2):
    """rd.bake_atlas_masks 的 GPU 等价实现(覆盖阈值 -1e-4, 后写覆盖先写)."""
    t = _torch()
    dev = "cuda"
    charts = pu["charts"]
    tris_list, refuv_list = [], []
    for c, uvc in zip(charts, uvs):
        cF = np.asarray(c["F"])
        g = c["gidx"]
        keep = valid[g]
        if not keep.any():
            continue
        uvp = uvc[cF[keep]]
        tris_list.append(np.stack([uvp[:, :, 0] * R, (1 - uvp[:, :, 1]) * R], -1))
        refuv_list.append(face_refuv[g[keep]])
    tex = t.zeros((R, R, 3), dtype=t.float64, device=dev)
    filled = t.zeros((R, R), dtype=t.bool, device=dev)
    if tris_list:
        P = t.as_tensor(np.concatenate(tris_list), dtype=t.float64, device=dev)
        refuv = t.as_tensor(np.concatenate(refuv_list), dtype=t.float64, device=dev)
        texT = t.as_tensor(texA, dtype=t.float64, device=dev)
        tri, px, py, bary = _raster_pairs(P, R, R, -1e-4)
        if len(tri):
            uv = t.einsum("nk,nkd->nd", bary, refuv[tri])
            col = _bilinear_t(texT, uv)
            win = _last_write_wins(py * R + px, tri)
            tex[py[win], px[win]] = col[win]
            filled[py, px] = True
    signal = filled.clone()
    tex, filled = _dilate_t(tex, filled, dilate_iters)
    return (tex.cpu().numpy(), signal.cpu().numpy(), filled.cpu().numpy())


def _dilate_t(tex, filled, iters):
    """rd._dilate_colors 的 GPU 等价实现(4 邻域平均)."""
    t = _torch()
    for _ in range(iters):
        empty = ~filled
        acc = t.zeros_like(tex)
        cnt = t.zeros(filled.shape, dtype=t.float64, device=tex.device)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sf = t.roll(filled, shifts=(dy, dx), dims=(0, 1))
            st = t.roll(tex, shifts=(dy, dx), dims=(0, 1))
            m = (sf & empty).to(t.float64)
            acc += st * m[..., None]
            cnt += m
        upd = cnt > 0
        tex[upd] = acc[upd] / cnt[upd][:, None]
        filled |= upd
    return tex, filled


# ---------------------------------------------------------------- owner/overlap
def rasterize_masks(charts, uvs, W, H):
    """budget.rasterize_masks 的 GPU 等价实现(阈值 -1e-6, 语义逐写入等价):
    owner=最后写入者; overlap=写入时已被其他 chart 占有的次数;
    per_chart[ci]=ci 的写入次数 - ci 的 clash 次数."""
    t = _torch()
    dev = "cuda"
    tris_list, cid_list = [], []
    for ci, (c, uvc) in enumerate(zip(charts, uvs)):
        cF = np.asarray(c["F"])
        if len(cF) == 0:
            continue
        uvp = uvc[cF]
        tris_list.append(np.stack([uvp[:, :, 0] * W, (1 - uvp[:, :, 1]) * H], -1))
        cid_list.append(np.full(len(cF), ci))
    owner = t.full((H, W), -1, dtype=t.int32, device=dev)
    per_chart = np.zeros(len(charts), np.int64)
    if not tris_list:
        return owner.cpu().numpy(), 0, per_chart
    P = t.as_tensor(np.concatenate(tris_list), dtype=t.float64, device=dev)
    cid = t.as_tensor(np.concatenate(cid_list), dtype=t.int64, device=dev)
    tri, px, py, _ = _raster_pairs(P, W, H, -1e-6)
    if not len(tri):
        return owner.cpu().numpy(), 0, per_chart
    p = py * W + px
    ch = cid[tri]
    srt = t.argsort(tri)                       # 写入顺序 = 三角形顺序
    srt = srt[t.argsort(p[srt], stable=True)]  # 按像素分组, 组内按写入顺序
    ps, cs = p[srt], ch[srt]
    same_p = t.zeros(len(ps), dtype=t.bool, device=dev)
    same_p[1:] = ps[1:] == ps[:-1]
    clash = same_p & (cs != t.roll(cs, 1))
    overlap = int(clash.sum())
    win = _last_write_wins(p, tri)
    owner.view(-1)[p[win]] = ch[win].to(t.int32)
    writes = t.bincount(cs, minlength=len(charts))
    clashes = t.bincount(cs[clash], minlength=len(charts))
    per_chart = (writes - clashes).cpu().numpy().astype(np.int64)
    return owner.cpu().numpy(), overlap, per_chart


def island_budget(Fo, uv, W, H, face_lbl):
    """api._island_budget 的 GPU 等价实现(阈值 -1e-6):
    B_unique=被任一岛覆盖的纹素数; B_surface=各岛 mask 纹素数之和."""
    t = _torch()
    dev = "cuda"
    uvc = np.clip(np.asarray(uv, float), 0, 1)
    tris = np.stack([uvc[Fo][:, :, 0] * W, (1 - uvc[Fo][:, :, 1]) * H], -1)
    P = t.as_tensor(tris, dtype=t.float64, device=dev)
    lbl = t.as_tensor(np.asarray(face_lbl), dtype=t.int64, device=dev)
    tri, px, py, _ = _raster_pairs(P, W, H, -1e-6)
    if not len(tri):
        return 0, 0
    key = (py * W + px) * (int(lbl.max()) + 1) + lbl[tri]
    uk = t.unique(key)                          # (像素, 岛) 去重
    b_surface = int(len(uk))
    b_unique = int(len(t.unique(t.div(uk, int(lbl.max()) + 1, rounding_mode="floor"))))
    return b_unique, b_surface


# ---------------------------------------------------------------- 预览渲染
def textured_render(V, F, uv_faces, ok, tex, view=(15, 45), px=900, pad=6):
    """GPU 逐像素纹理采样的正交预览渲染(替代 64× 细分 + matplotlib).
    视觉约定同 gen_dashboard_assets.render3d: 背面剔除 + 同款着色, 白底,
    按内容紧裁. ok=False 的面置灰 0.6. 返回 (h,w,3) float 数组."""
    t = _torch()
    dev = "cuda"
    V3 = t.as_tensor(np.asarray(V, float), dtype=t.float64, device=dev)
    Ft = t.as_tensor(np.asarray(F), dtype=t.int64, device=dev)
    UV = t.as_tensor(np.asarray(uv_faces, float), dtype=t.float64, device=dev)
    okt = t.as_tensor(np.asarray(ok, bool), device=dev)
    texT = t.as_tensor(np.asarray(tex, float), dtype=t.float64, device=dev)

    e, a = np.radians(view[0]), np.radians(view[1])
    cam = t.tensor([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
                   dtype=t.float64, device=dev)
    up = t.tensor([0.0, 0.0, 1.0], dtype=t.float64, device=dev)
    r = t.linalg.cross(up, cam); r = r / t.clamp(t.linalg.norm(r), min=1e-12)
    u = t.linalg.cross(cam, r)

    tris = V3[Ft]                                            # (nF,3,3)
    n = t.linalg.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n = n / t.clamp(t.linalg.norm(n, dim=1, keepdim=True), min=1e-12)
    keep = (n @ cam) > 1e-9                                  # 背面剔除(同 render3d)
    if int(keep.sum()) <= len(tris) * 0.1:                   # 绕向不一致保护
        keep = t.ones(len(tris), dtype=t.bool, device=dev)
    tris, n, UVk, okk = tris[keep], n[keep], UV[keep], okt[keep]

    X = tris @ r; Y = tris @ u; Z = tris @ cam               # (nK,3) 屏幕系
    mnx, mxx = float(X.min()), float(X.max())
    mny, mxy = float(Y.min()), float(Y.max())
    scale = (px - 2 * pad) / max(mxx - mnx, mxy - mny, 1e-12)
    W = int((mxx - mnx) * scale) + 2 * pad
    H = int((mxy - mny) * scale) + 2 * pad
    Ppx = t.stack([(X - mnx) * scale + pad, (mxy - Y) * scale + pad], -1)

    tri, ix, iy, bary = _raster_pairs(Ppx, W, H, -1e-6)
    img = t.ones((H, W, 3), dtype=t.float64, device=dev)
    if len(tri):
        depth = (bary * Z[tri]).sum(1)
        win = _last_write_wins(iy * W + ix, depth)           # 深度大 = 靠近相机
        tri, ix, iy, bary = tri[win], ix[win], iy[win], bary[win]
        light = t.tensor([0.4, 0.5, 0.77], dtype=t.float64, device=dev)
        shade = 0.72 + 0.28 * t.abs(n[tri] @ light)
        col = t.full((len(tri), 3), 0.6, dtype=t.float64, device=dev)
        m = okk[tri]
        if int(m.sum()):
            uv = t.einsum("nk,nkd->nd", bary[m], UVk[tri[m]])
            col[m] = _bilinear_t(texT, uv)
        img[iy, ix] = t.clamp(col * shade[:, None], 0, 1)
    out = img.cpu().numpy()
    fg = (out < 0.995).any(axis=2)                           # 按内容紧裁(同 render_img)
    ys, xs = np.where(fg)
    if len(ys) == 0:
        return out
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, out.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, out.shape[1])
    return out[y0:y1, x0:x1]


def visible_weight(V, F, w, view, px=96, pad=2):
    """正交视角下 z-buffer 实测可见性: Σ(胜出像素的所属面权重 w).
    像素计数天然给出投影面积因子, 遮挡被深度解析处理;
    用于遮挡感知选视角(gen_dashboard_assets.facing_view)."""
    t = _torch()
    dev = "cuda"
    V3 = t.as_tensor(np.asarray(V, float), dtype=t.float64, device=dev)
    Ft = t.as_tensor(np.asarray(F), dtype=t.int64, device=dev)
    wt = t.as_tensor(np.asarray(w, float), dtype=t.float64, device=dev)
    e, a = np.radians(view[0]), np.radians(view[1])
    cam = t.tensor([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)],
                   dtype=t.float64, device=dev)
    up = t.tensor([0.0, 0.0, 1.0], dtype=t.float64, device=dev)
    r = t.linalg.cross(up, cam); r = r / t.clamp(t.linalg.norm(r), min=1e-12)
    u = t.linalg.cross(cam, r)
    tris = V3[Ft]
    X = tris @ r; Y = tris @ u; Z = tris @ cam
    mnx, mxx = float(X.min()), float(X.max())
    mny, mxy = float(Y.min()), float(Y.max())
    scale = (px - 2 * pad) / max(mxx - mnx, mxy - mny, 1e-12)
    W = int((mxx - mnx) * scale) + 2 * pad
    H = int((mxy - mny) * scale) + 2 * pad
    Ppx = t.stack([(X - mnx) * scale + pad, (mxy - Y) * scale + pad], -1)
    tri, ix, iy, bary = _raster_pairs(Ppx, W, H, -1e-6)
    if not len(tri):
        return 0.0
    depth = (bary * Z[tri]).sum(1)
    win = _last_write_wins(iy * W + ix, depth)               # 深度大 = 靠近相机
    return float(wt[tri[win]].sum())
