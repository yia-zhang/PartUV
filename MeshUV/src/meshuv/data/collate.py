# -*- coding: utf-8 -*-
"""variable-chart collate: 逐 chart 特征来自原始可用信息(严禁诊断字段).

per-chart 特征(纯几何+原始颜色统计):
- log(chart surface-area fraction)
- log(baseline UV area fraction)
- chart 面数(log)
- 归一化质心(3) + 法线均值(3)
- 从 source RGB 采样的颜色均值(3)/std(3)   <- 原始颜色, 非 teacher 分数
"""
import numpy as np
from PIL import Image


def chart_features(item, tex_cache={}):
    z, tt = item["inputs"], item["targets"]
    f2c, fa = z["face_to_chart"], z["face_area"].astype(float)
    nC = len(tt["chart_surface_area"])
    m = f2c >= 0
    V, F = z["vertices"].astype(float), z["faces"]
    ctr = V[F].mean(1)
    e1, e2 = V[F][:, 1] - V[F][:, 0], V[F][:, 2] - V[F][:, 0]
    nrm = np.cross(e1, e2)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    # baseline UV 面积(local_uv)
    luv = z["local_uv"].astype(float)
    a2f = np.abs(np.cross(luv[:, 1] - luv[:, 0], luv[:, 2] - luv[:, 0])) / 2
    # 原始颜色采样(面质心 uv, nearest)
    p = item["basecolor"]
    if p not in tex_cache:
        tex_cache[p] = np.asarray(Image.open(p), float)[:, :, :3] / 255.0
        if len(tex_cache) > 16:
            tex_cache.pop(next(iter(tex_cache)))
    tex = tex_cache[p]
    suv = z["source_uv"].astype(float).mean(1)
    H, W = tex.shape[:2]
    xi = np.clip((suv[:, 0] * W).astype(int), 0, W - 1)
    yi = np.clip(((1 - suv[:, 1]) * H).astype(int), 0, H - 1)
    rgb = tex[yi, xi]

    def agg(vals, weights=None, how="mean"):
        out = np.zeros((nC,) + vals.shape[1:])
        wsum = np.zeros(nC)
        w = fa if weights is None else weights
        np.add.at(out, f2c[m], (vals[m].T * w[m]).T)
        np.add.at(wsum, f2c[m], w[m])
        return (out.T / np.maximum(wsum, 1e-12)).T

    A3 = tt["chart_surface_area"].astype(float)
    a2c = np.zeros(nC)
    np.add.at(a2c, f2c[m], a2f[m])
    nfc = np.bincount(f2c[m], minlength=nC).astype(float)
    bbox = V.max(0) - V.min(0)
    ctr_n = (agg(ctr) - V.min(0)) / np.maximum(bbox.max(), 1e-12)
    feats = np.concatenate([
        np.log(np.maximum(A3 / max(A3.sum(), 1e-12), 1e-9))[:, None],
        np.log(np.maximum(a2c / max(a2c.sum(), 1e-12), 1e-9))[:, None],
        np.log(np.maximum(nfc, 1))[:, None],
        ctr_n, agg(nrm), agg(rgb),
        agg((rgb - agg(rgb)[f2c]) ** 2) ** 0.5,
    ], axis=1)
    return feats.astype(np.float32)


def collate(items):
    """扁平 chart tokens + object 边界(变长)."""
    X, y, valid, obj_ix, obj_ranges = [], [], [], [], []
    p = 0
    for oi, it in enumerate(items):
        f = chart_features(it)
        X.append(f)
        y.append(it["targets"]["chart_log_density_ratio"])
        valid.append(it["targets"]["chart_valid_mask"])
        obj_ix.append(np.full(len(f), oi))
        obj_ranges.append((p, p + len(f)))
        p += len(f)
    return dict(features=np.concatenate(X).astype(np.float32),
                target=np.concatenate(y).astype(np.float32),
                valid=np.concatenate(valid).astype(bool),
                object_index=np.concatenate(obj_ix),
                object_ranges=obj_ranges)
