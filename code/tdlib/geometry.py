# -*- coding: utf-8 -*-
"""几何基元与面对应.

FaceMatcher: 孪生面(双面重合几何)安全的一对一面匹配
  - 质心分桶 + 候选>1 时按法线方向判别 + kd 兜底
  - P0.5: match_charts() 全局唯一性断言 (len(unique)==expected)
best_corner_perm: 三角形角点对应取 3! 全排列最小总代价双射 (P0.5, 禁止独立 argmin)
"""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def tri_area_2d(uv):
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])


def tri_area_3d(v):
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)


def face_normals(V, F):
    n = np.cross(V[F][:, 1] - V[F][:, 0], V[F][:, 2] - V[F][:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-16)


def best_corner_perm(src3, dst3):
    """src3, dst3: (3,3) 角点坐标. 返回使 Σ|src[perm[k]]-dst[k]|² 最小的 perm."""
    best, best_cost = _PERMS[0], np.inf
    for p in _PERMS:
        cost = float(((src3[list(p)] - dst3) ** 2).sum())
        if cost < best_cost:
            best, best_cost = p, cost
    return best, best_cost


class FaceMatcher:
    def __init__(self, V, F, scale=None, decimals=6):
        self.cent = V[F].mean(axis=1)
        self.nrm = face_normals(V, F)
        self.kd = cKDTree(self.cent)
        self.scale = float(scale if scale is not None
                           else np.linalg.norm(V.max(0) - V.min(0)))
        self.decimals = decimals
        self._fresh_buckets()

    def _fresh_buckets(self):
        self.buckets = defaultdict(list)
        for i, k in enumerate(map(tuple, np.round(self.cent / self.scale, self.decimals))):
            self.buckets[k].append(i)

    def match(self, cV, cF):
        """匹配一组面(如一个 chart); 桶内弹出保证跨调用一对一."""
        cents = cV[cF].mean(axis=1)
        nrms = face_normals(cV, cF)
        out = np.empty(len(cF), dtype=int)
        for j, k in enumerate(map(tuple, np.round(cents / self.scale, self.decimals))):
            lst = self.buckets.get(k)
            if lst:
                if len(lst) == 1:
                    out[j] = lst.pop()
                else:
                    dots = [float(self.nrm[i] @ nrms[j]) for i in lst]
                    out[j] = lst.pop(int(np.argmax(dots)))
            else:
                out[j] = int(self.kd.query(cents[j][None])[1][0])
        return out

    def match_charts(self, charts_VF, tol_rel=1e-5):
        """匹配全部 charts 并做 P0.5 校验.
        charts_VF: [(cV, cF), ...]
        返回 (list_of_gidx, report dict). 全局唯一性不满足时 report['unique']=False."""
        self._fresh_buckets()
        gidxs, all_idx, mismatch = [], [], 0
        for cV, cF in charts_VF:
            g = self.match(cV, cF)
            d = np.linalg.norm(cV[cF].mean(axis=1) - self.cent[g], axis=1)
            mismatch += int((d > tol_rel * self.scale).sum())
            gidxs.append(g)
            all_idx.append(g)
        all_idx = np.concatenate(all_idx) if all_idx else np.empty(0, int)
        unique_ok = len(np.unique(all_idx)) == len(all_idx)
        return gidxs, dict(n_matched=int(len(all_idx)),
                           n_unique=int(len(np.unique(all_idx))),
                           unique=bool(unique_ok),
                           mismatch=int(mismatch))
