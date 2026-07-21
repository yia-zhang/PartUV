# -*- coding: utf-8 -*-
"""192/32/32 拆分 —— 按 object + geometry hash + content hash + 近重复来源分组;
同一分组不得跨 split; calibration 对象(独立 UID 集)已在候选阶段排除。"""
import json
import os

import numpy as np

SPLIT_SIZES = dict(train=192, val=32, test=32)
SEED_SPLIT = 11


def _groups(records):
    """并查集: 相同 geometry_hash / content_phash 相同或汉明<=4 -> 同组."""
    n = len(records)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_geo = {}
    for i, r in enumerate(records):
        by_geo.setdefault(r["geometry_hash"], []).append(i)
    for idxs in by_geo.values():
        for j in idxs[1:]:
            union(idxs[0], j)
    ph = [int(r["content_phash"], 16) for r in records]
    for i in range(n):
        for j in range(i + 1, n):
            if bin(ph[i] ^ ph[j]).count("1") <= 4:
                union(i, j)
    out = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def make_splits(records, sizes=SPLIT_SIZES, seed=SEED_SPLIT):
    groups = _groups(records)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(groups))
    splits = {k: [] for k in sizes}
    quota = dict(sizes)
    seq = ["test", "val", "train"]                  # 先填小集合, 保证锁定集纯净
    for gi in order:
        g = [records[i]["object_id"] for i in groups[gi]]
        tgt = next((s for s in seq if quota[s] >= len(g)), "train")
        splits[tgt].extend(g)
        quota[tgt] = max(quota[tgt] - len(g), 0)
    info = dict(seed=seed, n_groups=len(groups),
                sizes={k: len(v) for k, v in splits.items()},
                rule="object+geometry_hash+content_phash(汉明<=4) 分组, 组不跨 split")
    return splits, info
