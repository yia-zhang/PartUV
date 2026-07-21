# -*- coding: utf-8 -*-
"""Clean V1 loader: 按角色分组; 诊断字段仅显式请求时暴露; 输入防泄漏断言."""
import glob
import json
import os

import numpy as np

from .schema import CORE, TARGETS, DIAGNOSTICS, FORBIDDEN_INPUTS


class CleanDataset:
    def __init__(self, root, object_ids=None, expose_diagnostics=False):
        self.root = os.path.abspath(root)
        self.expose = expose_diagnostics
        ids = object_ids
        if ids is None:
            ids = []
            for f in sorted(glob.glob(f"{self.root}/objects/*/manifest.json")):
                d = os.path.dirname(f)
                try:
                    st = json.load(open(f"{d}/status.json"))
                    if st.get("status") != "ACCEPTED":
                        continue
                    with np.load(f"{d}/arrays.npz") as z:
                        if set(CORE + TARGETS) - set(z.files):
                            continue          # schema 不完整不加载
                    ids.append(os.path.basename(d))
                except Exception:
                    continue
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = f"{self.root}/objects/{self.ids[i]}"
        z = dict(np.load(f"{d}/arrays.npz"))
        man = json.load(open(f"{d}/manifest.json"))
        item = dict(object_id=self.ids[i],
                    inputs={k: (z[k] if k in z else
                                np.zeros(len(z["faces"]), np.int64))
                            for k in CORE},   # 旧样本缺 face_source -> 0
                    targets={k: z[k] for k in TARGETS},
                    basecolor=f"{d}/basecolor.png", manifest=man)
        if self.expose:
            item["diagnostics"] = {k: z[k] for k in DIAGNOSTICS}
        assert not (FORBIDDEN_INPUTS & set(item["inputs"])), "输入泄漏"
        return item


def object_splits(root, sizes=(0.75, 0.125, 0.125), seed=11):
    """object 级拆分: geometry_hash 与 content_hash 双重隔离(并查集),
    同组不跨集; 禁止按 chart 切。返回 splits(附 _dup_audit)."""
    ds = CleanDataset(root)
    mans = [json.load(open(f"{root}/objects/{o}/manifest.json"))
            for o in ds.ids]
    parent = list(range(len(ds.ids)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for key in ("geometry_hash", "content_hash"):
        by = {}
        for i, man in enumerate(mans):
            by.setdefault(man.get(key, f"_{i}"), []).append(i)
        for idxs in by.values():
            for j in idxs[1:]:
                ra, rb = find(idxs[0]), find(j)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for i, oid in enumerate(ds.ids):
        groups.setdefault(find(i), []).append(oid)
    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    groups = {k: v for k, v in groups.items()}
    keys = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(keys)
    n = len(ds.ids)
    out, quota = {"train": [], "val": [], "test": []}, \
        {"train": sizes[0] * n, "val": sizes[1] * n, "test": sizes[2] * n}
    for k in keys:
        g = groups[k]
        tgt = max(quota, key=lambda s: quota[s])
        out[tgt].extend(g)
        quota[tgt] -= len(g)
    out["_dup_audit"] = dict(n_dup_groups=len(dup_groups),
                             dup_examples={str(k): v for k, v in
                                           list(dup_groups.items())[:5]})
    return out
