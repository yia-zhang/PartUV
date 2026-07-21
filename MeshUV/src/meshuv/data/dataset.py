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
            ids = sorted(os.path.basename(os.path.dirname(f)) for f in
                         glob.glob(f"{self.root}/objects/*/manifest.json"))
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        d = f"{self.root}/objects/{self.ids[i]}"
        z = dict(np.load(f"{d}/arrays.npz"))
        man = json.load(open(f"{d}/manifest.json"))
        item = dict(object_id=self.ids[i],
                    inputs={k: z[k] for k in CORE},
                    targets={k: z[k] for k in TARGETS},
                    basecolor=f"{d}/basecolor.png", manifest=man)
        if self.expose:
            item["diagnostics"] = {k: z[k] for k in DIAGNOSTICS}
        assert not (FORBIDDEN_INPUTS & set(item["inputs"])), "输入泄漏"
        return item


def object_splits(root, sizes=(0.75, 0.125, 0.125), seed=11):
    """按 geometry_hash 分组的 object 级拆分(禁止按 chart 切)."""
    ds = CleanDataset(root)
    groups = {}
    for oid in ds.ids:
        man = json.load(open(f"{root}/objects/{oid}/manifest.json"))
        groups.setdefault(man["geometry_hash"], []).append(oid)
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
    return out
