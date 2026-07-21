# -*- coding: utf-8 -*-
"""最小 dataset loader —— 按角色分组返回, 强制输入禁用名单."""
import json
import os

import numpy as np

from .schema import (FORBIDDEN_INPUTS, MODEL_INPUTS, TEACHER_DIAGNOSTICS,
                     TRAINING_TARGETS)


class MeshUVTDDataset:
    """索引自 dataset_index.jsonl; 路径均相对 dataset root(可迁移)."""

    def __init__(self, root, split=None, splits_file="splits.json",
                 expose_diagnostics=False):
        self.root = os.path.abspath(root)
        self.expose_diagnostics = expose_diagnostics
        idx = [json.loads(l) for l in
               open(os.path.join(self.root, "dataset_index.jsonl"))
               if l.strip()]
        if split is not None:
            sp = json.load(open(os.path.join(self.root, splits_file)))
            keep = set(sp[split])
            idx = [r for r in idx if r["object_id"] in keep]
        self.index = idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        rec = self.index[i]
        d = os.path.join(self.root, rec["sample_dir"])
        z = dict(np.load(os.path.join(d, "arrays.npz")))
        man = json.load(open(os.path.join(d, "manifest.json")))
        item = dict(
            object_id=rec["object_id"],
            model_inputs={k: z[k] for k in MODEL_INPUTS},
            training_targets={k: z[k] for k in TRAINING_TARGETS},
            qa_artifacts=dict(quality_json=os.path.join(d, "quality.json")),
            reference_texture=os.path.join(d, man["files"]["reference_texture"]),
            manifest=man)
        if self.expose_diagnostics:
            item["teacher_diagnostics"] = {k: z[k] for k in TEACHER_DIAGNOSTICS}
        # 防泄漏: 禁用名单绝不进入 model_inputs
        leak = FORBIDDEN_INPUTS & set(item["model_inputs"])
        assert not leak, f"标签泄漏字段进入输入: {leak}"
        return item
