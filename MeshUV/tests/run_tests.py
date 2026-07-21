# -*- coding: utf-8 -*-
"""MeshUV 单元测试(β 无关). 运行: python tests/run_tests.py"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meshuv.data.schema import (FORBIDDEN_INPUTS, MODEL_INPUTS, REQUIRED_NPZ,
                                TEACHER_DIAGNOSTICS, TRAINING_TARGETS)
from meshuv.data.splits import make_splits
from meshuv.data.collate import collate_charts
from meshuv.data.dataset import MeshUVTDDataset

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---- schema 角色分离 ----
check("禁用输入与 model_inputs 不相交",
      not (FORBIDDEN_INPUTS & set(MODEL_INPUTS)))
check("targets/diagnostics 不进 model_inputs",
      not (set(TRAINING_TARGETS) & set(MODEL_INPUTS))
      and not (set(TEACHER_DIAGNOSTICS) & set(MODEL_INPUTS)))
check("demand/scale 被列为禁用输入",
      {"chart_demand_normalized", "chart_target_scale"} <= FORBIDDEN_INPUTS)

# ---- splits 去重分组 ----
recs = [dict(object_id=f"o{i}", geometry_hash=f"g{i}",
             content_phash=f"{i:016x}") for i in range(20)]
recs[5]["geometry_hash"] = recs[3]["geometry_hash"]      # 3/5 同几何
recs[9]["content_phash"] = recs[8]["content_phash"]      # 8/9 同内容
splits, info = make_splits(recs, sizes=dict(train=12, val=4, test=4))
loc = {o: s for s, os_ in splits.items() for o in os_}
check("同几何 hash 不跨 split", loc["o3"] == loc["o5"])
check("同内容 phash 不跨 split", loc["o8"] == loc["o9"])
check("全部对象被分配", sum(len(v) for v in splits.values()) == 20)

# ---- 合成样本: loader + collate + 防泄漏 ----
with tempfile.TemporaryDirectory() as td:
    oid = "t0"
    d = f"{td}/objects/{oid}"
    os.makedirs(d)
    nF, nC = 12, 3
    arrays = dict(
        vertices=np.zeros((8, 3), np.float32),
        faces=np.zeros((nF, 3), np.int64),
        face_ids=np.arange(nF), face_to_chart=np.arange(nF) % nC,
        chart_ids=np.arange(nC),
        local_uv_before_td=np.zeros((nF, 3, 2), np.float32),
        train_face_mask=np.ones(nF, bool),
        source_uv=np.zeros((nF, 3, 2), np.float32),
        source_uv_valid=np.ones(nF, bool),
        chart_surface_area=np.ones(nC, np.float32),
        chart_uv_area_before_td=np.ones(nC, np.float32),
        chart_demand_normalized=np.full(nC, 1 / nC, np.float32),
        chart_target_area_fraction=np.full(nC, 1 / nC, np.float32),
        chart_log_density_ratio=np.zeros(nC, np.float32),
        chart_target_scale=np.ones(nC, np.float32),
        chart_valid_mask=np.ones(nC, bool),
        face_content_score=np.zeros(nF, np.float32),
        chart_content_score=np.zeros(nC, np.float32))
    assert set(REQUIRED_NPZ) <= set(arrays)
    np.savez(f"{d}/arrays.npz", **arrays)
    json.dump(dict(files=dict(reference_texture="ref.png"),
                   teacher=dict(beta=0.25)), open(f"{d}/manifest.json", "w"))
    json.dump(dict(eligible=True, quality_status="VALID_NO_OP"),
              open(f"{d}/quality.json", "w"))
    with open(f"{td}/dataset_index.jsonl", "w") as fp:
        fp.write(json.dumps(dict(object_id=oid, sample_dir=f"objects/{oid}")))
    ds = MeshUVTDDataset(td)
    it = ds[0]
    check("loader 角色分组正确",
          set(it["model_inputs"]) == set(MODEL_INPUTS)
          and set(it["training_targets"]) == set(TRAINING_TARGETS))
    check("diagnostics 默认不暴露", "teacher_diagnostics" not in it)
    b = collate_charts([it, it])
    check("collate 形状/掩码正确",
          b["features"].shape == (2 * nC, 3)
          and b["targets"]["chart_log_density_ratio"].shape == (2 * nC,)
          and b["chart_mask"].sum() == 2 * nC
          and (b["object_index"][:nC] == 0).all())

n_fail = RESULTS.count(False)
print(f"==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
