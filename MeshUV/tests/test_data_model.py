# -*- coding: utf-8 -*-
"""variable-chart collate / object 级 split 无泄漏 / Student forward+backward."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from meshuv.data.dataset import CleanDataset, object_splits  # noqa: E402
from meshuv.data.collate import collate  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def fake_obj(root, oid, nC, geo_hash):
    from PIL import Image
    d = f"{root}/objects/{oid}"
    os.makedirs(d)
    nF = nC * 4
    rng = np.random.RandomState(hash(oid) % 2**31)
    z = dict(vertices=rng.rand(nF * 3, 3).astype(np.float32),
             faces=np.arange(nF * 3).reshape(nF, 3),
             face_to_chart=(np.arange(nF) % nC).astype(np.int64),
             local_uv=rng.rand(nF, 3, 2).astype(np.float32),
             source_uv=rng.rand(nF, 3, 2).astype(np.float32),
             source_uv_valid=np.ones(nF, bool),
             train_face_mask=np.ones(nF, bool),
             face_area=np.ones(nF, np.float32),
             face_source=np.zeros(nF, np.int64),
             chart_surface_area=np.ones(nC, np.float32),
             chart_target_area_fraction=np.full(nC, 1 / nC, np.float32),
             chart_log_density_ratio=(rng.randn(nC) * 0.2).astype(np.float32),
             chart_valid_mask=np.ones(nC, bool),
             face_content_score=np.zeros(nF, np.float32),
             chart_content_score=np.zeros(nC, np.float32))
    z["chart_log_density_ratio"] -= z["chart_log_density_ratio"].mean()
    np.savez(f"{d}/arrays.npz", **z)
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(f"{d}/basecolor.png")
    json.dump(dict(geometry_hash=geo_hash, n_charts=nC), open(f"{d}/manifest.json", "w"))


with tempfile.TemporaryDirectory() as td:
    for i, (nc, gh) in enumerate([(3, "g0"), (7, "g1"), (5, "g1"), (4, "g2"),
                                  (6, "g3"), (2, "g4"), (9, "g5"), (3, "g6")]):
        fake_obj(td, f"o{i}", nc, gh)
    ds = CleanDataset(td)
    b = collate([ds[i] for i in range(len(ds))])
    check("variable-chart collate 形状",
          b["features"].shape == (3 + 7 + 5 + 4 + 6 + 2 + 9 + 3, 17)
          and len(b["object_ranges"]) == 8)
    sp = object_splits(td)
    loc = {o: s for s, os_ in sp.items() for o in os_}
    check("split: 同 geometry_hash 不跨集(o1/o2 同组)", loc["o1"] == loc["o2"])
    check("split: object 级无重复",
          sum(len(v) for v in sp.values()) == 8
          and len(set(sum(sp.values(), []))) == 8)

    import torch
    from meshuv.model.student_v0 import StudentV0
    m = StudentV0(d=32)
    X = torch.as_tensor(b["features"])
    v = torch.as_tensor(b["valid"])
    out = m(X, b["object_ranges"], v)
    loss = ((out[v] - torch.as_tensor(b["target"])[v]) ** 2).mean()
    loss.backward()
    check("Student forward/backward + object 均值居中",
          out.shape == (len(X),) and torch.isfinite(loss)
          and all(abs(float(out[a:b_][v[a:b_]].mean())) < 1e-5
                  for a, b_ in b["object_ranges"]))

n_fail = RESULTS.count(False)
print(f"==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
