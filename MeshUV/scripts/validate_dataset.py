# -*- coding: utf-8 -*-
"""数据集完整性验证(100% 门 + 回读 + hash + license 字段 + 去重).
用法: python scripts/validate_dataset.py --dataset datasets/processed/MeshUV-TD-PseudoGT-MVP-v0
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.schema import REQUIRED_NPZ, REQUIRED_FILES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    ds = ds if os.path.isabs(ds) else os.path.join(ROOT, ds)
    idx = [json.loads(l) for l in open(f"{ds}/dataset_index.jsonl") if l.strip()]
    fails, seen = [], {}
    for rec in idx:
        d = os.path.join(ds, rec["sample_dir"])
        try:
            for f in REQUIRED_FILES:
                assert os.path.exists(os.path.join(d, f)), f"缺文件 {f}"
            z = dict(np.load(os.path.join(d, "arrays.npz")))
            miss = [k for k in REQUIRED_NPZ if k not in z]
            assert not miss, f"缺字段 {miss}"
            man = json.load(open(os.path.join(d, "manifest.json")))
            q = json.load(open(os.path.join(d, "quality.json")))
            dsh = z["chart_demand_normalized"].astype(float)
            assert (dsh >= 0).all() and abs(dsh.sum() - 1) < 1e-5, "demand 非归一"
            assert np.isfinite(z["local_uv_before_td"]).all(), "local UV 非有限"
            assert np.isfinite(z["chart_log_density_ratio"]).all(), "label 非有限"
            assert z["face_to_chart"].max() < len(z["chart_ids"]), "chart 越界"
            assert len(z["faces"]) == len(z["face_to_chart"]), "面对应错位"
            assert q["eligible"] and q["quality_status"] in (
                "POSITIVE", "VALID_NO_OP"), "资格语义不符"
            assert man["teacher"]["beta"] is not None, "β 缺失"
            _ = rec.get("license_id", "")   # 可选 provenance(TexVerse 不做 license)
            h = hashlib.sha256()
            for k in ("chart_demand_normalized", "chart_log_density_ratio",
                      "chart_target_scale"):
                h.update(k.encode())
                h.update(np.ascontiguousarray(z[k]).tobytes())
            assert h.hexdigest() == man["hashes"]["label"], "label hash 不符"
            key = (man["hashes"]["geometry"], man["hashes"]["content_phash"])
            assert key not in seen, f"与 {seen.get(key)} 重复"
            seen[key] = rec["object_id"]
        except Exception as e:
            fails.append((rec["object_id"], str(e)[:120]))
    print(f"validated {len(idx)} objects, failures: {len(fails)}")
    for oid, msg in fails[:20]:
        print(f"  [FAIL] {oid}: {msg}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
