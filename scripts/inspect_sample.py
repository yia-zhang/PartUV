# -*- coding: utf-8 -*-
"""查看单个样本. 用法: python scripts/inspect_sample.py --dataset <root> [--object <oid>]"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.dataset import MeshUVTDDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--object", default=None)
    a = ap.parse_args()
    ds = MeshUVTDDataset(a.dataset if os.path.isabs(a.dataset)
                         else os.path.join(ROOT, a.dataset))
    i = next((k for k, r in enumerate(ds.index)
              if r["object_id"] == a.object), 0)
    it = ds[i]
    print("object:", it["object_id"])
    for grp in ("model_inputs", "training_targets"):
        print(f"-- {grp} --")
        for k, v in it[grp].items():
            print(f"  {k:28s} {getattr(v, 'shape', v)} {getattr(v, 'dtype', '')}")
    print("manifest.teacher:", json.dumps(it["manifest"]["teacher"])[:160])


if __name__ == "__main__":
    main()
