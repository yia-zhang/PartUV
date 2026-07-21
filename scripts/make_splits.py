# -*- coding: utf-8 -*-
"""生成 192/32/32 拆分. 用法: python scripts/make_splits.py --dataset <root>"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.splits import make_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    ds = ds if os.path.isabs(ds) else os.path.join(ROOT, ds)
    recs = [json.loads(l) for l in open(f"{ds}/dataset_index.jsonl") if l.strip()]
    splits, info = make_splits(recs)
    json.dump(dict(info=info, **splits), open(f"{ds}/splits.json", "w"),
              indent=1, ensure_ascii=False)
    print("splits:", info)


if __name__ == "__main__":
    main()
