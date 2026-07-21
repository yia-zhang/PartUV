# -*- coding: utf-8 -*-
"""Clean V1 数据集构建驱动(UID 断点/4 worker/实时统计行).
用法: python scripts/build_dataset.py --n-candidates 200 --target 20
      [--out datasets/processed/clean_v1] [--workers 4]
环境: MESHUV_DATA_ROOT 覆盖数据根; PARTUV_ROOT 指向 teacher checkout。"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.sources import TexVerse  # noqa: E402

PY = sys.executable
DATA_ROOT = os.environ.get("MESHUV_DATA_ROOT", os.path.join(ROOT, "datasets"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-candidates", type=int, default=200)
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--out", default="processed/clean_v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(DATA_ROOT, a.out)
    os.makedirs(f"{out}/objects", exist_ok=True)
    src = TexVerse(os.path.join(DATA_ROOT, "cache/texverse_1k"))
    cands = src.candidates(a.n_candidates)
    t0 = time.monotonic()
    state = dict(done=0, acc=0)

    def run_one(row):
        oid = f"tex_{row['uid'][:12]}"
        d = f"{out}/objects/{oid}"
        sp = f"{d}/status.json"
        if os.path.exists(sp):
            st = json.load(open(sp))
        elif state["acc"] >= a.target:
            return None
        else:
            glb = src.fetch(row)
            if glb is None:
                os.makedirs(d, exist_ok=True)
                st = dict(object_id=oid, status="DOWNLOAD_FAILED")
                json.dump(st, open(sp, "w"))
            else:
                try:
                    subprocess.run(
                        [PY, "-c",
                         "import sys; sys.path.insert(0, sys.argv[4]);"
                         "from meshuv.data.builder import build_object;"
                         "build_object(sys.argv[1], sys.argv[2], sys.argv[3])",
                         glb, oid, d, os.path.join(ROOT, "src")],
                        timeout=a.timeout, capture_output=True, text=True,
                        env=dict(os.environ))
                except subprocess.TimeoutExpired:
                    pass
                st = (json.load(open(sp)) if os.path.exists(sp)
                      else dict(object_id=oid, status="TIMEOUT"))
                if not os.path.exists(sp):
                    os.makedirs(d, exist_ok=True)
                    json.dump(st, open(sp, "w"))
        state["done"] += 1
        if st.get("status") == "ACCEPTED":
            state["acc"] += 1
        el_h = (time.monotonic() - t0) / 3600
        rate = state["acc"] / max(el_h, 1e-9)
        eta = (a.target - state["acc"]) / max(rate, 1e-9)
        print(f"  {st.get('status', '?')[:22]:22s} {oid} | "
              f"attempted={state['done']} accepted={state['acc']} "
              f"acc_rate={state['acc'] / max(state['done'], 1):.1%} "
              f"rolling={rate:.0f}/h ETA(target)={eta:.2f}h", flush=True)
        return st

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        sts = [s for s in ex.map(run_one, cands) if s]
    from collections import Counter
    cnt = Counter(s["status"] for s in sts)
    summary = dict(attempted=len(sts), yield_counts=dict(cnt),
                   accepted=cnt.get("ACCEPTED", 0),
                   wall_hours=round((time.monotonic() - t0) / 3600, 3))
    with open(f"{out}/summary.json", "w") as fp:
        json.dump(summary, fp, indent=1, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False))
    print("BUILD: DONE")


if __name__ == "__main__":
    main()
