# -*- coding: utf-8 -*-
"""TexVerse-1K 端到端 pilot(可从任意 cwd 启动, 逐对象断点恢复).

流程: metadata(增量 manifest) -> lazy 原子下载 -> quick preflight ->
PartUV+labels+QA(子进程, 阶段计时) -> 样本写盘 -> loader 回读。
停止: accepted >= target(16) 或 attempted >= cap(64); 失败不替换全记录。
teacher 未冻结时标签为 NON_CANONICAL_PILOT 且不写正式 dataset root。
用法: python scripts/pilot_texverse.py --config configs/pilot_texverse_1k.yaml
     [--smoke N] 只处理前 N 个对象(最小冒烟)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data_sources import get_source            # noqa: E402
from meshuv.preflight import quick_preflight          # noqa: E402
from meshuv.teacher_adapter import PARTUV_ROOT        # noqa: E402

PY = sys.executable


def rp(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_texverse_1k.yaml")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(rp(args.config)))
    fz = yaml.safe_load(open(rp(cfg["teacher_frozen_config"])))
    frozen = fz.get("status") == "FROZEN"
    beta = fz["beta"] if frozen else cfg["pilot_beta_non_canonical"]
    label_mode = "CANONICAL" if frozen else "NON_CANONICAL_PILOT"
    phash = fz.get("protocol_hash") or "PENDING_CALIBRATION"
    proot = rp(cfg["pilot_root"])
    os.makedirs(f"{proot}/objects", exist_ok=True)
    src = get_source(cfg["source"], cache_dir=rp(cfg["cache_dir"]))
    n_list = args.smoke or cfg["attempt_cap"]
    cands = src.list_candidates(n_list)
    print(f"[pilot] label_mode={label_mode} beta={beta} 候选={len(cands)}",
          flush=True)

    def n_accepted():
        acc = 0
        for c in cands:
            f = f"{proot}/objects/tex_{c['uid'][:12]}/status.json"
            if os.path.exists(f) and json.load(open(f)).get("status") == "ACCEPTED":
                acc += 1
        return acc

    def run_one(c):
        oid = f"tex_{c['uid'][:12]}"
        outd = f"{proot}/objects/{oid}"
        spath = f"{outd}/status.json"
        if os.path.exists(spath):
            return json.load(open(spath))
        if n_accepted() >= cfg["target_accepted"]:
            return None
        os.makedirs(outd, exist_ok=True)
        tm, t0 = {}, time.time()
        glb = src.ensure_local(c["uid"])
        tm["download"] = round(time.time() - t0, 2)
        if glb is None:
            st = dict(object_id=oid, uid=c["uid"], status="ERROR",
                      reason="download: " + src.read_status(c["uid"]).get(
                          "error", "?"), timings=tm)
            json.dump(st, open(spath, "w"), ensure_ascii=False)
            print(f"  -> [{oid}] DOWNLOAD_ERROR", flush=True)
            return st
        t0 = time.time()
        pf = quick_preflight(glb, max_faces=cfg["max_faces"])
        tm["preflight_fast"] = round(time.time() - t0, 2)
        if not pf["ok"]:
            st = dict(object_id=oid, uid=c["uid"], status="PRECHECK_REJECTED",
                      reason=pf["reason"], preflight=pf, timings=tm)
            json.dump(st, open(spath, "w"), ensure_ascii=False)
            print(f"  -> [{oid}] PRECHECK_REJECTED {pf['reason'][:50]}",
                  flush=True)
            return st
        try:
            subprocess.run(
                [PY, f"{ROOT}/scripts/_build_one_object.py", glb, oid, outd,
                 str(beta), phash, label_mode],
                timeout=cfg["timeout_s"], capture_output=True, text=True,
                env=dict(os.environ, PARTUV_ROOT=PARTUV_ROOT))
        except subprocess.TimeoutExpired:
            pass
        if not os.path.exists(spath):
            st = dict(object_id=oid, uid=c["uid"], status="TIMEOUT",
                      reason=f">{cfg['timeout_s']}s", timings=tm)
            json.dump(st, open(spath, "w"), ensure_ascii=False)
        st = json.load(open(spath))
        st.setdefault("timings", {}).update(tm)
        st["uid"] = c["uid"]
        st["preflight"] = pf
        json.dump(st, open(spath, "w"), ensure_ascii=False, indent=1)
        print(f"  -> [{oid}] {st['status']} {st.get('quality_status', '')} "
              f"{st.get('reason', '')[:44]}", flush=True)
        return st

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        list(ex.map(run_one, cands))

    # ---- 汇总(幂等从磁盘重算) ----
    sts = []
    for c in cands:
        f = f"{proot}/objects/tex_{c['uid'][:12]}/status.json"
        if os.path.exists(f):
            sts.append(json.load(open(f)))
    acc = [s for s in sts if s["status"] == "ACCEPTED"]
    yield_cnt = {}
    for s in sts:
        yield_cnt[s["status"]] = yield_cnt.get(s["status"], 0) + 1
    totals = [s["timings"].get("total", 0) + s["timings"].get("download", 0)
              for s in sts if "timings" in s]
    stage_sum = {}
    for s in sts:
        for k, v in s.get("timings", {}).items():
            stage_sum[k] = round(stage_sum.get(k, 0) + v, 1)
    wall_hours = sum(totals) / 3600 / max(cfg["workers"], 1)
    rate = len(acc) / max(wall_hours, 1e-6)
    summary = dict(
        label_mode=label_mode, beta=beta, workers=cfg["workers"],
        attempted=len(sts), accepted=len(acc),
        rejected=len(sts) - len(acc), yield_counts=yield_cnt,
        download_success_rate=round(
            1 - yield_cnt.get("ERROR", 0) / max(len(sts), 1), 3),
        preflight_yield=round(
            1 - yield_cnt.get("PRECHECK_REJECTED", 0) / max(len(sts), 1), 3),
        qa_acceptance=round(len(acc) / max(
            sum(v for k, v in yield_cnt.items()
                if k in ("ACCEPTED", "NOT_ELIGIBLE", "QA_REJECTED")), 1), 3),
        timing=dict(p50_total=round(float(np.percentile(totals, 50)), 1)
                    if totals else None,
                    p90_total=round(float(np.percentile(totals, 90)), 1)
                    if totals else None,
                    stage_seconds=stage_sum),
        accepted_per_hour=round(rate, 1),
        eta_256_accepted_hours=round(256 / max(rate, 1e-6), 1))
    json.dump(summary, open(f"{proot}/pilot_summary.json", "w"),
              indent=1, ensure_ascii=False)
    json.dump([s for s in sts if s["status"] != "ACCEPTED"],
              open(f"{proot}/rejection_report.json", "w"),
              indent=1, ensure_ascii=False)
    with open(f"{proot}/dataset_index.jsonl", "w") as fp:
        for s in acc:
            fp.write(json.dumps(dict(
                object_id=s["object_id"], uid=s["uid"],
                sample_dir=f"objects/{s['object_id']}",
                label_mode=label_mode,
                quality_status=s["quality_status"])) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("PILOT_TEXVERSE: DONE")


if __name__ == "__main__":
    main()
