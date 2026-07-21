# -*- coding: utf-8 -*-
"""MeshUV-TD-PseudoGT-MVP-v0 数据集构建驱动(可从任意 cwd 启动, 可断点续跑).

流程: 冻结 teacher 检查 -> (无则)构建 500 UID 有序候选清单(排除 dev/calibration/
已用; CC0/CC-BY 优先; 冻结后不按结果换样) -> 逐对象子进程生成(8 并发, 超时记
TIMEOUT) -> 达到 256 个 eligible 停止; 处理完 500 仍不足则保存并如实报告。
全部 rejected/failed 记入 rejection manifest; 不放宽 gate。
用法: python scripts/build_dataset_mvp.py --config configs/dataset_mvp_v0.yaml
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.teacher_adapter import PARTUV_ROOT, teacher_code_hash  # noqa: E402
from meshuv.data_sources import get_source  # noqa: E402

PY = sys.executable
OK_LIC = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "cc0", "by", "by-sa"}


def rp(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_frozen(cfg):
    fz = yaml.safe_load(open(rp(cfg["teacher_frozen_config"])))
    assert fz.get("status") == "FROZEN", (
        f"teacher status={fz.get('status')}: 未冻结, 拒绝生成正式 labels "
        f"(先跑 scripts/freeze_teacher.py)")
    ch = teacher_code_hash()
    assert ch == fz["code_hash"], f"tdlib code hash 漂移: {ch[:12]}"
    return fz


def build_candidates(cfg, mpath):
    print("[manifest] 构建 500 UID 候选清单…", flush=True)
    opaths = json.load(gzip.open(os.path.expanduser(cfg["object_paths"]), "rt"))
    used = set()
    for f in cfg["exclusion_uid_files"]:
        used |= {json.loads(l)["uid"] if f.endswith(".jsonl") else u
                 for l in open(rp(f))
                 for u in ([json.loads(l).get("uid")] if f.endswith(".jsonl")
                           else [l.strip()]) if u}
    calman = json.load(open(rp(cfg["calibration_manifest"])))
    used |= {a["uid"] for a in calman["assets"]}
    cached = {os.path.basename(p)[:-4]: p for p in
              glob.glob(os.path.expanduser(cfg["objaverse_cache"]) + "/*/*.glb")}
    rng = np.random.RandomState(cfg["seed_sample"])
    cand = [u for u in sorted(cached) if u not in used]
    fresh_pool = sorted(u for u in opaths if u not in used and u not in cached)
    need = cfg["candidate_max"] - len(cand) + 80          # 余量补许可占位
    cand += [fresh_pool[i] for i in
             rng.choice(len(fresh_pool), need, replace=False)]
    rows, snap = [], time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for u in cand:
        lic, creator = "UNKNOWN", "UNKNOWN"
        try:
            req = urllib.request.Request(
                f"https://api.sketchfab.com/v3/models/{u}",
                headers={"User-Agent": "meshuv-mvp/0.1"})
            meta = json.load(urllib.request.urlopen(req, timeout=12))
            creator = (meta.get("user", {}) or {}).get("username", "UNKNOWN")
            L = meta.get("license", {}) or {}
            lic = L.get("slug") or L.get("label") or "UNKNOWN"
        except Exception:
            pass
        rows.append(dict(uid=u, license_id=lic, creator=creator,
                         license_clear=lic in OK_LIC,
                         cached=u in cached,
                         hf_path=opaths.get(u, ""),
                         metadata_snapshot_time=snap))
        time.sleep(0.25)
        if len(rows) % 50 == 0:
            print(f"[manifest] license {len(rows)}/{len(cand)}", flush=True)
    rows.sort(key=lambda r: (not r["license_clear"], r["uid"]))
    rows = rows[:cfg["candidate_max"]]
    for i, r in enumerate(rows):
        r["selection_rank"] = i
        r["object_id"] = f"mv0_{r['uid'][:12]}"
    man = dict(schema="meshuv_mvp_candidates_v0", snapshot=snap,
               seed_sample=cfg["seed_sample"],
               domain="public-source Objaverse; CC0/CC-BY 优先排序",
               frozen_rule="冻结后不按结果换样; 全部失败计入 rejection manifest",
               candidates=rows)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)
    json.dump(man, open(mpath, "w"), indent=1, ensure_ascii=False)
    print(f"[manifest] 冻结 {len(rows)} 候选 -> {mpath}", flush=True)
    return man


def ensure_glb(c, cfg):
    if c["cached"]:
        p = os.path.expanduser(cfg["objaverse_cache"]) + "/" + \
            "/".join(c["hf_path"].split("/")[-2:])
        if os.path.exists(p):
            return p
    dst = rp(f"datasets/cache/{c['uid']}.glb")
    if os.path.exists(dst):
        return dst
    url = ("https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
           + c["hf_path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(url, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_mvp_v0.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验配置/冻结状态/候选清单, 不处理任何对象")
    ap.add_argument("--manifest-only", action="store_true",
                    help="只构建候选清单(β 无关), 不生成 labels")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(rp(args.config)))
    if args.manifest_only:
        mp = rp(cfg["candidate_manifest"])
        if os.path.exists(mp):
            print(f"[manifest] 已存在({len(json.load(open(mp))['candidates'])} 候选), 不重建")
        else:
            build_candidates(cfg, mp)
        return
    if args.dry_run:
        fzy = yaml.safe_load(open(rp(cfg["teacher_frozen_config"])))
        mp = rp(cfg["candidate_manifest"])
        print(f"[dry-run] teacher status = {fzy.get('status')} "
              f"(FROZEN 才会生成 labels)")
        print(f"[dry-run] 候选清单: "
              f"{'存在' if os.path.exists(mp) else '缺失(将构建 500 UID)'}")
        print(f"[dry-run] dataset_root = {rp(cfg['dataset_root'])}")
        print(f"[dry-run] target_accepted = {cfg['target_accepted']}, "
              f"workers = {cfg['workers']}, timeout = {cfg['timeout_s']}s")
        return
    fz = load_frozen(cfg)
    beta, phash = str(fz["beta"]), fz["protocol_hash"]
    dsroot = rp(cfg["dataset_root"])
    os.makedirs(dsroot, exist_ok=True)
    # 数据源: 配置含 source 时走数据源抽象(如 texverse_1k), 否则 Objaverse 遗留路径
    src = (get_source(cfg["source"], cache_dir=rp(cfg["source_cache"]))
           if cfg.get("source") else None)
    if src is not None:
        cands = [dict(uid=r["uid"], selection_rank=i,
                      object_id=f"mv0_{r['uid'][:12]}")
                 for i, r in enumerate(src.list_candidates(cfg["candidate_max"]))]
    else:
        mpath = rp(cfg["candidate_manifest"])
        man = (json.load(open(mpath)) if os.path.exists(mpath)
               else build_candidates(cfg, mpath))
        cands = sorted(man["candidates"], key=lambda c: c["selection_rank"])

    def n_accepted():
        return sum(1 for f in glob.glob(f"{dsroot}/objects/*/status.json")
                   if json.load(open(f)).get("status") == "ACCEPTED")

    def run_one(c):
        oid = c["object_id"]
        outd = f"{dsroot}/objects/{oid}"
        spath = f"{outd}/status.json"
        if os.path.exists(spath):
            return json.load(open(spath))
        if n_accepted() >= cfg["target_accepted"]:
            return None
        try:
            glb = (src.ensure_local(c["uid"]) if src is not None
                   else ensure_glb(c, cfg))
            if glb is None:
                raise RuntimeError(src.read_status(c["uid"]).get("error", "下载失败"))
        except Exception as e:
            os.makedirs(outd, exist_ok=True)
            st = dict(object_id=oid, status="ACQUISITION_FAILED",
                      reason=f"{type(e).__name__}: {str(e)[:120]}")
            json.dump(st, open(spath, "w"), ensure_ascii=False)
            return st
        try:
            subprocess.run([PY, f"{ROOT}/scripts/_build_one_object.py",
                            glb, oid, outd, beta, phash],
                           timeout=cfg["timeout_s"], capture_output=True,
                           text=True, env=dict(os.environ,
                                               PARTUV_ROOT=PARTUV_ROOT))
        except subprocess.TimeoutExpired:
            pass
        if not os.path.exists(spath):
            st = dict(object_id=oid, status="PROCESSING_TIMEOUT",
                      reason=f">{cfg['timeout_s']}s")
            os.makedirs(outd, exist_ok=True)
            json.dump(st, open(spath, "w"), ensure_ascii=False)
        st = json.load(open(spath))
        print(f"  -> [{oid}] {st['status']} {st.get('quality_status', '')} "
              f"{st.get('reason', '')[:50]}", flush=True)
        return st

    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        list(ex.map(run_one, cands))

    # ---- 汇总(幂等, 从磁盘重算) ----
    sts, dup_seen, accepted = [], {}, []
    for c in cands:
        spath = f"{dsroot}/objects/{c['object_id']}/status.json"
        if not os.path.exists(spath):
            continue
        st = json.load(open(spath)) | dict(uid=c["uid"],
                                           license_id=c["license_id"],
                                           selection_rank=c["selection_rank"])
        # object-level 去重(geometry+content hash)
        if st["status"] == "ACCEPTED":
            key = (st["hashes"]["geometry"], st["hashes"]["content_phash"])
            if key in dup_seen:
                st["status"] = "DUPLICATE_REJECTED"
                st["reason"] = f"dup of {dup_seen[key]}"
                json.dump(st, open(spath, "w"), indent=1, ensure_ascii=False)
            else:
                dup_seen[key] = st["object_id"]
                accepted.append(st)
        sts.append(st)
    yield_cnt = {}
    for st in sts:
        yield_cnt[st["status"]] = yield_cnt.get(st["status"], 0) + 1
    json.dump(dict(attempted=len(sts), yield_counts=yield_cnt,
                   accepted=len(accepted),
                   target=cfg["target_accepted"]),
              open(f"{dsroot}/processing_yield.json", "w"),
              indent=1, ensure_ascii=False)
    json.dump([st for st in sts if st["status"] != "ACCEPTED"],
              open(f"{dsroot}/rejection_summary.json", "w"),
              indent=1, ensure_ascii=False)
    with open(f"{dsroot}/dataset_index.jsonl", "w") as fp:
        for st in accepted[:cfg["target_accepted"]]:
            fp.write(json.dumps(dict(
                object_id=st["object_id"], uid=st["uid"],
                license_id=st["license_id"],
                selection_rank=st["selection_rank"],
                sample_dir=f"objects/{st['object_id']}",
                quality_status=st["quality_status"],
                geometry_hash=st["hashes"]["geometry"],
                content_phash=st["hashes"]["content_phash"]),
                ensure_ascii=False) + "\n")
    print(f"\nattempted={len(sts)} accepted={len(accepted)} "
          f"target={cfg['target_accepted']}")
    print("yield:", yield_cnt)
    print("BUILD_DATASET: DONE")


if __name__ == "__main__":
    main()
