# -*- coding: utf-8 -*-
"""幂等迁移 v2: 绑定 audit hash 的版本化 quarantine + 原地重标.

- quarantine/<migration_id>/<uid>(migration_id=audit_hash 前 12 位);
- 逐 UID stage 状态持久化(migration_state.json), 重跑跳过已完成;
- destination 已存在禁止嵌套移动;
- 已按当前 adapter 重建成功(ACCEPTED + canonicalizer_rgb_v2)的 UID 不再隔离;
- relabel: 原子更新 arrays+manifest, 重算 content_hash 与 label/code hash;
- 任何失败保持非 ACCEPTED(隔离), loader 不再读取。
支持 --dry-run 只输出计划。"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.density.signal import face_content_score, SIGNAL_VERSION
from meshuv.density.allocation import chart_targets, BETA, LABEL_SEMANTICS
from meshuv.asset.canonicalizer import ADAPTER_VERSION

DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"


def _needs_quarantine(d):
    """行动时重验条件(防 stale audit 二次隔离已重建对象)."""
    try:
        man = json.load(open(f"{d}/manifest.json"))
        st = json.load(open(f"{d}/status.json"))
    except Exception:
        return True
    if st.get("status") != "ACCEPTED":
        return True
    return man.get("source_adapter_version") != ADAPTER_VERSION


def relabel_inplace(d):
    from PIL import Image
    z = dict(np.load(f"{d}/arrays.npz"))
    atlas = np.asarray(Image.open(f"{d}/basecolor.png"), float)[:, :, :3] / 255
    sc = face_content_score(atlas, z["source_uv"], z["source_uv_valid"])
    cs = dict(face_to_chart=z["face_to_chart"], face_area=z["face_area"],
              covered=z["train_face_mask"],
              source_uv_valid=z["source_uv_valid"],
              n_charts=int(z["face_to_chart"].max()) + 1)
    lab = chart_targets(cs, sc)
    z.update(face_content_score=sc,
             chart_content_score=lab["chart_content_score"],
             chart_surface_area=lab["chart_surface_area"],
             chart_target_area_fraction=lab["chart_target_area_fraction"],
             chart_log_density_ratio=lab["chart_log_density_ratio"],
             chart_valid_mask=lab["chart_valid_mask"])
    np.savez_compressed(f"{d}/.arrays_tmp.npz", **z)
    man = json.load(open(f"{d}/manifest.json"))
    man["relabel"] = f"{SIGNAL_VERSION}+beta{BETA} 原地重标(未重跑 PartUV)"
    man["label_semantics"] = LABEL_SEMANTICS
    man["content_hash"] = hashlib.sha256(
        np.ascontiguousarray(sc).tobytes()).hexdigest()[:16]
    code = hashlib.sha256(b"".join(
        open(f"{ROOT}/src/meshuv/density/{f}.py", "rb").read()
        for f in ("signal", "allocation"))).hexdigest()[:16]
    man["label_code_hash"] = code
    with open(f"{d}/.manifest_tmp.json", "w") as fp:
        json.dump(man, fp, indent=1, ensure_ascii=False)
    os.replace(f"{d}/.arrays_tmp.npz", f"{d}/arrays.npz")     # 原子
    os.replace(f"{d}/.manifest_tmp.json", f"{d}/manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    audit = json.load(open(f"{DS}/audit_clean_256.json"))
    mid = audit["audit_hash"][:12]
    Q = f"{DATA}/quarantine/{mid}"
    state_p = f"{DS}/migration_state.json"
    state = (json.load(open(state_p)) if os.path.exists(state_p) else
             dict(migration_id=mid, audit_hash=audit["audit_hash"], stages={}))
    if state.get("audit_hash") != audit["audit_hash"]:
        state = dict(migration_id=mid, audit_hash=audit["audit_hash"],
                     stages={})
    rebuild = set(audit["rebuild_candidates"])
    relabel = set(audit["relabel_candidates"]) - rebuild
    plan = dict(migration_id=mid, quarantine=sorted(rebuild),
                relabel=sorted(relabel))
    if a.dry_run:
        print(json.dumps(dict(dry_run=True, n_quarantine=len(rebuild),
                              n_relabel=len(relabel), **plan),
                         ensure_ascii=False, indent=1)[:2000])
        return
    os.makedirs(Q, exist_ok=True)
    for oid in sorted(rebuild):
        if state["stages"].get(oid) in ("quarantined", "kept"):
            continue
        src, dst = f"{DS}/objects/{oid}", f"{Q}/{oid}"
        if not os.path.isdir(src):
            state["stages"][oid] = "absent"
        elif not _needs_quarantine(src):
            state["stages"][oid] = "kept"          # 已按新 adapter 重建
        elif os.path.exists(dst):
            state["stages"][oid] = "dest_exists_skip"   # 禁止嵌套移动
        else:
            shutil.move(src, dst)
            state["stages"][oid] = "quarantined"
        json.dump(state, open(state_p, "w"), indent=1)
    for oid in sorted(relabel):
        if state["stages"].get(oid) in ("relabeled", "quarantined"):
            continue
        d = f"{DS}/objects/{oid}"
        if not os.path.isdir(d):
            state["stages"][oid] = "absent"
        else:
            try:
                relabel_inplace(d)
                state["stages"][oid] = "relabeled"
            except Exception as e:
                dst = f"{Q}/{oid}"
                if not os.path.exists(dst):
                    shutil.move(d, dst)
                state["stages"][oid] = f"relabel_failed_quarantined:{type(e).__name__}"
        json.dump(state, open(state_p, "w"), indent=1)
    from collections import Counter
    cnt = Counter(v.split(":")[0] for v in state["stages"].values())
    rep = dict(migration_id=mid, audit_hash=audit["audit_hash"],
               stage_counts=dict(cnt), quarantine_dir=Q)
    json.dump(rep, open(f"{DS}/migration_report.json", "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(rep, ensure_ascii=False))
    print("MIGRATE: DONE")


if __name__ == "__main__":
    main()
