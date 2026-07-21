# -*- coding: utf-8 -*-
"""训练前最终硬闸门: 六条件全部满足才 exit 0, 否则 exit 1 (状态机即停).

1. 256 个 ACCEPTED 且 schema 完整;
2. adapter 100% canonicalizer_rgb_v2;
3. teacher/signal/code hash 100% 一致(= clean_teacher_v1 当前冻结);
4. 标签 round-trip drift(落盘 PNG 重算) max <= 1e-6;
5. rebuild/relabel 候选均为 0;
6. split 无对象重叠且并集完整。
依据最新 audit_clean_256.json + manifest 扫描 + object_splits。
产出 reports/final_gate.json。"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.asset.canonicalizer import ADAPTER_VERSION  # noqa: E402
from meshuv.density.signal import SIGNAL_VERSION  # noqa: E402
from meshuv.data.builder import _teacher_code_hash  # noqa: E402
from meshuv.data.dataset import CleanDataset, object_splits  # noqa: E402

DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"
TARGET = 256


def main():
    audit = json.load(open(f"{DS}/audit_clean_256.json"))
    checks = {}
    checks["1_256_accepted_schema_complete"] = (
        audit["n_accepted"] == TARGET and not audit.get("schema_bad"),
        f"accepted={audit['n_accepted']} schema_bad={len(audit.get('schema_bad', []))}")
    ad = audit.get("adapter_distribution", {})
    checks["2_adapter_100_v2"] = (
        set(ad) == {ADAPTER_VERSION} and sum(ad.values()) == TARGET,
        f"{ad}")
    thash = _teacher_code_hash()
    expect = f"clean_teacher_v1|{thash}|{SIGNAL_VERSION}"
    td = audit.get("teacher_distribution", {})
    checks["3_teacher_hash_uniform"] = (
        set(td) == {expect} and sum(td.values()) == TARGET,
        f"expect={expect} got={td}")
    drift = audit["label_drift"]["max"]
    checks["4_roundtrip_drift_le_1e6"] = (drift <= 1e-6, f"max={drift:.2e}")
    checks["5_zero_candidates"] = (
        not audit["rebuild_candidates"] and not audit["relabel_candidates"],
        f"rebuild={len(audit['rebuild_candidates'])} "
        f"relabel={len(audit['relabel_candidates'])}")
    ds = CleanDataset(DS)
    sp = object_splits(ds)
    ids = [set(sp[k]) for k in ("train", "val", "test")]
    overlap = (ids[0] & ids[1]) | (ids[0] & ids[2]) | (ids[1] & ids[2])
    union = ids[0] | ids[1] | ids[2]
    checks["6_split_disjoint_complete"] = (
        not overlap and len(union) == len(ds),
        f"overlap={len(overlap)} union={len(union)}/{len(ds)} "
        f"sizes={[len(i) for i in ids]}")
    all_pass = all(v[0] for v in checks.values())
    rep = dict(all_pass=all_pass, target=TARGET,
               teacher_code_hash=thash, audit_hash=audit["audit_hash"],
               checks={k: dict(ok=bool(v[0]), detail=v[1])
                       for k, v in checks.items()})
    json.dump(rep, open(f"{ROOT}/reports/final_gate.json", "w"), indent=1,
              ensure_ascii=False)
    for k, (ok, det) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {det}")
    print("FINAL_GATE:", "PASS" if all_pass else "FAIL")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
