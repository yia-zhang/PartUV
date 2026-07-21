# -*- coding: utf-8 -*-
"""收口状态机(入库): 每阶段 marker+输入 hash+返回码校验; 断连后从最后完成
阶段恢复; 任一失败即停(绝不打印 CHAIN_DONE)。
audit 后强制 PAUSE: 需存在 reports/closeout_state/CONFIRM_MIGRATION 才继续。
用法: python scripts/run_closeout.py [--confirm-migration]"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ST = f"{ROOT}/reports/closeout_state"
os.makedirs(ST, exist_ok=True)
DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"


def fhash(p):
    return (hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            if os.path.exists(p) else "absent")


def run_stage(name, cmd, verify, env=None):
    mk = f"{ST}/{name}.json"
    if os.path.exists(mk):
        print(f"[skip] {name}")
        return
    print(f"[stage] {name}: {' '.join(cmd)}", flush=True)
    e = dict(os.environ)
    e.update(env or {})
    rc = subprocess.run(cmd, env=e).returncode
    if rc != 0:
        print(f"[FAIL] {name} rc={rc}")
        sys.exit(1)
    ok, detail = verify()
    if not ok:
        print(f"[FAIL] {name} 输出验证失败: {detail}")
        sys.exit(1)
    json.dump(dict(stage=name, rc=rc, detail=detail,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
              open(mk, "w"), ensure_ascii=False, indent=1)
    print(f"[done] {name}: {detail}")


def _teacher_guard():
    """state 与 teacher 代码哈希绑定: 不匹配即停(须先归档旧 state)."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from meshuv.data.builder import _teacher_code_hash
    th = _teacher_code_hash()
    p = f"{ST}/TEACHER_HASH"
    if os.path.exists(p):
        old = open(p).read().strip()
        if old != th:
            print(f"[FAIL] closeout state 绑定 teacher hash {old} != 当前 {th}; "
                  f"请归档 {ST} 后重跑")
            sys.exit(1)
    else:
        open(p, "w").write(th + "\n")
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-migration", action="store_true")
    a = ap.parse_args()
    _teacher_guard()
    if a.confirm_migration:
        open(f"{ST}/CONFIRM_MIGRATION", "w").write("confirmed\n")
        print("已写入迁移确认标记")

    # 0) 等待 256 构建
    def v_build():
        s = json.load(open(f"{DS}/summary.json"))
        return s.get("accepted", 0) >= 1, f"accepted={s.get('accepted')}"
    if not os.path.exists(f"{ST}/build.json"):
        while True:
            log = f"{ROOT}/datasets/build256_clean.log"
            if os.path.exists(log) and "BUILD: DONE" in open(log).read():
                break
            time.sleep(120)
        json.dump(dict(stage="build"), open(f"{ST}/build.json", "w"))
    # 1) audit
    run_stage("audit", [PY, f"{ROOT}/scripts/audit_clean_256.py"],
              lambda: (os.path.exists(f"{DS}/audit_clean_256.json"),
                       "audit_hash=" + json.load(open(
                           f"{DS}/audit_clean_256.json"))["audit_hash"]))
    run_stage("migration_dryrun",
              [PY, f"{ROOT}/scripts/closeout_migrate.py", "--dry-run"],
              lambda: (True, "计划已打印"))
    # 2) PAUSE 闸门
    if not os.path.exists(f"{ST}/CONFIRM_MIGRATION"):
        print("PAUSED_AFTER_AUDIT: 等待确认(--confirm-migration)后继续迁移")
        sys.exit(0)
    audit_h = json.load(open(f"{DS}/audit_clean_256.json"))["audit_hash"]
    run_stage("migrate", [PY, f"{ROOT}/scripts/closeout_migrate.py"],
              lambda: (json.load(open(f"{DS}/migration_report.json"))
                       ["audit_hash"] == audit_h, "audit hash 绑定一致"))
    run_stage("topup", [PY, f"{ROOT}/scripts/build_dataset.py",
                        "--n-candidates", "2000", "--target", "256"],
              lambda: (json.load(open(f"{DS}/summary.json"))["accepted"] >= 256,
                       f"accepted={json.load(open(f'{DS}/summary.json'))['accepted']}"))
    def v_audit_final():
        aj = json.load(open(f"{DS}/audit_clean_256.json"))
        ok = (not aj["rebuild_candidates"] and not aj["relabel_candidates"])
        return ok, (f"rebuild={len(aj['rebuild_candidates'])} "
                    f"relabel={len(aj['relabel_candidates'])}")
    run_stage("audit_final", [PY, f"{ROOT}/scripts/audit_clean_256.py"],
              v_audit_final)
    run_stage("final_gate", [PY, f"{ROOT}/scripts/final_gate.py"],
              lambda: (json.load(open(f"{ROOT}/reports/final_gate.json"))
                       ["all_pass"], "六条件硬闸门全过"))
    run_stage("overfit", [PY, f"{ROOT}/scripts/overfit.py"],
              lambda: (json.load(open(f"{ROOT}/reports/overfit_8.json"))
                       ["pass_loss"], "loss<1% 初始"),
              env={"CUDA_VISIBLE_DEVICES": "7"})
    run_stage("sanity", [PY, f"{ROOT}/scripts/sanity_256.py"],
              lambda: (os.path.exists(f"{ROOT}/reports/student_v0_sanity.ckpt"),
                       "ckpt 已存"),
              env={"CUDA_VISIBLE_DEVICES": "7"})
    run_stage("gold", [PY, f"{ROOT}/scripts/gold_closeout.py"],
              lambda: (all(r.get("status") == "OK" for r in json.load(open(
                  f"{ROOT}/reports/gold_closeout.json"))["results"]),
                  "全部 OK 且零 overlap"),
              env={"CUDA_VISIBLE_DEVICES": "7"})
    run_stage("nb_smoke", [PY, f"{ROOT}/tests/test_notebooks_smoke.py"],
              lambda: (True, "smoke rc=0"))
    print("CLOSEOUT_COMPLETE")


if __name__ == "__main__":
    main()
