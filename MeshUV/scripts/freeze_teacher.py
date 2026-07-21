# -*- coding: utf-8 -*-
"""calibration 决策 -> 冻结 teacher(唯一一次性操作).

读 calibration_report_stratified.json 的 final_beta_candidate:
- 存在合格 β: 写 configs/teacher_frozen_v0.yaml(frozen=true, β/protocol_hash/
  code_hash), 并把 calibration 只读快照复制到 provenance/calibration_v1/;
- 无合格非零 β: 退出码 2, 不冻结不建数据集。
"""
import hashlib
import json
import os
import shutil
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.teacher_adapter import (teacher_code_hash, TEACHER_VERSION,
                                    EVALUATOR)  # noqa: E402

CAL = "/root/youjiaZhang/PartUV/code/notebook/outputs/calibration_v1"
rep = json.load(open(f"{CAL}/calibration_report_stratified.json"))
beta = rep["final_beta_candidate"]
if beta is None or float(beta) == 0.0:
    print("NO_QUALIFIED_NONZERO_BETA -> 不冻结, 不生成数据集")
    sys.exit(2)

protocol = dict(teacher_version=TEACHER_VERSION, evaluator=EVALUATOR,
                beta=float(beta), medium_frac=0.5, r_cap=2048,
                seed_eval=2, n_samples=150000, bsignal_dev_max=0.01,
                pos=dict(band_g=0.02, pos_hf=0.05, neg_g=-0.05),
                low_td_contrast=0.05, coverage_gate=0.999)
phash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
fz = dict(status="FROZEN", frozen=True, teacher_version=TEACHER_VERSION, evaluator=EVALUATOR,
          beta=float(beta), protocol=protocol, protocol_hash=phash,
          code_hash=teacher_code_hash(),
          calibration_report=f"{CAL}/calibration_report_stratified.json",
          decision_steps=rep["decision_steps"])
yaml.safe_dump(fz, open(f"{ROOT}/configs/teacher_frozen_v0.yaml", "w"),
               allow_unicode=True, sort_keys=False)

# 只读快照(不移动/不覆盖原始输出)
dst = f"{ROOT}/provenance/calibration_v1"
os.makedirs(dst, exist_ok=True)
for f in ("calibration_report_stratified.json", "calibration_summary.json",
          "calibration_manifest_v2.json", "determinism_check.json"):
    src = f"{CAL}/{f}"
    if os.path.exists(src):
        shutil.copy(src, f"{dst}/{f}")
lineage = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v2_rebaseline/metric_lineage.json"
if os.path.exists(lineage):
    shutil.copy(lineage, f"{dst}/metric_lineage.json")
for f in os.listdir(dst):
    os.chmod(f"{dst}/{f}", 0o444)
print(f"FROZEN: beta={beta} protocol_hash={phash[:16]} "
      f"code_hash={fz['code_hash'][:12]}")
print("快照 ->", dst)
