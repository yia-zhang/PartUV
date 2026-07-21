# -*- coding: utf-8 -*-
"""Object-level Pseudo-GT Exporter V1 —— 单资产(鞋)导出 + 自动验收.
鞋 -> map_partuv_td(auto, β=0.75)(PartUV 只运行一次)
   -> export_object_pseudo_gt ×2(同一 res, 验 exporter 确定性)
   -> 打印 gates 并停止。"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.api import map_partuv_td
from tdlib.dataset import export_object_pseudo_gt

ASSET = "/root/youjiaZhang/PartUV/code/data/objaverse_22b822c6520d4d49.glb"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pseudo_gt"
SAMPLE = f"{OUT}/shoe_22b822_v1"

res = map_partuv_td(ASSET, f"{OUT}/_teacher_run/", atlas_size="auto", beta=0.75)
print("teacher 运行完成:", {k: res["budget"][k] for k in
      ("B_target", "output_B_signal", "budget_ratio",
       "output_packing_fill", "E_alloc", "selected_atlas_size")}, flush=True)

m1 = export_object_pseudo_gt(res, SAMPLE, object_id="shoe_22b822")
# E. exporter 确定性: 同一 res 第二次导出(不重跑 PartUV)
m2 = export_object_pseudo_gt(res, SAMPLE + "_det2", object_id="shoe_22b822")

det_arrays = m1["arrays_content_sha256"] == m2["arrays_content_sha256"]
skip = {"files"}                       # zip 时间戳致 npz 文件级 hash 可不同
sem1 = {k: v for k, v in m1.items() if k not in skip}
sem2 = {k: v for k, v in m2.items() if k not in skip}
det_manifest = sem1 == sem2
shutil.rmtree(SAMPLE + "_det2")        # 确定性检查完成后清理副本

print("\n===== GATES =====")
for k, v in m1["gates"].items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
print(f"  [{'PASS' if det_arrays else 'FAIL'}] E_arrays_deterministic")
print(f"  [{'PASS' if det_manifest else 'FAIL'}] E_manifest_semantic_deterministic")
print(f"\nstatus={m1['status']}  train_ready={m1['train_ready']}")
print(f"sample_dir={SAMPLE}")
ok = m1["status"] == "ACCEPTED" and det_arrays and det_manifest
print("EXPORT_V1:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
