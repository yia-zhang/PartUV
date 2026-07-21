# -*- coding: utf-8 -*-
"""Calibration serial vs concurrent 确定性复核.

按冻结 selection_rank 顺序取前 4 个 processing OK 资产(不按完成顺序),
串行重跑 calib_one_asset 到 <oid>_det/, 比较:
  - chart_hash: charts_cache.pkl 中每 chart gidx 序列的 sha256(分解一致性)
  - target_scale_hash: 冻结公式重算 β=0.25 chart scales(round 1e-9)的 sha256
  - metrics_hash: result.json 的 betas 段 canonical json sha256
若 chart_hash 不一致 -> 归类 PartUV 跨运行非确定性(已知), 非并发问题;
chart 一致而下游不一致 -> 并发/数值问题, 必须上报。
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CODE = "/root/youjiaZhang/PartUV/code"
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
PY = "/root/miniconda3/envs/geomae/bin/python"
N_CHECK = 4

man = json.load(open(f"{OUTD}/calibration_manifest_v2.json"))


def chart_hash(pkl):
    pu = pickle.load(open(pkl, "rb"))
    h = hashlib.sha256()
    h.update(str(len(pu["charts"])).encode())
    for c in pu["charts"]:
        h.update(np.ascontiguousarray(np.sort(np.asarray(c["gidx"]))).tobytes())
    return h.hexdigest()[:16], pu


def scale_hash(pu, glb):
    from tdlib.pipeline import load_reference
    from tdlib.rd import prepare_face_ref_uv
    from tdlib.signal import demand_weights, luminance_std_heuristic
    ref = load_reference(glb, pu["V"], pu["F"], pu["mesh_scale"])
    face_refuv, valid, _ = prepare_face_ref_uv(pu, ref)
    cw = luminance_std_heuristic(ref["texA"], ref["uv0"], ref["Fo"], ref["f2o"],
                                 valid & pu["covered"])
    tris = pu["V"][pu["F"]]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    _, w = demand_weights(cw, valid & pu["covered"], fa3, beta=0.25)
    dem = np.array([float((fa3[c["gidx"]] * w[c["gidx"]]).sum())
                    for c in pu["charts"]])
    a2 = np.array([float(c["a2"]) for c in pu["charts"]])
    s = np.round(np.sqrt(dem / np.maximum(a2, 1e-12)), 9)
    return hashlib.sha256(s.tobytes()).hexdigest()[:16]


def metrics_hash(rpath):
    r = json.load(open(rpath))
    return hashlib.sha256(json.dumps(r.get("betas", {}), sort_keys=True)
                          .encode()).hexdigest()[:16]


picked = []
for a in sorted(man["assets"], key=lambda x: x["selection_rank"]):
    rp = f"{OUTD}/{a['object_id']}/result.json"
    if os.path.exists(rp) and json.load(open(rp))["processing_status"] == "OK":
        picked.append(a)
    if len(picked) == N_CHECK:
        break
print("复核资产(按 selection_rank):", [a["object_id"] for a in picked])

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

rows = []
for a in picked:
    oid = a["object_id"]
    det = f"{OUTD}/{oid}_det"
    print(f"\n== 串行重跑 {oid} ==", flush=True)
    subprocess.run([PY, f"{CODE}/scripts/calib_one_asset.py", a["glb"],
                    oid, det], timeout=1800, capture_output=True)
    row = dict(object_id=oid)
    try:
        h1, pu1 = chart_hash(f"{OUTD}/{oid}/charts_cache.pkl")
        h2, pu2 = chart_hash(f"{det}/charts_cache.pkl")
        row["chart_hash"] = dict(concurrent=h1, serial=h2, match=h1 == h2)
        if h1 == h2:
            s1 = scale_hash(pu1, a["glb"])
            s2 = scale_hash(pu2, a["glb"])
            row["target_scale_hash"] = dict(concurrent=s1, serial=s2,
                                            match=s1 == s2)
            m1 = metrics_hash(f"{OUTD}/{oid}/result.json")
            m2 = metrics_hash(f"{det}/result.json")
            row["metrics_hash"] = dict(concurrent=m1, serial=m2, match=m1 == m2)
            row["verdict"] = ("DETERMINISTIC" if s1 == s2 and m1 == m2
                              else "并发/数值不一致(chart 一致) -> 必须上报")
        else:
            row["verdict"] = "PartUV 跨运行非确定性(已知), 非并发问题"
    except Exception as e:
        row["verdict"] = f"复核异常: {type(e).__name__}: {str(e)[:120]}"
    print(json.dumps(row, ensure_ascii=False), flush=True)
    rows.append(row)

with open(f"{OUTD}/determinism_check.json", "w") as fp:
    json.dump(rows, fp, indent=1, ensure_ascii=False)
print("DETERMINISM: DONE")
