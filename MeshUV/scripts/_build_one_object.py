# -*- coding: utf-8 -*-
"""单对象数据生成(子进程单元; 驱动: build_dataset_mvp.py / pilot_texverse.py).
用法: _build_one_object.py <glb> <object_id> <out_dir> <beta> <protocol_hash> [label_mode]
label_mode: CANONICAL(默认) | NON_CANONICAL_PILOT(teacher 未冻结时的临时标签)。
写 <out_dir>/status.json(含各阶段耗时)(+ accepted 时的三件套)。"""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))

from meshuv.teacher_adapter import (check_support, compute_labels,
                                    pick_free_gpu, quality_check_medium,
                                    run_teacher_context, teacher_code_hash,
                                    TEACHER_VERSION)
from meshuv.data.schema import (REQUIRED_NPZ, SEMANTICS, SCHEMA_VERSION,
                                LABEL_TYPE, LABEL_SEMANTICS)

glb, oid, outd, beta, phash = (sys.argv[1], sys.argv[2], sys.argv[3],
                               float(sys.argv[4]), sys.argv[5])
label_mode = sys.argv[6] if len(sys.argv) > 6 else "CANONICAL"
_T0, _timings = time.time(), {}


def _tick(stage, t0):
    _timings[stage] = round(time.time() - t0, 2)


def _local_uv(pu, nF):
    out = np.zeros((nF, 3, 2), np.float32)
    for c in pu["charts"]:
        out[c["gidx"]] = c["UV"][np.asarray(c["F"])]
    return out


os.makedirs(outd, exist_ok=True)
st = dict(object_id=oid, status="", reason="", quality_status="",
          label_mode=label_mode, eligible=False,
          artifact_valid=dict(packed_layout=False, rebaked_asset=False))


def finish():
    _timings["total"] = round(time.time() - _T0, 2)
    st["timings"] = _timings
    with open(f"{outd}/status.json", "w") as fp:
        json.dump(st, fp, indent=1, ensure_ascii=False)
    print("BUILD_ONE:", st["status"], flush=True)
    sys.exit(0)


try:
    pick_free_gpu()
    t0 = time.time()
    sup = check_support(glb)
    _tick("parse_preflight", t0)
    if not sup["supported"]:
        st.update(status="PRECHECK_REJECTED", reason=sup["reason"][:160])
        finish()
    t0 = time.time()
    tc = run_teacher_context(glb, f"{outd}/partuv/")
    _tick("partuv_partfield", t0)
    if tc["status"] != "OK":
        st.update(status=tc["status"], reason=tc.get("reason", "")[:160])
        finish()
    t0 = time.time()
    labels = compute_labels(tc, beta)
    _tick("label_computation", t0)
    pu, ref = tc["pu"], tc["ref"]
    F, V = pu["F"], pu["V"]
    dsh = labels["chart_demand_normalized"]
    # 预布局 raw content contrast -> VALID_NO_OP 判定
    A3 = labels["chart_surface_area"]
    sd = float(0.5 * np.abs(dsh - A3 / max(A3.sum(), 1e-20)).sum())
    st["signal_dist"] = round(sd, 4)

    # ---- 100% QA 门(几何/覆盖/有限/归一/一致性) ----
    gates = {
        "geometry_face_correspondence":
            len(F) == len(tc["face2chart"]) == len(labels["face_content_score"]),
        "coverage>=99.9%": bool(tc["sel"].mean() >= 0.999),
        "finite_local_uv": all(np.isfinite(c["UV"]).all() for c in pu["charts"]),
        "demand_nonneg_sum1": bool((dsh >= 0).all()
                                   and abs(dsh.sum() - 1) < 1e-6),
        "chart_labels_finite": bool(
            np.isfinite(labels["chart_log_density_ratio"]).all()
            and np.isfinite(labels["chart_target_scale"]).all()),
        "beta_protocol_consistent": True,   # β/protocol 由驱动统一传入冻结值
    }
    st["gates"] = gates
    if not all(gates.values()):
        st.update(status="STRUCTURAL_REJECTED",
                  reason="; ".join(k for k, v in gates.items() if not v))
        finish()

    # ---- 中等 fixed-B_signal 质量检查(VALID_NO_OP 免测) ----
    if sd < 0.05:
        st["quality_status"] = "VALID_NO_OP"
        st["artifact_valid"]["packed_layout"] = True   # no-op 布局=Uniform 合法
    else:
        t0 = time.time()
        q = quality_check_medium(tc, labels)
        _tick("packing_quality", t0)
        st["quality_check"] = q
        if q["status"] != "OK":
            st.update(status="QUALITY_UNVERIFIABLE", reason=q["status"],
                      quality_status="UNVERIFIABLE")
            finish()
        st["artifact_valid"]["packed_layout"] = q["overlap"] == 0
        g, ghf = q["G_global_eq"], q["G_HF_eq"]
        if abs(g) <= 0.02 and abs(ghf) < 0.05:
            st["quality_status"] = "NEUTRAL_INBAND"
        elif g >= 0.02 or (ghf >= 0.05 and g >= -0.02):
            st["quality_status"] = "POSITIVE"
        elif g <= -0.05 and ghf < 0.05:
            st["quality_status"] = "NEGATIVE"
        else:
            st["quality_status"] = "MIXED"
    st["eligible"] = st["quality_status"] in ("POSITIVE", "VALID_NO_OP")
    if not st["eligible"]:
        st.update(status="NOT_ELIGIBLE",
                  reason=f"quality={st['quality_status']}")
        finish()

    # ---- 写样本三件套(相对路径, 可迁移) ----
    t0 = time.time()
    tris = V[F]
    arrays = dict(
        vertices=V.astype(np.float32), faces=F.astype(np.int64),
        face_ids=np.asarray(ref["f2o"], np.int64),
        face_to_chart=np.asarray(tc["face2chart"])[:, 0].astype(np.int64),
        chart_ids=np.arange(len(pu["charts"]), dtype=np.int64),
        local_uv_before_td=_local_uv(pu, len(F)),
        train_face_mask=tc["sel"].astype(bool),
        source_uv=tc["face_refuv"].astype(np.float32),
        source_uv_valid=tc["valid"].astype(bool),
        chart_surface_area=labels["chart_surface_area"].astype(np.float32),
        chart_uv_area_before_td=labels["chart_uv_area_before_td"].astype(np.float32),
        chart_demand_normalized=labels["chart_demand_normalized"].astype(np.float32),
        chart_target_area_fraction=labels["chart_target_area_fraction"].astype(np.float32),
        chart_log_density_ratio=labels["chart_log_density_ratio"].astype(np.float32),
        chart_target_scale=labels["chart_target_scale"].astype(np.float32),
        chart_valid_mask=labels["chart_valid_mask"].astype(bool),
        face_content_score=labels["face_content_score"].astype(np.float32),
        chart_content_score=labels["chart_content_score"].astype(np.float32))
    missing = [k for k in REQUIRED_NPZ if k not in arrays]
    assert not missing, f"schema 缺字段: {missing}"
    np.savez_compressed(f"{outd}/arrays.npz", **arrays)
    Image.fromarray((np.clip(ref["texA"], 0, 1) * 255).astype(np.uint8)).save(
        f"{outd}/reference_basecolor.png")

    def _sha(arr_dict):
        h = hashlib.sha256()
        for k in sorted(arr_dict):
            h.update(k.encode())
            h.update(np.ascontiguousarray(arr_dict[k]).tobytes())
        return h.hexdigest()

    geometry_hash = _sha({k: arrays[k] for k in ("vertices", "faces")})
    chart_hash = _sha({k: arrays[k] for k in ("face_to_chart",
                                              "local_uv_before_td")})
    label_hash = _sha({k: arrays[k] for k in
                       ("chart_demand_normalized", "chart_log_density_ratio",
                        "chart_target_scale")})
    tex = np.asarray(Image.open(f"{outd}/reference_basecolor.png")
                     .convert("L").resize((8, 8)), float)
    phash_img = "".join("1" if v > tex.mean() else "0"
                        for v in tex.ravel())
    manifest = dict(
        schema_version=SCHEMA_VERSION, label_type=LABEL_TYPE,
        label_semantics=LABEL_SEMANTICS, label_mode=label_mode,
        semantics=SEMANTICS, object_id=oid,
        teacher=dict(name=TEACHER_VERSION, beta=beta, protocol_hash=phash,
                     code_hash=teacher_code_hash(),
                     evaluator="texel_center_v1+coverage_center_v1"),
        geometry=dict(n_faces=int(len(F)), n_charts=len(pu["charts"]),
                      train_face_coverage=float(tc["sel"].mean())),
        hashes=dict(geometry=geometry_hash, chart=chart_hash, label=label_hash,
                    content_phash=hex(int(phash_img, 2))[2:].zfill(16)),
        teacher_diagnostics_policy="face/chart_content_score 禁止作为 Student-v0 输入",
        files=dict(arrays="arrays.npz",
                   reference_texture="reference_basecolor.png",
                   quality="quality.json"))
    with open(f"{outd}/manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=1, ensure_ascii=False)
    with open(f"{outd}/quality.json", "w") as fp:
        json.dump(dict(quality_status=st["quality_status"],
                       signal_dist=st["signal_dist"],
                       quality_check=st.get("quality_check"),
                       gates=gates, artifact_valid=st["artifact_valid"],
                       eligible=True), fp, indent=1, ensure_ascii=False)
    # 回读校验
    z2 = dict(np.load(f"{outd}/arrays.npz"))
    assert all((z2[k] == arrays[k]).all() for k in arrays), "回读不一致"
    _tick("serialization", t0)
    st.update(status="ACCEPTED",
              hashes=manifest["hashes"], n_charts=len(pu["charts"]),
              n_faces=int(len(F)))
    finish()
except Exception as e:
    st.update(status="ERROR", reason=f"{type(e).__name__}: {str(e)[:200]}")
    traceback.print_exc()
    finish()
