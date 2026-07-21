# -*- coding: utf-8 -*-
"""单对象构建: canonicalize -> baseline charts -> signal -> targets -> 落盘."""
import hashlib
import json
import os
import time

import numpy as np
from PIL import Image

from ..asset.canonicalizer import canonicalize
from ..baseline.partuv_adapter import PartUVGenerator
from ..density.signal import face_content_score, SIGNAL_VERSION
from ..density.allocation import chart_targets, BETA, LABEL_SEMANTICS
from .schema import SCHEMA_VERSION, REQUIRED_NPZ


def _git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


def _face_source_on_processed(canon, cs):
    """canonical face_source 映射到 PartUV 处理后 mesh(面心最近邻)."""
    Vc, Fc = canon["V"], canon["F"]
    cc = Vc[Fc].mean(1)
    Vp, Fp = cs["vertices"].astype(float), cs["faces"]
    pc = Vp[Fp].mean(1)
    try:
        from scipy.spatial import cKDTree
        idx = cKDTree(cc).query(pc, k=1)[1]
    except Exception:
        idx = np.argmin(((pc[:, None] - cc[None]) ** 2).sum(-1), 1)
    return canon["face_source"][idx].astype(np.int64)


def build_object(glb_path, object_id, out_dir):
    """返回 status dict(写盘 status.json; ACCEPTED 时写全套样本)."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.monotonic()
    st = dict(object_id=object_id, status="", warnings=[], timings={})

    def tick(k, t):
        st["timings"][k] = round(time.monotonic() - t, 2)

    def done(status, reason=""):
        st.update(status=status, reason=reason)
        st["timings"]["total"] = round(time.monotonic() - t0, 2)
        with open(f"{out_dir}/status.json.tmp", "w") as fp:
            json.dump(st, fp, ensure_ascii=False, indent=1)
        os.replace(f"{out_dir}/status.json.tmp", f"{out_dir}/status.json")
        return st

    t = time.monotonic()
    try:
        canon = canonicalize(glb_path)
    except ValueError as e:
        r = str(e)
        code = ("TILED_UV_UNSUPPORTED" if "TILED" in r else
                "TEXTURED_NO_UV" if "TEXTURED_NO_UV" in r else
                "PRECHECK_REJECTED")
        return done(code, r)
    except Exception as e:
        return done("UNPARSABLE", f"{type(e).__name__}: {str(e)[:100]}")
    st["warnings"] += canon["warnings"]
    tick("canonicalize", t)

    t = time.monotonic()
    cs = PartUVGenerator().generate(canon, f"{out_dir}/work")
    tick("baseline_charts", t)
    if cs.get("status") != "OK":
        return done(cs.get("status", "PARTUV_FAILED"), cs.get("reason", ""))

    t = time.monotonic()
    score = face_content_score(canon["atlas"], cs["source_uv"],
                               cs["source_uv_valid"])
    lab = chart_targets(cs, score)
    tick("labels", t)
    if not (np.isfinite(cs["vertices"]).all()
            and np.isfinite(lab["chart_log_density_ratio"]).all()):
        return done("NONFINITE", "几何或标签非有限")

    t = time.monotonic()
    arrays = dict(
        vertices=cs["vertices"], faces=cs["faces"],
        face_to_chart=cs["face_to_chart"], local_uv=cs["local_uv"],
        source_uv=cs["source_uv"], source_uv_valid=cs["source_uv_valid"],
        train_face_mask=(cs["covered"] & cs["source_uv_valid"]),
        face_area=cs["face_area"],
        face_source=_face_source_on_processed(canon, cs),
        chart_surface_area=lab["chart_surface_area"],
        chart_target_area_fraction=lab["chart_target_area_fraction"],
        chart_log_density_ratio=lab["chart_log_density_ratio"],
        chart_valid_mask=lab["chart_valid_mask"],
        face_content_score=score,
        chart_content_score=lab["chart_content_score"])
    assert not (set(REQUIRED_NPZ) - set(arrays)), "schema 缺字段"
    np.savez_compressed(f"{out_dir}/arrays.npz", **arrays)
    Image.fromarray((np.clip(canon["atlas"], 0, 1) * 255).astype(np.uint8)
                    ).save(f"{out_dir}/basecolor.png")
    h = lambda *ks: hashlib.sha256(b"".join(
        np.ascontiguousarray(arrays[k]).tobytes() for k in ks)).hexdigest()[:16]
    manifest = dict(schema_version=SCHEMA_VERSION, object_id=object_id,
                    label_semantics=LABEL_SEMANTICS, beta=BETA,
                    signal_version=SIGNAL_VERSION,
                    baseline_version=cs["baseline_version"],
                    source_adapter_version=canon["adapter_version"],
                    git_sha=_git_sha(),
                    n_faces=int(len(cs["faces"])), n_charts=cs["n_charts"],
                    coverage_area=cs["coverage_area"],
                    original=canon["original"], retained=canon["retained"],
                    retained_area_ratio=canon["retained_area_ratio"],
                    coverage_vs_original=cs["coverage_area"]
                    * canon["retained_area_ratio"],
                    geometry_hash=h("vertices", "faces"),
                    content_hash=h("face_content_score"),
                    diagnostics_policy="face/chart_content_score 禁作 Student 输入")
    with open(f"{out_dir}/manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=1, ensure_ascii=False)
    tick("serialize", t)
    st["n_charts"] = cs["n_charts"]
    st["n_faces"] = int(len(cs["faces"]))
    return done("ACCEPTED")
