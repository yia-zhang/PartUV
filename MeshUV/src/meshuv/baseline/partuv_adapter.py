# -*- coding: utf-8 -*-
"""PartUV baseline adapter(当前实现; 未来替换不影响下游).

依赖 PARTUV_ROOT 指向 PartUV teacher checkout(tdlib + PartUV 上游),
本模块是 clean-v1 里唯一接触 tdlib 的位置(除 density 的公式移植说明)。"""
import os
import sys

import numpy as np

from .interface import ChartGenerator, COVERAGE_MIN_AREA
from ..asset.canonicalizer import export_canonical_glb

PARTUV_ROOT = os.environ.get("PARTUV_ROOT", "/root/youjiaZhang/PartUV/code")
BASELINE_VERSION = "partuv_adapter_v1"


def _wire():
    for p in (PARTUV_ROOT, os.path.join(PARTUV_ROOT, "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


class PartUVGenerator(ChartGenerator):
    name = "partuv"

    def generate(self, canon, workdir):
        _wire()
        from tdlib.pipeline import load_reference, run_partuv
        from tdlib.rd import prepare_face_ref_uv
        os.makedirs(workdir, exist_ok=True)
        glb = export_canonical_glb(canon, f"{workdir}/canonical.glb")
        pu = run_partuv(glb, f"{workdir}/partuv/")
        if len(pu["charts"]) == 0 or not pu["covered"].any():
            return dict(status="PARTUV_FAILED",
                        reason=f"无可用 charts({len(pu['charts'])})")
        ref = load_reference(glb, pu["V"], pu["F"], pu["mesh_scale"])
        if not ref.get("has_tex"):
            return dict(status="PARTUV_FAILED", reason="reference 无贴图")
        face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)
        V, F = pu["V"], pu["F"]
        tris = V[F]
        fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                      tris[:, 2] - tris[:, 0]), axis=1) / 2
        covered = pu["covered"] & valid
        cov_area = float(fa3[covered].sum() / max(fa3.sum(), 1e-20))
        if cov_area < COVERAGE_MIN_AREA:
            return dict(status="COVERAGE_REJECTED",
                        reason=f"surface-area coverage {cov_area*100:.2f}% < 99%")
        f2c = np.asarray(face2chart)[:, 0].astype(np.int64)
        f2c[~covered] = -1
        local_uv = np.zeros((len(F), 3, 2), np.float32)
        for c in pu["charts"]:
            local_uv[c["gidx"]] = c["UV"][np.asarray(c["F"])]
        return dict(status="OK", vertices=V.astype(np.float32),
                    faces=F.astype(np.int64), face_to_chart=f2c,
                    local_uv=local_uv, covered=covered,
                    n_charts=len(pu["charts"]),
                    source_uv=np.asarray(face_refuv, np.float32),
                    source_uv_valid=np.asarray(valid, bool),
                    face_area=fa3.astype(np.float32),
                    coverage_area=cov_area,
                    baseline_version=BASELINE_VERSION)
