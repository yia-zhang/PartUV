# -*- coding: utf-8 -*-
"""Phase 1 只读审计: 已 accepted 对象 + 原始缓存 GLB 的 8 项统计.
产出 audit_clean_256.json + audit_clean_256.md; 不修改任何 manifest。"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.asset.loader import load_glb  # noqa: E402
from meshuv.asset.canonicalizer import uv_tile_violation  # noqa: E402
from meshuv.data.schema import CORE, TARGETS  # noqa: E402
from meshuv.density.signal import face_content_score  # noqa: E402
from meshuv.density.allocation import chart_targets  # noqa: E402

DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"
CACHE = f"{DATA}/cache/texverse_1k"


def main():
    from PIL import Image
    rows, cpo, timings = [], [], {}
    st_all = []
    for f in sorted(glob.glob(f"{DS}/objects/*/status.json")):
        st = json.load(open(f))
        st_all.append(st)
        for k, v in st.get("timings", {}).items():
            timings.setdefault(k, []).append(v)
    acc_dirs, schema_bad = [], []
    for f in sorted(glob.glob(f"{DS}/objects/*/manifest.json")):
        d = os.path.dirname(f)
        try:
            st = json.load(open(f"{d}/status.json"))
            if st.get("status") != "ACCEPTED":
                continue
            with np.load(f"{d}/arrays.npz") as zz:
                missing = set(CORE + TARGETS) - set(zz.files)
            if missing:
                schema_bad.append(os.path.basename(d))
                continue
            acc_dirs.append(d)
        except Exception:
            schema_bad.append(os.path.basename(d))
    n_factor = n_nouv_solid = n_nouv_tex = n_oob = n_shift = n_cross = 0
    label_drift = []
    for d in acc_dirs:
        man = json.load(open(f"{d}/manifest.json"))
        oid = man["object_id"]
        uid = oid.replace("tex_", "")
        glbs = glob.glob(f"{CACHE}/{uid}*.glb")
        row = dict(object_id=oid, n_charts=man["n_charts"],
                   n_faces=man["n_faces"],
                   coverage=man.get("coverage_area"),
                   coverage_vs_original=man.get("coverage_vs_original"),
                   retained_area_ratio=man.get("retained_area_ratio"),
                   has_v2_fields="retained_area_ratio" in man)
        cpo.append(man["n_charts"])
        if glbs:
            try:
                geoms = load_glb(glbs[0])
                row["orig_geoms"] = len(geoms)
                oarea = 0.0
                for g in geoms:
                    t = g["V"][g["F"]]
                    oarea += float(np.linalg.norm(np.cross(
                        t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), 2, 1).sum() / 2)
                    if not np.allclose(g["factor"], 1, atol=1e-3):
                        row["factor_ne_1"] = True
                    no_uv = g["uv"] is None or not len(np.atleast_1d(g["uv"]))
                    if no_uv:
                        if g["image"] is not None:
                            row["nouv_textured"] = True
                        else:
                            row["nouv_solid"] = True
                    elif g["uv"] is not None:
                        uv = np.asarray(g["uv"], float)
                        used = np.unique(g["F"])
                        if len(used):
                            u = uv[used]
                            if (u < -0.01).any() or (u > 1.01).any():
                                row["uv_oob"] = True
                                sh = np.floor(u.min(0) + 1e-9)
                                uvn = uv - sh
                                if (sh != 0).any():
                                    row["uv_tile_shift"] = True
                                if uv_tile_violation(uv, g["F"]):
                                    row["uv_cross_tile"] = True
                row["orig_area"] = oarea
            except Exception as e:
                row["orig_load_error"] = type(e).__name__   # 错误也进 rebuild
        n_factor += bool(row.get("factor_ne_1"))
        n_nouv_solid += bool(row.get("nouv_solid"))
        n_nouv_tex += bool(row.get("nouv_textured"))
        n_oob += bool(row.get("uv_oob"))
        n_shift += bool(row.get("uv_tile_shift"))
        n_cross += bool(row.get("uv_cross_tile"))
        # label drift: clean signal 重算 vs 已存标签
        try:
            z = dict(np.load(f"{d}/arrays.npz"))
            atlas = np.asarray(Image.open(f"{d}/basecolor.png"),
                               float)[:, :, :3] / 255.0
            sc = face_content_score(atlas, z["source_uv"], z["source_uv_valid"])
            cs = dict(face_to_chart=z["face_to_chart"], face_area=z["face_area"],
                      covered=z["train_face_mask"],
                      source_uv_valid=z["source_uv_valid"],
                      n_charts=len(z["chart_surface_area"]))
            lab = chart_targets(cs, sc)
            v = z["chart_valid_mask"]
            drift = float(np.abs(lab["chart_log_density_ratio"][v]
                                 - z["chart_log_density_ratio"][v]).max())
            label_drift.append(drift)
            row["label_drift_max"] = round(drift, 5)
        except Exception as e:
            row["label_drift_error"] = type(e).__name__
        rows.append(row)
    cpo = np.array(cpo)
    q = lambda p: float(np.percentile(cpo, p)) if len(cpo) else None
    yield_cnt = Counter(s.get("status", "?") for s in st_all)
    rebuild = sorted(set(
        [r["object_id"] for r in rows
         if r.get("uv_cross_tile") or r.get("nouv_textured")
         or not r.get("has_v2_fields") or r.get("orig_load_error")
         or r.get("label_drift_error")] + schema_bad))
    relabel = [r["object_id"] for r in rows
               if r.get("label_drift_max", 0) > 1e-4]
    audit = dict(
        n_accepted=len(rows), yield_counts=dict(yield_cnt),
        charts=dict(total=int(cpo.sum()), p50=q(50), p90=q(90), p95=q(95),
                    p99=q(99), max=int(cpo.max()) if len(cpo) else 0),
        counts=dict(factor_ne_1=n_factor, nouv_solid=n_nouv_solid,
                    nouv_textured=n_nouv_tex, uv_oob=n_oob,
                    uv_tile_shift=n_shift, uv_cross_tile=n_cross,
                    missing_v2_fields=sum(1 for r in rows
                                          if not r["has_v2_fields"])),
        label_drift=dict(max=float(np.max(label_drift)) if label_drift else 0,
                         p95=float(np.percentile(label_drift, 95))
                         if label_drift else 0,
                         n_gt_1e4=len(relabel)),
        timings={k: dict(sum_min=round(sum(v) / 60, 1),
                         p50=round(float(np.percentile(v, 50)), 1),
                         p90=round(float(np.percentile(v, 90)), 1))
                 for k, v in timings.items()},
        rebuild_candidates=rebuild, relabel_candidates=relabel,
        objects=rows)
    import hashlib, subprocess
    audit["adapter_distribution"] = dict(__import__("collections").Counter(
        json.load(open(f"{d}/manifest.json")).get("source_adapter_version", "?")
        for d in acc_dirs))
    audit["schema_bad"] = schema_bad
    audit["commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(ROOT),
        capture_output=True, text=True).stdout.strip()[:12]
    blob = json.dumps(audit, sort_keys=True, ensure_ascii=False).encode()
    audit["audit_hash"] = hashlib.sha256(blob).hexdigest()[:16]
    json.dump(audit, open(f"{DS}/audit_clean_256.json", "w"), indent=1,
              ensure_ascii=False)
    md = [f"# Clean 256 审计\n",
          f"- accepted {len(rows)}; yield {dict(yield_cnt)}",
          f"- charts: 总 {int(cpo.sum())}, P50 {q(50):.0f}, P90 {q(90):.0f}, "
          f"P95 {q(95):.0f}, P99 {q(99):.0f}, max {int(cpo.max())}",
          f"- factor≠1: {n_factor}; 纯色无UV: {n_nouv_solid}; "
          f"有纹理无UV: {n_nouv_tex}; UV 越界: {n_oob}; "
          f"整图平移: {n_shift}; 跨 tile: {n_cross}",
          f"- 缺 v2 面积字段(v1 构建): "
          f"{sum(1 for r in rows if not r['has_v2_fields'])}",
          f"- label drift(clean 重算 vs 已存): max "
          f"{audit['label_drift']['max']:.5f}, >1e-4 共 {len(relabel)}",
          f"- 需重建 UID(跨tile/有纹理无UV/v1字段/错误/schema): {len(rebuild)}",
          f"- adapter 分布: {audit['adapter_distribution']}",
          f"- audit_hash: {audit['audit_hash']}  commit: {audit['commit']}",
          ]
    open(f"{DS}/audit_clean_256.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("AUDIT: DONE")


if __name__ == "__main__":
    main()
