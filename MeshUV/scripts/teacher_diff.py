# -*- coding: utf-8 -*-
"""clean_teacher_v1 vs 旧 frozen teacher(tdlib canonical 实现)标签差异报告.
在代表性 accepted 对象上: 同一 source_uv/atlas 输入, 分别经
tdlib.luminance_std_heuristic+demand_weights 与 clean signal/allocation,
比较 chart_log_density_ratio。差异小 -> 记录并冻结 clean_teacher_v1。"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
PARTUV_ROOT = os.environ.get("PARTUV_ROOT", "/root/youjiaZhang/PartUV/code")
sys.path.insert(0, PARTUV_ROOT)
from meshuv.density.signal import face_content_score  # noqa: E402
from meshuv.density.allocation import chart_targets, BETA  # noqa: E402

DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"


def frozen_labels(z, atlas):
    from tdlib.signal import luminance_std_heuristic, demand_weights
    F = len(z["faces"])
    uv0 = z["source_uv"].reshape(-1, 2).astype(float)
    Fo = np.arange(F * 3).reshape(F, 3)
    sel = z["train_face_mask"].astype(bool)
    cw = luminance_std_heuristic(atlas, uv0, Fo, np.arange(F), sel)
    _, w = demand_weights(cw, sel, z["face_area"].astype(float), beta=BETA)
    f2c, fa = z["face_to_chart"], z["face_area"].astype(float)
    nC = int(f2c.max()) + 1
    dem, A3 = np.zeros(nC), np.zeros(nC)
    m = f2c >= 0
    np.add.at(dem, f2c[m], (fa * w)[m])
    np.add.at(A3, f2c[m], fa[m])
    valid = (dem > 0) & (A3 > 0)
    dsh = dem / max(dem.sum(), 1e-20)
    ash = A3 / max(A3.sum(), 1e-20)
    lr = np.zeros(nC)
    lr[valid] = 0.5 * np.log(np.maximum(dsh[valid], 1e-20)
                             / np.maximum(ash[valid], 1e-20))
    lr[valid] -= lr[valid].mean()
    return lr, valid


def main():
    from PIL import Image
    dirs = sorted(glob.glob(f"{DS}/objects/*/manifest.json"))[::max(
        1, len(glob.glob(f"{DS}/objects/*/manifest.json")) // 12)][:12]
    diffs = []
    for f in dirs:
        d = os.path.dirname(f)
        z = dict(np.load(f"{d}/arrays.npz"))
        atlas = np.asarray(Image.open(f"{d}/basecolor.png"),
                           float)[:, :, :3] / 255
        lr_old, v = frozen_labels(z, atlas)
        sc = face_content_score(atlas, z["source_uv"], z["source_uv_valid"])
        cs = dict(face_to_chart=z["face_to_chart"], face_area=z["face_area"],
                  covered=z["train_face_mask"],
                  source_uv_valid=z["source_uv_valid"],
                  n_charts=int(z["face_to_chart"].max()) + 1)
        lr_new = chart_targets(cs, sc)["chart_log_density_ratio"]
        dd = float(np.abs(lr_new[v] - lr_old[v]).max())
        diffs.append(dict(object_id=os.path.basename(d), max_abs_diff=round(dd, 5)))
    mx = max(d["max_abs_diff"] for d in diffs)
    # clean_teacher_v1 protocol/code hash
    files = ["src/meshuv/asset/canonicalizer.py", "src/meshuv/density/signal.py",
             "src/meshuv/density/allocation.py",
             "src/meshuv/baseline/partuv_adapter.py"]
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode()); h.update(open(f"{ROOT}/{rel}", "rb").read())
    h.update(open(f"{PARTUV_ROOT}/notebook/partuv_config.yaml", "rb").read())
    rep = dict(teacher="clean_teacher_v1", beta=BETA,
               code_hash=h.hexdigest()[:16],
               comparison="tdlib frozen(luminance_std+demand) vs clean 实现, "
                          "同一 canonical atlas/source_uv 输入",
               n_objects=len(diffs), max_abs_logr_diff=mx,
               verdict=("SMALL_DIFF_FROZEN" if mx < 0.05
                        else "SIGNIFICANT_REPORT_FIRST"),
               per_object=diffs)
    json.dump(rep, open(f"{ROOT}/reports/teacher_diff_report.json", "w"),
              indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ("max_abs_logr_diff", "verdict",
                                          "code_hash")}))
    print("TEACHER_DIFF: DONE")


if __name__ == "__main__":
    main()
