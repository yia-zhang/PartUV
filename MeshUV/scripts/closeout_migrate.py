# -*- coding: utf-8 -*-
"""审计驱动的安全迁移: quarantine(可恢复) / 原地重标 / 定向重建标记.
顺序: 读 audit_clean_256.json -> v1-schema 或 canonicalizer 受影响对象移
datasets/quarantine/ -> 仅标签漂移对象原地重标(不重跑 PartUV) -> 报告。
重建由 build_dataset.py 断点续跑自动完成(对象目录已不存在)。"""
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.density.signal import face_content_score  # noqa: E402
from meshuv.density.allocation import chart_targets  # noqa: E402

DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"
Q = f"{DATA}/quarantine"


def relabel_inplace(d):
    from PIL import Image
    z = dict(np.load(f"{d}/arrays.npz"))
    atlas = np.asarray(Image.open(f"{d}/basecolor.png"), float)[:, :, :3] / 255
    sc = face_content_score(atlas, z["source_uv"], z["source_uv_valid"])
    cs = dict(face_to_chart=z["face_to_chart"], face_area=z["face_area"],
              covered=z["train_face_mask"], source_uv_valid=z["source_uv_valid"],
              n_charts=int(z["face_to_chart"].max()) + 1)
    lab = chart_targets(cs, sc)
    z.update(face_content_score=sc,
             chart_content_score=lab["chart_content_score"],
             chart_surface_area=lab["chart_surface_area"],
             chart_target_area_fraction=lab["chart_target_area_fraction"],
             chart_log_density_ratio=lab["chart_log_density_ratio"],
             chart_valid_mask=lab["chart_valid_mask"])
    np.savez_compressed(f"{d}/arrays.npz.tmp.npz", **z)
    os.replace(f"{d}/arrays.npz.tmp.npz", f"{d}/arrays.npz")
    man = json.load(open(f"{d}/manifest.json"))
    man["relabel"] = "clean_teacher_v1 原地重标(signal/targets, 未重跑 PartUV)"
    json.dump(man, open(f"{d}/manifest.json", "w"), indent=1, ensure_ascii=False)


def main():
    audit = json.load(open(f"{DS}/audit_clean_256.json"))
    os.makedirs(Q, exist_ok=True)
    rebuild = set(audit["rebuild_candidates"])
    relabel = set(audit["relabel_candidates"]) - rebuild
    moved, relabeled = [], []
    for oid in sorted(rebuild):
        src = f"{DS}/objects/{oid}"
        if os.path.isdir(src):
            shutil.move(src, f"{Q}/{oid}")
            moved.append(oid)
    for oid in sorted(relabel):
        d = f"{DS}/objects/{oid}"
        if os.path.isdir(d):
            try:
                relabel_inplace(d)
                relabeled.append(oid)
            except Exception as e:
                shutil.move(d, f"{Q}/{oid}")     # 重标失败 -> 隔离, 不留坏样本
                moved.append(oid)
    rep = dict(quarantined=moved, relabeled=relabeled,
               n_quarantined=len(moved), n_relabeled=len(relabeled),
               note="重建经 build_dataset.py 断点补足; quarantine 可恢复")
    json.dump(rep, open(f"{DS}/migration_report.json", "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("n_")}))
    print("MIGRATE: DONE")


if __name__ == "__main__":
    main()
