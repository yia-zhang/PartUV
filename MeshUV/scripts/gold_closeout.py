# -*- coding: utf-8 -*-
"""held-out Gold closeout: 从 checkpoint 的 test UID 确定性选 >=5 对象,
Uniform/Teacher/Student 同 raw budget -> reference MSE/PSNR/HF + Gallery。"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.dataset import CleanDataset  # noqa: E402
from meshuv.data.collate import collate  # noqa: E402
from meshuv.model.student_v0 import StudentV0  # noqa: E402
from meshuv.evaluation.gold_evaluator import compare_methods  # noqa: E402

DATA = os.environ.get("MESHUV_DATA_ROOT", f"{ROOT}/datasets")
DS = f"{DATA}/processed/clean_v1"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    ck = torch.load(f"{ROOT}/reports/student_v0_sanity.ckpt",
                    map_location="cpu")
    test_ids = sorted(ck["uids"]["test"])[:5]      # 确定性选样
    model = StudentV0(d=ck["d"]); model.load_state_dict(ck["state"])
    model.eval()
    ds = CleanDataset(DS)
    out, gal = [], f"{ROOT}/reports/latest_gallery"
    os.makedirs(gal, exist_ok=True)
    for oid in test_ids:
        if oid not in ds.ids:
            out.append(dict(object_id=oid, status="MISSING_AFTER_MIGRATION"))
            continue
        item = ds[ds.ids.index(oid)]
        b = collate([item])
        with torch.no_grad():
            pred = model(torch.as_tensor(b["features"]), b["object_ranges"],
                         torch.as_tensor(b["valid"])).numpy()
        z = item["inputs"]
        fa, f2c = z["face_area"].astype(float), z["face_to_chart"]
        nC = int(f2c.max()) + 1
        A3 = np.zeros(nC); m = f2c >= 0
        np.add.at(A3, f2c[m], fa[m])
        ash = A3 / max(A3.sum(), 1e-12)
        sf = ash * np.exp(2 * pred)
        sf = sf / max(sf.sum(), 1e-12)
        r = compare_methods(DS, item, student_fraction=sf)
        if r["status"] != "OK":
            out.append(dict(object_id=oid, status=r["status"]))
            continue
        ov = sum(v["ov"] for v in r["metrics"].values())
        row = dict(object_id=oid, status="OK" if ov == 0 else "OVERLAP_FAIL",
                   overlap=ov,
                   **{f"{n}_{k}": r["metrics"][n][k] for n in r["metrics"]
                      for k in ("mse", "psnr", "hf_mse", "occ")})
        out.append(row)
        names = list(r["metrics"])
        fig = plt.figure(figsize=(4.6 * (len(names) + 1), 8.6))
        cols = len(names) + 1
        ax = fig.add_subplot(2, cols, 1)
        from PIL import Image
        ax.imshow(np.asarray(Image.open(item["basecolor"])))
        ax.set_axis_off(); ax.set_title("source basecolor", fontsize=9)
        r["draw"]["diff"](fig.add_subplot(2, cols, cols + 1))
        for k, n in enumerate(names):
            r["draw"][f"{n}_uv"](fig.add_subplot(2, cols, k + 2))
            r["draw"][f"{n}_tex"](fig.add_subplot(2, cols, cols + k + 2))
        plt.suptitle(f"{oid} | {r['metrics_text'].splitlines()[0]}", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{gal}/gold_{oid}.png", dpi=75)
        plt.close(fig)
    rep = dict(selection=dict(rule="checkpoint test UID 排序前 5",
                              uids=test_ids, commit=ck.get("git_sha"),
                              fingerprint=ck.get("dataset_fingerprint")),
               results=out)
    json.dump(rep, open(f"{ROOT}/reports/gold_closeout.json", "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(out, indent=1)[:1500])
    print("GOLD: DONE")


if __name__ == "__main__":
    main()
