# -*- coding: utf-8 -*-
"""8-object overfit 验收: loss<初始 1% + 非 no-op Spearman>0.95 + 可视化 PNG.
用法: python scripts/overfit.py [--dataset processed/clean_v1] [--n 8]"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.dataset import CleanDataset  # noqa: E402
from meshuv.data.collate import collate  # noqa: E402
from meshuv.model.student_v0 import StudentV0  # noqa: E402
from meshuv.training.trainer import train  # noqa: E402

DATA_ROOT = os.environ.get("MESHUV_DATA_ROOT", os.path.join(ROOT, "datasets"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="processed/clean_v1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8000)
    a = ap.parse_args()
    root = a.dataset if os.path.isabs(a.dataset) else os.path.join(DATA_ROOT,
                                                                   a.dataset)
    torch.manual_seed(3)
    ds = CleanDataset(root)
    # 确定性选 8 个 chart 数最接近中位数的对象(overfit 验证可训练性,
    # 非容量压力测试; 巨兽对象留给 256 sanity)
    import json as _j
    counts = [( _j.load(open(f"{root}/objects/{o}/manifest.json"))["n_charts"], o)
              for o in ds.ids]
    med = sorted(c for c, _ in counts)[len(counts) // 2]
    chosen = [o for _, o in sorted(counts, key=lambda t: (abs(t[0] - med), t[1]))][:a.n]
    print("selected(median-band):", chosen)
    items = [ds[ds.ids.index(o)] for o in sorted(chosen)]
    batch = collate(items)
    print(f"objects={len(items)} charts={len(batch['features'])} "
          f"valid={int(batch['valid'].sum())}")
    model = StudentV0(d=192)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    r = train(model, batch, steps=a.steps, lr=4e-3, device=dev, log_every=500)
    ratio = r["loss_last"] / max(r["loss_first"], 1e-12)
    out = dict(n_objects=len(items), n_charts=len(batch["features"]),
               loss_first=round(r["loss_first"], 6),
               loss_last=round(r["loss_last"], 6),
               loss_ratio=round(ratio, 5),
               pass_loss=bool(ratio < 0.01),
               spearman_active=round(r["spearman_active"], 4),
               pass_spearman=bool(r["spearman_active"] > 0.95))
    os.makedirs(f"{ROOT}/reports", exist_ok=True)
    json.dump(out, open(f"{ROOT}/reports/overfit_8.json", "w"), indent=1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].semilogy(r["losses"]); axs[0].set_title("overfit loss (SmoothL1)")
    m = batch["valid"]
    axs[1].scatter(batch["target"][m], r["pred"][m], s=8, alpha=0.6)
    lim = np.abs(batch["target"][m]).max() * 1.1
    axs[1].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axs[1].set_xlabel("target log-ratio"); axs[1].set_ylabel("prediction")
    axs[1].set_title(f"Spearman(active)={out['spearman_active']}")
    plt.tight_layout()
    os.makedirs(f"{ROOT}/reports/latest_gallery", exist_ok=True)
    plt.savefig(f"{ROOT}/reports/latest_gallery/overfit_8.png", dpi=100)
    print(json.dumps(out, indent=1))
    print("OVERFIT:", "PASS" if out["pass_loss"] and out["pass_spearman"]
          else "FAIL")


if __name__ == "__main__":
    main()
