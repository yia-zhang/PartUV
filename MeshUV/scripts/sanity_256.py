# -*- coding: utf-8 -*-
"""256-object 小范围泛化 sanity: object/geometry-hash 级 split(75/12.5/12.5),
train 训练 -> val/test 报告 masked loss + active-Spearman(不做调参搜索)。"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.dataset import CleanDataset, object_splits  # noqa: E402
from meshuv.data.collate import collate  # noqa: E402
from meshuv.model.student_v0 import StudentV0  # noqa: E402
from meshuv.training.trainer import (train_minibatch,
    evaluate as evaluate_full)  # noqa: E402

DATA_ROOT = os.environ.get("MESHUV_DATA_ROOT", os.path.join(ROOT, "datasets"))


def evaluate(model, items, device):
    return evaluate_full(model, items, collate, device=device)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="processed/clean_v1")
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    root = os.path.join(DATA_ROOT, a.dataset)
    torch.manual_seed(3)
    sp = object_splits(root)
    json.dump(sp, open(f"{root}/splits.json", "w"), indent=1)
    ds = {k: CleanDataset(root, object_ids=v) for k, v in sp.items()}
    print({k: len(v) for k, v in sp.items()})
    train_items = [ds["train"][i] for i in range(len(ds["train"]))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentV0(d=128)
    losses = train_minibatch(model, train_items, collate, steps=a.steps,
                             lr=2e-3, batch_objects=8, device=dev,
                             log_every=1000)
    out = dict(splits={k: len(v) for k, v in sp.items()},
               loss_first=round(losses[0], 6), loss_last=round(losses[-1], 6),
               train=evaluate(model, train_items, dev))
    torch.save(dict(state=model.state_dict(), d=128,
                    model=model.name, seed=3, lr=2e-3, steps=a.steps,
                    uids=sp, device=dev),
               f"{ROOT}/reports/student_v0_sanity.ckpt")
    for split in ("val", "test"):
        items = [ds[split][i] for i in range(len(ds[split]))]
        out[split] = evaluate(model, items, dev)
    json.dump(out, open(f"{ROOT}/reports/sanity_256.json", "w"), indent=1)
    # 可视化: val 第一个对象 target vs pred
    items = [ds["val"][i] for i in range(min(1, len(ds["val"])))]
    if items:
        from meshuv.visualization import pipeline_views as V
        import torch as T
        b = collate(items)
        with T.no_grad():
            pred = model(T.as_tensor(b["features"], device=dev),
                         b["object_ranges"],
                         T.as_tensor(b["valid"], device=dev)).cpu().numpy()
        fig = plt.figure(figsize=(13, 4.5))
        V.show_target(items[0], fig.add_subplot(1, 3, 1))
        V.show_prediction(items[0], pred, fig.add_subplot(1, 3, 2))
        V.show_charts(items[0], fig.add_subplot(1, 3, 3))
        plt.tight_layout()
        plt.savefig(f"{ROOT}/reports/latest_gallery/sanity_val_example.png",
                    dpi=90)
    print(json.dumps(out, indent=1))
    print("SANITY: DONE")


if __name__ == "__main__":
    main()
