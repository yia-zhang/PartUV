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
    dup = sp.pop("_dup_audit", {})
    ds = {k: CleanDataset(root, object_ids=v) for k, v in sp.items()}
    print({k: len(v) for k, v in sp.items()}, "| dup_groups:",
          dup.get("n_dup_groups"))
    train_items = [ds["train"][i] for i in range(len(ds["train"]))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = StudentV0(d=128)
    losses = train_minibatch(model, train_items, collate, steps=a.steps,
                             lr=2e-3, batch_objects=8, device=dev,
                             log_every=1000)
    out = dict(splits={k: len(v) for k, v in sp.items()},
               loss_first=round(losses[0], 6), loss_last=round(losses[-1], 6),
               train=evaluate(model, train_items, dev))
    import hashlib, subprocess
    fp = hashlib.sha256("".join(sorted(sum(sp.values(), []))).encode()).hexdigest()[:16]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(ROOT),
                         capture_output=True, text=True).stdout.strip()[:12]
    torch.save(dict(state=model.state_dict(), d=128, model=model.name,
                    seed=3, lr=2e-3, steps=a.steps, batch_objects=8,
                    loss="SmoothL1(object-mean)", scheduler="cosine",
                    schema="meshuv_clean_v1", feature_version="chart_feats_v2_17",
                    signal="luminance_std_v1", label="linear_texel_density_log_ratio_v1",
                    adapter="canonicalizer_rgb_v2+partuv_adapter_v1",
                    dataset_fingerprint=fp, git_sha=sha, uids=sp,
                    dup_audit=dup, device=dev),
               f"{ROOT}/reports/student_v0_sanity.ckpt")
    for split in ("val", "test"):
        items = [ds[split][i] for i in range(len(ds[split]))]
        out[split] = evaluate(model, items, dev)
    # 三个基线(相同训练协议, 特征变换): geometry-only / analytic proxy / RGB shuffle
    from meshuv.data import collate as C

    def make_cf(mode, seed=5):
        base = C.chart_features

        def cf(items_):
            import numpy as _np
            b = C.collate(items_)
            X = b["features"]
            if mode == "geometry_only":
                X[:, 9:] = 0                      # 去掉 RGB/亮度统计列
            elif mode == "analytic_proxy":
                X[:, :9] = 0                      # 只留 RGB/亮度(8 采样 proxy)
            elif mode == "rgb_shuffle":
                rng = _np.random.RandomState(seed)
                for a0, b0 in b["object_ranges"]:
                    perm = rng.permutation(b0 - a0)
                    X[a0:b0, 9:] = X[a0:b0, 9:][perm]
            b["features"] = X
            return b
        return cf

    out["baselines"] = {}
    for mode in ("geometry_only", "analytic_proxy", "rgb_shuffle"):
        mb = StudentV0(d=128)
        train_minibatch(mb, train_items,
                        make_cf(mode), steps=max(a.steps // 2, 1500), lr=2e-3,
                        batch_objects=8, device=dev, log_every=0)
        items_v = [ds["val"][i] for i in range(len(ds["val"]))]
        out["baselines"][mode] = evaluate_full(mb, items_v, make_cf(mode),
                                               device=dev)["macro"]
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
