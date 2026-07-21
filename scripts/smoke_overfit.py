# -*- coding: utf-8 -*-
"""8-object overfit smoke: batch 可加载 / target shape 正确 / loss 下降 /
无 NaN / mask 正确. 非正式训练。
用法: python scripts/smoke_overfit.py --config configs/student_v0.yaml"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.dataset import MeshUVTDDataset  # noqa: E402
from meshuv.data.collate import collate_charts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/student_v0.yaml")
    cfg = yaml.safe_load(open(os.path.join(ROOT, ap.parse_args().config)
                              if not os.path.isabs(ap.parse_args().config)
                              else ap.parse_args().config))
    import torch
    torch.manual_seed(cfg["seed"])
    ds = MeshUVTDDataset(os.path.join(ROOT, cfg["dataset_root"]), split="train")
    items = [ds[i] for i in range(min(cfg["n_objects"], len(ds)))]
    b = collate_charts(items)
    X = torch.as_tensor(b["features"])
    y = torch.as_tensor(b["targets"][cfg["target"]])
    m = torch.as_tensor(b["chart_mask"])
    assert X.ndim == 2 and y.shape == (X.shape[0],) and m.shape == y.shape, \
        "target/mask shape 错误"
    assert m.any(), "chart mask 全空"
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], cfg["hidden"]), torch.nn.ReLU(),
        torch.nn.Linear(cfg["hidden"], cfg["hidden"]), torch.nn.ReLU(),
        torch.nn.Linear(cfg["hidden"], 1))
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    losses = []
    for step in range(cfg["steps"]):
        pred = net(X).squeeze(-1)
        loss = ((pred - y)[m] ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss), f"NaN at step {step}"
        losses.append(float(loss))
    out = dict(n_objects=len(items), n_charts=int(X.shape[0]),
               n_valid=int(m.sum()), loss_first=round(losses[0], 5),
               loss_last=round(losses[-1], 5),
               loss_decreased=losses[-1] < losses[0] * 0.5,
               nan_free=True)
    os.makedirs(os.path.join(ROOT, "runs/smoke"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runs/smoke/overfit_smoke.json"),
                        "w"), indent=1)
    print(json.dumps(out, indent=1))
    assert out["loss_decreased"], "loss 未显著下降"
    print("SMOKE: PASS")


if __name__ == "__main__":
    main()
