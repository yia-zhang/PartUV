# -*- coding: utf-8 -*-
"""最小训练循环(masked SmoothL1), 附 Spearman 评估."""
import numpy as np
import torch


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def train(model, batch, steps=400, lr=1e-3, device="cpu", log_every=50):
    X = torch.as_tensor(batch["features"], device=device)
    y = torch.as_tensor(batch["target"], device=device)
    m = torch.as_tensor(batch["valid"], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = torch.nn.SmoothL1Loss()
    model.to(device).train()
    losses = []
    for step in range(steps):
        pred = model(X, batch["object_ranges"], m)
        loss = lossf(pred[m], y[m])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        assert torch.isfinite(loss), f"NaN@{step}"
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"  step {step}: loss={losses[-1]:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        pred = model(X, batch["object_ranges"], m)
    mn = m.cpu().numpy()
    # 非 no-op charts: |target|>1e-4
    act = mn & (np.abs(batch["target"]) > 1e-4)
    sp = spearman(pred.cpu().numpy()[act], batch["target"][act])
    return dict(losses=losses, loss_first=losses[0], loss_last=losses[-1],
                spearman_active=sp, pred=pred.cpu().numpy())
