# -*- coding: utf-8 -*-
"""object mini-batch 训练 + 宏/微指标(tie-aware Spearman)."""
import numpy as np
import torch


def spearman(a, b):
    """tie-aware(scipy); 常量输入返回 nan."""
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _object_loss(pred, y, m, lossf):
    """先对象内均值, 再跨对象均值."""
    if not m.any():
        return None
    return lossf(pred[m], y[m])


def train_minibatch(model, items, collate_fn, steps=3000, lr=2e-3,
                    batch_objects=8, device="cpu", seed=3, log_every=500):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = torch.nn.SmoothL1Loss()
    model.to(device).train()
    losses = []
    cache = {}
    for step in range(steps):
        ix = rng.choice(len(items), min(batch_objects, len(items)),
                        replace=False)
        key = tuple(sorted(ix))
        if key not in cache:
            cache[key] = collate_fn([items[i] for i in ix])
            if len(cache) > 64:
                cache.pop(next(iter(cache)))
        b = cache[key]
        X = torch.as_tensor(b["features"], device=device)
        y = torch.as_tensor(b["target"], device=device)
        m = torch.as_tensor(b["valid"], device=device)
        pred = model(X, b["object_ranges"], m)
        obj_losses = []
        for a, e in b["object_ranges"]:
            l = _object_loss(pred[a:e], y[a:e], m[a:e], lossf)
            if l is not None:
                obj_losses.append(l)
        loss = torch.stack(obj_losses).mean()      # 对象内均值 -> 对象间均值
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        assert torch.isfinite(loss)
        losses.append(float(loss))
        if log_every and step % log_every == 0:
            print(f"  step {step}: loss={losses[-1]:.5f}", flush=True)
    return losses


@torch.no_grad()
def evaluate(model, items, collate_fn, device="cpu"):
    """micro(chart 加权) + macro(object 加权) 指标 + per-object Spearman."""
    model.eval()
    mse_o, mae_o, sl1_o, sp_o = [], [], [], []
    allp, ally = [], []
    n_act_charts = 0
    sl1 = torch.nn.SmoothL1Loss(reduction="none")
    for it in items:
        b = collate_fn([it])
        X = torch.as_tensor(b["features"], device=device)
        m = torch.as_tensor(b["valid"], device=device)
        pred = model(X, b["object_ranges"], m).cpu().numpy()
        y, mn = b["target"], b["valid"]
        if not mn.any():
            continue
        e = pred[mn] - y[mn]
        mse_o.append(float((e ** 2).mean()))
        mae_o.append(float(np.abs(e).mean()))
        sl1_o.append(float(sl1(torch.as_tensor(pred[mn]),
                               torch.as_tensor(y[mn])).mean()))
        act = mn & (np.abs(y) > 1e-4)
        if act.sum() >= 3:
            sp_o.append(spearman(pred[act], y[act]))
            n_act_charts += int(act.sum())
        allp.append(pred[mn]); ally.append(y[mn])
    allp, ally = np.concatenate(allp), np.concatenate(ally)
    sp = np.array([s for s in sp_o if np.isfinite(s)])
    return dict(
        micro=dict(mse=float(((allp - ally) ** 2).mean()),
                   mae=float(np.abs(allp - ally).mean()),
                   spearman_all=spearman(allp, ally),
                   n_charts=len(ally)),
        macro=dict(mse=float(np.mean(mse_o)), mae=float(np.mean(mae_o)),
                   smooth_l1=float(np.mean(sl1_o)),
                   spearman_median=float(np.median(sp)) if len(sp) else None,
                   spearman_p25=float(np.percentile(sp, 25)) if len(sp) else None,
                   n_objects=len(mse_o), n_active_objects=len(sp),
                   n_active_charts=n_act_charts))
