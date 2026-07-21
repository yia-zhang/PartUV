# -*- coding: utf-8 -*-
"""Student-v0(handcrafted-signal baseline): O(C) 结构.
per-chart encoder -> object pooled context(valid 加权 mean/max) -> 拼接
-> head -> 每 chart 标量 -> object 内(valid) mean centering。
invalid charts 不参与 context 池化。无 O(C²) attention。"""
import torch
import torch.nn as nn

N_FEATS = 17


class StudentV0(nn.Module):
    name = "handcrafted_signal_baseline"

    def __init__(self, d=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_FEATS, d), nn.ReLU(),
                                 nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d * 3, d), nn.ReLU(),
                                  nn.Linear(d, 1))

    def forward(self, feats, object_ranges, valid):
        h = self.enc(feats)
        out = torch.zeros(len(feats), device=feats.device)
        for a, b in object_ranges:
            hv, m = h[a:b], valid[a:b]
            if m.any():
                ctx_mean = hv[m].mean(0)
                ctx_max = hv[m].max(0).values
            else:
                ctx_mean = ctx_max = torch.zeros_like(hv[0])
            ctx = torch.cat([ctx_mean, ctx_max]).expand(b - a, -1)
            y = self.head(torch.cat([hv, ctx], 1)).squeeze(-1)
            if m.any():
                y = y - y[m].mean()
            out[a:b] = y
        return out
