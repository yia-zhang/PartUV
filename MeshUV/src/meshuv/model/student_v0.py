# -*- coding: utf-8 -*-
"""Student-v0: chart-token MLP + 2 层 object 内 Transformer -> 每 chart 标量.
object-wise mean centering 使输出与标签同分布(标签均值居中)。"""
import torch
import torch.nn as nn

N_FEATS = 17


class StudentV0(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(N_FEATS, d), nn.ReLU(),
                                   nn.Linear(d, d), nn.ReLU())
        layer = nn.TransformerEncoderLayer(d, heads, d * 2, batch_first=True,
                                           dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(d, 1)

    def forward(self, feats, object_ranges, valid):
        """feats(T,F) 扁平 chart tokens; 逐 object 过 Transformer(变长)."""
        h = self.embed(feats)
        out = torch.zeros(len(feats), device=feats.device)
        for a, b in object_ranges:
            tok = self.tf(h[a:b].unsqueeze(0)).squeeze(0)
            y = self.head(tok).squeeze(-1)
            m = valid[a:b]
            if m.any():
                y = y - y[m].mean()               # object-wise mean centering
            out[a:b] = y
        return out
