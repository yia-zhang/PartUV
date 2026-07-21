# -*- coding: utf-8 -*-
"""纹理需求信号 v1: 每面亮度标准差(移植自已冻结 teacher 语义).

24 个固定 Dirichlet(1.2) 重心样本(seed=0), nearest 采样,
Y=0.299R+0.587G+0.114B —— 对比度启发式(已知局限: 幅值非频率)。
仅作 Teacher diagnostic 与 label 生成输入, 禁止作为 Student 输入。"""
import numpy as np

K_SAMPLES = 24
SIGNAL_VERSION = "luminance_std_v1"
_BARY = np.random.RandomState(0).dirichlet((1.2, 1.2, 1.2), K_SAMPLES)


def face_content_score(atlas, source_uv, valid):
    """atlas(H,W,3 float) + source_uv(F,3,2) -> (F,) 亮度 std."""
    H, W = atlas.shape[:2]
    lum = atlas @ np.array([0.299, 0.587, 0.114])
    F = len(source_uv)
    uv = np.einsum("kj,fjd->fkd", _BARY, source_uv.astype(float))  # (F,K,2)
    x = np.clip((uv[..., 0] * W).astype(int), 0, W - 1)
    y = np.clip(((1 - uv[..., 1]) * H).astype(int), 0, H - 1)
    s = lum[y, x]
    out = s.std(axis=1)
    out[~np.asarray(valid, bool)] = 0.0
    return out.astype(np.float32)
