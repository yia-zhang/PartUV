# -*- coding: utf-8 -*-
"""chart 级 batch 拼接: 变长 chart -> 扁平拼接 + object/chart 归属 mask."""
import numpy as np


def collate_charts(items, feature_fn=None):
    """items: MeshUVTDDataset.__getitem__ 输出列表.
    feature_fn(model_inputs)-> (C,F) chart 特征矩阵; 缺省用几何基线特征
    (log 面积份额 / log UV 面积 / log 面数) —— 不含任何禁用字段。
    返回 dict(features, targets{...}, object_index, chart_mask)。"""
    feats, tgts, obj_idx = [], {k: [] for k in
                               ("chart_demand_normalized",
                                "chart_target_area_fraction",
                                "chart_log_density_ratio",
                                "chart_target_scale")}, []
    valid = []
    for oi, it in enumerate(items):
        mi, tt = it["model_inputs"], it["training_targets"]
        A3 = np.maximum(mi["chart_surface_area"], 1e-12)
        a2 = np.maximum(mi["chart_uv_area_before_td"], 1e-12)
        nfc = np.bincount(mi["face_to_chart"],
                          minlength=len(A3)).astype(float)
        f = (feature_fn(mi) if feature_fn is not None else
             np.stack([np.log(A3 / A3.sum()), np.log(a2),
                       np.log(np.maximum(nfc, 1))], 1))
        feats.append(f)
        for k in tgts:
            tgts[k].append(tt[k])
        valid.append(tt["chart_valid_mask"].astype(bool))
        obj_idx.append(np.full(len(A3), oi))
    return dict(
        features=np.concatenate(feats).astype(np.float32),
        targets={k: np.concatenate(v).astype(np.float32)
                 for k, v in tgts.items()},
        chart_mask=np.concatenate(valid),
        object_index=np.concatenate(obj_idx))
