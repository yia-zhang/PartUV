# -*- coding: utf-8 -*-
"""可视化数据读取(只读; 缺文件/缺字段一律返回 None + 提示, 不抛不崩).
notebook 只调用这里的公开函数, 不复制 builder/dataset/Teacher 逻辑。"""
import json
import os

import numpy as np

MESHUV_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def resolve_root(path_str):
    """相对路径按 MeshUV 项目根解析(pilot/正式/未来 model run 均可切换)."""
    p = path_str if os.path.isabs(path_str) else os.path.join(MESHUV_ROOT,
                                                              path_str)
    if not os.path.isdir(p):
        print(f"[缺失] dataset root 不存在: {p}")
        return None
    return p


def _jload(p, label):
    if not os.path.exists(p):
        print(f"[缺失] {label}: {p}")
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"[损坏] {label}: {type(e).__name__}")
        return None


def load_reports(root):
    """yield/rejection/splits/run_manifest/summary(存在哪个读哪个)."""
    return dict(
        yield_=_jload(f"{root}/processing_yield.json", "processing_yield")
        or _jload(f"{root}/pilot_summary.json", "pilot_summary"),
        rejections=_jload(f"{root}/rejection_summary.json", "rejection_summary")
        or _jload(f"{root}/rejection_report.json", "rejection_report"),
        splits=_jload(f"{root}/splits.json", "splits"),
        run_manifest=_jload(f"{root}/run_manifest.json", "run_manifest"))


def load_index(root):
    p = f"{root}/dataset_index.jsonl"
    if not os.path.exists(p):
        print(f"[缺失] dataset_index.jsonl: {p}")
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def pick_object(index, uid=None, rank=None, seed=None):
    """按 UID / selection rank / 随机种子三选一挑对象."""
    if not index:
        print("[缺失] index 为空")
        return None
    if uid:
        r = next((r for r in index if r.get("uid", "").startswith(uid)
                  or r["object_id"].endswith(uid[:12])), None)
        if r is None:
            print(f"[未找到] uid={uid}")
        return r
    if rank is not None:
        by = sorted(index, key=lambda r: r.get("selection_rank", 0))
        return by[min(rank, len(by) - 1)]
    rng = np.random.RandomState(seed)
    r = index[rng.randint(len(index))]
    print(f"seed={seed} -> {r['object_id']}(复现: 固定该 seed)")
    return r


def open_sample(root, rec, diagnostics=True):
    """读取单个 accepted 样本(loader 语义, 含 teacher diagnostics 供 QA 查看)."""
    import sys
    sys.path.insert(0, os.path.join(MESHUV_ROOT, "src"))
    from meshuv.data.dataset import MeshUVTDDataset
    try:
        ds = MeshUVTDDataset(root, expose_diagnostics=diagnostics)
        i = next(k for k, r in enumerate(ds.index)
                 if r["object_id"] == rec["object_id"])
        return ds[i]
    except Exception as e:
        print(f"[读取失败] {rec.get('object_id')}: {type(e).__name__}: {e}")
        return None


def scan_label_stats(root, index, cap=64):
    """抽样(<=cap)扫描 npz 的 chart 数/logr/scale 分布(懒加载, 不改数据)."""
    stats = dict(n_charts=[], logr=[], scale=[], timing=[])
    for rec in index[:cap]:
        d = os.path.join(root, rec["sample_dir"])
        try:
            z = np.load(f"{d}/arrays.npz")
            m = z["chart_valid_mask"]
            stats["n_charts"].append(int(m.sum()))
            stats["logr"].extend(z["chart_log_density_ratio"][m].tolist())
            stats["scale"].extend(z["chart_target_scale"][m].tolist())
        except Exception:
            pass
        st = _jload(f"{d}/status.json", "status") or {}
        t = st.get("timings", {}).get("wall_total")
        if t:
            stats["timing"].append(t)
    if len(index) > cap:
        print(f"[提示] 分布统计基于前 {cap}/{len(index)} 个样本抽样")
    return stats
