# -*- coding: utf-8 -*-
"""统一可视化函数(notebook 与静态 gallery 共用; 缺数据软提示不崩)."""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DATA_ROOT = os.environ.get("MESHUV_DATA_ROOT", os.path.join(ROOT, "datasets"))


def _notice(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=10, color="#888")
    ax.set_axis_off()


def resolve(dataset):
    p = dataset if os.path.isabs(dataset) else os.path.join(DATA_ROOT, dataset)
    if not os.path.isdir(p):
        print(f"[missing] {p}")
        return None
    return p


def pick(ds, uid=None, rank=None, seed=None):
    if uid:
        j = next((i for i, o in enumerate(ds.ids) if uid in o), None)
        if j is None:
            print(f"[missing] uid={uid}")
        return j
    if rank is not None:
        return min(rank, len(ds) - 1)
    j = int(np.random.RandomState(seed).randint(len(ds)))
    print(f"seed={seed} -> {ds.ids[j]}")
    return j


def _uv_tri(ax, item, face_vals, title, cmap="viridis", sym=False, vmax=None):
    uv = item["inputs"]["source_uv"].reshape(-1, 2)
    tris = np.arange(len(uv)).reshape(-1, 3)
    kw = {}
    if sym:
        v = vmax if vmax is not None else max(float(np.abs(face_vals).max()), 1e-3)
        kw = dict(vmin=-v, vmax=v)
    tp = ax.tripcolor(uv[:, 0], uv[:, 1], tris, facecolors=face_vals,
                      cmap=cmap, **kw)
    ax.set_aspect("equal"); ax.set_axis_off(); ax.set_title(title, fontsize=9)
    plt.colorbar(tp, ax=ax, fraction=0.04)


def show_basecolor(item, ax):
    ax.imshow(np.asarray(Image.open(item["basecolor"])))
    ax.set_axis_off(); ax.set_title("canonical basecolor", fontsize=9)


def show_mesh(item, ax, max_faces=20000):
    V, F = item["inputs"]["vertices"], item["inputs"]["faces"]
    if len(F) > max_faces:
        F = F[np.linspace(0, len(F) - 1, max_faces).astype(int)]
    try:
        ax.plot_trisurf(V[:, 0], V[:, 2], V[:, 1], triangles=F,
                        color="#b8bcc4", edgecolor="none")
        ax.set_axis_off()
        ax.set_title(f"mesh ({len(item['inputs']['faces']):,} faces)", fontsize=9)
    except Exception as e:
        _notice(ax, f"mesh fail: {type(e).__name__}")


def show_charts(item, ax):
    f2c = item["inputs"]["face_to_chart"]
    _uv_tri(ax, item, (np.maximum(f2c, 0) % 20).astype(float),
            f"charts ({f2c.max() + 1})", cmap="tab20")


def show_coverage(item, ax):
    _uv_tri(ax, item, item["inputs"]["train_face_mask"].astype(float),
            f"train coverage {item['inputs']['train_face_mask'].mean()*100:.1f}%",
            cmap="RdYlGn")


def show_signal(item, ax):
    d = item.get("diagnostics")
    if d is None:
        return _notice(ax, "diagnostics not exposed")
    _uv_tri(ax, item, d["face_content_score"],
            "texture signal (diagnostic, not a Student input)", cmap="magma")


def show_target(item, ax, vmax=None):
    tt = item["targets"]
    logr = tt["chart_log_density_ratio"][np.maximum(
        item["inputs"]["face_to_chart"], 0)]
    logr = np.where(item["inputs"]["face_to_chart"] >= 0, logr, 0)
    _uv_tri(ax, item, logr, "target TD log-ratio (linear density)",
            cmap="coolwarm", sym=True, vmax=vmax)


def show_geometry_groups(item, ax=None):
    fs = item["inputs"].get("face_source")
    if fs is None or (hasattr(fs, "max") and fs.max() < 0):
        return _notice(ax, "face_source 缺失(v1 样本)")
    _uv_tri(ax, item, (fs % 10).astype(float),
            f"geometry groups ({int(fs.max()) + 1})", cmap="tab10")


def show_prediction(item, pred_chart, ax, vmax=None):
    p = pred_chart[np.maximum(item["inputs"]["face_to_chart"], 0)]
    p = np.where(item["inputs"]["face_to_chart"] >= 0, p, 0)
    _uv_tri(ax, item, p, "Student prediction log-ratio", cmap="coolwarm",
            sym=True, vmax=vmax)


def object_summary(item):
    man = item["manifest"]
    print(f"{item['object_id']}: faces={man['n_faces']:,} charts={man['n_charts']} "
          f"coverage={man['coverage_area']*100:.2f}% beta={man.get('beta')}")
    tt = item["targets"]
    v = tt["chart_valid_mask"]
    print(f"target_area_fraction sum={tt['chart_target_area_fraction'].sum():.6f} "
          f"logr range=[{tt['chart_log_density_ratio'][v].min():+.3f},"
          f"{tt['chart_log_density_ratio'][v].max():+.3f}]")


def export_gallery(ds, out_dir, n=5, pred_fn=None):
    """导出 n 张轻量 PNG(与 notebook 相同函数)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        fig = plt.figure(figsize=(15, 4.2))
        show_basecolor(item, fig.add_subplot(1, 4, 1))
        show_charts(item, fig.add_subplot(1, 4, 2))
        show_signal(item, fig.add_subplot(1, 4, 3))
        show_target(item, fig.add_subplot(1, 4, 4))
        plt.suptitle(item["object_id"], fontsize=10)
        plt.tight_layout()
        p = f"{out_dir}/{i:02d}_{item['object_id']}.png"
        plt.savefig(p, dpi=80)
        plt.close(fig)
        paths.append(p)
    return paths
