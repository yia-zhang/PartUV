# -*- coding: utf-8 -*-
"""可视化绘图函数(matplotlib; 缺数据显示提示图, 不崩)."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from PIL import Image

# 服务器普遍无 CJK 字体: 有则用, 无则图内标题回退英文(避免方框)
_HAS_CJK = any("CJK" in f.name or "WenQuanYi" in f.name
               for f in _fm.fontManager.ttflist)
if _HAS_CJK:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "DejaVu Sans"]


def _t(zh, en):
    return zh if _HAS_CJK else en


def _notice(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11,
            color="#888", wrap=True)
    ax.set_axis_off()


def plot_funnel(reports):
    y = reports.get("yield_")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    if not y:
        _notice(axs[0], "无 yield 报告")
        _notice(axs[1], "")
        return
    att, acc = y.get("attempted", 0), y.get("accepted", 0)
    axs[0].bar(["attempted", "accepted", "rejected"],
               [att, acc, att - acc], color=["#7a8", "#2a7", "#c66"])
    axs[0].set_title(_t("处理漏斗", "processing funnel"))
    cnt = y.get("yield_counts", {})
    ks = sorted(cnt, key=cnt.get)
    axs[1].barh(ks, [cnt[k] for k in ks], color="#c66")
    axs[1].set_title(_t("按状态分类", "by status"))
    for k in ("preflight_pass_rate", "structural_qa_pass_rate",
              "quality_eligibility_rate", "end_to_end_acceptance"):
        if k in y:
            print(f"  {k} = {y[k]}")
    plt.tight_layout()


def plot_rejections(reports, top=12):
    rej = reports.get("rejections")
    fig, ax = plt.subplots(figsize=(9, 4))
    if not rej:
        return _notice(ax, "无 rejection 报告")
    from collections import Counter
    c = Counter((r.get("reason", "?") or "?").split(":")[0].split("(")[0][:36]
                for r in rej)
    ks = [k for k, _ in c.most_common(top)]
    ax.barh(ks[::-1], [c[k] for k in ks][::-1], color="#c66")
    ax.set_title(_t(f"拒绝原因 top{len(ks)}(共 {len(rej)})", f"rejection reasons top{len(ks)} (of {len(rej)})"))
    plt.tight_layout()


def plot_label_distributions(stats):
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, key, title, bins in [
            (axs[0], "n_charts", _t("chart 数/对象", "charts/object"), 30),
            (axs[1], "logr", "chart_log_density_ratio", 40),
            (axs[2], "scale", "chart_target_scale", 40),
            (axs[3], "timing", _t("单对象耗时 s", "per-object seconds"), 30)]:
        v = stats.get(key, [])
        if not v:
            _notice(ax, f"无 {key} 数据")
            continue
        ax.hist(v, bins=bins, color="#478", alpha=0.85)
        if key == "timing":
            p50, p90 = np.percentile(v, [50, 90])
            ax.axvline(p50, color="#2a7", label=f"P50={p50:.0f}s")
            ax.axvline(p90, color="#c66", label=f"P90={p90:.0f}s")
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()


def plot_splits(reports):
    sp = reports.get("splits")
    fig, ax = plt.subplots(figsize=(5, 3))
    if not sp:
        return _notice(ax, "无 splits.json(构建完成后由 make_splits 生成)")
    ks = [k for k in ("train", "val", "test") if k in sp]
    ax.bar(ks, [len(sp[k]) for k in ks], color=["#478", "#7a8", "#a67"])
    ax.set_title("train/val/test")
    plt.tight_layout()


# ---- 单样本 ----
def show_basecolor(item, ax=None):
    ax = ax or plt.gca()
    p = item.get("reference_texture", "")
    if not os.path.exists(p):
        return _notice(ax, "无 reference 纹理")
    ax.imshow(np.asarray(Image.open(p)))
    ax.set_axis_off()
    ax.set_title("base color (1K)", fontsize=10)


def show_mesh_preview(item, ax=None, max_faces=20000):
    ax = ax or plt.gcf().add_subplot(projection="3d")
    mi = item["model_inputs"]
    V, F = mi["vertices"], mi["faces"]
    if len(F) > max_faces:
        F = F[np.linspace(0, len(F) - 1, max_faces).astype(int)]
    try:
        ax.plot_trisurf(V[:, 0], V[:, 2], V[:, 1], triangles=F,
                        color="#b8bcc4", edgecolor="none", shade=True)
        ax.set_axis_off()
        ax.set_title(_t(f"白模({len(mi['faces']):,} 面)", f"mesh ({len(mi['faces']):,} faces)"), fontsize=10)
    except Exception as e:
        _notice(ax, f"mesh 预览失败: {type(e).__name__}")


def _uv_tripcolor(ax, item, face_vals, title, cmap="viridis", sym=False):
    uv = item["model_inputs"]["source_uv"].reshape(-1, 2)
    tris = np.arange(len(uv)).reshape(-1, 3)
    if sym:
        vmax = max(float(np.abs(face_vals).max()), 1e-3)
        tp = ax.tripcolor(uv[:, 0], uv[:, 1], tris, facecolors=face_vals,
                          cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        tp = ax.tripcolor(uv[:, 0], uv[:, 1], tris, facecolors=face_vals,
                          cmap=cmap)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)
    plt.colorbar(tp, ax=ax, fraction=0.04)


def show_chart_segmentation(item, ax=None):
    ax = ax or plt.gca()
    f2c = item["model_inputs"]["face_to_chart"]
    _uv_tripcolor(ax, item, (f2c % 20).astype(float),
                  f"PartUV charts ({f2c.max() + 1})", cmap="tab20")


def show_demand_heatmap(item, ax=None):
    ax = ax or plt.gca()
    diag = item.get("teacher_diagnostics")
    if diag is None:
        return _notice(ax, "diagnostics 未暴露(loader expose_diagnostics=False)")
    _uv_tripcolor(ax, item, diag["face_content_score"],
                  _t("teacher content score(诊断, 禁作输入)", "teacher content score (diagnostic)"), cmap="magma")


def show_density_heatmap(item, ax=None):
    ax = ax or plt.gca()
    tt = item["training_targets"]
    logr = tt["chart_log_density_ratio"][item["model_inputs"]["face_to_chart"]]
    _uv_tripcolor(ax, item, logr, _t("target 线性纹素密度 log 比", "target linear texel-density log ratio"), cmap="coolwarm",
                  sym=True)


def show_source_uv_over_reference(item, ax=None, max_edges=60000):
    """SOURCE UV(非 target): reference 纹理 + 源 UV wireframe 叠加."""
    ax = ax or plt.gca()
    p = item.get("reference_texture", "")
    if not os.path.exists(p):
        return _notice(ax, "无 reference 纹理")
    img = np.asarray(Image.open(p))
    ax.imshow(img, extent=[0, 1, 0, 1], origin="upper")
    uv = item["model_inputs"]["source_uv"]          # (F,3,2)
    F = len(uv)
    step = max(1, F * 3 // max_edges)
    segs = np.concatenate([uv[::step, [0, 1]], uv[::step, [1, 2]],
                           uv[::step, [2, 0]]])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors="#00e5ff", linewidths=0.25,
                                     alpha=0.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(_t("SOURCE UV(非 target)", "SOURCE UV wireframe (not target)"),
                 fontsize=10)


def show_packed_atlas(item, ax=None):
    ax = ax or plt.gca()
    _notice(ax, "packed UV/atlas 不在 MVP 样本内\n"
                "(target_packed_uv 为 QA-only 产物, 数据集只含 allocation 标签)")


def show_quality(item):
    import json
    q = json.load(open(item["qa_artifacts"]["quality_json"])) \
        if os.path.exists(item["qa_artifacts"]["quality_json"]) else {}
    man = item["manifest"]
    print(f"quality = {q.get('quality_status', '?')}  "
          f"signal_dist = {q.get('signal_dist', '?')}")
    print(f"coverage = {man['geometry']['train_face_coverage']:.4f}  "
          f"charts = {man['geometry']['n_charts']}  "
          f"faces = {man['geometry']['n_faces']:,}")
    print(f"occupancy/fill: 样本内未存储(packed 布局为 QA-only)")
    qc = q.get("quality_check") or {}
    if qc:
        print(f"中等预算质量: G_global_eq={qc.get('G_global_eq')} "
              f"G_HF_eq={qc.get('G_HF_eq')}")
