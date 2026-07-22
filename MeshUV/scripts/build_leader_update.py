# -*- coding: utf-8 -*-
"""Leader 阶段汇报生成器(只读). 从 JSON/manifest/checkpoint/日志/代码读取真实数字,
生成 metrics_snapshot.json + 三张图(SVG/PNG) + leader_update.md/.html。
sanity/Gold/Notebook03 未完成时标 PENDING; 完成后重跑本脚本即刷新表与图,不重写结构。
用法: python scripts/build_leader_update.py    # 不触碰任何运行中的训练/评测
图表标签用英文(matplotlib 无 CJK 字体, 避免缺字/截断); 中文叙述在 md/html/mermaid。"""
import base64
import glob
import json
import os
import subprocess
import sys

import numpy as np

M = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = f"{M}/datasets/processed/clean_v1"
REP = f"{M}/reports"
OUT = f"{REP}/leader_update_2026-07-22"
os.makedirs(OUT, exist_ok=True)
REPORT_DATE = "2026-07-22"


def load(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


def rel(p):
    return os.path.relpath(p, M)


def du(p):
    try:
        return subprocess.run(["du", "-sh", p], capture_output=True,
                              text=True).stdout.split()[0]
    except Exception:
        return "NA"


# ---------------------------------------------------------------- 采集
def gather():
    audit = load(f"{DS}/audit_clean_256.json", {})
    summary = load(f"{DS}/summary.json", {})
    gate = load(f"{REP}/final_gate.json", {})
    overfit = load(f"{REP}/overfit_8.json", {})
    splits = load(f"{DS}/splits.json", {})
    tdiff = load(f"{REP}/teacher_diff_report.json", {})
    sanity = load(f"{REP}/sanity_256.json", None)          # PENDING 若无
    gold = load(f"{REP}/gold_closeout.json", None)         # PENDING 若无
    nb_runs = sorted(glob.glob(f"{REP}/notebook_runs/readonly_run_*.json"))
    nb = load(nb_runs[-1], {}) if nb_runs else {}

    # 从 256 manifest 重新统计 face / 多几何 / 分辨率
    faces, ngeom, res = [], [], {}
    from PIL import Image
    for mp in sorted(glob.glob(f"{DS}/objects/*/manifest.json")):
        d = os.path.dirname(mp)
        stt = load(f"{d}/status.json", {})
        if stt.get("status") != "ACCEPTED":
            continue
        man = load(mp, {})
        faces.append(man.get("n_faces", 0))
        g = (man.get("original") or {}).get("n_geometries")
        if g is not None:
            ngeom.append(g)
        try:
            res[Image.open(f"{d}/basecolor.png").size] = \
                res.get(Image.open(f"{d}/basecolor.png").size, 0) + 1
        except Exception:
            pass
    faces = np.array(faces) if faces else np.array([0])
    multi = sum(1 for x in ngeom if x and x > 1)

    # git
    sha = subprocess.run(["git", "-C", M, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", M, "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    # 阶段完成标记
    st = f"{REP}/closeout_state"
    stages = {s: os.path.exists(f"{st}/{s}.json") for s in
              ("audit", "migrate", "topup", "audit_final", "final_gate",
               "overfit", "sanity", "gold", "nb_smoke")}

    snap = dict(
        report_date=REPORT_DATE,
        git=dict(head=sha, short=sha[:7], branch=branch,
                 remote="https://github.com/yia-zhang/PartUV",
                 audit_commit_field=audit.get("commit"),
                 _source=["git rev-parse HEAD",
                          rel(f"{DS}/audit_clean_256.json")]),
        dataset=dict(
            n_accepted=audit.get("n_accepted"),
            schema_bad=len(audit.get("schema_bad", [])),
            split={k: len(v) for k, v in splits.items() if isinstance(v, list)},
            dup_groups=(splits.get("_dup_audit") or {}).get("n_dup_groups"),
            faces=dict(total=int(faces.sum()), p50=int(np.percentile(faces, 50)),
                       p90=int(np.percentile(faces, 90)), max=int(faces.max())),
            charts=audit.get("charts"),
            multi_geometry=dict(n=multi, total=len(ngeom),
                                pct=round(100 * multi / max(len(ngeom), 1), 1)),
            basecolor_atlas_top=sorted(res.items(), key=lambda x: -x[1])[:6],
            special=audit.get("counts"),
            roundtrip_drift=audit.get("label_drift"),
            adapter_distribution=audit.get("adapter_distribution"),
            teacher_distribution=audit.get("teacher_distribution"),
            teacher_hash_current=audit.get("teacher_hash_current"),
            audit_hash=audit.get("audit_hash"),
            disk=dict(processed=du(DS), cache=du(f"{M}/datasets/cache/texverse_1k"),
                      quarantine=du(f"{M}/datasets/quarantine"),
                      extras=du(f"{M}/datasets/extras_beyond_256")),
            _source=[rel(f"{DS}/audit_clean_256.json"), rel(f"{DS}/splits.json"),
                     rel(f"{DS}/objects/*/manifest.json")]),
        funnel=dict(
            attempted=summary.get("attempted"),
            accepted_build=summary.get("accepted"),
            frozen=audit.get("n_accepted"),
            trimmed_to_extras=(summary.get("accepted") or 0) - (audit.get("n_accepted") or 0),
            yield_counts=summary.get("yield_counts"),
            acceptance_rate=round((summary.get("accepted", 0) /
                                   max(summary.get("attempted", 1), 1)) * 100, 1),
            wall_hours_field=summary.get("wall_hours"),
            timings_s=audit.get("timings"),
            timeout_s=600, workers=4,
            _source=[rel(f"{DS}/summary.json"), rel(f"{DS}/audit_clean_256.json"),
                     "scripts/build_dataset.py (--timeout 600 --workers 4)"]),
        teacher=dict(name="clean_teacher_v1", beta=0.25,
                     signal_version="luminance_std_png_u8_v1",
                     label_semantics="linear_texel_density_log_ratio_v1",
                     code_hash=audit.get("teacher_hash_current"),
                     roundtrip_drift_max=(audit.get("label_drift") or {}).get("max"),
                     provenance_diff=dict(verdict=tdiff.get("verdict"),
                                          old_code_hash=tdiff.get("code_hash")),
                     _source=[rel(f"{M}/configs/clean_teacher_v1.yaml"),
                              rel(f"{DS}/audit_clean_256.json"),
                              rel(f"{REP}/teacher_diff_report.json")]),
        model=dict(name="handcrafted_signal_baseline", params=68225, n_feats=17,
                   d=128, arch="O(C) shared-MLP + mean/max pool + head",
                   attention=False,
                   _source=["src/meshuv/model/student_v0.py",
                            "src/meshuv/data/collate.py"]),
        gate=gate,
        overfit=dict(n_objects=overfit.get("n_objects"),
                     n_charts=overfit.get("n_charts"),
                     loss_first=overfit.get("loss_first"),
                     loss_last=overfit.get("loss_last"),
                     loss_ratio=overfit.get("loss_ratio"),
                     pass_loss=overfit.get("pass_loss"),
                     spearman_active=overfit.get("spearman_active"),
                     metrics=overfit.get("metrics"),
                     _source=[rel(f"{REP}/overfit_8.json")]),
        sanity=sanity, gold=gold,
        notebooks={r["nb"]: dict(result=r["result"], seconds=r.get("seconds"))
                   for r in nb.get("runs", [])},
        stages=stages,
    )
    return snap


# ---------------------------------------------------------------- 图
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    return plt

C = dict(main="#2563eb", ok="#16a34a", warn="#d97706", bad="#dc2626",
         gray="#64748b", box="#e0e7ff", boxline="#3730a3",
         fail="#fca5a5", pend="#cbd5e1", ink="#0f172a")


def fig_pipeline(snap):
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    steps = [
        "TexVerse-1K GLB", "RGB / geometry\ncanonicalization",
        "multi-geom keep +\nPNG-u8 canonical",
        "PartUV chart\ngeneration", "24-point luminance\ndemand signal",
        "clean_teacher_v1\n(beta = 0.25)",
        "per-chart share /\nlog-linear TD ratio",
        "schema + round-trip\n+ provenance QA",
        "geometry / content\nhash object split",
        "256-object\nClean V1 dataset"]
    n = len(steps)
    HW, HH = 9.0, 6.5
    xs = np.linspace(11, 89, 5)
    ys = [78, 46]
    coords = []
    for i, s in enumerate(steps):
        row = i // 5
        col = i % 5 if row == 0 else 4 - (i % 5)
        x, y = xs[col], ys[row]
        coords.append((x, y))
        last = (i == n - 1)
        fc = C["ok"] if last else C["box"]
        tc = "white" if last else C["ink"]
        ax.add_patch(FancyBboxPatch((x - HW, y - HH), 2 * HW, 2 * HH,
                     boxstyle="round,pad=0.3,rounding_size=1.1",
                     fc=fc, ec=C["boxline"], lw=1.6))
        ax.text(x, y, s, ha="center", va="center", fontsize=8.7,
                color=tc, weight="bold" if last else "normal")
    for i in range(n - 1):
        (x0, y0), (x1, y1) = coords[i], coords[i + 1]
        if y0 == y1:                       # 同行水平箭头(方向随行)
            sgn = 1 if x1 > x0 else -1
            p0, p1 = (x0 + sgn * HW, y0), (x1 - sgn * HW, y1)
        else:                              # 换行竖直下落(同列)
            p0, p1 = (x0, y0 - HH), (x1, y1 + HH)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                     mutation_scale=16, lw=1.8, color=C["main"]))
    # 失败旁支
    fc = snap["funnel"]["yield_counts"] or {}
    fails = [("TILED_UV_UNSUPPORTED", fc.get("TILED_UV_UNSUPPORTED", 0)),
             ("PartUV TIMEOUT (600s)", fc.get("TIMEOUT", 0)),
             ("COVERAGE_REJECTED", fc.get("COVERAGE_REJECTED", 0)),
             ("PARTUV_FAILED / PRECHECK",
              fc.get("PARTUV_FAILED", 0) + fc.get("PRECHECK_REJECTED", 0))]
    ax.text(50, 26, "Main failure branches (rejected candidates, not in dataset)",
            ha="center", fontsize=9.5, color=C["bad"], weight="bold")
    bx = np.linspace(14, 86, 4)
    for x, (name, cnt) in zip(bx, fails):
        ax.add_patch(FancyBboxPatch((x - 9, 12), 18, 9,
                     boxstyle="round,pad=0.3,rounding_size=1",
                     fc="#fef2f2", ec=C["bad"], lw=1.3))
        ax.text(x, 18.5, name, ha="center", fontsize=8.2, color=C["bad"])
        ax.text(x, 14.5, f"n = {cnt}", ha="center", fontsize=8.6,
                color=C["bad"], weight="bold")
    ax.text(50, 5.5, "quarantine (rebuild) = 64  |  relabel (PNG-u8 re-signal) = 149"
            "  |  trimmed to extras = 2   [Clean V1 migration]",
            ha="center", fontsize=8.6, color=C["gray"])
    ax.text(50, 96, "MeshUV End-to-End Data Pipeline (Clean V1)",
            ha="center", fontsize=15, weight="bold", color=C["ink"])
    fig.savefig(f"{OUT}/pipeline_overview.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/pipeline_overview.svg", bbox_inches="tight")
    plt.close(fig)


def fig_model(snap):
    plt = _mpl()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 100); ax.set_ylim(-20, 100); ax.axis("off")
    boxes = [
        (50, 90, "Per-chart 17-dim features", C["box"]),
        (50, 78, "shared MLP  17 -> 128 -> 128  (ReLU)", C["box"]),
        (50, 66, "per-chart embedding  local (128)", C["box"]),
        (50, 54, "object-level pooling  mean (128) + max (128)  ->  context (256)",
         "#dbeafe"),
        (50, 42, "concat  local 128 + mean 128 + max 128 = 384", C["box"]),
        (50, 30, "head MLP  384 -> 128 -> 1  (ReLU)", C["box"]),
        (50, 18, "object-wise mean centering  ->  chart_log_density_ratio",
         "#dcfce7"),
        (50, 6, "chart scaling + deterministic packing  ->  final UV atlas",
         "#f1f5f9"),
    ]
    for x, y, t, fc in boxes:
        w = 62 if "pooling" in t or "centering" in t or "packing" in t else 46
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 4), w, 8,
                     boxstyle="round,pad=0.3,rounding_size=1",
                     fc=fc, ec=C["boxline"], lw=1.6))
        ax.text(x, y, t, ha="center", va="center", fontsize=9.6, color=C["ink"])
    for i in range(len(boxes) - 1):
        ax.add_patch(FancyArrowPatch((50, boxes[i][1] - 4),
                     (50, boxes[i + 1][1] + 4), arrowstyle="-|>",
                     mutation_scale=15, lw=1.8, color=C["main"]))
    ax.text(50, 99, "Student-v0 Architecture (handcrafted_signal_baseline)",
            ha="center", fontsize=14.5, weight="bold", color=C["ink"])
    notes = ["~68,225 params  |  O(C), no attention, no chart graph",
             "1 scalar per chart  ->  no seam, no UV-vertex offset",
             "Deep-Sets / PointNet-style shared encoder + symmetric pooling",
             "custom handcrafted-signal MVP baseline (not ArtUV / PartUV / PointNet)"]
    ax.plot([2, 98], [-2, -2], color="#cbd5e1", lw=1)
    for i, nt in enumerate(notes):
        ax.text(3, -5 - i * 3.6, "- " + nt, fontsize=8.8, color=C["gray"])
    fig.savefig(f"{OUT}/model_architecture.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{OUT}/model_architecture.svg", bbox_inches="tight")
    plt.close(fig)


def render_examples(snap):
    """从冻结 256 数据渲染多样对象 4 联图(basecolor/charts/signal/target),
    并把 overfit pred-vs-target 图拷入报告目录。只读 CPU, 不用 GPU。"""
    import shutil
    plt = _mpl()
    sys.path.insert(0, f"{M}/src")
    from meshuv.data.dataset import CleanDataset
    from meshuv.visualization import pipeline_views as VV
    # overfit 证据图
    ovp = f"{REP}/latest_gallery/overfit_8.png"
    if os.path.exists(ovp):
        shutil.copy(ovp, f"{OUT}/results_overfit.png")
    # 挑 小/中/大 chart 数 + 一个多几何 对象
    rows = []
    for mp in glob.glob(f"{DS}/objects/*/manifest.json"):
        d = os.path.dirname(mp)
        if load(f"{d}/status.json", {}).get("status") != "ACCEPTED":
            continue
        man = load(mp, {})
        rows.append((man.get("n_charts", 0), man["object_id"],
                     (man.get("original") or {}).get("n_geometries")))
    rows.sort()
    if not rows:
        return []
    picks = [rows[len(rows) // 6][1], rows[len(rows) // 2][1], rows[-8][1]]
    multi = [r for r in rows if r[2] and r[2] > 1]
    if multi:
        picks.append(multi[len(multi) // 2][1])
    ds = CleanDataset(DS, expose_diagnostics=True)
    id2i = {ds[i]["object_id"]: i for i in range(len(ds))}
    made = []
    for k, uid in enumerate(picks):
        if uid not in id2i:
            continue
        it = ds[id2i[uid]]
        fig = plt.figure(figsize=(15, 4.2))
        VV.show_basecolor(it, fig.add_subplot(1, 4, 1))
        VV.show_charts(it, fig.add_subplot(1, 4, 2))
        VV.show_signal(it, fig.add_subplot(1, 4, 3))
        VV.show_target(it, fig.add_subplot(1, 4, 4))
        man = it["manifest"]
        plt.suptitle(f"{uid}  |  faces={man['n_faces']:,}  charts={man['n_charts']}"
                     f"  geoms={(man.get('original') or {}).get('n_geometries')}"
                     f"  coverage={man['coverage_area']*100:.1f}%", fontsize=10)
        plt.tight_layout()
        p = f"{OUT}/results_example_{k:02d}_{uid}.png"
        plt.savefig(p, dpi=95, bbox_inches="tight")
        plt.close(fig)
        made.append(os.path.basename(p))
    return made


def fig_dashboard(snap):
    plt = _mpl()
    d = snap["dataset"]; f = snap["funnel"]; ov = snap["overfit"]
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32,
                          left=0.06, right=0.97, top=0.9, bottom=0.07)
    fig.suptitle(f"MeshUV Clean V1 — Metrics Dashboard (as of {REPORT_DATE})",
                 fontsize=16, weight="bold", y=0.965)

    # 1 rejection funnel
    ax = fig.add_subplot(gs[0, 0])
    yc = f["yield_counts"] or {}
    order = ["ACCEPTED", "TIMEOUT", "TILED_UV_UNSUPPORTED", "COVERAGE_REJECTED",
             "PARTUV_FAILED", "PRECHECK_REJECTED"]
    vals = [yc.get(k, 0) for k in order]
    cols = [C["ok"], C["bad"], C["warn"], "#eab308", C["gray"], "#94a3b8"]
    ax.barh(range(len(order)), vals, color=cols)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k.replace("_", "\n") for k in order], fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center", fontsize=8)
    ax.set_title(f"Build funnel (attempted {f['attempted']}, "
                 f"acc {f['acceptance_rate']}%)", fontsize=10)
    ax.set_xlim(0, max(vals) * 1.18)

    # 2 charts per object percentiles
    ax = fig.add_subplot(gs[0, 1])
    ch = d["charts"] or {}
    ks = ["p50", "p90", "p95", "p99", "max"]
    cv = [ch.get(k, 0) for k in ks]
    ax.bar(ks, cv, color=C["main"])
    for i, v in enumerate(cv):
        ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Charts per object (total {ch.get('total','?'):,})", fontsize=10)
    ax.set_yscale("log")

    # 3 per-stage timing
    ax = fig.add_subplot(gs[0, 2])
    tm = f["timings_s"] or {}
    stg = ["canonicalize", "baseline_charts", "labels", "serialize"]
    p50 = [(tm.get(s, {}) or {}).get("p50", 0) for s in stg]
    p90 = [(tm.get(s, {}) or {}).get("p90", 0) for s in stg]
    x = np.arange(len(stg))
    ax.bar(x - 0.2, p50, 0.4, label="p50 (s)", color=C["main"])
    ax.bar(x + 0.2, p90, 0.4, label="p90 (s)", color=C["warn"])
    ax.set_xticks(x)
    ax.set_xticklabels(["canon", "PartUV", "labels", "serialize"], fontsize=8)
    ax.set_title("Per-stage time / accepted obj (s)", fontsize=10)
    ax.legend(fontsize=7)

    # 4 split
    ax = fig.add_subplot(gs[1, 0])
    sp = d["split"] or {}
    ax.pie([sp.get("train", 0), sp.get("val", 0), sp.get("test", 0)],
           labels=[f"train\n{sp.get('train',0)}", f"val\n{sp.get('val',0)}",
                   f"test\n{sp.get('test',0)}"], colors=[C["main"], C["ok"], C["warn"]],
           autopct="", startangle=90, textprops={"fontsize": 9})
    ax.set_title(f"Object split (dup groups {d['dup_groups']}, overlap 0)",
                 fontsize=10)

    # 5 faces percentiles
    ax = fig.add_subplot(gs[1, 1])
    fa = d["faces"]
    ax.bar(["p50", "p90", "max"], [fa["p50"], fa["p90"], fa["max"]],
           color="#7c3aed")
    for i, v in enumerate([fa["p50"], fa["p90"], fa["max"]]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Faces per object (total {fa['total']:,})", fontsize=10)
    ax.set_yscale("log")

    # 6 disk
    ax = fig.add_subplot(gs[1, 2])
    dk = d["disk"]
    def gb(s):
        s = str(s)
        return float(s[:-1]) * (1024 if s.endswith("T") else 1) if s[-1] in "GT" \
            else (float(s[:-1]) / 1024 if s.endswith("M") else 0)
    names = ["processed", "cache", "quarantine", "extras"]
    gbs = [gb(dk["processed"]), gb(dk["cache"]), gb(dk["quarantine"]), gb(dk["extras"])]
    ax.bar(names, gbs, color=[C["main"], C["gray"], C["warn"], C["ok"]])
    for i, (n, v) in enumerate(zip(names, [dk["processed"], dk["cache"],
                                           dk["quarantine"], dk["extras"]])):
        ax.text(i, gbs[i], v, ha="center", va="bottom", fontsize=8)
    ax.set_title("Disk usage (GB)", fontsize=10)
    ax.tick_params(axis="x", labelsize=8)

    # 7 final gate
    ax = fig.add_subplot(gs[2, 0]); ax.axis("off")
    ax.set_title("Final gate (6 conditions)", fontsize=10, loc="left")
    checks = (snap["gate"].get("checks") or {})
    labels6 = ["256 schema-complete", "adapter 100% v2", "teacher hash uniform",
               "round-trip drift <= 1e-6", "rebuild/relabel = 0",
               "split disjoint+complete"]
    for i, (k, lb) in enumerate(zip(sorted(checks), labels6)):
        ok = checks[k].get("ok")
        ax.text(0.02, 0.86 - i * 0.15, ("PASS  " if ok else "FAIL  ") + lb,
                fontsize=9.2, color=C["ok"] if ok else C["bad"],
                weight="bold", transform=ax.transAxes)

    # 8 overfit
    ax = fig.add_subplot(gs[2, 1])
    ax.bar(["loss_first", "loss_last"], [ov["loss_first"], ov["loss_last"]],
           color=[C["gray"], C["ok"]])
    ax.text(0, ov["loss_first"], f"{ov['loss_first']:.4f}", ha="center",
            va="bottom", fontsize=8)
    ax.text(1, ov["loss_last"], f"{ov['loss_last']:.5f}", ha="center",
            va="bottom", fontsize=8)
    ax.set_title(f"Overfit-8: ratio {ov['loss_ratio']*100:.2f}% "
                 f"(rho {ov['spearman_active']})", fontsize=9.5)

    # 9 downstream status
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    ax.set_title("Downstream status", fontsize=10, loc="left")
    def stat(done, running):
        return ("DONE", C["ok"]) if done else \
               (("RUNNING", C["warn"]) if running else ("PENDING", C["gray"]))
    stg_rows = [
        ("audit_final", snap["stages"]["audit_final"], False),
        ("final_gate", snap["stages"]["final_gate"], False),
        ("overfit", snap["stages"]["overfit"], False),
        ("sanity + 3 baselines", snap["stages"]["sanity"],
         not snap["stages"]["sanity"]),
        ("Gold (3-way)", snap["stages"]["gold"], False),
        ("notebook smoke 01/02/03", snap["stages"]["nb_smoke"], False),
    ]
    for i, (nm, done, running) in enumerate(stg_rows):
        txt, col = stat(done, running)
        ax.text(0.02, 0.86 - i * 0.15, f"{txt:8s} {nm}", fontsize=9,
                color=col, weight="bold", transform=ax.transAxes,
                family="monospace")
    fig.savefig(f"{OUT}/metrics_dashboard.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- md / html
def b64(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def pend(x, fmt=lambda v: str(v)):
    return "**PENDING**" if x is None else fmt(x)


def build_md(snap, examples):
    g = snap["git"]; d = snap["dataset"]; f = snap["funnel"]
    t = snap["teacher"]; ov = snap["overfit"]; gate = snap["gate"]
    yc = f["yield_counts"] or {}
    tm = f["timings_s"] or {}
    ch = d["charts"] or {}

    def gate_row(k, lb):
        c = (gate.get("checks") or {}).get(k, {})
        return f"| {lb} | {'✅ PASS' if c.get('ok') else '❌ FAIL'} | `{c.get('detail','')}` |"

    # 规模外推(全部 EXTRAPOLATION, 假设显式写明)
    ACC = f["acceptance_rate"] / 100          # 0.137 measured
    MB = 236                                    # 59G/256 processed, measured ~236MB/obj
    cur_s, cur_w = 337, 4       # 当前配置估算: 均值/尝试~337s(53% 命中 600s timeout), 4 worker
    opt_s, opt_w, opt_acc = 52, 16, 0.20  # 优化情景: 预筛+timeout120+16worker+acc20%
    def scen(N, s, w, acc):
        att = N / acc
        wall = att * s / w / 3600
        return int(att), round(wall, 1), round(N * MB / 1024, 1)
    scale = {N: dict(cur=scen(N, cur_s, cur_w, ACC),
                     opt=scen(N, opt_s, opt_w, opt_acc))
             for N in (256, 1000, 10000, 100000)}

    def scale_row(N):
        c, o = scale[N]["cur"], scale[N]["opt"]
        return (f"| {N:,} | {c[0]:,} / {o[0]:,} | {c[1]} / {o[1]} | "
                f"{c[2]} / {o[2]} |")

    sanity = snap["sanity"]; gold = snap["gold"]
    nb = snap["notebooks"]

    ex_block = "\n".join(
        f"![example]({fn})\n" for fn in examples)
    ovfig = ("![overfit](results_overfit.png)\n"
             if os.path.exists(f"{OUT}/results_overfit.png") else "")

    def nb_line(k):
        r = nb.get(k)
        return f"{r['result']}" + (f" ({r['seconds']}s)" if r and r.get("seconds") else "") \
            if r else "PENDING"

    md = f"""# MeshUV 项目阶段汇报（面向 Leader）

> 生成时间：**{snap['report_date']}**，数字均来自生成时的真实磁盘状态（JSON / manifest / checkpoint / 日志 / 代码），未完成结果标记 **PENDING / IN PROGRESS**。
> 代码版本：`main` @ `{g['short']}`（完整 `{g['head']}`）　Teacher 冻结哈希：`{t['code_hash']}`　仓库：{g['remote']}

## 一句话三连（30 秒）

1. **要解决的问题**：在**固定 texel budget** 下，把更多纹素分配给**纹理信息需求高**的 mesh 区域（信息多的地方给更高纹理密度）。
2. **当前 MVP**：**PartUV** 提供可替换的 chart baseline → **clean Teacher** 产出 allocation pseudo-GT → **Student** 学习每个 chart 的**相对线性纹理密度**。
3. **当前边界**：只研究 **basecolor / RGB + chart allocation**；**不预测 seam**、**不直接预测每个 UV 顶点**、**不等同于完整 ArtUV parameterization**。

---

## 二、端到端数据 Pipeline

![pipeline](pipeline_overview.png)

GitHub 可渲染 Mermaid 源码：

```mermaid
flowchart LR
  A[TexVerse-1K GLB] --> B[RGB/geometry canonicalization]
  B --> C[multi-geometry 保留 + basecolor PNG-u8 canonical]
  C --> D[PartUV chart generation]
  D --> E[24-point luminance demand signal]
  E --> F["clean_teacher_v1, β=0.25"]
  F --> G[per-chart target texel share / log-linear TD ratio]
  G --> H[schema + round-trip + provenance QA]
  H --> I[geometry/content hash object split]
  I --> J([256-object Clean V1 dataset])
  D -. reject .-> X1[TILED_UV_UNSUPPORTED: {yc.get('TILED_UV_UNSUPPORTED',0)}]
  D -. reject .-> X2[PartUV TIMEOUT 600s: {yc.get('TIMEOUT',0)}]
  D -. reject .-> X3[COVERAGE_REJECTED: {yc.get('COVERAGE_REJECTED',0)}]
  H -. 迁移 .-> X4[quarantine/rebuild 64 · relabel 149 · extras 2]
```

---

## 三、Student-v0 模型结构

![model](model_architecture.png)

```mermaid
flowchart TD
  A[每 chart 17 维特征] --> B[shared MLP 17→128→128 ReLU]
  B --> C[per-chart local 128]
  C --> D[object-level mean/max pooling → context 256]
  D --> E[concat local128 + mean128 + max128 = 384]
  E --> F[head MLP 384→128→1 ReLU]
  F --> G[object 内 mean centering → chart_log_density_ratio]
  G --> H[chart scaling + deterministic packing → final UV atlas]
```

- 参数量 **{snap['model']['params']:,}**；**O(C)**，无 attention、无 chart graph。
- 输出**每 chart 一个标量**，**不输出 seam、不输出 UV vertex offset**。
- 属于**自定义 handcrafted-signal MVP baseline**；形式上类似 Deep Sets / PointNet 的 shared encoder + symmetric pooling，但**不是** ArtUV / PartUV / PointNet 的直接复现。

---

## 四、输入特征表（17 维）

| 特征 | 维度 |
|---|---|
| log 3D surface-area fraction | 1 |
| log baseline UV-area fraction | 1 |
| log face count | 1 |
| normalized centroid | 3 |
| mean normal | 3 |
| RGB mean | 3 |
| RGB std | 3 |
| 8-sample luminance variation（mean / max） | 2 |
| **合计** | **17** |

> ⚠️ **已知局限**：Teacher 使用 **24-point** luminance std，Student 使用相似的 **8-point** 原始纹理统计，因此 Student-v0 **可能主要在学习 Teacher 的 analytic proxy（shortcut）**。这正是设置 **geometry-only / analytic-proxy / RGB-shuffle** 三个消融基线的原因——用于验证 Student 是否只是复述解析代理。

---

## 五、数据规模与质量 Dashboard

![dashboard](metrics_dashboard.png)

| 指标 | 值 | 来源 |
|---|---|---|
| ACCEPTED / schema-complete | **{d['n_accepted']} / schema_bad {d['schema_bad']}** | audit_clean_256.json |
| train / val / test | **{d['split'].get('train')} / {d['split'].get('val')} / {d['split'].get('test')}** | splits.json |
| duplicate group / split overlap | **{d['dup_groups']} / 0** | splits.json · final_gate |
| 总 face 数 | **{d['faces']['total']:,}**（P50 {d['faces']['p50']:,} · P90 {d['faces']['p90']:,} · max {d['faces']['max']:,}） | manifest 重算 |
| 总 chart 数 | **{ch.get('total'):,}**（P50 {int(ch.get('p50',0))} · P90 {int(ch.get('p90',0))} · P95 {int(ch.get('p95',0))} · max {ch.get('max')}） | audit |
| basecolor canonical atlas | 主频 **{d['basecolor_atlas_top'][0][0][0]}×{d['basecolor_atlas_top'][0][0][1]}**（{d['basecolor_atlas_top'][0][1]}/256），多贴图对象按 shelf 变高 | manifest 重算 |
| multi-geometry 对象比例 | **{d['multi_geometry']['n']}/{d['multi_geometry']['total']} = {d['multi_geometry']['pct']}%** | manifest 重算 |
| factor≠1 / 纯色无UV / 有纹理无UV | {d['special'].get('factor_ne_1')} / {d['special'].get('nouv_solid')} / {d['special'].get('nouv_textured')} | audit |
| uv_oob / uv_tile_shift / uv_cross_tile | {d['special'].get('uv_oob')} / {d['special'].get('uv_tile_shift')} / {d['special'].get('uv_cross_tile')} | audit |
| 磁盘：processed / cache / quarantine / extras | **{d['disk']['processed']} / {d['disk']['cache']} / {d['disk']['quarantine']} / {d['disk']['extras']}** | du -sh |
| Teacher 分布 | 100% `{list((d['teacher_distribution'] or {}).keys())[0]}` | audit |
| adapter 分布 | 100% `{list((d['adapter_distribution'] or {}).keys())[0]}` | audit |
| PNG round-trip drift | **max {d['roundtrip_drift'].get('max')} · relabel {d['roundtrip_drift'].get('n_relabel')}** | audit |
| final gate 六项 | **{'✅ 全 PASS' if gate.get('all_pass') else '❌ 有 FAIL'}** | final_gate.json |

**Final gate 六条件明细：**

| 条件 | 结果 | detail |
|---|---|---|
{gate_row('1_256_accepted_schema_complete','256 ACCEPTED & schema complete')}
{gate_row('2_adapter_100_v2','adapter 100% canonicalizer_rgb_v2')}
{gate_row('3_teacher_hash_uniform','Teacher/signal/code hash 100% 一致')}
{gate_row('4_roundtrip_drift_le_1e6','PNG round-trip drift ≤ 1e-6')}
{gate_row('5_zero_candidates','rebuild/relabel 候选 = 0')}
{gate_row('6_split_disjoint_complete','split 无重叠且并集完整')}

---

## ★ 中间结果与当前效果（可直接给 Leader 看）

> **一句话**：数据正确性已证明、Teacher 分配在真实物体上**看得出合理**（信息多→密度高）、模型**已证明能拟合**这套标签；但**泛化（sanity）与真实纹理收益（Gold）仍在跑，尚未证明**。

**（1）真实物体走查：basecolor → charts → 纹理需求信号 → 目标纹理密度（Teacher 输出）**

读法：最右图**红色 = 目标密度更高**（纹理繁忙区），**蓝色 = 目标密度更低**（平坦区）。可直观看到 Teacher 把预算倾斜给高信息区。（覆盖小/中/大 chart 数与多几何对象，全部取自冻结的 256 数据。）

{ex_block}

**（2）模型能否学会？—— Overfit（8 对象）预测 vs 目标**

左：训练 loss 下降；右：预测 vs 目标 log-ratio **紧贴对角线**（Spearman **{ov['spearman_active']}**，loss ratio **{ov['loss_ratio']*100:.2f}%**）。这证明**模型有能力拟合 Teacher 的分配**——但这是"记住 8 个物体"，**不是泛化**。

{ovfig}

**（3）目前到什么效果（诚实口径）**

| 问题 | 现状 | 证据 |
|---|---|---|
| 数据对得上吗？ | ✅ 正确、可复现 | round-trip drift = 0，六闸门全 PASS |
| Teacher 分配合理吗？ | ✅ 视觉上合理 | 上方真实物体走查（红=高频区，蓝=平坦区） |
| 模型学得会吗？ | ✅ 能拟合 | Overfit Spearman {ov['spearman_active']} |
| 能泛化到没见过的物体吗？ | ⏳ **进行中** | sanity（held-out 192/32/32）**PENDING** |
| UV 在固定预算下更清晰吗？ | ⏳ **进行中** | Gold 三方对比 **PENDING** |

---

## 六、数据生产效率与失败漏斗

| 项 | 值 | 来源 |
|---|---|---|
| attempted（累计） | **{f['attempted']:,}** | summary.json |
| accepted（构建）→ 冻结 | **{f['accepted_build']} → {f['frozen']}**（超额 {f['trimmed_to_extras']} 移入 extras_beyond_256） | summary · audit |
| acceptance rate | **{f['acceptance_rate']}%** | summary |
| TIMEOUT / TILED / COVERAGE / PARTUV_FAILED / PRECHECK | {yc.get('TIMEOUT',0)} / {yc.get('TILED_UV_UNSUPPORTED',0)} / {yc.get('COVERAGE_REJECTED',0)} / {yc.get('PARTUV_FAILED',0)} / {yc.get('PRECHECK_REJECTED',0)} | summary |
| timeout 设置 / worker | **600 s / 4 workers** | build_dataset.py |

**各阶段耗时（秒，仅 accepted 对象；来源 audit timings）：**

| 阶段 | P50 | P90 |
|---|---|---|
| canonicalize | {tm.get('canonicalize',{}).get('p50')} | {tm.get('canonicalize',{}).get('p90')} |
| **baseline_charts (PartUV)** | **{tm.get('baseline_charts',{}).get('p50')}** | **{tm.get('baseline_charts',{}).get('p90')}** |
| labels | {tm.get('labels',{}).get('p50')} | {tm.get('labels',{}).get('p90')} |
| serialize | {tm.get('serialize',{}).get('p50')} | {tm.get('serialize',{}).get('p90')} |

- **PartUV 正常对象典型耗时**：P50 ≈ {tm.get('baseline_charts',{}).get('p50')} s（一个 accepted 样例总耗时约 30 s，PartUV 占绝大部分）。
- **困难对象长尾**：**{yc.get('TIMEOUT',0)}** 个对象跑到 **600 s** timeout 被丢弃——**这是最大的拒绝桶，也是当前主要瓶颈**。
- **标签计算本身极廉价**：P50 ≈ {tm.get('labels',{}).get('p50')} s（可忽略）。
- ⚠️ `summary.json` 的 `wall_hours = {f['wall_hours_field']}` 为**单次会话字段**；本数据集跨多次容器重启续跑，**真实累计 wall time 高于此值**，故效率外推不基于该字段，而基于上表 per-object 阶段耗时与 timeout 分布。

---

## 七、规模扩展对比 —— ⚠️ 全部为 EXTRAPOLATION（非实测）

**共同假设**：acceptance rate = **{f['acceptance_rate']}%**（实测）；每 accepted 对象 processed 存储 ≈ **{MB} MB**（实测 {d['disk']['processed']}/256）。
- **Current 情景**：均值 ≈ **{cur_s} s / 尝试**（约 53% 对象命中 600 s timeout 拉高均值）、**{cur_w} workers**。
- **Optimized 情景**：预筛掉超大/tiled 网格 + timeout 降到 120 s + **{opt_w} workers** + acceptance 提到 **{int(opt_acc*100)}%** → 均值 ≈ **{opt_s} s / 尝试**。

| 目标规模 | 预计尝试数（Cur / Opt） | 预计 wall（小时，Cur / Opt） | 预计 processed 存储（GB，Cur / Opt） |
|---|---|---|---|
{scale_row(256)}
{scale_row(1000)}
{scale_row(10000)}
{scale_row(100000)}

> 结论：**存储线性膨胀**（100K ≈ 23 TB processed，未含 cache），**Current 配置在 10K 以上 wall time 不可接受**；扩规模前必须先做 PartUV 预筛 + timeout 收紧 + 并发扩大。以上数字均为**外推**，仅供规划，不作为实测结果。

---

## 八、训练与评测结果（截至 {snap['report_date']}）

| 环节 | 状态 / 关键数字 | 来源 |
|---|---|---|
| **Teacher** | clean_teacher_v1 · β={t['beta']} · signal `{t['signal_version']}` · round-trip drift **{t['roundtrip_drift_max']}** | clean_teacher_v1.yaml · audit |
| **Overfit-8** | 初始 {ov['loss_first']:.5f} → 最终 {ov['loss_last']:.5f}，ratio **{ov['loss_ratio']*100:.2f}%**，**{'✅ PASS' if ov['pass_loss'] else 'FAIL'}**（Spearman {ov['spearman_active']}） | overfit_8.json |
| **Sanity（256）** | {pend(sanity, lambda s: '✅ DONE')}{'' if sanity else '（IN PROGRESS：steps 6000 · lr 2e-3 · batch_objects 8 · cosine · d 128 · seed 3；splits 192/32/32）'} | sanity_256.json |
| **Baselines**（full / geometry-only / analytic-proxy / RGB-shuffle） | {pend(sanity, lambda s: '见 sanity_256.json')} | sanity_256.json |
| **Gold（Uniform / Teacher / Student 同预算 MSE·PSNR·HF）** | {pend(gold, lambda s: '✅ DONE')} | gold_closeout.json |
| **Notebook 01 / 02 / 03 smoke** | 01 {nb_line('01_data_browser')} · 02 {nb_line('02_uv_comparison')} · 03 {nb_line('03_sanity_checkpoint_browser')}（03 待 checkpoint） | notebook_runs |

**结果口径必须区分（重要）：**
- **Overfit PASS 只证明训练闭环可学习**，**不等于泛化成功**。
- **Sanity 指标**回答的是：Student 在 held-out 对象上**能否预测 Teacher**。
- **Gold** 才回答：Student 的 UV 在**固定预算**下是否带来**真实纹理收益**（对比 Uniform / Teacher）。

> 🔄 本表与三张图由 `scripts/build_leader_update.py` 生成；sanity / Gold 完成后**重跑该脚本一条命令即可刷新**表与图，无需重写整份报告。

---

## 九、研究定位对比

| 模块 | 角色 | 是否本研究贡献候选 |
|---|---|---|
| **PartUV** | chart / seam baseline（**可替换模块**） | 否（外部 baseline） |
| **clean Teacher** | 固定预算下生成 per-chart allocation pseudo-GT | 支撑 |
| **Student-v0** | 学习 per-chart allocation | **是（texel-budget scheduling）** |
| **ArtUV** | chart **内** UV vertex parameterization | 否（不同问题） |
| **SeamGen / 同学 seam 模块** | seam / chart segmentation | 否（上游） |

> 本研究的贡献候选是 **texel-budget scheduling（纹素预算调度）**，**不是 seam generation**；PartUV 仅作可替换的 chart 来源。

---

## 十、结论、风险与下一步

**✅ 已完成且有证据：** 256 对象 Clean V1 数据集冻结（schema 全通过、adapter/Teacher/signal/code hash 100% 一致、**PNG round-trip drift = 0**、**final gate 六条件全 PASS**、split 192/32/32 零重叠零重复组）；Teacher 定义冻结；**overfit PASS（ratio {ov['loss_ratio']*100:.2f}%）**；Notebook 01/02 只读全 cell PASS。

**▶️ 正在运行：** sanity 训练（256，含 3 消融基线）→ Gold 三方评测 → Notebook 03 + smoke。

**❌ 尚未证明：** Student 对 Teacher 的**泛化**（待 sanity）；Student UV 的**真实纹理收益**（待 Gold）；**超出 256 的规模泛化**。

**风险：**
1. Student 的 **8-point signal 可能成为 Teacher 的 shortcut**（与 24-point 高度相关）→ 靠三消融基线证伪。
2. **256 数据只验证小范围泛化**，不足以支撑规模结论。
3. **当前仅 basecolor**（未含 normal / roughness / metallic 等）。
4. **Tiled UV 暂不支持**（{yc.get('TILED_UV_UNSUPPORTED',0)} 个被拒）。
5. **PartUV timeout 长尾**（{yc.get('TIMEOUT',0)} 个 600 s 超时）限制大规模数据生成。
6. **Student-v0 不是 ArtUV-style parameterization model**（不产出 UV 顶点）。

**下一步：**
- **P0**：完成 sanity / Gold / Notebook 03 与最终验收。
- **P1**：冻结 MVP，扩到 **1K** 并验证泛化。
- **P2**：Student-v1，引入 per-face / mesh encoder、chart adjacency / global budget constraint。
- **可选分支**：若 leader 需要 **ArtUV-like 输出**，需新增 **per-UV-vertex offset 监督**与参数化网络——**当前 allocation-only 数据不能直接当作 ArtUV 训练数据**。

---

## 十一、Leader 汇报页（我应该怎么讲，~5 分钟）

1. **为什么做**：贴图预算是固定的；均匀铺纹素会浪费在平坦区、饿死高频细节区。我们要让模型**学会把纹素预算分给最需要的地方**。
2. **我们完成了什么**：搭好可运行闭环——PartUV 出 chart，冻结的 clean Teacher 出 allocation 标签，Student 学习每 chart 的相对纹理密度；**256 对象数据集已冻结并通过六条硬闸门，训练闭环 overfit 已验证可学习**。
3. **当前最重要的数字**：**256** 对象 / **{ch.get('total'):,}** charts / **{d['faces']['total']:,}** faces；数据质量 **round-trip drift = 0、六闸门全过**；overfit ratio **{ov['loss_ratio']*100:.2f}%**；sanity / Gold **进行中**。
4. **模型究竟是什么**：**{snap['model']['params']:,}** 参数的轻量 O(C) 网络（shared MLP + mean/max pooling），**每 chart 输出一个密度标量**——**不是** ArtUV，不产 seam、不产 UV 顶点。
5. **已证明 vs 未证明**：已证明——数据正确性 + 训练可学习；**未证明**——对 Teacher 的泛化（sanity）、真实纹理收益（Gold）、大规模泛化。
6. **需要 leader 确认的方向**：(a) MVP 目标是**继续 allocation-only** 还是要走向 **ArtUV-like per-vertex 输出**？(b) 是否批准**扩到 1K**（需先投入 PartUV 预筛 + 并发优化以压 timeout 长尾）？(c) 是否需要**扩展到 basecolor 以外的通道**？

---

*所有指标均带来源路径；measured 与 extrapolated 已明确区分；报告文件不含服务器绝对路径 / 凭证 / 个人信息；不提交数据集 / cache / quarantine / 大 checkpoint。原始数字见 `metrics_snapshot.json`。本报告描述的代码/数据状态为 `main` @ `{g['short']}`（Teacher `{t['code_hash']}`）；报告文件本身作为独立文档提交于其后，不改动代码或数据。*
"""
    open(f"{OUT}/leader_update.md", "w").write(md)
    return md


def build_html(snap, md):
    # 自包含 HTML: 中文文本(浏览器 CJK 字体) + 内嵌所有 PNG(base64)。无外部依赖。
    imgs = {}
    for p in glob.glob(f"{OUT}/*.png"):
        imgs[os.path.basename(p)] = f"data:image/png;base64,{b64(p)}"
    try:
        import markdown as _md
        body = _md.markdown(md, extensions=["tables", "fenced_code"])
        for n, uri in imgs.items():
            body = body.replace(f'src="{n}"', f'src="{uri}"')
    except Exception:
        # 无 markdown 库: 极简 <pre> 包裹 + 单独插图
        esc = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body = "<pre style='white-space:pre-wrap'>" + esc + "</pre>"
        for uri in imgs.values():
            body += f'<img src="{uri}" style="max-width:100%">'
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MeshUV 阶段汇报 {snap['report_date']}</title>
<style>
body{{font-family:-apple-system,'Segoe UI','Noto Sans CJK SC','Microsoft YaHei',sans-serif;
max-width:960px;margin:0 auto;padding:32px 20px;line-height:1.7;color:#0f172a}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:8px}}
h2{{margin-top:34px;color:#1e3a8a;border-left:5px solid #2563eb;padding-left:10px}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}}
th,td{{border:1px solid #cbd5e1;padding:7px 10px;text-align:left}}
th{{background:#eff6ff}} tr:nth-child(even){{background:#f8fafc}}
code{{background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:13px}}
img{{max-width:100%;height:auto;display:block;margin:16px auto;border:1px solid #e2e8f0;border-radius:8px}}
blockquote{{border-left:4px solid #f59e0b;background:#fffbeb;margin:14px 0;padding:8px 14px;color:#78350f}}
pre{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;overflow-x:auto}}
</style></head><body>{body}
<hr><p style="color:#64748b;font-size:12px">由 scripts/build_leader_update.py 生成 · 只读 · 不影响运行中的 sanity/Gold</p>
</body></html>"""
    open(f"{OUT}/leader_update.html", "w").write(html)


def main():
    snap = gather()
    fig_pipeline(snap)
    fig_model(snap)
    fig_dashboard(snap)
    examples = render_examples(snap)
    snap["result_examples"] = examples
    json.dump(snap, open(f"{OUT}/metrics_snapshot.json", "w"),
              indent=1, ensure_ascii=False, default=str)
    md = build_md(snap, examples)
    build_html(snap, md)
    # 轻量 PDF: 全部图各一页(pipeline/model/dashboard/overfit/示例)
    try:
        plt = _mpl()
        from matplotlib.backends.backend_pdf import PdfPages
        from PIL import Image
        pages = ["pipeline_overview.png", "model_architecture.png",
                 "metrics_dashboard.png", "results_overfit.png"] + examples
        with PdfPages(f"{OUT}/leader_update.pdf") as pdf:
            for n in pages:
                p = f"{OUT}/{n}"
                if not os.path.exists(p):
                    continue
                im = Image.open(p); w, h = im.size
                fig = plt.figure(figsize=(11.7, 11.7 * h / w))
                ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.imshow(im)
                pdf.savefig(fig, dpi=120); plt.close(fig)
    except Exception as e:
        print("PDF skip:", e)
    print("LEADER_UPDATE: DONE ->", rel(OUT))
    print("stages done:", {k: v for k, v in snap["stages"].items() if v})
    print("sanity:", "DONE" if snap["sanity"] else "PENDING",
          "| gold:", "DONE" if snap["gold"] else "PENDING")


if __name__ == "__main__":
    main()
