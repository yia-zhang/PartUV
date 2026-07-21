# -*- coding: utf-8 -*-
"""构建只读人工抽查 notebook: pseudo_gt_case_browser.ipynb.

QA 浏览器: 只读取已保存的 pilot/pseudo-GT/quality artifacts(JSON/PNG/NPZ/GLB
元信息), 不运行 PartUV/packing/rebake/quality evaluation, 不改变 teacher/β/
gate/任何实验结论。人工标注只追加写 outputs/manual_review/manual_reviews.jsonl。
"""
import nbformat as nbf

NB = "/root/youjiaZhang/PartUV/code/notebook/pseudo_gt_case_browser.ipynb"
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md("""# Pseudo-GT Case Browser（只读人工随机抽查）

目的：从已保存的 pilot / pseudo-GT / quality artifacts 中随机查看任意 case，
帮助人工发现**自动指标未捕获的 pipeline bug**。

规则（冻结）：
- **只读**：不运行 PartUV / packing / rebake / quality evaluation；
- 人工结果只**追加**写 `outputs/manual_review/manual_reviews.jsonl`，
  不修改 manifest / 自动 metrics / pseudo-GT 数据；
- 这只是 QA 浏览器，不改变 teacher、β、gate 或任何实验结论；
- 随机选择使用显式 `RANDOM_SEED`，seed 与最终 case_id 都会打印，可复现。"""))

cells.append(code(r'''import glob, json, os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

BASE = "/root/youjiaZhang/PartUV/code/notebook/outputs"
REVIEW_PATH = f"{BASE}/manual_review/manual_reviews.jsonl"

def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None

_v1 = _load(f"{BASE}/pilot_v1/pilot_summary.json")
_v11 = _load(f"{BASE}/pilot_v1_1/rejudge_summary.json")
_diag = _load(f"{BASE}/pilot_v1_1/diagnosis.json")

CASES = {}
for r in (_v1 or {}).get("objects", []):
    CASES[r["object_id"]] = dict(case_id=r["object_id"], category=r["category"],
                                 v1=r, v11=None,
                                 root=f"{BASE}/pilot_v1/{r['object_id']}")
for r in (_v11 or {}).get("objects", []):
    c = CASES.setdefault(r["object_id"], dict(
        case_id=r["object_id"], category=r["category"], v1=None,
        root=f"{BASE}/pilot_v1/{r['object_id']}"))
    c["v11"] = r
# development case(鞋): 不属于 pilot 统计, 仅供 QA 浏览
if os.path.exists(f"{BASE}/pseudo_gt/shoe_22b822_v1/manifest.json"):
    CASES["shoe_22b822_dev"] = dict(
        case_id="shoe_22b822_dev", category="development(不计入统计)",
        v1=None, v11=None, root="", dev=dict(
            sample=f"{BASE}/pseudo_gt/shoe_22b822_v1",
            quality=glob.glob(f"{BASE}/pseudo_gt_quality/shoe_22b822*")))

def case_fields(c):
    """三轴状态(优先 V1.1 语义) + 指标摘要."""
    v11, v1 = c.get("v11"), c.get("v1")
    f = dict(case_id=c["case_id"], category=c["category"],
             processing_status="OK", structural_status="-",
             label_quality="NOT_EVALUATED", borderline=False,
             signal_dist=None, reason="", warnings=[],
             G_global={}, G_HF={}, ssim_delta_mean={}, fill={}, B_signal={})
    if v11:
        f.update(processing_status=v11["processing_status"],
                 structural_status=v11["structural_status"] or "-",
                 label_quality=v11["label_quality"],
                 borderline=v11.get("borderline", False),
                 signal_dist=v11.get("signal_dist"),
                 reason=v11.get("reason", ""), warnings=v11.get("warnings", []),
                 G_global=v11.get("G_global", {}), G_HF=v11.get("G_HF", {}),
                 ssim_delta_mean=v11.get("ssim_delta_mean", {}),
                 fill=v11.get("fill", {}))
    elif v1:
        f.update(structural_status=v1.get("structural_status") or "-",
                 label_quality="(V1 legacy) " + v1.get("quality_status", ""),
                 signal_dist=v1.get("signal_dist"),
                 reason=v1.get("failure_reason", ""),
                 G_global=v1.get("G_global", {}), G_HF=v1.get("G_HF", {}),
                 fill=v1.get("fill", {}))
    if v1:
        f["B_signal"] = v1.get("B_signal", {})
    if "dev" in c:
        man = _load(f"{c['dev']['sample']}/manifest.json") or {}
        f.update(structural_status=man.get("status", "?"),
                 label_quality="development case(见 quality 报告)")
    return f

def case_paths(c):
    """全部已存在 artifact 的绝对路径(只列不改)."""
    cid = c["case_id"]
    if "dev" in c:
        cand = dict(sample=c["dev"]["sample"],
                    quality_v1=(c["dev"]["quality"] or [""])[0])
    else:
        q1 = glob.glob(f"{c['root']}/quality/*")
        q11 = glob.glob(f"{BASE}/pilot_v1_1/{cid}/quality/*")
        cand = dict(teacher=f"{c['root']}/teacher",
                    teacher_retry=f"{BASE}/pilot_v1_1/{cid}/teacher",
                    sample=f"{c['root']}/sample",
                    quality_v1=(q1 or [""])[0], quality_v11=(q11 or [""])[0],
                    diag_fig=f"{BASE}/pilot_v1_1/{cid}/demand_extremes.png")
    return {k: v for k, v in cand.items() if v and os.path.exists(v)}

def load_reviews():
    revs = []
    if os.path.exists(REVIEW_PATH):
        for ln in open(REVIEW_PATH):
            ln = ln.strip()
            if ln:
                revs.append(json.loads(ln))
    return revs

_revs = load_reviews()
print(f"catalog: {len(CASES)} cases | 已人工标注 {len({r['case_id'] for r in _revs})} 个 "
      f"(记录 {len(_revs)} 条) | rejudge_v1_1: {'有' if _v11 else '无'} | "
      f"diagnosis: {'有' if _diag else '无'}")
_lab = {}
for c in CASES.values():
    L = case_fields(c)["label_quality"]
    _lab[L] = _lab.get(L, 0) + 1
print("label_quality 分布:", _lab)'''))

cells.append(md("""## 1. 选 case（纯参数 cell，无 widget 也可用）

- `MODE`: `random` / `by_id` / `by_processing_status` / `by_label_quality` /
  `by_category` / `unreviewed_only`
- `SAMPLING`（`random`/`unreviewed_only` 时生效）:
  `uniform`（object 级均匀）/ `stratified`（先均匀抽类别再抽 case）/
  `failure_mixed`（仅 MIXED/NEGATIVE/处理失败/结构拒绝池）
- `by_*` 模式配 `FILTER_VALUE`；`by_id` 配 `CASE_ID`。
**换 case = 重跑下面这个 cell + 展示 cell**（`RANDOM_SEED=None` 时每次自动换新 seed）；
复现某次抽样：把打印出的 seed 填回 `RANDOM_SEED`。"""))

cells.append(code(r'''RANDOM_SEED = None       # None = 每次重跑本 cell 抽一个新 case(seed 会打印, 可复现);
                         # 填固定整数 = 精确复现那一次抽样
MODE = "random"          # random | by_id | by_processing_status | by_label_quality
                         # | by_category | unreviewed_only
SAMPLING = "uniform"     # uniform | stratified | failure_mixed
CASE_ID = "sample_WaterBottle"   # MODE="by_id" 时用
FILTER_VALUE = None      # by_processing_status/by_label_quality/by_category 用

SEED = RANDOM_SEED if RANDOM_SEED is not None else int.from_bytes(os.urandom(4), "little")
if "HISTORY" not in globals():
    HISTORY, HIST_POS = [], -1

def _is_failure_mixed(f):
    return (f["processing_status"] != "OK" or f["structural_status"] == "REJECTED"
            or f["label_quality"] in ("MIXED", "NEGATIVE"))

def pick_case():
    rng = np.random.RandomState(SEED)
    fields = {cid: case_fields(c) for cid, c in CASES.items()}
    if MODE == "by_id":
        assert CASE_ID in CASES, f"未知 case_id: {CASE_ID}(可选: {sorted(CASES)})"
        return CASES[CASE_ID]
    pool = sorted(CASES)
    if MODE == "by_processing_status":
        pool = [i for i in pool if fields[i]["processing_status"] == FILTER_VALUE]
    elif MODE == "by_label_quality":
        pool = [i for i in pool if fields[i]["label_quality"] == FILTER_VALUE]
    elif MODE == "by_category":
        pool = [i for i in pool if CASES[i]["category"] == FILTER_VALUE]
    elif MODE == "unreviewed_only":
        seen = {r["case_id"] for r in load_reviews()}
        pool = [i for i in pool if i not in seen]
    if MODE in ("random", "unreviewed_only"):
        if SAMPLING == "failure_mixed":
            pool = [i for i in pool if _is_failure_mixed(fields[i])]
        elif SAMPLING == "stratified":
            cats = sorted({CASES[i]["category"] for i in pool})
            cat = cats[rng.randint(len(cats))]
            pool = [i for i in pool if CASES[i]["category"] == cat]
    assert pool, f"筛选后无 case(MODE={MODE}, FILTER_VALUE={FILTER_VALUE})"
    return CASES[pool[rng.randint(len(pool))]]

CURRENT = pick_case()
HISTORY.append(CURRENT["case_id"]); HIST_POS = len(HISTORY) - 1
print(f"seed={SEED}{'(自动抽取)' if RANDOM_SEED is None else ''}  MODE={MODE}  "
      f"SAMPLING={SAMPLING}  ->  case_id = {CURRENT['case_id']}")
print(f"复现方式: RANDOM_SEED = {SEED}")

def prev_case():
    """无 widget 的历史导航: 在任意 cell 调用 prev_case()/next_case() 后重跑展示 cell."""
    global CURRENT, HIST_POS
    HIST_POS = max(HIST_POS - 1, 0); CURRENT = CASES[HISTORY[HIST_POS]]
    print("case_id =", CURRENT["case_id"])

def next_case():
    global CURRENT, HIST_POS
    HIST_POS = min(HIST_POS + 1, len(HISTORY) - 1); CURRENT = CASES[HISTORY[HIST_POS]]
    print("case_id =", CURRENT["case_id"])'''))

cells.append(md("""## 2. 展示当前 case（轻量读取，不执行 pipeline）"""))

cells.append(code(r'''def _show_png(p, figw=13, title=""):
    im = imread(p)
    plt.figure(figsize=(figw, figw * im.shape[0] / im.shape[1]))
    plt.imshow(im); plt.axis("off")
    if title:
        plt.title(title, fontsize=10)
    plt.show()

def _microchart_stats(sample_dir):
    """从已存 NPZ + manifest 读 chart/coverage/subpixel(teacher atlas 分辨率下)."""
    man = _load(f"{sample_dir}/manifest.json")
    npz_p = f"{sample_dir}/arrays.npz"
    if not (man and os.path.exists(npz_p)):
        return None
    z = np.load(npz_p)
    R = int(np.max(man["teacher"]["atlas_size"]))  # 标量或 [W,H] 都兼容
    f2c, tpu = z["face_to_chart"], z["target_packed_uv"]
    C = int(f2c.max()) + 1
    spans = np.full((C, 2), np.nan)
    for ci in range(C):
        uv = tpu[f2c == ci].reshape(-1, 2)
        if len(uv):
            spans[ci] = uv.max(0) - uv.min(0)
    m = spans.min(1) * R
    return dict(n_charts=C, atlas_size=R,
                coverage=man["geometry"]["train_face_coverage"],
                subpixel_ratio=float((m < 1).mean()),
                microchart_lt2px_ratio=float((m < 2).mean()))

def show_case(c):
    f = case_fields(c)
    paths = case_paths(c)
    print("=" * 78)
    print(f"case_id: {f['case_id']}   类别: {f['category']}")
    print(f"processing_status: {f['processing_status']}   "
          f"structural: {f['structural_status']}   "
          f"label_quality: {f['label_quality']}"
          f"{'(BORDERLINE)' if f['borderline'] else ''}")
    if f["processing_status"] != "OK" or f["structural_status"] == "REJECTED":
        stage = ("PartUV" if f["processing_status"] == "PARTUV_FAILED" else
                 "packing" if f["processing_status"] == "PACKING_FAILED" else
                 "precheck" if f["processing_status"] == "PRECHECK_REJECTED" else
                 "结构验收(exporter)" if f["structural_status"] == "REJECTED"
                 else f["processing_status"])
        print(f"!! 失败阶段: {stage}")
        print(f"!! 原因: {f['reason'] or '(无)'}")
    elif f["reason"]:
        print(f"原因/证据: {f['reason']}")
    for w in f["warnings"]:
        print(f"warning: {w}")
    if f["signal_dist"] is not None:
        print(f"signal_dist = {f['signal_dist']}")

    tiers = sorted(set(f["G_global"]) | set(f["fill"]))
    if tiers:
        print("\n---- 两档指标 (Uniform vs TD) ----")
        for t in tiers:
            fills = f["fill"].get(t, {})
            bs = f["B_signal"].get(t, {})
            print(f"  {t}: G_global={f['G_global'].get(t, '-'):>8}  "
                  f"G_HF={f['G_HF'].get(t, '-'):>8}  "
                  f"dSSIM(paired mean)={f['ssim_delta_mean'].get(t, '-'):>9}  "
                  f"fill={ {m: round(v * 100, 1) for m, v in fills.items()} }  "
                  f"B_signal={bs if bs else '-'}")

    sd = paths.get("sample")
    if sd:
        st = _microchart_stats(sd)
        if st:
            print(f"\ncharts={st['n_charts']}  teacher_atlas={st['atlas_size']}  "
                  f"coverage={st['coverage'] * 100:.3f}%  "
                  f"subpixel(<1px)={st['subpixel_ratio'] * 100:.1f}%  "
                  f"microchart(<2px)={st['microchart_lt2px_ratio'] * 100:.1f}%")
    if _diag and f["case_id"] in _diag.get("cases", {}):
        d = _diag["cases"][f["case_id"]]
        print("诊断(V1.1): TD subpixel_ratio="
              f"{d.get('TD', {}).get('subpixel_chart_ratio', '-')}, "
              f"padding/signal={d.get('TD', {}).get('padding_over_signal', '-')}, "
              f"seam_error_share={d.get('TD', {}).get('seam_error_share', '-')}"
              + (f", fixed_B_signal={d['fixed_B_signal']['verdict']}"
                 if "fixed_B_signal" in d else ""))

    print("\n---- 源文件绝对路径 ----")
    for k, p in paths.items():
        print(f"  {k:14s} {p}")
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                fp = os.path.join(p, fn)
                print(f"      - {fp}  ({os.path.getsize(fp):,} B)")

    qd = paths.get("quality_v1") or paths.get("quality_v11")
    shown = False
    if qd:
        for fig, title in [("render_comparison.png", "Reference / Uniform / TD 同视角"),
                           ("layout_comparison.png", "UV layout (Uniform vs TD)"),
                           ("error_heatmap.png", "误差热图"),
                           ("detail_crops.png", "细节裁剪")]:
            p = os.path.join(qd, fig)
            if os.path.exists(p):
                _show_png(p, title=title); shown = True
    if os.path.exists(paths.get("diag_fig", "")):
        _show_png(paths["diag_fig"], figw=9, title="诊断: 最高/最低需求 chart 内容")
        shown = True
    if not shown:
        # 失败/拒绝 case: 展示已有预览(teacher atlas / reference), 不静默跳过
        prev = []
        for k in ("teacher", "teacher_retry", "sample"):
            if k in paths:
                prev += sorted(glob.glob(f"{paths[k]}/*.png"))[:3]
        if prev:
            for p in prev:
                _show_png(p, figw=8, title=os.path.basename(p))
        else:
            print("\n(该 case 无任何可视 artifact —— 仅有上方的失败阶段与原因)")

show_case(CURRENT)'''))

cells.append(md("""## 3. 人工标注（追加写 JSONL，不改任何已有数据）

- verdict: `OK` / `SUSPICIOUS` / `CONFIRMED_BUG`
- bug_type（CONFIRMED_BUG 必填）: `PARTUV` / `MATERIAL_MAPPING` / `TD_ALLOCATION` /
  `PACKING` / `REBAKE_OR_SEAM` / `METRIC_DISAGREEMENT` / `GEOMETRY_EXPORT` / `OTHER`

取消注释并运行下方调用即可保存一条对**当前 case** 的标注。"""))

cells.append(code(r'''VERDICTS = {"OK", "SUSPICIOUS", "CONFIRMED_BUG"}
BUG_TYPES = {"PARTUV", "MATERIAL_MAPPING", "TD_ALLOCATION", "PACKING",
             "REBAKE_OR_SEAM", "METRIC_DISAGREEMENT", "GEOMETRY_EXPORT", "OTHER"}

def save_review(verdict, bug_type=None, note=""):
    assert verdict in VERDICTS, f"verdict 必须是 {VERDICTS}"
    assert bug_type is None or bug_type in BUG_TYPES, f"bug_type 必须是 {BUG_TYPES}"
    if verdict == "CONFIRMED_BUG":
        assert bug_type, "CONFIRMED_BUG 必须给 bug_type"
    os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
    rec = dict(ts=datetime.now().isoformat(timespec="seconds"),
               case_id=CURRENT["case_id"], verdict=verdict, bug_type=bug_type,
               note=note, reviewer=os.environ.get("USER", "?"),
               reviewer_type="human",   # 浏览器为人工标注入口; agent 记录用 "agent"

               seed=SEED, mode=MODE, sampling=SAMPLING)
    with open(REVIEW_PATH, "a") as fp:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("已追加 ->", REVIEW_PATH); print(json.dumps(rec, ensure_ascii=False))

# save_review("SUSPICIOUS", bug_type="REBAKE_OR_SEAM", note="接缝处颜色渗出")
n_rev = load_reviews()
print(f"当前标注总数 {len(n_rev)} 条, 覆盖 {len({r['case_id'] for r in n_rev})}/"
      f"{len(CASES)} cases")'''))

cells.append(md("""## 4. 可选按钮界面（ipywidgets 可用时；无则用上方参数 cell）"""))

cells.append(code(r'''try:
    import ipywidgets as W
    from IPython.display import display, clear_output

    _out = W.Output()
    _verd = W.Dropdown(options=sorted(VERDICTS), description="verdict")
    _bug = W.Dropdown(options=[None] + sorted(BUG_TYPES), description="bug_type")
    _note = W.Text(description="note", layout=W.Layout(width="50%"))

    def _refresh():
        with _out:
            clear_output(wait=True); show_case(CURRENT)

    def _rand(_):
        global CURRENT, SEED, MODE, HIST_POS
        MODE = "random"; SEED = int.from_bytes(os.urandom(4), "little")
        CURRENT = pick_case()
        HISTORY.append(CURRENT["case_id"]); HIST_POS = len(HISTORY) - 1
        print(f"seed={SEED} -> {CURRENT['case_id']}"); _refresh()

    def _prev(_):
        prev_case(); _refresh()

    def _next(_):
        next_case(); _refresh()

    def _save(_):
        save_review(_verd.value, _bug.value, _note.value)

    bts = [W.Button(description=d) for d in
           ("Random Case", "Previous", "Next", "保存标注")]
    for b, h in zip(bts, (_rand, _prev, _next, _save)):
        b.on_click(h)
    display(W.HBox(bts[:3]), W.HBox([_verd, _bug, _note, bts[3]]), _out)
    _refresh()
except ImportError:
    print("ipywidgets 不可用 —— 使用上方参数 cell(MODE/RANDOM_SEED)与 "
          "prev_case()/next_case() 即可, 功能等价。")'''))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
with open(NB, "w") as f:
    nbf.write(nb, f)
print(f"written: {NB} ({len(cells)} cells)")
