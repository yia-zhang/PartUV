# -*- coding: utf-8 -*-
"""Stratified Pilot V1 —— 17 个未见资产(排除鞋/车轮/所有参与过 β 选择与调试的资产).

每资产: 一次 PartUV teacher 运行(map_partuv_td, 冻结 β=0.75/auto 平价)
      -> export_object_pseudo_gt(结构验收)
      -> quality_gate(冻结两档 fixed-B_raw 协议)
统计单位 = object。逐资产容错: 任一环节异常 -> FAIL + failure_reason, 继续。
不修改 teacher/β/gates/协议; 不新增指标。

状态语义(冻结):
  development cases   = 鞋/车轮(不计入 pilot 统计)
  pilot metric status = 尚未校准
  pipeline validation = CASE_STUDY_ONLY
  training eligibility 仅限逐样本 td_allocation label
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.api import map_partuv_td
from tdlib.dataset import export_object_pseudo_gt
from run_pseudo_gt_quality_gate import quality_gate

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"

# 分层资产清单(均未参与 β 选择/前期调试; 排除鞋/车轮/Corset/8d1b/Duck/
# DamagedHelmet/clock_2mat/synthetic_freq 等 development/debug 资产)
ASSETS = [
    # 1. 低纹理/纯色
    ("synth_flat_solid",        "synth_flat_solid.glb",              "low_texture"),
    ("synth_gradient",          "synth_gradient.glb",                "low_texture"),
    ("objav_7f9212",            "objaverse_7f92127194344e30.glb",    "low_texture"),
    # 2. 局部 logo/文字
    ("objav_138900",            "objaverse_1389004d1eb94507.glb",    "local_logo"),
    ("objav_dd2030",            "objaverse_dd20303e82654cbd.glb",    "local_logo"),
    ("sample_WaterBottle",      "sample_WaterBottle.glb",            "local_logo"),
    # 3. 分布式高频
    ("synth_multiscale_noise",  "synth_multiscale_noise.glb",        "distributed_hf"),
    ("sample_BarramundiFish",   "sample_BarramundiFish.glb",         "distributed_hf"),
    ("sample_Avocado",          "sample_Avocado.glb",                "distributed_hf"),
    # 4. 大量小 chart
    ("sample_Fox",              "sample_Fox.glb",                    "many_charts"),
    ("sample_Lantern",          "sample_Lantern.glb",                "many_charts"),
    ("synth_many_islands",      "synth_many_islands_144.glb",        "many_charts"),
    # 5. 多材质 / UV overlap warning
    ("synth_two_materials",     "synth_two_materials.glb",           "multimat_overlap"),
    ("synth_trimsheet_reuse6",  "synth_trimsheet_reuse6.glb",        "multimat_overlap"),
    # 6. 不同几何类别
    ("sample_BoomBox",          "sample_BoomBox.glb",                "geometry_misc"),
    ("synth_open_terrain",      "synth_open_terrain.glb",            "geometry_misc"),
    ("objav_2de1dd_hipoly",     "objaverse_2de1dd3830864ade.glb",    "geometry_misc"),
]

rows = []
for oid, fn, cat in ASSETS:
    print(f"\n================ {oid} ({cat}) ================", flush=True)
    row = dict(object_id=oid, category=cat, asset=fn,
               structural_status="", quality_status="", signal_dist=None,
               G_global={}, G_HF={}, masked_ssim={}, fill={}, B_signal={},
               failure_reason="")
    try:
        res = map_partuv_td(f"{DATA}/{fn}", f"{OUT}/{oid}/teacher/")
        m = export_object_pseudo_gt(res, f"{OUT}/{oid}/sample", object_id=oid)
        row["structural_status"] = m["status"]
        if m["status"] != "ACCEPTED":
            row["quality_status"] = "FAIL"
            row["failure_reason"] = "structural: " + "; ".join(
                k for k, v in m["gates"].items() if not v)
        else:
            rep, met = quality_gate(f"{OUT}/{oid}/sample", f"{OUT}/{oid}/quality")
            row["quality_status"] = rep["quality_status"]
            row["signal_dist"] = round(rep["signal_dist"], 4)
            row["failure_reason"] = rep["failure_reason"]
            for t, r_ in met["tiers"].items():
                row["G_global"][t] = r_["G_global"]
                row["G_HF"][t] = r_["G_HF"]
                row["masked_ssim"][t] = {
                    m_: r_["methods"][m_]["masked_ssim_mean"]
                    for m_ in r_["methods"]}
                row["fill"][t] = {m_: r_["methods"][m_]["packing_fill"]
                                  for m_ in r_["methods"]}
                row["B_signal"][t] = {m_: r_["methods"][m_]["B_signal"]
                                      for m_ in r_["methods"]}
        print(f"  -> structural={row['structural_status']} "
              f"quality={row['quality_status']} "
              f"G_global={row['G_global']} G_HF={row['G_HF']} "
              f"reason={row['failure_reason'] or '-'}", flush=True)
    except Exception as e:
        row["quality_status"] = "FAIL"
        row["failure_reason"] = f"exception: {type(e).__name__}: {str(e)[:200]}"
        print(f"  -> FAIL {row['failure_reason']}", flush=True)
        traceback.print_exc()
    rows.append(row)

# ---- object-level 汇总(统计单位=object; 鞋/车轮为 development cases 不在此列) ----
done = [r for r in rows if r["G_global"]]
gg50 = np.array([r["G_global"].get("50pct", np.nan) for r in done], float)
gh50 = np.array([r["G_HF"].get("50pct", np.nan) for r in done], float)
gg25 = np.array([r["G_global"].get("25pct", np.nan) for r in done], float)
gh25 = np.array([r["G_HF"].get("25pct", np.nan) for r in done], float)


def stats(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return dict(median=round(float(np.median(x)), 4),
                q25=round(float(np.percentile(x, 25)), 4),
                q75=round(float(np.percentile(x, 75)), 4),
                win_rate=round(float((x > 0).mean()), 3),
                worst10pct=round(float(np.percentile(x, 10)), 4),
                n=int(len(x)))


by_cat = {}
for cat in sorted({r["category"] for r in rows}):
    sub = [r for r in rows if r["category"] == cat]
    by_cat[cat] = {s: sum(1 for r in sub if r["quality_status"] == s)
                   for s in ("PASS", "LOW_SIGNAL", "FAIL")}

summary = dict(
    semantics=dict(
        development_cases=["shoe_22b822", "wheel_92ff6"],
        pilot_metric_status="尚未校准",
        pipeline_validation="CASE_STUDY_ONLY",
        training_eligibility="仅限逐样本 td_allocation label",
        statistics_unit="object"),
    n_objects=len(rows),
    counts={s: sum(1 for r in rows if r["quality_status"] == s)
            for s in ("PASS", "LOW_SIGNAL", "FAIL")},
    stats=dict(G_global_50=stats(gg50), G_HF_50=stats(gh50),
               G_global_25=stats(gg25), G_HF_25=stats(gh25)),
    by_category=by_cat,
    objects=rows)
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/pilot_summary.json", "w") as fp:
    json.dump(summary, fp, indent=1, ensure_ascii=False)

print("\n============ PILOT 汇总(object-level) ============")
hdr = (f"{'object':24s} {'类别':16s} {'结构':9s} {'质量':11s} {'sig':>6s} "
       f"{'G_g50':>7s} {'G_hf50':>7s} {'G_g25':>7s} {'G_hf25':>7s}  失败原因")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['object_id']:24s} {r['category']:16s} "
          f"{r['structural_status'] or '-':9s} {r['quality_status']:11s} "
          f"{r['signal_dist'] if r['signal_dist'] is not None else '':>6} "
          f"{r['G_global'].get('50pct', ''):>7} {r['G_HF'].get('50pct', ''):>7} "
          f"{r['G_global'].get('25pct', ''):>7} {r['G_HF'].get('25pct', ''):>7}  "
          f"{r['failure_reason'][:44] or '-'}")
print("\ncounts:", summary["counts"])
print("stats:", json.dumps(summary["stats"], ensure_ascii=False))
print("by_category:", json.dumps(by_cat, ensure_ascii=False))
print("PILOT_V1: DONE")
