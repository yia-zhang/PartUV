# -*- coding: utf-8 -*-
"""V1.1 Gate and Failure Attribution —— 用新状态语义重判原 17 个 pilot 资产.

processing_status: OK / PRECHECK_REJECTED / PARTUV_FAILED / PACKING_FAILED
structural_status: ACCEPTED / REJECTED (exporter 结构验收, 独立轴)
label_quality:     POSITIVE / NEUTRAL(LOW_TD_CONTRAST) / MIXED / NEGATIVE /
                   NOT_EVALUATED

复用 pilot 的已导出 sample(不重跑 PartUV); 仅对 3 个此前崩溃/打包失败的资产
重新走一次 map_partuv_td 以捕获 fail-graceful 的明确状态(不做 chart merge/split)。
fill 为 diagnostic warning; SSIM 报 paired per-view delta。
原 17 个资产此后属于 development set, 不得用于最终 held-out validation。
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

from tdlib.api import (map_partuv_td, PartUVFailedError, UnsupportedAssetError,
                       NeedsMultiAtlasError)
from tdlib.layout import PackingFailedError
from run_pseudo_gt_quality_gate import quality_gate

# 与 run_pilot_v1.ASSETS 一致(该脚本无 main 守卫, 不可 import)
DATA = "/root/youjiaZhang/PartUV/code/data"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
ASSETS = [
    ("synth_flat_solid",        "synth_flat_solid.glb",              "low_texture"),
    ("synth_gradient",          "synth_gradient.glb",                "low_texture"),
    ("objav_7f9212",            "objaverse_7f92127194344e30.glb",    "low_texture"),
    ("objav_138900",            "objaverse_1389004d1eb94507.glb",    "local_logo"),
    ("objav_dd2030",            "objaverse_dd20303e82654cbd.glb",    "local_logo"),
    ("sample_WaterBottle",      "sample_WaterBottle.glb",            "local_logo"),
    ("synth_multiscale_noise",  "synth_multiscale_noise.glb",        "distributed_hf"),
    ("sample_BarramundiFish",   "sample_BarramundiFish.glb",         "distributed_hf"),
    ("sample_Avocado",          "sample_Avocado.glb",                "distributed_hf"),
    ("sample_Fox",              "sample_Fox.glb",                    "many_charts"),
    ("sample_Lantern",          "sample_Lantern.glb",                "many_charts"),
    ("synth_many_islands",      "synth_many_islands_144.glb",        "many_charts"),
    ("synth_two_materials",     "synth_two_materials.glb",           "multimat_overlap"),
    ("synth_trimsheet_reuse6",  "synth_trimsheet_reuse6.glb",        "multimat_overlap"),
    ("sample_BoomBox",          "sample_BoomBox.glb",                "geometry_misc"),
    ("synth_open_terrain",      "synth_open_terrain.glb",            "geometry_misc"),
    ("objav_2de1dd_hipoly",     "objaverse_2de1dd3830864ade.glb",    "geometry_misc"),
]

OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"
os.makedirs(OUT11, exist_ok=True)

rows = []
for oid, fn, cat in ASSETS:
    print(f"\n================ {oid} ({cat}) ================", flush=True)
    row = dict(object_id=oid, category=cat, asset=fn,
               processing_status="OK", structural_status="",
               label_quality="NOT_EVALUATED", borderline=False,
               signal_dist=None, G_global={}, G_HF={}, ssim_delta_mean={},
               fill={}, warnings=[], reason="")
    sample = f"{OUT}/{oid}/sample"
    try:
        mpath = f"{sample}/manifest.json"
        if os.path.exists(mpath):
            man = json.load(open(mpath))
            row["structural_status"] = man["status"]
            if man["status"] != "ACCEPTED":
                row["reason"] = "structural: " + "; ".join(
                    k for k, v in man["gates"].items() if not v)
            else:
                rep, met = quality_gate(sample, f"{OUT11}/{oid}/quality",
                                        make_figs=False)
                row["label_quality"] = rep["label_quality"]
                row["borderline"] = rep["label_quality_borderline"]
                row["signal_dist"] = round(rep["signal_dist"], 4)
                row["warnings"] = rep["warnings"]
                row["reason"] = rep["failure_reason"]
                for t, r_ in met["tiers"].items():
                    row["G_global"][t] = r_["G_global"]
                    row["G_HF"][t] = r_["G_HF"]
                    row["ssim_delta_mean"][t] = r_["ssim_delta_mean"]
                    row["fill"][t] = {m: r_["methods"][m]["packing_fill"]
                                      for m in r_["methods"]}
        else:
            # pilot 中未产出 sample 的失败资产: 重跑一次以捕获明确状态
            res = map_partuv_td(f"{DATA}/{fn}", f"{OUT11}/{oid}/teacher/")
            row["reason"] = "pilot 中失败但本次重跑成功(需人工复核非确定性)"
            row["processing_status"] = "OK_ON_RETRY"
    except PartUVFailedError as e:
        row["processing_status"] = "PARTUV_FAILED"
        row["reason"] = str(e)[:160]
    except PackingFailedError as e:
        row["processing_status"] = "PACKING_FAILED"
        row["reason"] = str(e)[:160]
    except (UnsupportedAssetError, NeedsMultiAtlasError) as e:
        row["processing_status"] = "PRECHECK_REJECTED"
        row["reason"] = str(e)[:160]
    except Exception as e:
        row["processing_status"] = "ERROR"
        row["reason"] = f"{type(e).__name__}: {str(e)[:160]}"
        traceback.print_exc()
    print(f"  -> proc={row['processing_status']} "
          f"struct={row['structural_status'] or '-'} "
          f"label={row['label_quality']}"
          f"{'(BORDERLINE)' if row['borderline'] else ''} "
          f"warn={len(row['warnings'])} reason={row['reason'][:80] or '-'}",
          flush=True)
    rows.append(row)

counts_label = {}
for r in rows:
    counts_label[r["label_quality"]] = counts_label.get(r["label_quality"], 0) + 1
counts_proc = {}
for r in rows:
    counts_proc[r["processing_status"]] = counts_proc.get(r["processing_status"], 0) + 1

summary = dict(
    semantics=dict(
        schema="pilot_v1_1_rejudge",
        development_cases=["shoe_22b822", "wheel_92ff6"],
        original_17_assets="development set(经 V1/V1.1 检视), 不得用于最终 held-out validation",
        pilot_metric_status="尚未校准",
        pipeline_validation="CASE_STUDY_ONLY",
        neutral_alias="NEUTRAL = LOW_TD_CONTRAST(chart 间无明显重分配, 不表示纹理低频)",
        fill_gate="diagnostic warning(overlap/OOB/预算不满足仍为硬失败)",
        ssim_gate="paired per-view delta; 轻微降(-0.002..-0.01)=BORDERLINE, 明显降(<-0.01)=负向证据"),
    counts_processing=counts_proc, counts_label=counts_label,
    objects=rows)
with open(f"{OUT11}/rejudge_summary.json", "w") as fp:
    json.dump(summary, fp, indent=1, ensure_ascii=False)

print("\n============ V1.1 重判汇总 ============")
hdr = (f"{'object':24s} {'类别':16s} {'processing':16s} {'struct':9s} "
       f"{'label':14s} {'G_g50':>7s} {'G_hf50':>7s} {'dSSIM50':>8s}  原因/警告")
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['object_id']:24s} {r['category']:16s} {r['processing_status']:16s} "
          f"{r['structural_status'] or '-':9s} "
          f"{r['label_quality'] + ('*' if r['borderline'] else ''):14s} "
          f"{r['G_global'].get('50pct', ''):>7} {r['G_HF'].get('50pct', ''):>7} "
          f"{r['ssim_delta_mean'].get('50pct', ''):>8}  "
          f"{(r['reason'] or '; '.join(r['warnings']))[:56] or '-'}")
print("\ncounts_processing:", counts_proc)
print("counts_label:", counts_label)
print("REJUDGE_V1_1: DONE")
