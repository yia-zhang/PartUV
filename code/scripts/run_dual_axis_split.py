# -*- coding: utf-8 -*-
"""Allocation-vs-Delivery Split —— 双质量轴重报 17 个 development 资产.

delivery_quality_fixed_braw   = V1.1 重判的 label_quality(相同 B_raw, 两档,
                                含 SSIM/warning 证据) —— 直接复用, 不重算。
allocation_quality_fixed_bsignal = 等 B_signal 对比(50pct 档): 对 TD 单独搜
                                atlas 边长使 B_signal ≈ Uniform@R50, 比表面域
                                MSE/HF —— 隔离 content ranking, 排除 packing。
分类(allocation, 线性 RGB, 无渲染/SSIM —— SSIM 属 delivery 轴):
  NOT_EVALUATED: 无 ACCEPTED 样本
  NEUTRAL: signal_dist<0.05, 或等预算下 |G_global_eq|<=2% 且 |G_HF_eq|<5%
  POSITIVE: G_global_eq>=2%, 或 (G_HF_eq>=5% 且 G_global_eq>=-2%)
  NEGATIVE: G_global_eq<=-5% 且无正向轴
  MIXED: 其余冲突情形
training_eligible 三分开: td_allocation(allocation 轴) / local_uv_refinement(false) /
  packed_uv_regression(当前全部 false)。
artifact_valid: packed_layout(几何合法/无 overlap/OOB) / rebaked_asset(需
  delivery+seam integrity 双通过, baker 审计通过前全部 false)。
边界: 不改 β/signal/PartUV, 不进入 Signal V2。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.layout import PackingFailedError
from diag_common import load_sample, eval_samples, pack_only, bake_layout, surface_err

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUT11 = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1_1"
rej = json.load(open(f"{OUT11}/rejudge_summary.json"))

LOW_SIGNAL_DIST = 0.05
BAND_G, POS_HF, NEG_G = 0.02, 0.05, -0.05


def alloc_axis(sample_dir):
    """等 B_signal 的 allocation 轴度量(50pct 档)."""
    ctx = load_sample(sample_dir)
    if ctx["signal_dist"] < LOW_SIGNAL_DIST:
        return dict(label="NEUTRAL", why="signal_dist<0.05(无明显重分配)")
    ev = eval_samples(ctx)
    R = max(int(round(np.sqrt(0.50 * ctx["B_source"]))), 64)
    pu = pack_only(ctx, ctx["scales_uni"], R)
    tex_u, nuv_u = bake_layout(ctx, pu["uvs"], R)
    d_u = surface_err(tex_u, nuv_u, ev)
    # 对 TD 搜边长使 B_signal ≈ Uniform 的 B_signal
    S_target, lo, hi, best = pu["b_signal"], int(R * 0.7), int(R * 1.6), None
    for _ in range(8):
        mid = (lo + hi) // 2
        try:
            pt = pack_only(ctx, ctx["scales_td"], mid)
        except PackingFailedError:
            lo = mid + 1
            continue
        if best is None or abs(pt["b_signal"] - S_target) < abs(best[1]["b_signal"] - S_target):
            best = (mid, pt)
        if pt["b_signal"] < S_target:
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return dict(label="NOT_EVALUATED", why="等 B_signal 搜索全部打包失败")
    R_td, pt = best
    tex_t, nuv_t = bake_layout(ctx, pt["uvs"], R_td)
    d_t = surface_err(tex_t, nuv_t, ev)
    g_eq = 1 - float(d_t.mean()) / max(float(d_u.mean()), 1e-20)
    ghf_eq = 1 - float(d_t[ev["hi"]].mean()) / max(float(d_u[ev["hi"]].mean()), 1e-20)
    m = dict(R_uniform=R, R_td=R_td,
             bsignal_match=round(pt["b_signal"] / max(S_target, 1), 4),
             G_global_eq=round(g_eq, 4), G_HF_eq=round(ghf_eq, 4),
             overlap=pt["overlap"] + pu["overlap"])
    if m["overlap"]:
        m.update(label="NOT_EVALUATED", why="等 B_signal 布局出现 overlap")
    elif abs(g_eq) <= BAND_G and abs(ghf_eq) < POS_HF:
        m.update(label="NEUTRAL", why="等预算下差异在中性带内")
    elif g_eq >= BAND_G or (ghf_eq >= POS_HF and g_eq >= -BAND_G):
        m.update(label="POSITIVE", why="等预算下 TD 明确更好")
    elif g_eq <= NEG_G and ghf_eq < POS_HF:
        m.update(label="NEGATIVE", why="等预算下 TD 明确更差(content ranking 嫌疑)")
    else:
        m.update(label="MIXED", why="等预算下证据冲突")
    return m


rows = []
for r in rej["objects"]:
    oid = r["object_id"]
    print(f"\n================ {oid} ================", flush=True)
    delivery = r["label_quality"]
    row = dict(object_id=oid, category=r["category"],
               processing_status=r["processing_status"],
               structural_status=r["structural_status"] or "-",
               delivery_quality_fixed_braw=delivery,
               delivery_borderline=r.get("borderline", False),
               delivery_warnings=r.get("warnings", []),
               allocation_quality_fixed_bsignal="NOT_EVALUATED",
               allocation_detail={}, signal_dist=r.get("signal_dist"))
    sd = f"{OUT}/{oid}/sample"
    if (r["processing_status"] == "OK" and r["structural_status"] == "ACCEPTED"
            and os.path.exists(f"{sd}/manifest.json")):
        try:
            m = alloc_axis(sd)
            row["allocation_quality_fixed_bsignal"] = m.pop("label")
            row["allocation_detail"] = m
        except Exception as e:
            row["allocation_detail"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
    ok_geom = (r["processing_status"] == "OK"
               and r["structural_status"] == "ACCEPTED"
               and row["allocation_detail"].get("overlap", 0) == 0)
    row["training_eligible"] = dict(
        td_allocation=row["allocation_quality_fixed_bsignal"] in ("POSITIVE", "NEUTRAL"),
        local_uv_refinement=False,
        packed_uv_regression=False)   # 当前全部样本 false
    row["artifact_valid"] = dict(
        packed_layout=bool(ok_geom),  # 仅几何合法/无 overlap/OOB(确定性派生产物)
        rebaked_asset=False)          # 需 delivery+seam integrity 双通过才可 true
    print(f"  alloc={row['allocation_quality_fixed_bsignal']} "
          f"{row['allocation_detail']}  delivery={delivery}", flush=True)
    rows.append(row)

cnt = lambda key: {s: sum(1 for r in rows if r[key] == s) for s in
                   ("POSITIVE", "NEUTRAL", "MIXED", "NEGATIVE", "NOT_EVALUATED")}
summary = dict(
    semantics=dict(
        schema="dual_axis_split_v1",
        allocation="fixed-B_signal(50pct 档, 等交付预算, 隔离 content ranking)",
        delivery="fixed-B_raw(V1.1 label_quality, 两档, 含 SSIM/packing 交付)",
        notes=["gradient 等 B_signal 下 TD/U≈0.98 -> 不得称为明确 content failure",
               "two_materials 等 B_signal 下 TD/U≈0.82 -> allocation POSITIVE, "
               "delivery MIXED(fill 崩塌是 packing delivery 问题)",
               "原 17 资产=development set, 不得用于最终 held-out validation"],
        training_eligible="三分开: td_allocation/final_packed_uv/final_rebaked_asset"),
    counts_allocation=cnt("allocation_quality_fixed_bsignal"),
    counts_delivery=cnt("delivery_quality_fixed_braw"),
    objects=rows)
with open(f"{OUT11}/dual_axis_summary.json", "w") as fp:
    json.dump(summary, fp, indent=1, ensure_ascii=False)

print("\n============ 双轴汇总 ============")
hdr = (f"{'object':24s} {'alloc(fixed-Bsig)':18s} {'delivery(fixed-Braw)':21s} "
       f"{'G_g_eq':>7s} {'G_hf_eq':>8s}  td_elig/layout_valid")
print(hdr); print("-" * len(hdr))
for r in rows:
    d = r["allocation_detail"]
    e = r["training_eligible"]
    print(f"{r['object_id']:24s} {r['allocation_quality_fixed_bsignal']:18s} "
          f"{r['delivery_quality_fixed_braw'] + ('*' if r['delivery_borderline'] else ''):21s} "
          f"{d.get('G_global_eq', ''):>7} {d.get('G_HF_eq', ''):>8}  "
          f"{int(e['td_allocation'])}/{int(r['artifact_valid']['packed_layout'])}")
print("\ncounts_allocation:", summary["counts_allocation"])
print("counts_delivery:", summary["counts_delivery"])
print("DUAL_AXIS: DONE")
