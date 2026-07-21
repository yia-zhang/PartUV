# -*- coding: utf-8 -*-
"""Global β Calibration V1 驱动 —— 50 冻结资产 × β∈{0,0.125,0.25}.

逐资产子进程(超时 1800s -> PROCESSING_TIMEOUT), 不替换失败资产(全计入 yield)。
口径修正: 全部候选 β 使用完全相同的 object denominator(= processing OK 的资产);
NEUTRAL(=VALID_NO_OP) 保留在 denominator, paired gain 记 0.0(no-op 的真实增益);
paired median 在相同对象集合上计算。
全局 β 决策顺序(相同 denominator): NEGATIVE 少 -> MIXED 少 -> POSITIVE 多 ->
median global 非劣 -> median HF 高 -> 更小 β。
PUBLIC-DOMAIN CALIBRATION(Objaverse), 不能代表 Meshy target-domain。
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CODE = "/root/youjiaZhang/PartUV/code"
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
PY = "/root/miniconda3/envs/geomae/bin/python"
BETAS = ["0.125", "0.25"]
TIMEOUT = 1800

manifest = json.load(open(f"{OUTD}/calibration_manifest.json"))


def run_one(a):
    oid = a["object_id"]
    rdir = f"{OUTD}/{oid}"
    rpath = f"{rdir}/result.json"
    if os.path.exists(rpath):                      # 断点续跑(资产级)
        print(f"[skip 已完成] {oid}", flush=True)
        return json.load(open(rpath)) | dict(group=a["group"])
    dpath = f"{rdir}/result_driver.json"
    if os.path.exists(dpath):                      # 超时/异常已记录过 -> 不再重烧
        print(f"[skip 已记录失败] {oid}", flush=True)
        return json.load(open(dpath)) | dict(group=a["group"])
    try:
        p = subprocess.run([PY, f"{CODE}/scripts/calib_one_asset.py",
                            a["glb"], oid, rdir],
                           timeout=TIMEOUT, capture_output=True, text=True)
        if os.path.exists(rpath):
            r = json.load(open(rpath))
        else:
            r = dict(object_id=oid, processing_status="ERROR",
                     reason=(p.stderr or "")[-200:], betas={})
    except subprocess.TimeoutExpired:
        r = dict(object_id=oid, processing_status="PROCESSING_TIMEOUT",
                 reason=f">{TIMEOUT}s", betas={})
    r["group"] = a["group"]
    os.makedirs(rdir, exist_ok=True)
    with open(f"{rdir}/result_driver.json", "w") as fp:
        json.dump(r, fp, indent=1, ensure_ascii=False)
    labs = {b: r["betas"].get(b, {}).get("label", "-") for b in BETAS}
    print(f"  -> [{oid}] {r['processing_status']} {labs} "
          f"{r.get('reason', '')[:60]}", flush=True)
    return r


with ThreadPoolExecutor(max_workers=8) as ex:      # 纯吞吐并行, 协议零变化
    rows = list(ex.map(run_one, manifest["assets"]))

# ---- yield ----
yield_cnt = {}
for r in rows:
    yield_cnt[r["processing_status"]] = yield_cnt.get(r["processing_status"], 0) + 1
D = [r for r in rows if r["processing_status"] == "OK"]
print(f"\nprocessing yield: {yield_cnt}  denominator |D|={len(D)}/50", flush=True)

# ---- 同口径 per-β 统计(NEUTRAL=VALID_NO_OP 计 0 增益, 留在 denominator) ----
TIER_NAMES = ["50pct", "25pct"]


def obj_gain(r, b, f):
    row = r["betas"].get(b, {})
    if row.get("valid_no_op") or row.get("label") == "NEUTRAL":
        return {t: 0.0 for t in TIER_NAMES}
    out = {}
    for t in TIER_NAMES:
        v = row.get("tiers", {}).get(t, {})
        if f in v:
            out[t] = v[f]
    return out


summary = {}
for b in BETAS:
    labs = [r["betas"].get(b, {}).get("label", "NOT_EVALUATED") for r in D]
    cnt = {s: labs.count(s) for s in
           ("POSITIVE", "NEUTRAL", "MIXED", "NEGATIVE", "NOT_EVALUATED")}
    gains_g = {t: [] for t in TIER_NAMES}
    gains_h = {t: [] for t in TIER_NAMES}
    harmed, ssim_ok = [], 0
    for r in D:
        gg = obj_gain(r, b, "G_global_eq")
        gh = obj_gain(r, b, "G_HF_eq")
        for t in TIER_NAMES:
            if t in gg:
                gains_g[t].append(gg[t])
            if t in gh:
                gains_h[t].append(gh[t])
        if any(v < -0.02 for v in gg.values()):
            harmed.append(r["object_id"])
        row = r["betas"].get(b, {})
        if row.get("valid_no_op") or row.get("label") == "NEUTRAL":
            ssim_ok += 1
        else:
            tt = [v for v in row.get("tiers", {}).values() if "ssim_not_worse" in v]
            ssim_ok += int(bool(tt) and all(v["ssim_not_worse"] for v in tt))
    by_group = {}
    for g in sorted({r["group"] for r in D}):
        sub = [r["betas"].get(b, {}).get("label", "NOT_EVALUATED")
               for r in D if r["group"] == g]
        by_group[g] = {s: sub.count(s) for s in
                       ("POSITIVE", "NEUTRAL", "MIXED", "NEGATIVE")}
    summary[b] = dict(
        counts=cnt, positive_rate=round(cnt["POSITIVE"] / max(len(D), 1), 3),
        harm_rate=round(len(harmed) / max(len(D), 1), 3), harmed=harmed,
        paired_median_G_global={t: round(float(np.median(v)), 4)
                                for t, v in gains_g.items() if v},
        paired_median_G_HF={t: round(float(np.median(v)), 4)
                            for t, v in gains_h.items() if v},
        ssim_not_worse_rate=round(ssim_ok / max(len(D), 1), 3),
        by_group=by_group)

# ---- 决策链(相同 denominator) ----
b1, b2 = BETAS
s1, s2 = summary[b1], summary[b2]
steps, winner = [], None
for step, k1, k2, better in [
        ("1.NEGATIVE 少", s1["counts"]["NEGATIVE"], s2["counts"]["NEGATIVE"], min),
        ("2.MIXED 少", s1["counts"]["MIXED"], s2["counts"]["MIXED"], min),
        ("3.POSITIVE 多", s1["counts"]["POSITIVE"], s2["counts"]["POSITIVE"], max)]:
    steps.append(f"{step}: β={b1}:{k1} vs β={b2}:{k2}")
    if k1 != k2:
        winner = b1 if better(k1, k2) == k1 else b2
        steps.append(f"  -> 决出 β={winner}")
        break
if winner is None:
    m1 = np.mean(list(s1["paired_median_G_global"].values()) or [0])
    m2 = np.mean(list(s2["paired_median_G_global"].values()) or [0])
    steps.append(f"4.median global 非劣: {m1:.4f} vs {m2:.4f}")
    ok1, ok2 = m1 >= -0.005, m2 >= -0.005
    h1 = np.mean(list(s1["paired_median_G_HF"].values()) or [0])
    h2 = np.mean(list(s2["paired_median_G_HF"].values()) or [0])
    steps.append(f"5.median HF: {h1:.4f} vs {h2:.4f}")
    if ok1 != ok2:
        winner = b1 if ok1 else b2
        steps.append(f"  -> 仅一方 global 非劣, 决出 β={winner}")
    elif abs(h1 - h2) > 0.005 and ok1 and ok2:
        winner = b1 if h1 > h2 else b2
        steps.append(f"  -> 决出 β={winner}")
    else:
        winner = min(float(b1), float(b2))
        winner = b1 if float(b1) == winner else b2
        steps.append(f"6.难以区分 -> 较小 β={winner}")

out = dict(
    semantics=dict(
        schema="global_beta_calibration_v1",
        domain=manifest["domain_label"],
        denominator=f"processing OK 的 {len(D)}/50 资产, 全部 β 相同口径; "
                    "NEUTRAL=VALID_NO_OP 计 0 增益保留在 denominator",
        split="calibration set(选 β 用), 非最终验证集; holdout>=20 未动",
        frozen="PartUV/signal/xatlas/padding/baker/gate 零修改"),
    processing_yield=yield_cnt, denominator=len(D),
    per_beta=summary, decision_steps=steps,
    final_beta_candidate=winner,
    note="仅本轮决策产物; 冻结动作与 holdout 由用户确认后执行")
with open(f"{OUTD}/calibration_summary.json", "w") as fp:
    json.dump(out, fp, indent=1, ensure_ascii=False)
print("\n======== Calibration 汇总 ========")
for b in BETAS:
    s = summary[b]
    print(f"β={b}: {s['counts']} pos_rate={s['positive_rate']} "
          f"harm={s['harm_rate']} medG={s['paired_median_G_global']} "
          f"medHF={s['paired_median_G_HF']} ssim_ok={s['ssim_not_worse_rate']}")
for st in steps:
    print(st)
print("final_beta_candidate:", winner)
print("CALIBRATION_V1: DONE")
