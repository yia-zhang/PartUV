# -*- coding: utf-8 -*-
"""Calibration V1 最终分层汇报(权威口径, 驱动 pooled 汇总仅存档参考).

口径:
- random-30 = 全局 β 主校准(六步决策链只在其 common-support 上运行);
- challenge-20 = 独立压力测试与类别性风险检查(不参与平局裁决;
  某候选在同一 subtype 出现 >=2 个 NEGATIVE -> 显式风险标记);
- all-50 = 仅描述性汇总。
packing 失败按 asset×β 粒度细分: ALL_BETA_FAILED / BETA_SPECIFIC_FAILED,
subtype: BUDGET_INFEASIBLE / NEEDS_MULTI_ATLAS / PACKER_BACKEND_ERROR /
PROBE_SEARCH_ERROR。全 β 失败对象计入 pipeline yield 不进 β 比较;
部分 β 失败计入对应 β 风险。失败资产未替换。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

CODE = "/root/youjiaZhang/PartUV/code"
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
BETAS = ["0.125", "0.25"]
TIER_NAMES = ["50pct", "25pct"]

mpath = f"{OUTD}/calibration_manifest_v2.json"
if not os.path.exists(mpath):
    mpath = f"{OUTD}/calibration_manifest.json"
man = json.load(open(mpath))


def load_result(oid):
    for f in (f"{OUTD}/{oid}/result.json", f"{OUTD}/{oid}/result_driver.json"):
        if os.path.exists(f):
            return json.load(open(f))
    return dict(object_id=oid, processing_status="MISSING", betas={})


def pack_subtype(msg):
    if "无法装入单一" in msg:
        return "NEEDS_MULTI_ATLAS"
    if "超尺寸" in msg or "hull" in msg or "直径" in msg:
        return "BUDGET_INFEASIBLE"
    return "PACKER_BACKEND_ERROR"


rows = []
for a in sorted(man["assets"], key=lambda x: x.get("selection_rank", 0)):
    r = load_result(a["object_id"])
    r["group"] = a["group"]
    r["stratum"] = "random" if a["group"] == "random" else "challenge"
    r["selection_rank"] = a.get("selection_rank")
    # asset×β packing 失败细分
    st, reason = r["processing_status"], r.get("reason", "")
    if st == "PACKING_FAILED" or (st == "ERROR" and "PackingFailedError" in reason):
        r["processing_status"] = "PACKING_FAILED"
        r["packing_failure"] = dict(scope="ALL_BETA_FAILED",
                                    subtype=pack_subtype(reason))
    beta_fail = {}
    for b in BETAS:
        row = r.get("betas", {}).get(b, {})
        errs = [(t, v["error"], v.get("bsignal_match"))
                for t, v in row.get("tiers", {}).items() if "error" in v]
        if errs and row.get("label") == "NOT_EVALUATED":
            sub = ("BUDGET_INFEASIBLE" if any("bsignal_dev" in e for _, e, _ in errs)
                   else "PROBE_SEARCH_ERROR")
            beta_fail[b] = dict(scope="BETA_SPECIFIC_FAILED", subtype=sub,
                                tiers=[t for t, _, _ in errs])
    if beta_fail:
        r["beta_specific_failures"] = beta_fail
    rows.append(r)


def stat_block(sub, denom_note):
    """在给定资产子集上做同口径 per-β 统计(denominator=processing OK)."""
    D = [r for r in sub if r["processing_status"] == "OK"]
    out = dict(n_assets=len(sub), denominator=len(D), note=denom_note,
               yield_={}, per_beta={})
    for r in sub:
        out["yield_"][r["processing_status"]] = \
            out["yield_"].get(r["processing_status"], 0) + 1
    # common-support: 两个 β 都可评(label != NOT_EVALUATED)
    CS = [r for r in D if all(
        r["betas"].get(b, {}).get("label", "NOT_EVALUATED") != "NOT_EVALUATED"
        for b in BETAS)]
    out["common_support"] = len(CS)
    out["all_beta_failed"] = sum(1 for r in sub
                                 if r["processing_status"] == "PACKING_FAILED")
    for b in BETAS:
        labs = [r["betas"][b]["label"] for r in CS]
        cnt = {s: labs.count(s) for s in
               ("POSITIVE", "NEUTRAL", "MIXED", "NEGATIVE")}
        n_bsf = sum(1 for r in D if b in r.get("beta_specific_failures", {}))
        gains_g, gains_h, harmed, ssim_ok = {t: [] for t in TIER_NAMES}, \
            {t: [] for t in TIER_NAMES}, [], 0
        for r in CS:
            row = r["betas"][b]
            noop = row.get("valid_no_op") or row["label"] == "NEUTRAL"
            for t in TIER_NAMES:
                v = row.get("tiers", {}).get(t, {})
                gains_g[t].append(0.0 if noop else v.get("G_global_eq", np.nan))
                gains_h[t].append(0.0 if noop else v.get("G_HF_eq", np.nan))
            gg = [v.get("G_global_eq", 0) for v in row.get("tiers", {}).values()]
            if not noop and any(g < -0.02 for g in gg):
                harmed.append(r["object_id"])
            if noop:
                ssim_ok += 1
            else:
                tt = [v for v in row.get("tiers", {}).values()
                      if "ssim_not_worse" in v]
                ssim_ok += int(bool(tt) and all(v["ssim_not_worse"] for v in tt))
        med = lambda d: {t: round(float(np.nanmedian(v)), 4)
                         for t, v in d.items() if len(v)}
        out["per_beta"][b] = dict(
            counts=cnt,
            success_rate=round((len(CS)) / max(len(D), 1), 3),
            beta_specific_failures=n_bsf,
            positive_rate=round(cnt["POSITIVE"] / max(len(CS), 1), 3),
            harm_rate=round(len(harmed) / max(len(CS), 1), 3), harmed=harmed,
            paired_median_G_global=med(gains_g),
            paired_median_G_HF=med(gains_h),
            ssim_not_worse_rate=round(ssim_ok / max(len(CS), 1), 3))
    return out, CS


rand = [r for r in rows if r["stratum"] == "random"]
chal = [r for r in rows if r["stratum"] == "challenge"]
blk_r, CS_r = stat_block(rand, "random-30: 全局 β 主校准")
blk_c, _ = stat_block(chal, "challenge-20: 独立压力测试, 不参与平局裁决")
blk_all, _ = stat_block(rows, "all-50: 仅描述性")

# ---- 决策链(仅 random-30 common-support) ----
b1, b2 = BETAS
s1, s2 = blk_r["per_beta"][b1], blk_r["per_beta"][b2]
steps, winner = [f"决策集 = random-30 common-support(n={blk_r['common_support']})"], None
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
    ok1, ok2 = m1 >= -0.005, m2 >= -0.005
    h1 = np.mean(list(s1["paired_median_G_HF"].values()) or [0])
    h2 = np.mean(list(s2["paired_median_G_HF"].values()) or [0])
    steps.append(f"4.median global 非劣: {m1:.4f} vs {m2:.4f}")
    steps.append(f"5.median HF: {h1:.4f} vs {h2:.4f}")
    if ok1 != ok2:
        winner = b1 if ok1 else b2
        steps.append(f"  -> 仅一方 global 非劣, 决出 β={winner}")
    elif abs(h1 - h2) > 0.005 and ok1 and ok2:
        winner = b1 if h1 > h2 else b2
        steps.append(f"  -> 决出 β={winner}")
    else:
        winner = b1 if float(b1) < float(b2) else b2
        steps.append(f"6.难以区分 -> 较小 β={winner}(challenge 均值不参与裁决)")

# ---- challenge 系统性风险(同 subtype >=2 NEGATIVE) ----
risks = {}
for b in BETAS:
    neg = {}
    for r in chal:
        if r["processing_status"] == "OK" \
                and r["betas"].get(b, {}).get("label") == "NEGATIVE":
            neg.setdefault(r["group"], []).append(r["object_id"])
    risks[b] = {g: v for g, v in neg.items() if len(v) >= 2}

report = dict(
    semantics=dict(
        schema="calibration_v1_stratified_report",
        domain=man.get("domain_label", ""),
        rule="random-30 主决策; challenge-20 压力测试; all-50 描述性; "
             "全 β 失败对象只计 pipeline yield; 部分 β 失败计入对应 β 风险; "
             "失败资产未替换"),
    random_30=blk_r, challenge_20=blk_c, all_50_descriptive=blk_all,
    packing_failures={r["object_id"]: r["packing_failure"] for r in rows
                      if "packing_failure" in r},
    beta_specific_failures={r["object_id"]: r["beta_specific_failures"]
                            for r in rows if "beta_specific_failures" in r},
    challenge_systematic_negative_risk=risks,
    decision_steps=steps, final_beta_candidate=winner,
    objects=[{k: r.get(k) for k in
              ("object_id", "selection_rank", "stratum", "group",
               "processing_status", "reason", "n_charts", "n_faces")}
             | {"labels": {b: r.get("betas", {}).get(b, {}).get("label", "-")
                           for b in BETAS}} for r in rows])
with open(f"{OUTD}/calibration_report_stratified.json", "w") as fp:
    json.dump(report, fp, indent=1, ensure_ascii=False)

print("=" * 30, "random-30(主校准)", "=" * 30)
print("yield:", blk_r["yield_"], f"| denominator={blk_r['denominator']} "
      f"common_support={blk_r['common_support']}")
for b in BETAS:
    s = blk_r["per_beta"][b]
    print(f"  β={b}: {s['counts']} pos={s['positive_rate']} harm={s['harm_rate']} "
          f"medG={s['paired_median_G_global']} medHF={s['paired_median_G_HF']} "
          f"ssim_ok={s['ssim_not_worse_rate']} β特定失败={s['beta_specific_failures']}")
print("=" * 30, "challenge-20(压力测试)", "=" * 28)
print("yield:", blk_c["yield_"], f"| common_support={blk_c['common_support']}")
for b in BETAS:
    s = blk_c["per_beta"][b]
    print(f"  β={b}: {s['counts']} harm={s['harm_rate']} harmed={s['harmed']}")
print("challenge 系统性 NEGATIVE 风险:", json.dumps(risks, ensure_ascii=False))
print("packing 失败细分:", json.dumps(report["packing_failures"],
                                   ensure_ascii=False))
for st in steps:
    print(st)
print("final_beta_candidate:", winner)
print("REPORT_V2: DONE")
