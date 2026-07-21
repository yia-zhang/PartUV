# -*- coding: utf-8 -*-
"""结果 gate (任务书 §3.2 / P0.7): 机器判定, 不靠人工看图."""


def evaluate_gates(m):
    """m: 测量字典. 返回 gate dict.
    需要键: coverage, n_quarantined, overlap_texels, nan_count, zero_uv_count,
            budget_gap_frac, budget_policy_ok, beta0_pass, matcher_unique,
            e_chart_L1, e_chart_L2, quality_evaluated(bool), quality_pass(bool|None)
    """
    validity = (
        (m["coverage"] >= 0.9999 or m["n_quarantined_reported"])
        and m["overlap_texels"] == 0
        and m["nan_count"] == 0
        and m["zero_uv_count"] == 0
        and m["budget_policy_ok"]
        and m["beta0_pass"]
        and m["matcher_unique"]
    )
    mechanism = ("NOT_EVALUATED" if not validity
                 else ("PASS" if m["e_chart_L2"] < m["e_chart_L1"] else "FAIL"))
    if not m.get("quality_evaluated", False):
        quality = "NOT_EVALUATED"
    else:
        quality = "PASS" if m.get("quality_pass") else "FAIL"

    if not validity:
        final = "INVALID"
    elif quality == "PASS":
        final = "QUALITY_IMPROVED"
    elif mechanism == "PASS":
        final = "TARGET_MATCH_IMPROVED"
    else:
        final = "VALID_BUT_NOT_IMPROVED"
    return dict(VALIDITY_GATE="PASS" if validity else "FAIL",
                MECHANISM_GATE=mechanism,
                QUALITY_GATE=quality,
                FINAL_STATUS=final)
