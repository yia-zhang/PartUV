# -*- coding: utf-8 -*-
"""notebook smoke: 顺序执行三本 notebook 的全部 code cell API(合成/真实小数据).
真实数据不存在时跳过对应 notebook 并 FAIL 提示(不能静默)。"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "notebooks")
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def run_nb(name, skip_if=None):
    p = f"{NB_DIR}/{name}"
    cells = [c for c in json.load(open(p))["cells"]
             if c["cell_type"] == "code"]
    g = {"__name__": "__main__"}
    os.chdir(NB_DIR)
    for i, c in enumerate(cells):
        src = c["source"]
        src = "".join(src) if isinstance(src, list) else src
        src = src.replace("%matplotlib inline", "")
        if skip_if and skip_if(src):
            return "SKIP"
        exec(compile(src, f"{name}#cell{i}", "exec"), g)
    return "OK"


ds_root = os.path.join(os.path.dirname(NB_DIR), "datasets/processed/clean_v1")
have_data = os.path.isdir(f"{ds_root}/objects")
have_ckpt = os.path.exists(os.path.join(os.path.dirname(NB_DIR),
                                        "reports/student_v0_sanity.ckpt"))
try:
    r = run_nb("01_data_browser.ipynb") if have_data else "NO_DATA"
    check("nb01 全 cell 执行", r == "OK", r)
except Exception as e:
    check("nb01 全 cell 执行", False, f"{type(e).__name__}: {str(e)[:90]}")
try:
    r = run_nb("02_uv_comparison.ipynb") if have_data else "NO_DATA"
    check("nb02 全 cell 执行", r == "OK", r)
except Exception as e:
    check("nb02 全 cell 执行", False, f"{type(e).__name__}: {str(e)[:90]}")
try:
    r = (run_nb("03_sanity_checkpoint_browser.ipynb")
         if (have_data and have_ckpt) else "MISSING_CKPT_OR_DATA")
    check("nb03 全 cell 执行(缺 ckpt 即 FAIL)", r == "OK", r)
except Exception as e:
    check("nb03 全 cell 执行(缺 ckpt 即 FAIL)", False,
          f"{type(e).__name__}: {str(e)[:90]}")

n_fail = RESULTS.count(False)
print(f"==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
