# -*- coding: utf-8 -*-
"""TD 标签测试: 常数纹理→近均匀; 高频 chart→更大 target area; 合法性不变量。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))
from meshuv.density.signal import face_content_score  # noqa: E402
from meshuv.density.allocation import chart_targets  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def make_cs(nF=120, nC=4):
    f2c = np.arange(nF) % nC
    return dict(face_to_chart=f2c, face_area=np.ones(nF, np.float32),
                covered=np.ones(nF, bool), source_uv_valid=np.ones(nF, bool),
                n_charts=nC)


# 常数纹理 -> 全面 std=0 -> q 均一 -> target 接近面积份额(均匀)
cs = make_cs()
lab = chart_targets(cs, np.zeros(120, np.float32))
check("常数纹理 -> TD 接近均匀",
      np.abs(lab["chart_target_area_fraction"] - 0.25).max() < 1e-6
      and np.abs(lab["chart_log_density_ratio"]).max() < 1e-6)

# chart0 高频(高 std) -> 更大 target area
score = np.zeros(120, np.float32)
score[cs["face_to_chart"] == 0] = 0.3
lab2 = chart_targets(cs, score)
t = lab2["chart_target_area_fraction"]
check("高频 chart 获得更大 target area", t[0] > t[1] + 0.02,
      f"t0={t[0]:.3f} t1={t[1]:.3f}")
check("不变量: sum=1/非负/有限/均值居中",
      abs(t.sum() - 1) < 1e-5 and (t >= 0).all()
      and abs(lab2["chart_log_density_ratio"][lab2["chart_valid_mask"]].mean()) < 1e-5)

# signal: 棋盘 vs 常数
atlas = np.zeros((64, 64, 3))
atlas[::2, ::2] = 1
suv = np.zeros((10, 3, 2), np.float32)
suv[:, :, 0] = np.random.RandomState(1).rand(10, 3)
suv[:, :, 1] = np.random.RandomState(2).rand(10, 3)
s_hi = face_content_score(atlas, suv, np.ones(10, bool))
s_lo = face_content_score(np.full((64, 64, 3), 0.5), suv, np.ones(10, bool))
check("signal: 棋盘 std > 常数 std", s_hi.mean() > 0.1 and s_lo.max() < 1e-9)

n_fail = RESULTS.count(False)
print(f"==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
