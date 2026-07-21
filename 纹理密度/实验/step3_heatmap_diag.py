# STEP3 热力图诊断: cw 分布 + log 归一化被零值破坏的验证 (2026-07-15)
# 运行: geomae env, cwd 任意
import os, sys

sys.path.insert(0, "/root/youjiaZhang/PartUV/code")
sys.path.insert(0, "/root/youjiaZhang/PartUV/code/scripts")

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tdlib.api import map_partuv_td
from tdlib.signal import demand_weights
from gen_dashboard_assets import render_img

OUT = os.path.dirname(os.path.abspath(__file__))
res = map_partuv_td("/root/youjiaZhang/PartUV/code/data/objaverse_22b822c6520d4d49.glb",
                    os.path.join(OUT, "step3_diag_out"),
                    atlas_size="auto", beta=0.75, max_atlas=8192)

cw, sel = np.asarray(res["cw"]), np.asarray(res["sel"])
pu = res["pu"]; V, F = pu["V"], pu["F"]
area = np.asarray(pu["area"])

c = cw[sel]
print(f"sel 面数: {sel.sum()} / {len(cw)}")
print(f"cw==0 精确零值占比: {(c == 0).mean()*100:.1f}%")
print(f"cw<=1e-6 占比:      {(c <= 1e-6).mean()*100:.1f}%")
qs = np.quantile(c, [0.05, 0.25, 0.5, 0.75, 0.99])
print("cw 分位 P5/P25/P50/P75/P99:", np.array2string(qs, precision=5))

# 复现 notebook 的归一化
z = np.log(np.maximum(cw, 1e-6))
lo, hi = np.quantile(z[sel], [0.05, 0.99])
print(f"notebook 归一窗口: lo={lo:.2f}, hi={hi:.2f}  (log 1e-6 = {np.log(1e-6):.2f})")
t_cur = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)
print(f"归一后 t>0.5 的 sel 面占比: {(t_cur[sel] > 0.5).mean()*100:.1f}% (橙色主导的原因)")

# 对 STEP4 分配的影响: q 的分布
q, w = demand_weights(cw, sel, area, beta=0.75)
print(f"q 分位 P5/P50/P95: {np.quantile(q[sel], [0.05, 0.5, 0.95])}")
print(f"q 撞下限0.5占比: {(np.isclose(q[sel], 0.5)).mean()*100:.1f}% | "
      f"撞上限2.83占比: {(np.isclose(q[sel], 2.83, atol=0.01)).mean()*100:.1f}%")

# 修正版归一: 分位数只在"有意义"的面上取(cw>1e-6, 排掉精确零+浮点碎屑),
# 零值/碎屑自然落到窗口下界之外 -> 贴到最暗端, 不再撑爆窗口
znz = z[sel & (cw > 1e-6)]
lo2, hi2 = np.quantile(znz, [0.05, 0.99])
t_fix = np.clip((z - lo2) / max(hi2 - lo2, 1e-9), 0, 1)
print(f"修正窗口: lo={lo2:.2f}, hi={hi2:.2f}; 修正后 t>0.5 占比: {(t_fix[sel] > 0.5).mean()*100:.1f}%")

np.savez(os.path.join(OUT, "step3_diag_cache.npz"), cw=cw, sel=sel, area=area, V=V, F=F)

heat_cur = plt.cm.magma(t_cur)[:, :3]; heat_cur[~sel] = 0.5
heat_fix = plt.cm.magma(t_fix)[:, :3]; heat_fix[~sel] = 0.5
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for ax, h, t in zip(axs, [heat_cur, heat_fix],
                    ["当前: P5 被 cw=0 拉爆 (椒盐+橙底)", "修正: 分位仅取 cw>0 (平坦=暗, 细节=亮)"]):
    ax.imshow(render_img(V, F, h)); ax.set_axis_off(); ax.set_title(t, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "step3_heatmap_diag.png"), dpi=110, bbox_inches="tight")
print("图已存:", os.path.join(OUT, "step3_heatmap_diag.png"))
