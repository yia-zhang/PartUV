# -*- coding: utf-8 -*-
"""P0 自动化测试 (CPU, 秒级). 运行: python tests/run_tests.py  全部 PASS 才算过."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tdlib.budget import (ALLOWED_SPECS, budget_accounting, choose_multi,
                          choose_resolution, rasterize_masks)
from tdlib.geometry import FaceMatcher, best_corner_perm
from tdlib.metrics import chart_mean_w, e_metrics
from tdlib.signal import demand_weights

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---------- P0.1: 预算吸附政策 (任务书反例 R_exact=1081) ----------
print("== test_budget_snap")
target = 1081 * 1081                                     # 1,168,561
spec, actual, gap = choose_resolution(target, "preserve_at_least")
check("preserve_at_least 不向下吸附", actual >= target, f"spec={spec} gap={gap:+.1%}")
spec2, actual2, gap2 = choose_resolution(target, "hard_cap")
check("hard_cap 允许向下但必须报缺口", actual2 <= target and gap2 < 0,
      f"spec={spec2} gap={gap2:+.1%}")
check("hard_cap 缺口被显式量化(≈-10%)", abs(gap2 + 0.103) < 0.02, f"gap={gap2:+.3f}")

# ---------- P0.2: multi-atlas 联合预算 (任务书反例 target=2M) ----------
print("== test_budget_multi")
combo, actual, gap = choose_multi(2_000_000, k_max=3, policy="hard_cap")
check("multi hard_cap 不劣于旧组合(-21.4%)", gap > -0.214, f"combo={combo} gap={gap:+.1%}")
combo2, actual2, gap2 = choose_multi(2_000_000, k_max=3, policy="preserve_at_least")
check("multi preserve_at_least ≥ 目标", actual2 >= 2_000_000, f"combo={combo2} gap={gap2:+.1%}")
check("multi 联合误差 <= 5%", abs(gap2) <= 0.05 or abs(gap) <= 0.05,
      f"best |gap|={min(abs(gap), abs(gap2)):.1%}")

# ---------- P0.6: β=0 一致性 ----------
print("== test_beta0")
rng = np.random.RandomState(1)
cw = rng.rand(1000)
area = rng.rand(1000) + 0.1
sel = np.ones(1000, bool)
q, w = demand_weights(cw, sel, area, beta=0.0)
check("β=0 ⇒ q≡1 (精确)", np.all(q == 1.0))
check("β=0 ⇒ w≡1 (精确)", np.all(w == 1.0))
q2, w2 = demand_weights(cw, sel, area, beta=0.4)
check("β>0 预算归一 mean_A(q²)≈1", abs(np.average(q2 ** 2, weights=area) - 1) < 0.05,
      f"mean={np.average(q2**2, weights=area):.4f}")

# ---------- P0.5: FaceMatcher 孪生面唯一性 ----------
print("== test_matcher_twins")
tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
V = np.vstack([tri, tri + [3, 0, 0]])
F = np.array([[0, 1, 2], [3, 4, 5], [0, 2, 1], [3, 5, 4]])   # 每处一对法线相反的孪生
fm = FaceMatcher(V, F)
charts_VF = [(V, F[:2]), (V, F[2:])]                         # 正面 chart + 背面 chart
gidxs, rep = fm.match_charts(charts_VF)
check("孪生面全局一对一", rep["unique"], str(rep))
nrm_ok = all(g in ([0, 1] if i == 0 else [2, 3]) for i, gs in enumerate(gidxs) for g in gs)
check("法线判别选对孪生层", nrm_ok, f"gidxs={[list(g) for g in gidxs]}")

# ---------- P0.5: 角点 3! 双射 ----------
print("== test_corner_perm")
src = np.array([[0, 0, 0], [2, 0, 0], [0, 3, 0]], float)
for p_true in [(1, 2, 0), (2, 0, 1), (0, 2, 1)]:
    dst = src[list(p_true)]
    perm, cost = best_corner_perm(src, dst)
    check(f"恢复排列 {p_true}", src[list(perm)].tolist() == dst.tolist() and cost < 1e-12)

# ---------- P0.3: e_face/e_chart/e_irreducible 合成信号验证 ----------
print("== test_e_metrics_synthetic")
# 两个 chart: A 内部需求均匀(w=4), B 内部需求二元(w∈{1,9}, 均值5) —— 已知解析行为
nf = 400
area = np.ones(nf)
gA, gB = np.arange(0, 200), np.arange(200, 400)
w = np.ones(nf)
w[gA] = 4.0
w[gB] = np.where(np.arange(200) % 2 == 0, 1.0, 9.0)
charts = [dict(gidx=gA), dict(gidx=gB)]
wbar = chart_mean_w(charts, w, area, nf)
check("w̄_c 计算正确", abs(wbar[gA][0] - 4) < 1e-9 and abs(wbar[gB][0] - 5) < 1e-9)
# 完美 chart-level 分配: TD² = w̄_c ⇒ e_chart=0, e_face=within>0
td = np.sqrt(wbar)
m = np.ones(nf, bool)
em = e_metrics(td, w, wbar, area, m, charts=charts)
check("完美 chart 分配 ⇒ e_chart≈0", em["e_chart"] < 1e-9, f"{em['e_chart']:.2e}")
check("e_face ≈ within (相对交付目标)", abs(em["e_face"] - em["within_chart_heterogeneity"]) < 1e-9)
check("chartB 异质 ⇒ within>0", em["within_chart_heterogeneity"] > 0.5,
      f"{em['within_chart_heterogeneity']:.3f}")
check("分解恒等式 e_face²=e_chart²+within²+2·cross",
      abs(em["e_face"]**2 - (em["e_chart"]**2 + em["within_chart_heterogeneity"]**2
          + 2*em["cross_term"])) < 1e-9)
check("log 下界 ≤ within (几何均值最优)",
      em["e_irreducible_log"] <= em["within_chart_heterogeneity"] + 1e-12,
      f"log={em['e_irreducible_log']:.3f} within={em['within_chart_heterogeneity']:.3f}")
# uniform L1: TD²=1 ⇒ e_chart = RMS log w̄ > 0 (机制指标能区分 L1/L2)
em1 = e_metrics(np.ones(nf), w, wbar, area, m, charts=charts)
check("L1 的 e_chart > 完美分配的 e_chart", em1["e_chart"] > em["e_chart"] + 0.05,
      f"L1={em1['e_chart']:.3f} (chart 均值 4 vs 5, between 分量小属预期)")

# ---------- P0.1: rasterized mask 预算核算 ----------
print("== test_raster_budget")
sq = np.array([[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]])
chart = dict(F=np.array([[0, 1, 2], [0, 2, 3]]), gidx=np.array([0, 1]))
owner, overlap, per = rasterize_masks([chart], [sq], 128, 128)
acc = budget_accounting(owner, gutter_px=2)
exp = (0.3 * 128) ** 2
check("signal texels ≈ 解析面积", abs(acc["B_signal"] - exp) / exp < 0.1,
      f"{acc['B_signal']} vs {exp:.0f}")
check("无重叠", overlap == 0)
check("B_raw = signal+pad+empty", acc["B_raw"] == acc["B_signal"] + acc["B_pad"] + acc["B_empty"])

# ---------- 绝对预算: island mask 的 B_unique/B_surface/reuse_factor ----------
print("== test_island_budget")
from tdlib.api import _island_budget

# 两个完全不重叠 island(各为一个 [0.1,0.4]/[0.6,0.9] 方块, 两三角):
uv_a = np.array([[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4],
                 [0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]])
Fq = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
lbl = np.array([0, 0, 1, 1])
bu, bs = _island_budget(Fq, uv_a, 256, 256, lbl)
check("不重叠两岛: reuse_factor = 1", bu == bs and bu > 0,
      f"B_unique={bu} B_surface={bs} reuse={bs / max(bu, 1):.3f}")

# 两个完全重叠且面积相同的 island:
uv_b = np.array([[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4],
                 [0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]])
bu2, bs2 = _island_budget(Fq, uv_b, 256, 256, lbl)
check("完全重叠两岛: reuse_factor = 2", bs2 == 2 * bu2 and bu2 > 0,
      f"B_unique={bu2} B_surface={bs2} reuse={bs2 / max(bu2, 1):.3f}")
check("重叠岛 B_unique = 单岛面积", bu2 * 2 == bu,
      f"union(重叠)={bu2} vs union(分离)={bu}")

# ---------- xatlas 刚体合同: 超尺寸 chart 不得被静默逐 chart 缩小 ----------
print("== test_xatlas_rigid_contract")
from tdlib.layout import PackingFailedError, xatlas_pack

# 审计构造的逃逸用例: 45° 旋转长条(直径远超 atlas) + 3 个微 chart。
# 连续旋转的最小外接正方形会误判可行(xatlas 只有主轴对齐+90° 取向),
# 凸包直径预检必须拒绝所有 prefill -> PACKING_FAILED, 绝不静默缩小。
th = np.pi / 4
Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
long_uv = np.array([[0, 0], [1000, 0], [1000, 131], [0, 131]], float) @ Rm.T
charts_x = [dict(UV=long_uv, F=np.array([[0, 1, 2], [0, 2, 3]]), a2=131000.0)]
for k in range(3):
    charts_x.append(dict(UV=np.array([[0, 0], [5, 0], [5, 5], [0, 5]], float),
                         F=np.array([[0, 1, 2], [0, 2, 3]]), a2=25.0))
try:
    xatlas_pack(charts_x, np.ones(4), resolution=256, padding_px=4)
    check("超尺寸 chart 触发 PACKING_FAILED(非静默缩小)", False, "未抛出")
except PackingFailedError:
    check("超尺寸 chart 触发 PACKING_FAILED(非静默缩小)", True)

# 正常小场景: 刚体 + 面积比一致(相对期望值)
charts_ok = [dict(UV=np.array([[0, 0], [40, 0], [40, 20], [0, 20]], float),
                  F=np.array([[0, 1, 2], [0, 2, 3]]), a2=800.0),
             dict(UV=np.array([[0, 0], [20, 0], [20, 20], [0, 20]], float),
                  F=np.array([[0, 1, 2], [0, 2, 3]]), a2=400.0)]
uvs_ok = xatlas_pack(charts_ok, np.ones(2), resolution=256, padding_px=4)
a_out = [abs(np.cross(u[c["F"]][:, 1] - u[c["F"]][:, 0],
             u[c["F"]][:, 2] - u[c["F"]][:, 0])).sum() / 2
         for u, c in zip(uvs_ok, charts_ok)]
ratio2 = (a_out[0] / 800.0) / (a_out[1] / 400.0)
check("正常场景相对面积精确保持", abs(ratio2 - 1) < 0.02, f"比值比={ratio2:.4f}")

# ---- Baker Convergence Audit 回归: 双 chart 平面 + 跨 seam 线性渐变 ----
# 保护不变量: 覆盖感知降采样(bake_atlas_ss)下 seam-band/global MSE 随 SS 收敛,
# 不复发「未覆盖子纹素被当黑色平均 -> SS 越高 seam 越差」的评测 baker bug。
from tdlib.rd import bake_atlas_ss, _srgb2lin

_Rb, _G, _SC = 128, 8, 0.8
_xs = np.linspace(0, 1, _G + 1)
_V = np.array([[x, y, 0.0] for y in _xs for x in _xs])
_F = []
for i in range(_G):
    for j in range(_G):
        a, b = i * (_G + 1) + j, i * (_G + 1) + j + 1
        c, d = (i + 1) * (_G + 1) + j + 1, (i + 1) * (_G + 1) + j
        _F += [[a, b, c], [a, c, d]]
_F = np.array(_F)
_left = _V[_F][:, :, 0].mean(1) < 0.5


def _mk_chart(m):
    fids = np.where(m)[0]
    cor = _V[_F[fids]][:, :, :2].reshape(-1, 2)
    uq, inv = np.unique(cor, axis=0, return_inverse=True)
    return dict(UV=uq, F=inv.reshape(-1, 3), gidx=fids, a2=0.5)


_charts = [_mk_chart(_left), _mk_chart(~_left)]
_tris = _V[_F]
_fa = np.linalg.norm(np.cross(_tris[:, 1] - _tris[:, 0],
                              _tris[:, 2] - _tris[:, 0]), axis=1) / 2
_pu = dict(charts=_charts, F=_F, area=_fa)
_uvs = [np.stack([0.06 + c["UV"][:, 0] * _SC, 0.10 + c["UV"][:, 1] * _SC], 1)
        if ci == 0 else
        np.stack([0.54 + (c["UV"][:, 0] - 0.5) * _SC, 0.10 + c["UV"][:, 1] * _SC], 1)
        for ci, c in enumerate(_charts)]
_gx = (np.arange(256) + 0.5) / 256
_texA = np.repeat(np.stack([_gx, 1 - _gx, np.full_like(_gx, .5)], -1)[None], 256, 0)
_refuv = _V[_F][:, :, :2].copy()
_ii, _jj = np.meshgrid(np.arange(_Rb), np.arange(_Rb), indexing="ij")
_ax, _ay = (_jj + .5) / _Rb, 1 - (_ii + .5) / _Rb
_gt = np.full((_Rb, _Rb, 3), np.nan)
_spx = np.full((_Rb, _Rb), np.inf)
for _ci in (0, 1):
    _x0, _x1 = (0.06, 0.46) if _ci == 0 else (0.54, 0.94)
    _in = (_ax >= _x0) & (_ax <= _x1) & (_ay >= .10) & (_ay <= .90)
    _u = (_ax - 0.06) / _SC if _ci == 0 else 0.5 + (_ax - 0.54) / _SC
    _gt[_in] = _srgb2lin(np.stack([_u, 1 - _u, np.full_like(_u, .5)], -1))[_in]
    _spx[_in] = np.minimum(_spx[_in],
                           np.abs(_ax - (0.46 if _ci == 0 else 0.54))[_in] * _Rb)
_glob, _seam2 = [], []
for _ss in (1, 2, 4, 8):
    _t, _sig, _ = bake_atlas_ss(_pu, _uvs, _Rb, _ss, _refuv,
                                np.ones(len(_F), bool), _texA)
    _ok = _sig & np.isfinite(_gt[:, :, 0])
    _e = ((_srgb2lin(_t) - _gt) ** 2).mean(-1)
    _glob.append(float(_e[_ok].mean()))
    _seam2.append(float(_e[_ok & (_spx <= 2)].mean()))
_conv = lambda s: all(b <= a * 1.05 or (a < 1e-10 and b < 1e-10)
                      for a, b in zip(s, s[1:])) \
    and (s[-1] <= s[0] * 1.05 or s[-1] < 1e-10)
check("baker 收敛: seam-2px MSE 随 SS 非增(覆盖感知降采样)", _conv(_seam2),
      f"{['%.2e' % v for v in _seam2]}")
check("baker 收敛: global MSE 终点<=起点且 SS4->SS8 收敛",
      (_glob[-1] <= _glob[0] * 1.05 and _glob[-1] <= _glob[-2] * 1.05)
      or _glob[-1] < 1e-10,
      f"{['%.2e' % v for v in _glob]}")

# ---- Coordinate Rebaseline: 四类确定性测试(texel-center 唯一约定) ----
from tdlib.rd import bilinear, bake_atlas_masks

# T1: texel-center 精确采样 —— u=(i+0.5)/W 必须精确取到第 i 个纹素(多分辨率)
for (Wt, Ht) in [(7, 5), (64, 64), (256, 128)]:
    img = np.random.RandomState(Wt).rand(Ht, Wt, 3)
    jj, ii = np.meshgrid(np.arange(Wt), np.arange(Ht))
    uv = np.stack([(jj.ravel() + 0.5) / Wt, 1 - (ii.ravel() + 0.5) / Ht], 1)
    got = bilinear(img, uv).reshape(Ht, Wt, 3)
    e1 = np.abs(got - img).max()
    check(f"T1 texel-center 精确采样 {Wt}x{Ht}", e1 < 1e-12, f"maxerr={e1:.1e}")
# 边界语义: u=0/1 clamp-to-edge 取边缘纹素
img = np.random.RandomState(0).rand(8, 8, 3)
edge = bilinear(img, np.array([[0.0, 1 - 0.5 / 8], [1.0, 1 - 0.5 / 8]]))
check("T1 边界 clamp-to-edge(u=0/1)", np.abs(edge[0] - img[0, 0]).max() < 1e-12
      and np.abs(edge[1] - img[0, 7]).max() < 1e-12)


def _plane_identity(Rt):
    """单 chart 全平面, packed uv == source uv(identity, 同分辨率 rebake 用)."""
    G = 8
    gxs = np.linspace(0, 1, G + 1)
    Vp = np.array([[x, y, 0.0] for y in gxs for x in gxs])
    Fp = []
    for i in range(G):
        for j in range(G):
            a, b = i * (G + 1) + j, i * (G + 1) + j + 1
            c, d = (i + 1) * (G + 1) + j + 1, (i + 1) * (G + 1) + j
            Fp += [[a, b, c], [a, c, d]]
    Fp = np.array(Fp)
    ch = dict(UV=Vp[:, :2].copy(), F=Fp.copy(), gidx=np.arange(len(Fp)), a2=1.0)
    tr = Vp[Fp]
    fa = np.linalg.norm(np.cross(tr[:, 1] - tr[:, 0], tr[:, 2] - tr[:, 0]), 1) / 2
    pu = dict(charts=[ch], F=Fp, area=fa)
    return pu, [Vp[:, :2].copy()], Vp[Fp][:, :, :2].copy(), np.ones(len(Fp), bool)


# T2: identity UV 同分辨率 rebake —— production(bake_atlas_masks@R) 与
#     evaluation(bake_atlas_ss ss=1) 都必须逐纹素还原源纹理
for Rt in (64, 128):
    pu_i, uvs_i, refuv_i, val_i = _plane_identity(Rt)
    texR = np.random.RandomState(Rt).rand(Rt, Rt, 3)
    tp, sp, _ = bake_atlas_masks(pu_i, uvs_i, Rt, refuv_i, val_i, texR)
    e_prod = np.abs(tp[sp] - texR[sp]).max()
    te, se, _ = bake_atlas_ss(pu_i, uvs_i, Rt, 1, refuv_i, val_i, texR)
    e_eval = np.abs(te[se] - texR[se]).max()
    check(f"T2 identity 同分辨率 rebake R={Rt}(production)", e_prod < 1e-9,
          f"maxerr={e_prod:.1e} 覆盖={sp.mean() * 100:.0f}%")
    check(f"T2 identity 同分辨率 rebake R={Rt}(evaluation ss=1 一致)",
          e_eval < 1e-9 and (se == sp).all(), f"maxerr={e_eval:.1e}")

# T3: 连续线性渐变 —— 上方 baker 收敛回归即 T3(bands+SS 收敛);
#     此处补 production SS1 interior 精确性(texel-center 下 bilinear-of-linear 精确)
check("T3 线性渐变 production(SS1) global≈0(约定统一后)", _glob[0] < 1e-9,
      f"global={_glob[0]:.2e} (修复前~1.6e-06)")

# T4: checkerboard + 单纹素 impulse —— 同分辨率 identity rebake 必须精确保持;
#     evaluation 超采样路径 impulse 位置不漂移且 SS4->SS8 收敛
for Rt in (64, 128):
    pu_i, uvs_i, refuv_i, val_i = _plane_identity(Rt)
    texC = ((np.add.outer(np.arange(Rt) // 8, np.arange(Rt) // 8)) % 2
            ).astype(float)[..., None].repeat(3, -1)
    texC[int(Rt * 0.3), int(Rt * 0.4)] = [1, 0, 0]      # 单纹素 impulse
    tp, sp, _ = bake_atlas_masks(pu_i, uvs_i, Rt, refuv_i, val_i, texC)
    e4 = np.abs(tp[sp] - texC[sp]).max()
    check(f"T4 checker+impulse 同分辨率精确(production) R={Rt}", e4 < 1e-9,
          f"maxerr={e4:.1e}")
    t4, s4m, _ = bake_atlas_ss(pu_i, uvs_i, Rt, 4, refuv_i, val_i, texC)
    t8, _, _ = bake_atlas_ss(pu_i, uvs_i, Rt, 8, refuv_i, val_i, texC)
    r4 = t4[:, :, 0] - t4[:, :, 1]                       # impulse 红色残差图
    pos = np.unravel_index(np.argmax(r4), r4.shape)
    conv = np.abs(t8[s4m] - t4[s4m]).mean()
    check(f"T4 impulse 位置不漂移+SS4->SS8 收敛 R={Rt}",
          pos == (int(Rt * 0.3), int(Rt * 0.4)) and conv < 0.02,
          f"pos={pos} conv={conv:.4f}")

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
