# -*- coding: utf-8 -*-
"""Production-Protocol β Preflight —— 正式生产协议下是否存在可冻结的共享 β.

冻结: 每资产全部 β 复用同一 cached charts/chart hash/local UV/reference/
surface samples(seed=2)/production xatlas/padding=4/texel-center baker/
luminance-std 信号与 demand 公式(逐字调用既有函数, 零修改)。
协议: 仅 allocation 轴(fixed-B_signal), 25%/50% 两档, B_signal 偏差<=1%;
B_raw/fill 只记录不判断; 统计单位=object。
候选: β ∈ {0, 0.125, 0.25, 0.5, 0.75, 1.0}(β=0 为 Uniform baseline)。
12 个有 allocation 指标的 development assets 正式汇总; 鞋/车轮(p1b 缓存分解)
为 sanity cases 单独展示不计入。
提名顺序: harm rate 低 -> median global gain 非劣(>=0) -> median HF gain 高 ->
更小 β(minimum sufficient strength)。全部非零 β 不稳定优于 β=0 =>
NO_SHARED_BETA_FOUND(不自动进 Signal V2, 不做 adaptive β)。
本轮只提名不冻结。
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.layout import PackingFailedError
from tdlib.pipeline import load_reference
from tdlib.rd import prepare_face_ref_uv
from tdlib.signal import demand_weights, luminance_std_heuristic
from diag_common import load_sample, eval_samples, pack_only, bake_layout, surface_err
from run_pseudo_gt_quality_gate import masked_ssim, SSIM_VIEWS

OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/pilot_v1"
OUTB = "/root/youjiaZhang/PartUV/code/notebook/outputs/beta_preflight"
os.makedirs(OUTB, exist_ok=True)

BETAS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
TIERS = [("50pct", 0.50), ("25pct", 0.25)]
LOW_SIGNAL_DIST = 0.05
BAND_G, POS_HF, NEG_G = 0.02, 0.05, -0.05
SSIM_TOL = -0.002

ASSETS_12 = ["synth_flat_solid", "synth_gradient", "objav_7f9212",
             "sample_WaterBottle", "synth_multiscale_noise",
             "sample_BarramundiFish", "sample_Avocado", "synth_many_islands",
             "synth_two_materials", "synth_trimsheet_reuse6", "sample_BoomBox",
             "synth_open_terrain"]
SANITY = [("shoe_22b822", "objaverse_22b822c6520d4d49.glb"),
          ("wheel_92ff6", "objaverse_92ff65712c62408d.glb")]
DATA = "/root/youjiaZhang/PartUV/code/data"
P1B = "/root/youjiaZhang/PartUV/code/notebook/outputs/p1b"


def face_cw_from_ctx(ctx):
    """冻结信号: 用样本内 face_refuv 无损重建 luminance_std_heuristic 输入
    (OUV3 = uv0[Fo[f2o]] 恒等于 face_refuv, 数学逐字相同)."""
    nF = len(ctx["F"])
    uv0 = ctx["face_refuv"].reshape(-1, 2)
    Fo = np.arange(nF * 3).reshape(nF, 3)
    f2o = np.arange(nF)
    ok = ctx["valid"].astype(bool)
    return luminance_std_heuristic(ctx["texA"], uv0, Fo, f2o, ok)


def scales_for_beta(ctx, cw, sel, beta):
    """冻结 demand 公式: demand_weights -> f_c = sqrt(Σ a3·w / a2)."""
    _, w = demand_weights(cw, sel, ctx["fa3"], beta=beta)
    dem = np.array([float((ctx["fa3"][c["gidx"]] * w[c["gidx"]]).sum())
                    for c in ctx["charts"]])
    scales = np.sqrt(dem / np.maximum(ctx["a2"], 1e-12))
    return scales, dem / max(dem.sum(), 1e-20)


def render_set(ctx, nuv, ok, tex):
    return [tdgpu.textured_render(ctx["V"], ctx["F"], nuv, ok, tex, view=v)
            for v in SSIM_VIEWS]


def eval_ctx(oid, ctx, betas=BETAS):
    """一个资产上全部 β 的 allocation 轴指标(两档 fixed-B_signal)."""
    ev = eval_samples(ctx)
    cw = face_cw_from_ctx(ctx)
    sel = ctx.get("sel", ctx["valid"].astype(bool))
    okm = np.ones(len(ctx["F"]), bool)
    refs = [tdgpu.textured_render(ctx["V"], ctx["F"], ctx["face_refuv"],
                                  ctx["valid"], ctx["texA"], view=v)
            for v in SSIM_VIEWS]
    out = dict(object_id=oid, betas={})
    uni = {}
    for tname, frac in TIERS:
        R = max(int(round(np.sqrt(frac * ctx["B_source"]))), 64)
        p = pack_only(ctx, ctx["scales_uni"], R)
        tex, nuv = bake_layout(ctx, p["uvs"], R)
        ss = [masked_ssim(a, b) for a, b in zip(refs, render_set(ctx, nuv, okm, tex))]
        uni[tname] = dict(R=R, S=p["b_signal"], fill=p["fill"],
                          d=surface_err(tex, nuv, ev), ssim=ss)
    # β=0.75 与 exporter 标签的一致性自检
    s075, _ = scales_for_beta(ctx, cw, sel, 0.75)
    if "label_scale" in ctx:
        rel = float(np.abs(s075 / ctx["label_scale"] - 1).max())
        out["scale_check_beta075_max_rel_diff"] = rel
    for beta in betas:
        row = dict(beta=beta, tiers={}, label="NEUTRAL", signal_dist=0.0)
        if beta == 0.0:
            row["label"] = "BASELINE(=Uniform)"
            out["betas"]["0"] = row
            continue
        scales_b, dem_norm = scales_for_beta(ctx, cw, sel, beta)
        sd = float(0.5 * np.abs(dem_norm - ctx["area_norm"]).sum())
        row["signal_dist"] = round(sd, 4)
        if sd < LOW_SIGNAL_DIST:
            row["label"] = "NEUTRAL"
            out["betas"][str(beta)] = row
            continue
        pos_any = neg_any = False
        for tname, frac in TIERS:
            u = uni[tname]
            lo, hi, best = int(u["R"] * 0.6), int(u["R"] * 1.8), None
            for _ in range(9):
                mid = (lo + hi) // 2
                try:
                    pt = pack_only(ctx, scales_b, mid)
                except PackingFailedError:
                    lo = mid + 1
                    continue
                if best is None or abs(pt["b_signal"] - u["S"]) < abs(
                        best[1]["b_signal"] - u["S"]):
                    best = (mid, pt)
                if pt["b_signal"] < u["S"]:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best is None:
                row["tiers"][tname] = dict(error="packing failed 全部候选")
                neg_any = True
                continue
            R_td, pt = best
            match = pt["b_signal"] / max(u["S"], 1)
            tex, nuv = bake_layout(ctx, pt["uvs"], R_td)
            d_t = surface_err(tex, nuv, ev)
            g = 1 - float(d_t.mean()) / max(float(u["d"].mean()), 1e-20)
            ghf = 1 - float(d_t[ev["hi"]].mean()) / max(
                float(u["d"][ev["hi"]].mean()), 1e-20)
            ss_t = [masked_ssim(a, b)
                    for a, b in zip(refs, render_set(ctx, nuv, okm, tex))]
            dss = float(np.mean([a - b for a, b in zip(ss_t, uni[tname]["ssim"])]))
            row["tiers"][tname] = dict(
                G_global_eq=round(g, 4), G_HF_eq=round(ghf, 4),
                bsignal_match=round(match, 4),
                bsignal_dev_ok=bool(abs(match - 1) <= 0.01),
                ssim_delta_mean=round(dss, 5),
                ssim_not_worse=bool(dss >= SSIM_TOL),
                B_raw_record=int(R_td * R_td), fill_record=round(pt["fill"], 4),
                overlap=pt["overlap"])
            pos_any |= g >= BAND_G or (ghf >= POS_HF and g >= -BAND_G)
            neg_any |= g <= NEG_G and ghf < POS_HF
        t_ok = [t for t in row["tiers"].values() if "G_global_eq" in t]
        in_band = all(abs(t["G_global_eq"]) <= BAND_G
                      and abs(t["G_HF_eq"]) < POS_HF for t in t_ok) if t_ok else False
        row["label"] = ("NEUTRAL" if in_band else
                        "POSITIVE" if pos_any and not neg_any else
                        "NEGATIVE" if neg_any and not pos_any else "MIXED")
        out["betas"][str(beta)] = row
        print(f"    β={beta}: {row['label']} sd={row['signal_dist']} "
              + " ".join(f"{t}:g={v.get('G_global_eq')},hf={v.get('G_HF_eq')}"
                         for t, v in row["tiers"].items()), flush=True)
    return out


def ctx_from_p1b_cache(name, glb):
    """sanity 资产: p1b 缓存 chart 分解 + 标准 reference 重载(非 exporter 样本)."""
    with open(f"{P1B}/{name}/charts_cache.pkl", "rb") as fp:
        pu = pickle.load(fp)
    ref = load_reference(f"{DATA}/{glb}", pu["V"], pu["F"], pu["mesh_scale"])
    face_refuv, valid, _ = prepare_face_ref_uv(pu, ref)
    charts = pu["charts"]
    F = pu["F"]
    tris = pu["V"][F]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    A3 = np.array([fa3[c["gidx"]].sum() for c in charts])
    a2 = np.array([float(c["a2"]) for c in charts])
    texA = ref["texA"]
    ctx = dict(charts=charts, F=F, V=pu["V"], fa3=fa3, A3=A3, a2=a2,
               face_refuv=face_refuv, valid=valid,
               scales_uni=np.sqrt(A3 / a2), area_norm=A3 / A3.sum(),
               pu_like=dict(charts=charts, F=F, area=fa3),
               ch_masks=[dict(F=np.asarray(c["F"]), gidx=c["gidx"])
                         for c in charts],
               B_source=texA.shape[0] * texA.shape[1], texA=texA,
               sel=valid & pu["covered"])
    return ctx


results, sanity_results = [], []
for oid in ASSETS_12:
    print(f"\n================ {oid} ================", flush=True)
    ctx = load_sample(f"{OUT}/{oid}/sample")
    ctx["label_scale"] = ctx["scales_td"]
    ctx["sel"] = ctx["z"]["train_face_mask"].astype(bool) & ctx["valid"].astype(bool)
    ctx["V"] = ctx["z"]["vertices"]
    results.append(eval_ctx(oid, ctx))
for name, glb in SANITY:
    print(f"\n================ [sanity] {name} ================", flush=True)
    sanity_results.append(eval_ctx(name, ctx_from_p1b_cache(name, glb)))

# ---- object-level 汇总(12 资产正式; NEUTRAL 单独, 不计成功/失败) ----
summary_rows = []
for beta in BETAS[1:]:
    key = str(beta)
    labs = [r["betas"][key]["label"] for r in results]
    pairs = [(r["object_id"], r["betas"][key]) for r in results
             if r["betas"][key]["tiers"]]
    evald = [e for _, e in pairs]
    med = lambda f: {t: round(float(np.median([e["tiers"][t][f] for e in evald
                                               if t in e["tiers"] and f in e["tiers"][t]])), 4)
                     for t, _ in TIERS} if evald else {}
    harmed = [o for o, e in pairs
              if any(t.get("G_global_eq", 0) < -BAND_G for t in e["tiers"].values())]
    ssim_ok = [o for o, e in pairs
               if all(t.get("ssim_not_worse", True) for t in e["tiers"].values())]
    summary_rows.append(dict(
        beta=beta,
        counts={s: labs.count(s) for s in
                ("POSITIVE", "NEUTRAL", "MIXED", "NEGATIVE")},
        n_evaluated=len(evald),
        median_G_global_eq=med("G_global_eq"),
        median_G_HF_eq=med("G_HF_eq"),
        global_harm_rate=round(len(harmed) / max(len(pairs), 1), 3),
        harmed_objects=harmed,
        ssim_not_worse_rate=round(len(ssim_ok) / max(len(pairs), 1), 3)))

# ---- 提名(不冻结): harm rate -> median global 非劣 -> median HF -> 小 β ----
def sort_key(r):
    mg = np.mean(list(r["median_G_global_eq"].values())) if r["median_G_global_eq"] else -9
    mh = np.mean(list(r["median_G_HF_eq"].values())) if r["median_G_HF_eq"] else -9
    return (r["global_harm_rate"], -mh, r["beta"]), mg


cand = [r for r in summary_rows if r["n_evaluated"] > 0]
cand = [r for r in cand
        if all(v >= 0 for v in r["median_G_global_eq"].values())]  # 非劣
nominated = None
if cand:
    cand.sort(key=lambda r: sort_key(r)[0])
    best = cand[0]
    # 稳定优于 β=0: 至少有正向证据且 harm 未过半
    med_hf = np.mean(list(best["median_G_HF_eq"].values()))
    if (best["counts"]["POSITIVE"] >= 1 and best["global_harm_rate"] < 0.5
            and med_hf > 0):
        nominated = best["beta"]
verdict = (f"NOMINATED_SHARED_BETA={nominated}(仅提名, 未冻结)"
           if nominated is not None else "NO_SHARED_BETA_FOUND")

report = dict(
    semantics=dict(
        protocol="allocation 轴 fixed-B_signal 25%/50% 两档, 偏差<=1%; "
                 "B_raw/fill 仅记录; 统计单位=object; NEUTRAL 单独不计成败",
        evaluator="texel_center_v1 + coverage_center_v1(修复后)",
        nomination_rule="harm rate -> median global 非劣 -> median HF -> 小 β",
        sanity="鞋/车轮为 p1b 缓存分解的 sanity cases, 不计入 12 资产汇总",
        frozen="PartUV/signal/demand 公式/xatlas/padding/质量门 零修改"),
    verdict=verdict, summary=summary_rows,
    objects=results, sanity=sanity_results)
with open(f"{OUTB}/beta_preflight.json", "w") as fp:
    json.dump(report, fp, indent=1, ensure_ascii=False)

# ---- 曲线图 ----
bs = [r["beta"] for r in summary_rows]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for t, _ in TIERS:
    axs[0].plot(bs, [r["median_G_global_eq"].get(t, np.nan) for r in summary_rows],
                "o-", label=f"global {t}")
    axs[0].plot(bs, [r["median_G_HF_eq"].get(t, np.nan) for r in summary_rows],
                "s--", label=f"HF {t}")
axs[0].axhline(0, color="gray", lw=0.7)
axs[0].set_xlabel("β"); axs[0].set_ylabel("median gain(object-level)")
axs[0].legend(fontsize=8); axs[0].set_title("等 B_signal 增益中位数", fontsize=10)
axs[1].plot(bs, [r["global_harm_rate"] for r in summary_rows], "o-", color="crimson")
axs[1].set_xlabel("β"); axs[1].set_title("global harm rate(G_g_eq<-2% 任一档)",
                                          fontsize=10)
axs[1].set_ylim(0, 1)
bot = np.zeros(len(bs))
for s, c in [("POSITIVE", "#2a9d8f"), ("NEUTRAL", "#bbb"),
             ("MIXED", "#e9c46a"), ("NEGATIVE", "#e76f51")]:
    v = [r["counts"][s] for r in summary_rows]
    axs[2].bar([str(b) for b in bs], v, bottom=bot, label=s, color=c)
    bot += np.array(v, float)
axs[2].legend(fontsize=8); axs[2].set_title("12 资产标签分布", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTB}/beta_preflight_curves.png", dpi=115, bbox_inches="tight")
plt.close(fig)

print("\n============ β Preflight 汇总(12 development assets) ============")
for r in summary_rows:
    print(f"β={r['beta']:<6} {r['counts']}  n_eval={r['n_evaluated']}  "
          f"medG={r['median_G_global_eq']}  medHF={r['median_G_HF_eq']}  "
          f"harm={r['global_harm_rate']}  ssim_ok={r['ssim_not_worse_rate']}")
print("verdict:", verdict)
print("BETA_PREFLIGHT: DONE")
