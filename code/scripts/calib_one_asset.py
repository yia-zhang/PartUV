# -*- coding: utf-8 -*-
"""Calibration V1 单资产协议(子进程单元, 驱动: run_calibration_v1.py).

冻结协议: 每资产一次 PartUV(缓存), 全部 β 复用同一 charts/local UV/reference/
samples(seed=2)/xatlas/padding=4/texel-center baker/两档 fixed-B_signal(50%/25%,
偏差<=1% 否则该档判 dev_fail); B_raw/fill 仅记录。β ∈ {0(基线), 0.125, 0.25}。
LOW_TD_CONTRAST(signal_dist<0.05, 布局与渲染前的 raw content contrast)
-> NEUTRAL=VALID_NO_OP, 保留在 denominator 中。
tier R 上限 2048(GPU 防线, 对全部 β 一致, 命中时记录 tier_capped)。
用法: calib_one_asset.py <glb> <oid> <outdir>
"""
import json
import os
import pickle
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

from tdlib.api import check_asset_support
from tdlib.layout import PackingFailedError
from tdlib.pipeline import load_reference, run_partuv
from tdlib.rd import prepare_face_ref_uv
from tdlib.signal import demand_weights, luminance_std_heuristic
from diag_common import eval_samples, pack_only, bake_layout, surface_err
from run_pseudo_gt_quality_gate import masked_ssim, SSIM_VIEWS

BETAS = [0.125, 0.25]
TIERS = [("50pct", 0.50), ("25pct", 0.25)]
LOW_SIGNAL_DIST = 0.05
BAND_G, POS_HF, NEG_G = 0.02, 0.05, -0.05
SSIM_TOL = -0.002
R_CAP = 2048

glb, oid, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(outdir, exist_ok=True)
res = dict(object_id=oid, processing_status="OK", reason="", betas={},
           tier_capped=False)


def finish():
    with open(f"{outdir}/result.json", "w") as fp:
        json.dump(res, fp, indent=1, ensure_ascii=False)
    print("CALIB_ONE: DONE", flush=True)
    sys.exit(0)


try:
    sup = check_asset_support(glb)
    if not sup["supported"]:
        res["processing_status"] = "PRECHECK_REJECTED"
        res["reason"] = sup["reason"][:160]
        finish()
    pu = run_partuv(glb, f"{outdir}/partuv/")
    charts, F = pu["charts"], pu["F"]
    if len(charts) == 0 or not pu["covered"].any():
        res["processing_status"] = "PARTUV_FAILED"
        res["reason"] = f"PartUV 未产生可用 charts({len(charts)})"
        finish()
    ref = load_reference(glb, pu["V"], F, pu["mesh_scale"])
    if not ref.get("has_tex"):
        res["processing_status"] = "PRECHECK_REJECTED"
        res["reason"] = "load_reference 无贴图"
        finish()
    face_refuv, valid, _ = prepare_face_ref_uv(pu, ref)
    texA = ref["texA"]
    tris = pu["V"][F]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    A3 = np.array([fa3[c["gidx"]].sum() for c in charts])
    a2 = np.array([float(c["a2"]) for c in charts])
    ctx = dict(charts=charts, F=F, V=pu["V"], fa3=fa3, a2=a2,
               face_refuv=face_refuv, valid=valid,
               scales_uni=np.sqrt(A3 / np.maximum(a2, 1e-12)),
               area_norm=A3 / A3.sum(),
               pu_like=dict(charts=charts, F=F, area=fa3),
               ch_masks=[dict(F=np.asarray(c["F"]), gidx=c["gidx"])
                         for c in charts],
               B_source=texA.shape[0] * texA.shape[1], texA=texA)
    with open(f"{outdir}/charts_cache.pkl", "wb") as fp:
        pickle.dump(pu, fp)
    res["n_charts"] = len(charts)
    res["n_faces"] = int(len(F))

    cw = luminance_std_heuristic(texA, ref["uv0"], ref["Fo"], ref["f2o"],
                                 valid & pu["covered"])
    sel = valid & pu["covered"]
    ev = eval_samples(ctx)
    okm = np.ones(len(F), bool)
    refs = [tdgpu.textured_render(ctx["V"], F, face_refuv, valid, texA, view=v)
            for v in SSIM_VIEWS]
    uni = {}
    for tname, frac in TIERS:
        R = max(int(round(np.sqrt(frac * ctx["B_source"]))), 64)
        if R > R_CAP:
            R = R_CAP
            res["tier_capped"] = True
        p = pack_only(ctx, ctx["scales_uni"], R)
        tex, nuv = bake_layout(ctx, p["uvs"], R)
        ss = [masked_ssim(a, b) for a, b in zip(
            refs, [tdgpu.textured_render(ctx["V"], F, nuv, okm, tex, view=v)
                   for v in SSIM_VIEWS])]
        uni[tname] = dict(R=R, S=p["b_signal"], fill=p["fill"],
                          d=surface_err(tex, nuv, ev), ssim=ss)
        res.setdefault("uniform", {})[tname] = dict(
            R=R, B_signal=p["b_signal"], fill_record=round(p["fill"], 4))

    for beta in BETAS:
        _, w = demand_weights(cw, sel, fa3, beta=beta)
        dem = np.array([float((fa3[c["gidx"]] * w[c["gidx"]]).sum())
                        for c in charts])
        scales_b = np.sqrt(dem / np.maximum(a2, 1e-12))
        sd = float(0.5 * np.abs(dem / max(dem.sum(), 1e-20)
                                - ctx["area_norm"]).sum())
        row = dict(beta=beta, signal_dist=round(sd, 4), tiers={},
                   label="NEUTRAL", valid_no_op=False)
        if sd < LOW_SIGNAL_DIST:
            row["valid_no_op"] = True     # LOW_TD_CONTRAST(布局/渲染前判定)
            res["betas"][str(beta)] = row
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
                row["tiers"][tname] = dict(error="packing failed")
                neg_any = True
                continue
            R_td, pt = best
            match = pt["b_signal"] / max(u["S"], 1)
            if abs(match - 1) > 0.01:
                row["tiers"][tname] = dict(error="bsignal_dev>1%",
                                           bsignal_match=round(match, 4))
                continue
            tex, nuv = bake_layout(ctx, pt["uvs"], R_td)
            d_t = surface_err(tex, nuv, ev)
            g = 1 - float(d_t.mean()) / max(float(u["d"].mean()), 1e-20)
            ghf = 1 - float(d_t[ev["hi"]].mean()) / max(
                float(u["d"][ev["hi"]].mean()), 1e-20)
            ss_t = [masked_ssim(a, b) for a, b in zip(
                refs, [tdgpu.textured_render(ctx["V"], F, nuv, okm, tex, view=v)
                       for v in SSIM_VIEWS])]
            dss = float(np.mean([a - b for a, b in zip(ss_t, u["ssim"])]))
            row["tiers"][tname] = dict(
                G_global_eq=round(g, 4), G_HF_eq=round(ghf, 4),
                bsignal_match=round(match, 4), ssim_delta_mean=round(dss, 5),
                ssim_not_worse=bool(dss >= SSIM_TOL),
                B_raw_record=int(R_td * R_td), fill_record=round(pt["fill"], 4),
                overlap=pt["overlap"])
            pos_any |= g >= BAND_G or (ghf >= POS_HF and g >= -BAND_G)
            neg_any |= g <= NEG_G and ghf < POS_HF
        t_ok = [t for t in row["tiers"].values() if "G_global_eq" in t]
        if not t_ok:
            row["label"] = "NOT_EVALUATED"
        else:
            in_band = all(abs(t["G_global_eq"]) <= BAND_G
                          and abs(t["G_HF_eq"]) < POS_HF for t in t_ok)
            row["label"] = ("NEUTRAL" if in_band else
                            "POSITIVE" if pos_any and not neg_any else
                            "NEGATIVE" if neg_any and not pos_any else "MIXED")
        res["betas"][str(beta)] = row
    finish()
except PackingFailedError as e:
    res["processing_status"] = "PACKING_FAILED"
    res["reason"] = str(e)[:200]
    finish()
except Exception as e:
    res["processing_status"] = "ERROR"
    res["reason"] = f"{type(e).__name__}: {str(e)[:200]}"
    traceback.print_exc()
    finish()
