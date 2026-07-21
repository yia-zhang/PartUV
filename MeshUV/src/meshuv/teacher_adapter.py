# -*- coding: utf-8 -*-
"""冻结 teacher 的唯一调用入口(MeshUV 侧不复制/不分叉/不重实现 tdlib 逻辑).

集中管理: PARTUV_ROOT / TEACHER_VERSION / 冻结 β / protocol hash / code hash。
其余 MeshUV 代码一律 `from meshuv.teacher_adapter import ...`, 不得自行
sys.path 指向 PartUV。
"""
import hashlib
import os
import sys

import numpy as np

PARTUV_ROOT = os.environ.get("PARTUV_ROOT", "/root/youjiaZhang/PartUV/code")
TEACHER_VERSION = "partuv_td_teacher_pseudo_gt_v1"
EVALUATOR = "sampler=texel_center_v1 + reduce=coverage_center_v1"
# canonical label 语义: 线性纹理密度(texels/单位长度)的 mean-centered log 比,
# 与 TD(f)=R·sqrt(A_UV/A_3D) 及 chart_target_scale(线性 sqrt)约定一致。
LABEL_SEMANTICS = "linear_texel_density_log_ratio_v1"

_wired = False


def _ensure_path():
    global _wired
    if not _wired:
        for p in (PARTUV_ROOT, os.path.join(PARTUV_ROOT, "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        _wired = True


def _repo_root():
    """仓库根 = PARTUV_ROOT 的上级(hash 用仓库相对路径, 与绝对位置无关)."""
    return os.path.dirname(os.path.abspath(PARTUV_ROOT))


def teacher_hash_files():
    """Teacher provenance 覆盖的文件(仓库相对路径, 排序稳定):
    - code/tdlib/**/*.py         teacher 计算核心
    - MeshUV/src/meshuv/teacher_adapter.py   canonical label 实现(本文件)
    - code/notebook/partuv_config.yaml       PartUV 管线配置
    不含生成的 frozen YAML(避免循环依赖)。"""
    root = _repo_root()
    files = ["MeshUV/src/meshuv/teacher_adapter.py",
             "code/notebook/partuv_config.yaml"]
    td = os.path.join(root, "code", "tdlib")
    for dirpath, _, fns in os.walk(td):
        for fn in fns:
            if fn.endswith(".py"):
                files.append(os.path.relpath(os.path.join(dirpath, fn), root))
    return sorted(files)


def _hash_files(root, rel_files):
    """内容 sha256: 只依赖相对路径与文件字节, 与遍历顺序/时间戳/绝对路径无关."""
    h = hashlib.sha256()
    for rel in sorted(rel_files):
        h.update(rel.encode())
        h.update(open(os.path.join(root, rel), "rb").read())
    return h.hexdigest()


def teacher_code_hash():
    """Teacher 代码+配置的 provenance hash(冻结校验用)."""
    return _hash_files(_repo_root(), teacher_hash_files())


def pick_free_gpu():
    _ensure_path()
    from tdlib import gpu as tdgpu
    tdgpu.pick_free_gpu()


def check_support(glb):
    _ensure_path()
    from tdlib.api import check_asset_support
    return check_asset_support(glb)


def run_teacher_context(glb, workdir):
    """PartUV 一次 + reference + 内容分数 —— 返回 label/QA 共用的 ctx.
    失败以 dict(status=...) 返回, 不抛(调用方决定 yield 记账)。"""
    _ensure_path()
    from tdlib.pipeline import load_reference, run_partuv
    from tdlib.rd import prepare_face_ref_uv
    from tdlib.signal import luminance_std_heuristic
    pu = run_partuv(glb, workdir)
    if len(pu["charts"]) == 0 or not pu["covered"].any():
        return dict(status="PARTUV_FAILED",
                    reason=f"PartUV 未产生可用 charts({len(pu['charts'])})")
    ref = load_reference(glb, pu["V"], pu["F"], pu["mesh_scale"])
    if not ref.get("has_tex"):
        return dict(status="PRECHECK_REJECTED", reason="load_reference 无贴图")
    face_refuv, valid, face2chart = prepare_face_ref_uv(pu, ref)
    tris = pu["V"][pu["F"]]
    fa3 = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0],
                                  tris[:, 2] - tris[:, 0]), axis=1) / 2
    sel = valid & pu["covered"]
    cw = luminance_std_heuristic(ref["texA"], ref["uv0"], ref["Fo"],
                                 ref["f2o"], sel)
    return dict(status="OK", pu=pu, ref=ref, face_refuv=face_refuv,
                valid=valid, face2chart=face2chart, fa3=fa3, sel=sel, cw=cw)


def compute_labels(tc, beta):
    """冻结 demand 公式 -> canonical targets(不涉及 packing)."""
    _ensure_path()
    from tdlib.signal import demand_weights
    pu, fa3, cw, sel = tc["pu"], tc["fa3"], tc["cw"], tc["sel"]
    charts = pu["charts"]
    _, w = demand_weights(cw, sel, fa3, beta=beta)
    dem = np.array([float((fa3[c["gidx"]] * w[c["gidx"]]).sum())
                    for c in charts])
    A3 = np.array([float(fa3[c["gidx"]].sum()) for c in charts])
    a2 = np.array([float(c["a2"]) for c in charts])
    valid_c = (dem > 0) & (A3 > 0) & (a2 > 0)
    dshare = dem / max(dem.sum(), 1e-20)
    ashare = A3 / max(A3.sum(), 1e-20)
    # canonical: mean-centered log LINEAR texel-density ratio
    # (LABEL_SEMANTICS=linear_texel_density_log_ratio_v1)。
    # 0.5 因子把面积纹素分配比(texels/面积)转为线性密度(texels/长度),
    # 恒等式: 0.5*log(dshare/ashare) == log(target_scale/uniform_scale)+const
    logr = np.zeros(len(charts))
    logr[valid_c] = 0.5 * np.log(np.maximum(dshare[valid_c], 1e-20)
                                 / np.maximum(ashare[valid_c], 1e-20))
    logr[valid_c] -= logr[valid_c].mean()          # mean-centered
    chart_cw = np.array([float(np.average(
        cw[c["gidx"]], weights=np.maximum(fa3[c["gidx"]], 1e-12)))
        for c in charts])
    return dict(
        chart_demand_normalized=dshare,
        chart_target_area_fraction=dshare.copy(),   # 目标 UV 面积份额==需求份额
        chart_log_density_ratio=logr,
        chart_target_scale=np.sqrt(np.maximum(dem, 0)
                                   / np.maximum(a2, 1e-12)),
        chart_valid_mask=valid_c,
        chart_surface_area=A3, chart_uv_area_before_td=a2,
        face_content_score=cw,                      # teacher diagnostic(禁作输入)
        chart_content_score=chart_cw)               # teacher diagnostic(禁作输入)


def quality_check_medium(tc, labels, frac=0.5, r_cap=2048, seed=2,
                         n_samples=150_000):
    """固定中等 fixed-B_signal 预算: Uniform vs teacher 的 global/HF 表面误差.
    等 B_signal(偏差<=1%), texel-center baker。返回 dict(status=..., 指标)。"""
    _ensure_path()
    from tdlib.budget import rasterize_masks
    from tdlib.layout import xatlas_pack, PackingFailedError
    from tdlib.rd import (bake_atlas_ss, bilinear, ref_gradient_at_samples,
                          surface_samples)
    pu, ref = tc["pu"], tc["ref"]
    charts, F, fa3 = pu["charts"], pu["F"], tc["fa3"]
    texA = ref["texA"]
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    pu_like = dict(charts=charts, F=F, area=fa3)

    def srgb2lin(x):
        x = np.clip(x, 0, 1)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    def build(scales, R):
        uvs = xatlas_pack(charts, scales, resolution=R, padding_px=4)
        owner, ov, _ = rasterize_masks(ch_masks, uvs, R, R)
        tex, _, _ = bake_atlas_ss(pu_like, uvs, R, 4, tc["face_refuv"],
                                  tc["valid"], texA)
        nuv = np.zeros((len(F), 3, 2))
        for ci, c in enumerate(charts):
            nuv[c["gidx"]] = uvs[ci][np.asarray(c["F"])]
        return dict(S=int((owner >= 0).sum()), ov=int(ov), tex=tex, nuv=nuv)

    a2 = labels["chart_uv_area_before_td"]
    A3 = labels["chart_surface_area"]
    scales_u = np.sqrt(np.maximum(A3, 0) / np.maximum(a2, 1e-12))
    B = texA.shape[0] * texA.shape[1]
    R = min(max(int(round(np.sqrt(frac * B))), 64), r_cap)
    try:
        pu_b = build(scales_u, R)
    except PackingFailedError as e:
        return dict(status="PACKING_FAILED", reason=str(e)[:160])
    s = surface_samples(pu_like, tc["face_refuv"], tc["valid"], texA,
                        n_samples, seed=seed)
    g = ref_gradient_at_samples(texA, tc["face_refuv"], s)
    hi = g >= np.quantile(g, 0.9)
    ref_lin = srgb2lin(np.asarray(s["ref_color"]))

    def err(p):
        uvq = np.einsum("nk,nkd->nd", s["bary"], p["nuv"][s["fid"]])
        return ((srgb2lin(bilinear(p["tex"], uvq)) - ref_lin) ** 2).mean(1)

    d_u = err(pu_b)
    lo, hi_r, best = int(R * 0.6), int(R * 1.8), None
    for _ in range(9):
        mid = (lo + hi_r) // 2
        try:
            pt = build(labels["chart_target_scale"], mid)
        except PackingFailedError:
            lo = mid + 1
            continue
        if best is None or abs(pt["S"] - pu_b["S"]) < abs(best[1]["S"] - pu_b["S"]):
            best = (mid, pt)
        if pt["S"] < pu_b["S"]:
            lo = mid + 1
        else:
            hi_r = mid - 1
    if best is None:
        return dict(status="TEACHER_PACKING_FAILED")
    R_t, pt = best
    match = pt["S"] / max(pu_b["S"], 1)
    if abs(match - 1) > 0.01:
        return dict(status="BSIGNAL_DEV_FAIL", bsignal_match=round(match, 4))
    d_t = err(pt)
    g_eq = 1 - float(d_t.mean()) / max(float(d_u.mean()), 1e-20)
    ghf_eq = 1 - float(d_t[hi].mean()) / max(float(d_u[hi].mean()), 1e-20)
    return dict(status="OK", R_uniform=R, R_teacher=R_t,
                bsignal_match=round(match, 4),
                G_global_eq=round(g_eq, 4), G_HF_eq=round(ghf_eq, 4),
                overlap=pu_b["ov"] + pt["ov"])
