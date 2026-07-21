# -*- coding: utf-8 -*-
"""Object-level teacher-generated pseudo-GT exporter (V1).

export_object_pseudo_gt(res, sample_dir, *, object_id=None) -> manifest dict

- 只消费 map_partuv_td 的同一次运行结果 res, 不重算任何 teacher 逻辑;
- 固定输出: manifest.json / arrays.npz / reference_basecolor.png /
  target_atlas.png / target_mesh.glb;
- 自动验收 gates A(几何对应)/B(UV 合法)/C(TD 与预算)/D(磁盘回读+SHA256);
  全过才写 status=ACCEPTED, 否则 REJECTED(不留伪装成品);
- 标签语义: teacher-generated pseudo-GT, 仅 TD allocation supervision,
  无 ArtUV-like local refinement, 不是 artist GT(artist_gt 恒 false)。
"""
import hashlib
import json
import os
import shutil

import numpy as np


def _json_default(o):
    """numpy 标量 -> python 原生(禁止静默丢数据, 未知类型仍抛错)."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"不可序列化: {type(o)}")

SCHEMA_VERSION = "partuv_td_object_v1"
LABEL_TYPE = "partuv_td_teacher_pseudo_gt_v1"
PADDING_PX = 4                     # layout.xatlas_pack 默认(teacher 固定值)
LUMSTD_SEED = 0                    # signal.luminance_std_heuristic 默认 seed

REQUIRED_KEYS = [
    "vertices", "faces", "face_ids", "source_uv", "source_uv_valid",
    "face_to_chart", "train_face_mask", "chart_ids",
    "local_uv_before_td", "target_packed_uv",
    "face_content_score", "chart_content_score",
    "chart_demand_normalized", "chart_target_texels", "chart_target_scale",
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_arrays(arrs):
    """数组内容的规范化 hash(与 zip 时间戳无关, 供确定性 gate 用)."""
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _build_arrays(res):
    """从 res 组装全部 npz 数组(仅读取, 不重算 teacher 逻辑)."""
    pu = res["pu"]
    charts = pu["charts"]
    F = np.asarray(pu["F"], np.int64)
    Nf, C = len(F), len(charts)
    verts_world = np.asarray(res["to_world"](pu["V"]), np.float64)

    face_to_chart = np.asarray(pu["face_chart"], np.int64)      # -1 = 未覆盖
    valid = np.asarray(res["valid"], bool)
    train_mask = (face_to_chart >= 0) & valid

    local_uv = np.zeros((Nf, 3, 2))
    packed_uv = np.zeros((Nf, 3, 2))
    for ci, c in enumerate(charts):
        cF = np.asarray(c["F"])
        local_uv[c["gidx"]] = np.asarray(c["UV"], float)[cF]    # TD 缩放/打包前
        packed_uv[c["gidx"]] = res["uvs_td"][ci][cF]

    demand = np.asarray(res["td_chart_demand"], np.float64)
    demand_norm = demand / max(demand.sum(), 1e-20)
    b_target = float(res["budget"]["B_target"])

    return dict(
        vertices=verts_world,
        faces=F,
        face_ids=np.arange(Nf, dtype=np.int64),
        source_uv=np.asarray(res["face_refuv"], np.float64),
        source_uv_valid=valid,
        face_to_chart=face_to_chart,
        train_face_mask=train_mask,
        chart_ids=np.arange(C, dtype=np.int64),
        local_uv_before_td=local_uv,
        target_packed_uv=packed_uv,
        face_content_score=np.asarray(res["cw"], np.float64),
        chart_content_score=np.asarray(res["mean_cw"], np.float64),
        chart_demand_normalized=demand_norm,
        chart_target_texels=demand_norm * b_target,
        chart_target_scale=np.asarray(res["td_chart_scales"], np.float64),
    )


def _gates_abc(arrs, res):
    """内存内 gates A/B/C. 返回 (gates dict, warnings list)."""
    import trimesh
    from .budget import rasterize_masks

    g, warn = {}, []
    pu, budget, integ = res["pu"], res["budget"], res["integrity"]
    charts = pu["charts"]
    Nf, C = len(arrs["faces"]), len(arrs["chart_ids"])
    tm = arrs["train_face_mask"]

    # ---- A. 几何与对应关系 ----
    orig = trimesh.load(res["input_mesh"], force="mesh")
    a_area = float(arrs_area(arrs) / max(orig.area, 1e-20))
    scale_w = float(np.linalg.norm(np.asarray(orig.vertices).ptp(0)))
    bbox_dev = float(np.abs(np.asarray(
        [arrs["vertices"].min(0) - orig.vertices.min(0),
         arrs["vertices"].max(0) - orig.vertices.max(0)])).max() / scale_w)
    unc = np.where(arrs["face_to_chart"] < 0)[0]
    cov = float(tm.mean())
    g["A_face_count"] = (Nf == len(orig.faces) == integ["n_faces_in"])
    g["A_face_ids"] = bool((arrs["face_ids"] == np.arange(Nf)).all())
    g["A_uv_shapes"] = all(arrs[k].shape == (Nf, 3, 2) for k in
                           ("source_uv", "local_uv_before_td", "target_packed_uv"))
    g["A_area_ratio"] = abs(a_area - 1) <= 1e-6
    g["A_bbox_dev"] = bbox_dev <= 1e-6
    g["A_one_chart_per_train_face"] = bool(
        (arrs["face_to_chart"][tm] >= 0).all()
        and (arrs["face_to_chart"][tm] < C).all())
    g["A_train_coverage>=99.9%"] = cov >= 0.999
    g["A_glb_reload"] = bool(integ.get("reload_ok"))
    warn.append(f"未覆盖 face 数={len(unc)}, 示例 face id={unc[:8].tolist()}; "
                f"train_face_coverage={cov*100:.3f}%")
    excl = np.where(~tm)[0]
    warn.append(f"train_face_mask=False 共 {len(excl)} 面(未覆盖或无源 UV 对应), "
                f"示例={excl[:8].tolist()}, 不进入训练")

    # ---- B. UV 合法性 ----
    puv = arrs["target_packed_uv"][tm]
    g["B_finite"] = bool(np.isfinite(puv).all())
    g["B_in_unit_range"] = bool(puv.min() >= -1e-6 and puv.max() <= 1 + 1e-6)
    R = int(budget["selected_atlas_size"])
    ch_masks = [dict(F=np.asarray(c["F"]), gidx=c["gidx"]) for c in charts]
    _, overlap, _ = rasterize_masks(ch_masks, res["uvs_td"], R, R)
    g["B_no_overlap"] = (int(overlap) == 0)
    memb_ok = True
    for ci, c in enumerate(charts):
        if not (arrs["face_to_chart"][c["gidx"]] == ci).all():
            memb_ok = False
            break
    g["B_chart_membership_matches_teacher"] = memb_ok

    # ---- C. TD 与预算 ----
    ratio = budget["output_B_signal"] / max(budget["B_target"], 1)
    g["C_label_type"] = True                       # 由本模块常量写入
    g["C_artist_gt_false"] = True
    g["C_budget_ratio_band"] = 1.00 <= ratio <= 1.05
    g["C_E_alloc<=1%"] = budget["E_alloc"] <= 0.01
    g["C_B_raw==W*H"] = True                       # 方形 atlas: R*R, 回读时复核
    tt = arrs["chart_target_texels"]
    g["C_target_texels_sum"] = abs(tt.sum() - budget["B_target"]) <= \
        max(1e-6 * budget["B_target"], 1.0)
    g["C_demand_norm_sum1"] = abs(arrs["chart_demand_normalized"].sum() - 1) <= 1e-9
    g["C_chart_arrays_len"] = all(len(arrs[k]) == C for k in
                                  ("chart_content_score", "chart_demand_normalized",
                                   "chart_target_texels", "chart_target_scale"))
    return g, warn


def arrs_area(arrs):
    t = arrs["vertices"][arrs["faces"]]
    return float(np.linalg.norm(
        np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)


def _gate_d_reload(sample_dir, manifest_draft, arrs_mem):
    """D. 磁盘回读验收(全部从磁盘重新读取验证)."""
    import trimesh
    from PIL import Image

    g = {}
    npz_path = os.path.join(sample_dir, "arrays.npz")
    with np.load(npz_path) as z:
        keys = set(z.files)
        g["D_npz_keys"] = all(k in keys for k in REQUIRED_KEYS)
        reload_ok = True
        for k in REQUIRED_KEYS:
            a, b = z[k], arrs_mem[k]
            if a.shape != b.shape or a.dtype != b.dtype or \
                    not np.array_equal(a, b):
                reload_ok = False
                break
        g["D_npz_roundtrip_identical"] = reload_ok
        tm = z["train_face_mask"]
        g["D_reload_uv_finite_masked"] = bool(
            np.isfinite(z["target_packed_uv"][tm]).all())
    ref_png = Image.open(os.path.join(sample_dir, "reference_basecolor.png"))
    atlas_png = Image.open(os.path.join(sample_dir, "target_atlas.png"))
    g["D_textures_readable"] = True
    R = manifest_draft["teacher"]["atlas_size"]
    g["D_atlas_size_matches"] = list(atlas_png.size) == [R[0], R[1]]
    glb = trimesh.load(os.path.join(sample_dir, "target_mesh.glb"),
                       force="mesh", process=False)
    geo = manifest_draft["geometry"]
    npz_area = arrs_area(arrs_mem)
    g["D_glb_faces"] = len(glb.faces) == geo["n_faces"]
    g["D_glb_area_vs_npz"] = abs(glb.area / max(npz_area, 1e-20) - 1) <= 1e-6
    g["D_glb_bbox_vs_npz"] = bool(np.abs(
        np.asarray(glb.bounds, float)
        - np.stack([arrs_mem["vertices"].min(0), arrs_mem["vertices"].max(0)])
    ).max() / max(np.linalg.norm(arrs_mem["vertices"].ptp(0)), 1e-20) <= 1e-6)
    files = {}
    for fn in ("arrays.npz", "reference_basecolor.png",
               "target_atlas.png", "target_mesh.glb"):
        files[fn] = _sha256_file(os.path.join(sample_dir, fn))
    return g, files


def export_object_pseudo_gt(res, sample_dir, *, object_id=None):
    """导出 object-level pseudo-GT 样本并自动验收. 返回 manifest dict."""
    from PIL import Image

    os.makedirs(sample_dir, exist_ok=True)
    object_id = object_id or res["name"]
    arrs = _build_arrays(res)
    budget, integ = res["budget"], res["integrity"]
    R = int(budget["selected_atlas_size"])

    # ---- 写文件(manifest 最后写) ----
    np.savez(os.path.join(sample_dir, "arrays.npz"), **arrs)
    Image.fromarray((np.clip(res["texA"], 0, 1) * 255).astype(np.uint8)).save(
        os.path.join(sample_dir, "reference_basecolor.png"))
    shutil.copyfile(res["atlas_path"], os.path.join(sample_dir, "target_atlas.png"))
    shutil.copyfile(res["glb_path"], os.path.join(sample_dir, "target_mesh.glb"))

    gates, warnings = _gates_abc(arrs, res)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{object_id}__{LABEL_TYPE}",
        "object_id": object_id,
        "status": "REJECTED",                     # 回读全过才改 ACCEPTED
        "label_type": LABEL_TYPE,
        "artist_gt": False,
        "sample_unit": "object",
        "supervised_scope": "td_allocation_only",
        "local_uv_refinement": "none",
        "train_ready": {"td_allocation": False, "artist_refinement": False},
        "teacher": {
            "name": "PartUV+TD Simple V1",
            "signal": "luminance_std",
            "beta": float(res["beta"]),
            "packer": budget["packer"],
            "atlas_size": [R, R],
            "padding_px": PADDING_PX,
            "code_version": "unknown",            # code/ 非 git 仓库, 不伪造
            "geometry_hash": hashlib.sha256(
                np.ascontiguousarray(arrs["vertices"]).tobytes()
                + np.ascontiguousarray(arrs["faces"]).tobytes()).hexdigest(),
            "chart_hash": hashlib.sha256(
                np.ascontiguousarray(arrs["face_to_chart"]).tobytes()
                + np.ascontiguousarray(arrs["local_uv_before_td"]).tobytes()
            ).hexdigest(),
            "seed": f"luminance_std_seed={LUMSTD_SEED}; partuv/partfield 推理无外部种子控制(跨运行 chart 漂移已知), 记为 unknown",
        },
        "budget": {
            "B_target": int(budget["B_target"]),
            "B_signal": int(budget["output_B_signal"]),
            "B_raw": R * R,
            "budget_ratio": float(budget["budget_ratio"]),
            "packing_fill": float(budget["output_packing_fill"]),
            "E_alloc": float(budget["E_alloc"]),
        },
        "geometry": {
            "n_vertices": int(len(arrs["vertices"])),
            "n_faces": int(len(arrs["faces"])),
            "n_charts": int(len(arrs["chart_ids"])),
            "train_face_coverage": float(arrs["train_face_mask"].mean()),
            "area_ratio": float(integ["area_ratio"]),
            "bbox_deviation": float(integ["bbox_dev"]),
            "reload_ok": bool(integ["reload_ok"]),
        },
        "warnings": warnings + list(res.get("warnings", [])),
        "gates": gates,
        "files": {},
        "arrays_content_sha256": _sha256_arrays(arrs),
        "notes": ("teacher-generated pseudo-GT; 仅适用于 TD allocation "
                  "supervision; 无 ArtUV-like local refinement 标签; 非 artist GT。"
                  " target_packed_uv 为最终资产/QA 产物, 非默认回归标签;"
                  " 主要可学习标签为 chart_demand_normalized/"
                  "chart_target_texels/chart_target_scale。"),
    }

    # ---- D. 磁盘回读 ----
    gates_d, files = _gate_d_reload(sample_dir, manifest, arrs)
    manifest["gates"].update(gates_d)
    manifest["files"] = files
    all_pass = all(bool(v) for v in manifest["gates"].values())
    manifest["status"] = "ACCEPTED" if all_pass else "REJECTED"
    manifest["train_ready"]["td_allocation"] = bool(all_pass)
    if not all_pass:
        manifest["warnings"].append(
            "GATES FAILED: " + ", ".join(k for k, v in manifest["gates"].items()
                                         if not v))
    with open(os.path.join(sample_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False, default=_json_default)
    return manifest
