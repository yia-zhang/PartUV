# -*- coding: utf-8 -*-
"""Simple V1 Core Integrity smoke test.
1) 鞋/车轮 端到端: map_partuv_td -> 独立回读校验(面数/UV/贴图/尺度/面积);
2) 多材质资产 输入读取 smoke: check_asset_support 必须给出明确 supported/UNSUPPORTED.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from tdlib.api import UnsupportedAssetError, check_asset_support, map_partuv_td

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/smoke_v1"

E2E = [
    ("shoe", f"{DATA}/objaverse_22b822c6520d4d49.glb"),
    ("wheel", f"{DATA}/objaverse_92ff65712c62408d.glb"),
]
READ_ONLY = [
    ("multi_tex_8d1b6fc", f"{DATA}/8d1b6fc369484f4c4517d1f44d88471fa2083a75a9d4d7ecef9a880e107729ee.glb"),
    ("clock_2mat", f"{DATA}/clock_2mat.glb"),
]

fails = []

for tag, path in E2E:
    print(f"\n===== e2e: {tag} =====", flush=True)
    res = map_partuv_td(path, f"{OUT}/{tag}/")
    print("integrity:", res["integrity"])
    for wmsg in res["warnings"]:
        print("warning:", wmsg)

    # ---- 独立回读校验(不信任入口自己的检查) ----
    orig = trimesh.load(path, force="mesh")
    back = trimesh.load(res["glb_path"], force="mesh", process=False)
    checks = {
        "faces_preserved": len(back.faces) == len(orig.faces),
        "uv_present": back.visual.uv is not None and len(back.visual.uv) == len(back.vertices),
        "uv_in_01": bool((np.asarray(back.visual.uv) >= 0).all()
                         and (np.asarray(back.visual.uv) <= 1).all()),
        "tex_size": tuple(np.asarray(back.visual.material.baseColorTexture).shape[:2])
                    == (res["atlas_size"], res["atlas_size"]),
        "normals": back.vertex_normals is not None and len(back.vertex_normals) == len(back.vertices),
        "bbox_match": bool(np.abs(back.bounds - orig.bounds).max()
                           / np.linalg.norm(orig.extents) < 2e-3),
        "area_match": abs(back.area / orig.area - 1) < 2e-3,
        "seam_split_only": len(back.vertices) < 2.0 * len(orig.vertices),  # 远小于 3*faces
        "atlas_png": os.path.exists(res["atlas_path"]),
    }
    for k, v in checks.items():
        print(f"  {k:18s} {'PASS' if v else 'FAIL'}")
        if not v:
            fails.append(f"{tag}:{k}")
    print(f"  顶点数: 原 {len(orig.vertices):,} -> 出 {len(back.vertices):,} "
          f"(3*faces={3 * len(orig.faces):,})")

for tag, path in READ_ONLY:
    print(f"\n===== read-only: {tag} =====", flush=True)
    try:
        s = check_asset_support(path)
        verdict = "supported" if s["supported"] else s["reason"]
        print(f"  faces={s['n_faces']:,} tex={s['tex_shape']}  -> {verdict}")
        if not s["supported"] and "UNSUPPORTED" not in s["reason"]:
            fails.append(f"{tag}:reason_missing_UNSUPPORTED_tag")
    except Exception as e:
        print(f"  读取抛异常(应返回明确结论而非崩溃): {type(e).__name__}: {e}")
        fails.append(f"{tag}:crash")

print("\n===== SMOKE RESULT =====")
print("ALL PASS" if not fails else f"FAILS: {fails}")
sys.exit(0 if not fails else 1)
