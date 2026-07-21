# -*- coding: utf-8 -*-
"""为 50 个 calibration 资产渲染缩略图(只读源 GLB, 不触碰运行中的任务).
巨型网格(>500k 面)与加载失败者出占位图。输出 outputs/calibration_v1/thumbs/."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from PIL import Image

from tdlib import gpu as tdgpu
tdgpu.pick_free_gpu()

CODE = "/root/youjiaZhang/PartUV/code"
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
TH = f"{OUTD}/thumbs"
os.makedirs(TH, exist_ok=True)
man = json.load(open(f"{OUTD}/calibration_manifest_v2.json"))

for a in man["assets"]:
    oid = a["object_id"]
    dst = f"{TH}/{oid}.jpg"
    if os.path.exists(dst):
        continue
    try:
        if a["descriptors"]["n_faces"] > 500_000:
            raise RuntimeError("巨型网格, 跳过渲染")
        m = trimesh.load(a["glb"], force="mesh", process=False)
        uv = np.asarray(m.visual.uv, float)
        img = (getattr(m.visual.material, "baseColorTexture", None)
               or getattr(m.visual.material, "image", None))
        texA = np.asarray(img.convert("RGB"), float) / 255.0
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
        fuv = uv[F]
        ok = np.ones(len(F), bool)
        im = tdgpu.textured_render(V, F, fuv, ok, texA, view=(18, 40), px=340)
        Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).convert(
            "RGB").save(dst, quality=72)
        print(f"[ok] {oid}", flush=True)
    except Exception as e:
        ph = Image.new("RGB", (340, 340), (48, 52, 60))
        ph.save(dst, quality=60)
        print(f"[placeholder] {oid}: {type(e).__name__}: {str(e)[:60]}", flush=True)
print("THUMBS: DONE")
