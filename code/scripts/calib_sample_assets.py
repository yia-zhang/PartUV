# -*- coding: utf-8 -*-
"""Global β Calibration V1 —— 采样 50 个全新 Objaverse 纹理资产(冻结清单).

PUBLIC-DOMAIN CALIBRATION: Meshy 内部目标域数据在本机不可访问, 按协议改用
Objaverse; 结果不能代表 Meshy target-domain performance(报告必须标注)。

采样帧: ~/.objaverse 本地缓存 201 个 GLB(早期数据准备阶段随机抓取, 未参与
pipeline 开发) 减去 data/ 中全部 33 个开发帧 id(含鞋/车轮/17 dev cases 用到的
objaverse 资产); synthetic/sample_* 本就不在帧内。候选先做纹理可用性筛
(有 UV + basecolor 贴图, 属采样帧定义), 之后冻结 50 个:
  30 随机 + 20 challenge(六类, 描述符分层: 低纹理/局部logo/分布高频/
  many-chart/多材质/几何复杂)。
冻结后不得因 PartUV/结构/质量失败替换 —— 全部计入 processing yield。
SEED_SAMPLE=7。产物: outputs/calibration_v1/calibration_manifest.json,
资产复制到 data_calib/。
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
from PIL import Image

CODE = "/root/youjiaZhang/PartUV/code"
CACHE = os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs")
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
DATD = f"{CODE}/data_calib"
os.makedirs(OUTD, exist_ok=True)
os.makedirs(DATD, exist_ok=True)
SEED_SAMPLE = 7

dev_ids = {os.path.basename(p).replace("objaverse_", "").replace(".glb", "")
           for p in glob.glob(f"{CODE}/data/objaverse_*.glb")}
pool_paths = sorted(glob.glob(f"{CACHE}/*/*.glb"))
funnel = dict(cached_total=len(pool_paths), dev_frame_excluded=0,
              unloadable=0, untextured=0, textured_pool=0)

cands = []
for p in pool_paths:
    uid = os.path.basename(p).replace(".glb", "")
    if uid[:16] in dev_ids or uid in dev_ids:
        funnel["dev_frame_excluded"] += 1
        continue
    try:
        m = trimesh.load(p, force="mesh", process=False)
        uv = getattr(m.visual, "uv", None)
        img = None
        try:
            mat = m.visual.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        except Exception:
            pass
        if uv is None or len(np.atleast_1d(uv)) == 0 or img is None:
            funnel["untextured"] += 1
            continue
        # 描述符(仅用于 challenge 分层, 不进入协议)
        tex = np.asarray(img.convert("RGB").resize((256, 256)), float) / 255
        lum = tex @ [0.299, 0.587, 0.114]
        gy, gx = np.gradient(lum)
        g = np.sqrt(gx ** 2 + gy ** 2)
        gsum = g.sum() + 1e-12
        thr = np.quantile(g, 0.95)
        conc = float(g[g >= thr].sum() / gsum)     # top-5% 像素的梯度质量占比
        # UV 岛计数(union-find, 与 api._uv_islands 同思路)
        Fo = np.asarray(m.faces)
        parent = np.arange(len(uv))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for tri in Fo:
            r0 = find(tri[0])
            for v in tri[1:]:
                r = find(v)
                if r != r0:
                    parent[r] = r0
        n_isl = len(np.unique([find(v) for v in Fo[:, 0]]))
        try:
            sc = trimesh.load(p, process=False)
            n_mat = (len(sc.geometry) if isinstance(sc, trimesh.Scene) else 1)
        except Exception:
            n_mat = 1
        cands.append(dict(uid=uid, path=p, n_faces=int(len(Fo)),
                          n_islands=int(n_isl), n_geoms=int(n_mat),
                          tex_std=float(lum.std()), grad_density=float(g.mean()),
                          grad_conc=conc))
    except Exception:
        funnel["unloadable"] += 1
funnel["textured_pool"] = len(cands)
print("采样漏斗:", funnel)
assert len(cands) >= 60, f"纹理候选不足({len(cands)}), 需补下载"

rng = np.random.RandomState(SEED_SAMPLE)
by_uid = {c["uid"]: c for c in cands}
chosen, chall = {}, []


def pick_top(key, n, reverse=True, flt=None, tag=""):
    pool = [c for c in cands if c["uid"] not in chosen and (flt is None or flt(c))]
    pool.sort(key=lambda c: c[key], reverse=reverse)
    for c in pool[:n]:
        chosen[c["uid"]] = tag
        chall.append((c["uid"], tag))


# 六类 challenge, 共 20 个(3/3/3/4/3/4)
pick_top("tex_std", 3, reverse=False, tag="low_texture")
pick_top("grad_conc", 3, reverse=True,
         flt=lambda c: c["grad_density"] > 0.005, tag="local_logo")
pick_top("grad_density", 3, reverse=True,
         flt=lambda c: c["grad_conc"] < 0.35, tag="distributed_hf")
pick_top("n_islands", 4, reverse=True, tag="many_charts")
pick_top("n_geoms", 3, reverse=True,
         flt=lambda c: c["n_geoms"] >= 2, tag="multimat_overlap")
pick_top("n_faces", 4, reverse=True, tag="geometry_misc")
assert len(chall) == 20, f"challenge 数量 {len(chall)} != 20"

rest = sorted(c["uid"] for c in cands if c["uid"] not in chosen)
rand30 = [rest[i] for i in rng.choice(len(rest), 30, replace=False)]
for u in rand30:
    chosen[u] = "random"

manifest = dict(
    schema="calibration_v1_manifest",
    domain_label="PUBLIC-DOMAIN CALIBRATION(Objaverse); 不能代表 Meshy "
                 "target-domain performance(内部数据本机不可访问)",
    seed_sample=SEED_SAMPLE, funnel=funnel,
    frozen_rule="冻结后不因 PartUV/结构/质量失败替换; 全部失败计入 processing yield",
    excluded="data/ 全部 33 个开发帧 objaverse id + 17 dev cases + 鞋/车轮 + synthetics",
    assets=[])
for uid, tag in chall + [(u, "random") for u in rand30]:
    c = by_uid[uid]
    dst = f"{DATD}/objv_{uid[:12]}.glb"
    shutil.copy(c["path"], dst)
    manifest["assets"].append(dict(
        object_id=f"objv_{uid[:12]}", uid=uid, glb=dst, group=tag,
        descriptors={k: c[k] for k in ("n_faces", "n_islands", "n_geoms",
                                       "tex_std", "grad_density", "grad_conc")}))
with open(f"{OUTD}/calibration_manifest.json", "w") as fp:
    json.dump(manifest, fp, indent=1, ensure_ascii=False)
print(f"冻结 50 资产 -> {OUTD}/calibration_manifest.json")
from collections import Counter
print(Counter(a["group"] for a in manifest["assets"]))
print("SAMPLE: DONE")
