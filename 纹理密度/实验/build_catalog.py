# -*- coding: utf-8 -*-
"""Build code/data/CATALOG.md: per-GLB stats to pick test meshes quickly.

Columns:
  faces / has UV+texture(+res) / orig-UV TD CVw (area-weighted; 高=原始分配不均)
  content contrast = P90(cw)/median(cw) (cw=面内8点亮度std; 高=细节分布悬殊 -> L2 演示效果好)
  推荐用途 tag.
"""
import glob, os
import numpy as np
import trimesh

DATA = "/root/youjiaZhang/PartUV/code/data"

def tri_a2(uv):
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

def tri_a3(v):
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

rows = []
for p in sorted(glob.glob(f"{DATA}/*.glb")):
    name = os.path.basename(p)
    try:
        m = trimesh.load(p, force="mesh")
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
        uv = getattr(m.visual, "uv", None)
        tex = None
        try:
            mat = m.visual.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None:
                tex = np.asarray(img.convert("RGB"), float) / 255.0
        except Exception:
            pass
        a3 = tri_a3(V[F]); ok3 = a3 > 1e-14
        cvw_td = contrast = None
        if uv is not None:
            uv = np.asarray(uv, float)
            a2 = tri_a2(uv[F])
            sel = ok3 & (a2 > 1e-16)
            td = np.sqrt(a2[sel] / a3[sel]); wgt = a3[sel]
            mu = np.average(td, weights=wgt)
            cvw_td = float(np.sqrt(np.average((td - mu) ** 2, weights=wgt)) / mu)
        if tex is not None and uv is not None:
            rng = np.random.RandomState(0)
            bar = rng.dirichlet((1.2, 1.2, 1.2), 8)
            lum = tex @ np.array([0.299, 0.587, 0.114])
            UV3 = uv[F]
            acc = acc2 = None
            for s in range(len(bar)):
                pt = bar[s] @ UV3.transpose(1, 0, 2) if False else np.einsum("k,fkd->fd", bar[s], UV3)
                x = np.clip(pt[:, 0], 0, 1) * (lum.shape[1] - 1)
                y = np.clip(1 - pt[:, 1], 0, 1) * (lum.shape[0] - 1)
                v = lum[y.astype(int), x.astype(int)]
                acc = v.copy() if acc is None else acc + v
                acc2 = v ** 2 if acc2 is None else acc2 + v ** 2
            cw = np.sqrt(np.maximum(acc2 / 8 - (acc / 8) ** 2, 0))[ok3]
            p90 = float(np.quantile(cw, 0.9))
            med = max(float(np.median(cw)), 1e-3 * max(p90, 1e-9))   # 大片纯色时 median~0, 加下限
            contrast = min(p90 / max(med, 1e-9), 99.0)               # 封顶显示
        rows.append(dict(name=name, faces=len(F),
                         uv=uv is not None,
                         tex=(f"{tex.shape[1]}x{tex.shape[0]}" if tex is not None else "-"),
                         cvw=cvw_td, contrast=contrast))
        print(f"{name:40s} F={len(F):6d} uv={uv is not None} tex={rows[-1]['tex']:>10s} "
              f"cvw={'' if cvw_td is None else f'{cvw_td:.3f}'} "
              f"contrast={'' if contrast is None else f'{contrast:.1f}'}")
    except Exception as e:
        rows.append(dict(name=name, faces=-1, uv=False, tex="ERR", cvw=None, contrast=None))
        print(f"{name}: ERR {e}")

def tag(r):
    t = []
    if not r["uv"] or r["tex"] in ("-", "ERR"):
        t.append("无纹理(只能看Baseline)")
        return "; ".join(t)
    if r["contrast"] is not None and r["contrast"] >= 90:
        t.append("★★ L2极佳(细节高度集中,大片纯色)")
    elif r["contrast"] is not None and r["contrast"] >= 4:
        t.append("★ L2演示佳(细节反差大)")
    elif r["contrast"] is not None and r["contrast"] >= 2.5:
        t.append("L2可用")
    else:
        t.append("内容较均匀(L2增益小)")
    if r["cvw"] is not None and r["cvw"] > 0.3:
        t.append("原始UV密度不均(heatmap before/after 好看)")
    if r["faces"] < 500:
        t.append("网格过小")
    if r["faces"] > 50000:
        t.append("大网格(跑得慢)")
    return "; ".join(t)

rows.sort(key=lambda r: -(r["contrast"] or 0))
lines = [
    "# code/data 测试网格目录（自动生成, 2026-07-13）",
    "",
    "- **orig TD CVw**: 原始 UV 的面积加权密度变异系数(高=原始分配不均匀)",
    "- **content contrast**: P90(cw)/median(cw), cw=面内采样亮度std(高=细节分布悬殊, L2 演示效果好)",
    "- 生成脚本: 纹理密度/实验/build_catalog.py (从 scratchpad 归档)",
    "",
    "| mesh | faces | UV | texture | orig TD CVw | content contrast | 推荐 |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    lines.append(f"| {r['name']} | {r['faces']} | {'Y' if r['uv'] else 'N'} | {r['tex']} "
                 f"| {('%.3f' % r['cvw']) if r['cvw'] is not None else '-'} "
                 f"| {('%.1f' % r['contrast']) if r['contrast'] is not None else '-'} "
                 f"| {tag(r)} |")
lines += ["", "OBJ 文件（官方 demo，无 UV/纹理，只能看 Baseline）: "
          + ", ".join(os.path.basename(x) for x in sorted(glob.glob(f"{DATA}/*.obj")))]
open(f"{DATA}/CATALOG.md", "w").write("\n".join(lines))
print(f"\nCATALOG.md written with {len(rows)} glbs")
