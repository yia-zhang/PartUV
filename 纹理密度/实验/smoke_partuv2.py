# -*- coding: utf-8 -*-
"""Smoke test v2: PartUV output -> global face correspondence via face centroids.

Verifies on a real textured objaverse GLB:
  1. end-to-end runtime on this machine;
  2. face correspondence: centroid-match combined output faces -> input mesh faces
     (the mapping our density layer needs for content weights + rebake);
  3. per-part face labels recoverable from individual_parts the same way;
  4. raw per-chart TD spread of the non-packed output (what density allocation will fix).
"""
import sys, time
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import partuv
from partuv.preprocess_utils.partfield_official.run_PF import PFInferenceModel

SC = "/tmp/claude-0/-root-youjiaZhang-PartUV/c580f3d2-b821-4d16-9a61-42de5ae8b19f/scratchpad"
CKPT = "/root/zhaotianhao/PartField/model/model_objaverse.ckpt"
CFG = SC + "/partuv_config.yaml"
MESH = sys.argv[1] if len(sys.argv) > 1 else "/root/youjiaZhang/纹理密度/data/objaverse_6a4e8c22280b46ea.glb"

t0 = time.time()
pf = PFInferenceModel(device="cuda", checkpoint_path=CKPT)
mesh, tf, tree, _ = partuv.preprocess(MESH, pf, SC + "/smoke_out2/", merge_vertices_epsilon=None)
V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32)
t1 = time.time()
final, parts = partuv.pipeline_numpy(V=V, F=F, tree_dict=tree, configPath=CFG, threshold=1.25)
t2 = time.time()
print(f"[mesh] F={len(F)}  [time] preprocess+PF={t1-t0:.1f}s pipeline={t2-t1:.1f}s")
print(f"[result] charts={final.num_components} parts={len(parts)} distortion={final.distortion:.3f}")

# --- correspondence via face centroids (input processed mesh <-> chart meshes)
tree_in = cKDTree(V[F].mean(axis=1))

def match_faces(cV, cF):
    d, idx = tree_in.query(cV[cF].mean(axis=1))
    return d, idx

n_total, n_bad, seen = 0, 0, np.zeros(len(F), bool)
chart_rows = []
for i, c in enumerate(final.components):
    cV, cF, cUV = np.asarray(c.V), np.asarray(c.F), np.asarray(c.UV)
    d, idx = match_faces(cV, cF)
    n_total += len(cF); n_bad += int((d > 1e-8).sum())
    dup = seen[idx].sum(); seen[idx] = True
    e1, e2 = cUV[cF][:, 1] - cUV[cF][:, 0], cUV[cF][:, 2] - cUV[cF][:, 0]
    a2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    v3 = cV[cF]
    a3 = 0.5 * np.linalg.norm(np.cross(v3[:, 1] - v3[:, 0], v3[:, 2] - v3[:, 0]), axis=1)
    chart_rows.append(dict(chart=i, nf=len(cF),
                           scale=float(np.sqrt(a2.sum() / max(a3.sum(), 1e-12))),
                           dup=int(dup)))
cov = seen.sum() / len(F)
print(f"[correspondence] centroid-matched {n_total} chart-faces; mismatch(d>1e-8)={n_bad}; "
      f"coverage of input faces={cov:.4f}")

# --- per-part labels the same way
part_label = np.full(len(F), -1)
for pi, p in enumerate(parts):
    for c in p.components:
        cV, cF = np.asarray(c.V), np.asarray(c.F)
        d, idx = match_faces(cV, cF)
        part_label[idx[d < 1e-8]] = pi
print(f"[parts] labeled faces={float((part_label >= 0).mean()):.4f}  n_parts={len(parts)}  "
      f"part sizes={np.bincount(part_label[part_label >= 0]).tolist()}")

# --- raw TD spread across charts (non-packed output)
s = np.array([r['scale'] for r in chart_rows])
print(f"[raw TD] charts={len(s)} linear-scale max/min={s.max()/max(s.min(),1e-12):.2f} "
      f"CV={s.std()/s.mean():.3f}  (uncontrolled, as expected for non-packed output)")
ok = (n_bad == 0) and cov > 0.999 and (part_label >= 0).mean() > 0.999
print("SMOKE2_PASSED" if ok else "SMOKE2_CHECK_DETAILS")
