# -*- coding: utf-8 -*-
"""Smoke test: can we consume PartUV output for a texel-density layer?

Checks, on one small textured GLB:
  1. preprocess (PartField tree) + pipeline_numpy run end-to-end on this machine;
  2. face correspondence: do per-Component `faces` indices tile the processed mesh's F
     exactly (needed to attach per-face content weights / rebake)?
  3. per-part grouping available (individual_parts);
  4. compute per-chart raw TD stats from the non-packed output (the quantity our
     density layer will control).
"""
import sys, time, json
import numpy as np

CKPT = "/root/zhaotianhao/PartField/model/model_objaverse.ckpt"
CFG = "/tmp/claude-0/-root-youjiaZhang-PartUV/c580f3d2-b821-4d16-9a61-42de5ae8b19f/scratchpad/partuv_config.yaml"
MESH = sys.argv[1] if len(sys.argv) > 1 else "/root/youjiaZhang/纹理密度/data/synthetic_freq.glb"
OUT = "/tmp/claude-0/-root-youjiaZhang-PartUV/c580f3d2-b821-4d16-9a61-42de5ae8b19f/scratchpad/smoke_out"

import trimesh
import partuv
from partuv.preprocess_utils.partfield_official.run_PF import PFInferenceModel

orig = trimesh.load(MESH, force='mesh')
print(f"[orig] V={orig.vertices.shape} F={orig.faces.shape} has_uv={orig.visual.kind}")

t0 = time.time()
pf = PFInferenceModel(device="cuda", checkpoint_path=CKPT)
print(f"[pf] model loaded in {time.time()-t0:.1f}s")

t0 = time.time()
mesh, tree_file, tree_dict, times = partuv.preprocess(
    MESH, pf, OUT, save_tree_file=False, save_processed_mesh=False,
    sample_on_faces=10, sample_batch_size=100_000, merge_vertices_epsilon=None)
V, F = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)
print(f"[preprocess] {time.time()-t0:.1f}s  processed V={V.shape} F={F.shape}  "
      f"faces_changed_vs_orig={len(F) != len(orig.faces)}")

t0 = time.time()
final, parts = partuv.pipeline_numpy(V=V, F=F, tree_dict=tree_dict,
                                     configPath=CFG, threshold=1.25)
print(f"[pipeline] {time.time()-t0:.1f}s  charts={final.num_components} "
      f"parts={len(parts)} distortion={final.distortion:.4f}")

# --- correspondence check: union of component faces == all face indices exactly once
all_idx = np.concatenate([np.asarray(c.faces) for c in final.components])
ok = (len(all_idx) == len(F)) and (np.sort(all_idx) == np.arange(len(F))).all()
print(f"[correspondence] union(component.faces) covers F exactly once: {ok}")

# --- per-chart raw TD (non-packed; each chart lives in its own unit square)
def tri_area2(uv):   # uv: (m,3,2)
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    return 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
def tri_area3(v):    # v: (m,3,3)
    return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

rows = []
for i, c in enumerate(final.components):
    cV, cF, cUV = np.asarray(c.V), np.asarray(c.F), np.asarray(c.UV)
    a2, a3 = tri_area2(cUV[cF]), tri_area3(cV[cF])
    scale = np.sqrt(a2.sum() / max(a3.sum(), 1e-12))  # chart's UV-per-3D linear scale
    rows.append(dict(chart=i, n_faces=int(len(cF)), scale=float(scale),
                     distortion=float(c.distortion)))
scales = np.array([r["scale"] for r in rows])
print(f"[raw TD] chart linear scales: min={scales.min():.3g} max={scales.max():.3g} "
      f"max/min={scales.max()/max(scales.min(),1e-12):.2f} CV={scales.std()/scales.mean():.3f}")
print(f"[raw TD] (>1.25x spread confirms charts need density normalization before packing)")
print(json.dumps(rows[:8], indent=1))
print("SMOKE_TEST_PASSED" if ok else "SMOKE_TEST_FAILED_CORRESPONDENCE")
