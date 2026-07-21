# -*- coding: utf-8 -*-
"""PartUV 管线封装: preprocess + pipeline_numpy + 孪生面安全对应 + 原 mesh 对齐."""
import os

import numpy as np
import trimesh

from .geometry import FaceMatcher, tri_area_2d, tri_area_3d

CFG_YAML = "/root/youjiaZhang/PartUV/code/notebook/partuv_config.yaml"
CKPT = "/root/zhaotianhao/PartField/model/model_objaverse.ckpt"

_PF_MODEL = None


def _pf():
    global _PF_MODEL
    if _PF_MODEL is None:
        from partuv.preprocess_utils.partfield_official.run_PF import PFInferenceModel
        _PF_MODEL = PFInferenceModel(device="cuda", checkpoint_path=CKPT)
    return _PF_MODEL


def run_partuv(mesh_path, out_dir, threshold=1.25):
    """返回 dict: V,F,charts,face_chart,covered,area(A_3D per face),match_report."""
    import partuv
    os.makedirs(out_dir, exist_ok=True)
    mesh, _, tree_dict, _ = partuv.preprocess(
        mesh_path, _pf(), out_dir if out_dir.endswith("/") else out_dir + "/",
        save_tree_file=False, save_processed_mesh=False,
        sample_on_faces=10, sample_batch_size=100_000, merge_vertices_epsilon=None)
    V = np.asarray(mesh.vertices, float)
    F = np.asarray(mesh.faces, np.int32)
    final, parts = partuv.pipeline_numpy(
        V=V, F=F, tree_dict=tree_dict, configPath=CFG_YAML, threshold=threshold)

    fm = FaceMatcher(V, F)
    charts_VF, comps = [], []
    for pi, p in enumerate(parts):
        for c in p.components:
            cV, cF, cUV = np.asarray(c.V), np.asarray(c.F), np.asarray(c.UV)
            charts_VF.append((cV, cF))
            comps.append((pi, cV, cF, cUV, float(c.distortion)))
    gidxs, rep = fm.match_charts(charts_VF)

    charts, face_chart = [], np.full(len(F), -1)
    for (pi, cV, cF, cUV, dist), g in zip(comps, gidxs):
        face_chart[g] = len(charts)
        charts.append(dict(part=pi, V=cV, F=cF, UV=cUV, gidx=g,
                           a2=float(tri_area_2d(cUV[cF]).sum()),
                           distortion=dist))
    covered = face_chart >= 0
    return dict(V=V, F=F, charts=charts, face_chart=face_chart, covered=covered,
                area=tri_area_3d(V[F]), n_parts=len(parts),
                match_report=rep, mesh_scale=float(np.linalg.norm(V.max(0) - V.min(0))))


def tile_normalized_uv(uv, tol=0.01):
    """整数 tile 平移归一(输入读取层): 部分导出器把整套 UV 放在偏移的整数 tile
    (如 v∈[-1,0], REPEAT 采样下与 [0,1] 等价)。若整体平移整数后能落入 [0,1](±tol)
    则平移; 否则原样返回(真正跨 tile 的 tiled/UDIM 由支持性检查拒绝)."""
    uv = np.asarray(uv, float)
    if not np.isfinite(uv).all():
        return uv
    shift = np.floor(uv.min(axis=0) + tol)
    uvn = uv - shift
    if uvn.min() >= -tol and uvn.max() <= 1 + tol:
        return uvn
    return uv


def load_reference(mesh_path, V, F, mesh_scale):
    """原 mesh(世界坐标) + 纹理 + processed->orig 面映射 (孪生面安全)."""
    orig = trimesh.load(mesh_path, force="mesh")
    uv0 = getattr(orig.visual, "uv", None)
    if uv0 is not None and len(np.atleast_1d(uv0)):
        uv0 = tile_normalized_uv(uv0)
    texA = None
    try:
        mat = orig.visual.material
        img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        if img is not None:
            texA = np.asarray(img.convert("RGB"), float) / 255.0
    except Exception:
        pass
    if uv0 is None or texA is None:
        return dict(has_tex=False)
    Vo = np.asarray(orig.vertices, float)
    Fo = np.asarray(orig.faces)
    co = (Vo.max(0) + Vo.min(0)) / 2
    cp = (V.max(0) + V.min(0)) / 2
    s_align = (np.linalg.norm(V.max(0) - V.min(0))
               / max(np.linalg.norm(Vo.max(0) - Vo.min(0)), 1e-12))
    Vo_al = (Vo - co) * s_align + cp
    fmo = FaceMatcher(Vo_al, Fo, scale=mesh_scale)
    f2o = fmo.match(V, F)
    d = np.linalg.norm(V[F].mean(axis=1) - fmo.cent[f2o], axis=1)
    ok_map = d < 1e-4 * mesh_scale
    return dict(has_tex=True, texA=texA, uv0=np.asarray(uv0, float),
                Vo=Vo, Fo=Fo, f2o=f2o, ok_map=ok_map, s_align=s_align)
