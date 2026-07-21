# -*- coding: utf-8 -*-
"""MeshUV 单元测试(β 无关). 运行: python tests/run_tests.py"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from meshuv.data.schema import (FORBIDDEN_INPUTS, MODEL_INPUTS, REQUIRED_NPZ,
                                TEACHER_DIAGNOSTICS, TRAINING_TARGETS)
from meshuv.data.splits import make_splits
from meshuv.data.collate import collate_charts
from meshuv.data.dataset import MeshUVTDDataset

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ---- schema 角色分离 ----
check("禁用输入与 model_inputs 不相交",
      not (FORBIDDEN_INPUTS & set(MODEL_INPUTS)))
check("targets/diagnostics 不进 model_inputs",
      not (set(TRAINING_TARGETS) & set(MODEL_INPUTS))
      and not (set(TEACHER_DIAGNOSTICS) & set(MODEL_INPUTS)))
check("demand/scale 被列为禁用输入",
      {"chart_demand_normalized", "chart_target_scale"} <= FORBIDDEN_INPUTS)

# ---- splits 去重分组 ----
recs = [dict(object_id=f"o{i}", geometry_hash=f"g{i}",
             content_phash=f"{i:016x}") for i in range(20)]
recs[5]["geometry_hash"] = recs[3]["geometry_hash"]      # 3/5 同几何
recs[9]["content_phash"] = recs[8]["content_phash"]      # 8/9 同内容
splits, info = make_splits(recs, sizes=dict(train=12, val=4, test=4))
loc = {o: s for s, os_ in splits.items() for o in os_}
check("同几何 hash 不跨 split", loc["o3"] == loc["o5"])
check("同内容 phash 不跨 split", loc["o8"] == loc["o9"])
check("全部对象被分配", sum(len(v) for v in splits.values()) == 20)

# ---- 合成样本: loader + collate + 防泄漏 ----
with tempfile.TemporaryDirectory() as td:
    oid = "t0"
    d = f"{td}/objects/{oid}"
    os.makedirs(d)
    nF, nC = 12, 3
    arrays = dict(
        vertices=np.zeros((8, 3), np.float32),
        faces=np.zeros((nF, 3), np.int64),
        face_ids=np.arange(nF), face_to_chart=np.arange(nF) % nC,
        chart_ids=np.arange(nC),
        local_uv_before_td=np.zeros((nF, 3, 2), np.float32),
        train_face_mask=np.ones(nF, bool),
        source_uv=np.zeros((nF, 3, 2), np.float32),
        source_uv_valid=np.ones(nF, bool),
        chart_surface_area=np.ones(nC, np.float32),
        chart_uv_area_before_td=np.ones(nC, np.float32),
        chart_demand_normalized=np.full(nC, 1 / nC, np.float32),
        chart_target_area_fraction=np.full(nC, 1 / nC, np.float32),
        chart_log_density_ratio=np.zeros(nC, np.float32),
        chart_target_scale=np.ones(nC, np.float32),
        chart_valid_mask=np.ones(nC, bool),
        face_content_score=np.zeros(nF, np.float32),
        chart_content_score=np.zeros(nC, np.float32))
    assert set(REQUIRED_NPZ) <= set(arrays)
    np.savez(f"{d}/arrays.npz", **arrays)
    json.dump(dict(files=dict(reference_texture="ref.png"),
                   teacher=dict(beta=0.25)), open(f"{d}/manifest.json", "w"))
    json.dump(dict(eligible=True, quality_status="VALID_NO_OP"),
              open(f"{d}/quality.json", "w"))
    with open(f"{td}/dataset_index.jsonl", "w") as fp:
        fp.write(json.dumps(dict(object_id=oid, sample_dir=f"objects/{oid}")))
    ds = MeshUVTDDataset(td)
    it = ds[0]
    check("loader 角色分组正确",
          set(it["model_inputs"]) == set(MODEL_INPUTS)
          and set(it["training_targets"]) == set(TRAINING_TARGETS))
    check("diagnostics 默认不暴露", "teacher_diagnostics" not in it)
    b = collate_charts([it, it])
    check("collate 形状/掩码正确",
          b["features"].shape == (2 * nC, 3)
          and b["targets"]["chart_log_density_ratio"].shape == (2 * nC,)
          and b["chart_mask"].sum() == 2 * nC
          and (b["object_index"][:nC] == 0).all())

# ---- canonical label 恒等式: 线性纹理密度语义 ----
# chart_log_density_ratio == mean_centered(log(chart_target_scale/uniform_scale)),
# uniform_scale = sqrt(A3/a2)(当前 UV/3D 面积约定); 多 chart + 非均匀 demand。
from meshuv.teacher_adapter import compute_labels, LABEL_SEMANTICS
from meshuv.data.schema import LABEL_SEMANTICS as SCHEMA_LS

rng = np.random.RandomState(0)
nC, nF = 6, 60
gidx = np.array_split(np.arange(nF), nC)
charts = [dict(gidx=g, a2=float(rng.rand() + 0.2),
               UV=np.zeros((3, 2)), F=np.zeros((1, 3), int)) for g in gidx]
fa3 = rng.rand(nF) + 0.05
cw = rng.rand(nF) * 3.0                     # 非均匀内容 -> 非均匀 demand
sel = np.ones(nF, bool)
tc_fake = dict(pu=dict(charts=charts), fa3=fa3, cw=cw, sel=sel)
lab = compute_labels(tc_fake, beta=0.25)
A3 = lab["chart_surface_area"]
a2 = np.array([c["a2"] for c in charts])
uniform_scale = np.sqrt(A3 / a2)
ref = np.log(lab["chart_target_scale"] / uniform_scale)
ref -= ref.mean()
err = np.abs(lab["chart_log_density_ratio"] - ref).max()
check("标签恒等式: log_density_ratio == mc(log(target/uniform scale))",
      err < 1e-9, f"maxerr={err:.2e}")
check("demand 确实非均匀(测试有效性)",
      np.ptp(lab["chart_demand_normalized"]) > 0.01)
check("label_semantics 一致(adapter==schema)",
      LABEL_SEMANTICS == SCHEMA_LS == "linear_texel_density_log_ratio_v1")

# ---- provenance hash: 内容敏感 + 顺序无关 ----
from meshuv.teacher_adapter import _hash_files, teacher_hash_files

with tempfile.TemporaryDirectory() as td:
    os.makedirs(f"{td}/a")
    open(f"{td}/a/x.py", "w").write("v1")
    open(f"{td}/b.yaml", "w").write("cfg")
    h1 = _hash_files(td, ["a/x.py", "b.yaml"])
    h2 = _hash_files(td, ["b.yaml", "a/x.py"])       # 传入顺序无关
    open(f"{td}/a/x.py", "w").write("v2")
    h3 = _hash_files(td, ["a/x.py", "b.yaml"])
    check("hash 与传入顺序无关", h1 == h2)
    check("关键文件内容改变 -> hash 改变", h1 != h3)
hf = teacher_hash_files()
check("hash 覆盖 adapter+config+tdlib(仓库相对路径)",
      "MeshUV/src/meshuv/teacher_adapter.py" in hf
      and "code/notebook/partuv_config.yaml" in hf
      and any(f.startswith("code/tdlib/") for f in hf)
      and hf == sorted(hf), f"n={len(hf)}")
check("hash 不含 frozen YAML(避免循环)",
      not any("teacher_frozen" in f for f in hf))

# ---- 数据源抽象(纯单元, 无网络; 网络测试见 tests/net_tests.py) ----
from meshuv.data_sources import get_source
from meshuv.data_sources.texverse import TexVerse1K
from meshuv.preflight import quick_preflight


class _FakeTexVerse(TexVerse1K):
    """无网络: 伪造两个 shard 的 tree API 响应."""
    CALLS = []

    def _api_json(self, path):
        self.CALLS.append(path)
        if path.endswith("glbs_1k"):
            return [dict(type="directory", path="glbs/glbs_1k/000-000"),
                    dict(type="directory", path="glbs/glbs_1k/000-001")]
        sh = path.rsplit("/", 1)[-1]
        return [dict(type="file", path=f"{path}/u{sh[-1]}{i}_1024.glb")
                for i in range(3)]


with tempfile.TemporaryDirectory() as td:
    srcx = _FakeTexVerse(cache_dir=td)
    c1 = srcx.list_candidates(2)
    check("TexVerse UID/URL 解析", c1[0]["uid"] == "u00"
          and srcx.resolve_url("u00").endswith(
              "TexVerse-1K/resolve/main/glbs/glbs_1k/000-000/u00_1024.glb"))
    check("manifest 增量落盘", os.path.exists(f"{td}/candidates.jsonl"))
    # 恢复: 新实例读 manifest, 不重复枚举已完成 shard
    _FakeTexVerse.CALLS.clear()
    src2 = _FakeTexVerse(cache_dir=td)
    c2 = src2.list_candidates(3)
    check("恢复后不重复枚举已完成 shard",
          [c["uid"] for c in c2] == ["u00", "u01", "u02"]
          and "glbs/glbs_1k/000-000" not in _FakeTexVerse.CALLS)
    # 下载复用与临时文件语义
    dst = src2.local_path("u00")
    open(dst, "wb").write(b"GLBDATA")
    src2._write_status("u00", status="DOWNLOADED", size=7)
    check("已下载文件校验后复用", src2.ensure_local("u00") == dst)
    open(f"{td}/u01.glb.part", "wb").write(b"HALF")   # 半成品
    check("临时下载文件不伪装成完成文件",
          src2.read_status("u01")["status"] == "PENDING"
          and not os.path.exists(src2.local_path("u01")))
    check("数据源注册表按名切换",
          type(get_source("texverse_1k", cache_dir=td)).__name__ == "TexVerse1K")

# preflight 分类(合成 glb)
import trimesh
with tempfile.TemporaryDirectory() as td:
    box = trimesh.creation.box()
    p_noUV = f"{td}/nouv.glb"
    box.export(p_noUV)
    r = quick_preflight(p_noUV)
    check("preflight: 无 UV/贴图 拒绝", not r["ok"]
          and r["reason"].startswith(("NO_UV", "NO_BASECOLOR")))
    open(f"{td}/bad.glb", "wb").write(b"not a glb")
    check("preflight: 不可解析拒绝",
          quick_preflight(f"{td}/bad.glb")["reason"].startswith("UNPARSABLE"))
    from PIL import Image as _Im
    uv = np.random.rand(len(box.vertices), 2)
    box2 = box.copy()
    box2.visual = trimesh.visual.TextureVisuals(
        uv=uv, image=_Im.new("RGB", (64, 64), (120, 30, 30)))
    p_ok = f"{td}/ok.glb"
    box2.export(p_ok)
    r2 = quick_preflight(p_ok)
    check("preflight: 单贴图小网格通过", r2["ok"] and r2["tex_size"] == (64, 64),
          r2["reason"])

# ---- 正式 TexVerse 闭环补充测试 ----
import yaml as _yaml

cfg_tex = _yaml.safe_load(open(os.path.join(ROOT, "configs/dataset_texverse_1k_mvp_v0.yaml")))
check("正式配置 source=texverse_1k(不落 Objaverse)",
      cfg_tex["source"] == "texverse_1k"
      and "objaverse" not in cfg_tex["source_cache"]
      and cfg_tex["target_accepted"] == 256 and cfg_tex["candidate_max"] == 3000)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "bdm", os.path.join(ROOT, "scripts/build_dataset_mvp.py"))
_bdm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bdm)
check("正式 builder 引用 quick_preflight",
      "quick_preflight" in open(os.path.join(
          ROOT, "scripts/build_dataset_mvp.py")).read()
      and hasattr(_bdm, "atomic_json"))
with tempfile.TemporaryDirectory() as td:
    _bdm.atomic_json(f"{td}/x.json", dict(a=1))
    check("原子状态写入(无 .tmp 残留)", json.load(open(f"{td}/x.json")) == dict(a=1)
          and not os.path.exists(f"{td}/x.json.tmp"))
# manifest 中断恢复无重复 UID
with tempfile.TemporaryDirectory() as td:
    srcd = _FakeTexVerse(cache_dir=td)
    srcd.list_candidates(2)                # shard 000-000 完整写入(3 uid)
    srcd2 = _FakeTexVerse(cache_dir=td)
    c = srcd2.list_candidates(6)           # 续跑: 000-000 skip, 只补 000-001
    uids = [r["uid"] for r in c]
    check("manifest 中断恢复无重复 UID", len(uids) == len(set(uids)), uids)
# 无 license_id 的候选可通过 index/validator 语义
check("候选缺 license_id 不阻塞(builder .get 语义)",
      ".get(\"license_id\"" in open(os.path.join(
          ROOT, "scripts/build_dataset_mvp.py")).read()
      and "license 字段缺失" not in open(os.path.join(
          ROOT, "scripts/validate_dataset.py")).read())
# 非零子进程返回码 -> ERROR 分类(builder 源码路径断言)
_src = open(os.path.join(ROOT, "scripts/build_dataset_mvp.py")).read()
check("非零 rc 立即记 ERROR(不等 TIMEOUT)",
      "rc not in (None, 0)" in _src and "stderr_tail" in _src)
# summary 比例语义
check("统计语义字段齐备",
      all(k in _src for k in ("preflight_pass_rate", "structural_qa_pass_rate",
                              "quality_eligibility_rate",
                              "post_preflight_acceptance",
                              "end_to_end_acceptance")))

n_fail = RESULTS.count(False)
print(f"==== {len(RESULTS) - n_fail}/{len(RESULTS)} PASS ====")
sys.exit(1 if n_fail else 0)
