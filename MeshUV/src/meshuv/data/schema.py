# -*- coding: utf-8 -*-
"""MeshUV-TD-PseudoGT-MVP-v0 样本 schema 与字段角色(唯一权威定义).

角色分离(loader 按此分组; FORBIDDEN_INPUTS 不得作为 Student-v0 输入):
- model_inputs: 几何/chart 结构/源 UV/参考纹理路径(特征在下游从原始纹理提取)
- training_targets: canonical allocation 标签
- teacher_diagnostics: 内容分数(标签泄漏源, 仅诊断)
- qa_artifacts: 质量检查产物引用
"""
SCHEMA_VERSION = "meshuv_td_mvp_v0"
LABEL_TYPE = "partuv_td_allocation_pseudo_gt"

# chart_log_density_ratio = mean-centered log LINEAR texel-density ratio
# (0.5*log(demand_share/surface_area_share); 线性密度=texels/单位长度,
#  非面积分配比 —— 语义版本随 protocol hash 冻结)
LABEL_SEMANTICS = "linear_texel_density_log_ratio_v1"

SEMANTICS = dict(
    label_type=LABEL_TYPE,
    label_semantics=LABEL_SEMANTICS,
    quality_scope="td_allocation_only",
    artist_gt=False,
    local_uv_refinement="none",
    seam_topology_target=False,
    packed_uv_regression_target=False)

MODEL_INPUTS = [
    "vertices", "faces", "face_ids",
    "face_to_chart", "chart_ids",
    "local_uv_before_td", "train_face_mask",
    "source_uv", "source_uv_valid",
    "chart_surface_area", "chart_uv_area_before_td",
]
TRAINING_TARGETS = [
    "chart_demand_normalized",
    "chart_target_area_fraction",
    "chart_log_density_ratio",
    "chart_target_scale",
    "chart_valid_mask",
]
TEACHER_DIAGNOSTICS = ["face_content_score", "chart_content_score"]
QA_ARTIFACTS = ["quality.json"]

# 不得暴露为 Student-v0 输入(demand/scale=target; content score=诊断/泄漏)
FORBIDDEN_INPUTS = {"face_content_score", "chart_content_score",
                    "chart_demand_normalized", "chart_target_scale",
                    "target_packed_uv"}

REQUIRED_NPZ = MODEL_INPUTS + TRAINING_TARGETS + TEACHER_DIAGNOSTICS
REQUIRED_FILES = ["manifest.json", "arrays.npz", "quality.json"]
