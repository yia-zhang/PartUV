# -*- coding: utf-8 -*-
"""Clean V1 最小样本 schema(每 object 一目录: arrays.npz + manifest.json +
basecolor.png + status.json)。诊断字段显式标记, 禁作 Student 输入。"""
SCHEMA_VERSION = "meshuv_clean_v1"
LABEL_SEMANTICS = "linear_texel_density_log_ratio_v1"

CORE = ["vertices", "faces", "face_to_chart", "local_uv", "source_uv",
        "source_uv_valid", "train_face_mask", "face_area", "face_source"]
TARGETS = ["chart_surface_area", "chart_target_area_fraction",
           "chart_log_density_ratio", "chart_valid_mask"]
DIAGNOSTICS = ["face_content_score", "chart_content_score"]   # 禁作 Student 输入
FORBIDDEN_INPUTS = set(DIAGNOSTICS) | {"chart_target_area_fraction",
                                       "chart_log_density_ratio"}
REQUIRED_NPZ = CORE + TARGETS + DIAGNOSTICS
# hard rejection 仅限: UNPARSABLE / NO_USABLE_RGB_UV / PARTUV_FAILED /
# NONFINITE / COVERAGE_REJECTED; 其余记 warning
