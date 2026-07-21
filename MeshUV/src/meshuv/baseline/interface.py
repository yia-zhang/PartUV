# -*- coding: utf-8 -*-
"""Baseline chart generator 接口: PartUV 只是可替换实现之一。

ChartSet 契约(所有下游 density/data/model/notebook 只依赖这些字段):
- vertices (N,3) / faces (F,3): 处理后 mesh
- face_to_chart (F,): 每面 chart id(-1=未覆盖)
- local_uv (F,3,2): 每面角点的 chart 内局部 UV
- covered (F,) bool: 有效覆盖面
- n_charts: chart 数
- source_uv (F,3,2) + source_uv_valid (F,): 与 canonical asset 的对应
coverage 判定使用表面积: covered 面面积占比 >=99% 接受(未覆盖走 mask 排除),
<99% 拒绝 —— 不再用 99.9% face-count 硬门。"""
COVERAGE_MIN_AREA = 0.99


class ChartGenerator:
    name = "base"

    def generate(self, canonical_asset, workdir):
        """canonical_asset: canonicalizer 输出 dict. 返回 ChartSet dict
        或 dict(status=..., reason=...) 表示失败。"""
        raise NotImplementedError
