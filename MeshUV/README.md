# MeshUV — TD Allocation Student 项目

从 `MeshUV-TD-PseudoGT-MVP-v0` 数据集开始的独立项目目录。teacher（PartUV +
luminance-std + TD 分配 + xatlas + texel-center baker）保留在
`/root/youjiaZhang/PartUV/code/tdlib`，本项目**不复制、不分叉、不重新实现**其中
任何逻辑；唯一调用入口是 `src/meshuv/teacher_adapter.py`（集中管理
`PARTUV_ROOT`、teacher version、冻结 β、protocol hash、code hash）。

## 数据集语义（冻结）

```json
{
  "label_type": "partuv_td_allocation_pseudo_gt",
  "quality_scope": "td_allocation_only",
  "artist_gt": false,
  "local_uv_refinement": "none",
  "seam_topology_target": false,
  "packed_uv_regression_target": false
}
```

只监督：给定冻结 PartUV charts 后，每个 chart 应获得多少相对 texel budget。
canonical targets = `chart_demand_normalized / chart_target_area_fraction /
chart_log_density_ratio`（`chart_target_scale` 亦为 target）。
`face_content_score / chart_content_score` 是 teacher diagnostics，
**禁止作为 Student-v0 输入**（标签泄漏）；`target_packed_uv` 等打包产物仅 QA。

## 可复现入口

```bash
# 构建 MVP 数据集(候选清单 -> teacher -> QA -> accepted, 可断点续跑)
python scripts/build_dataset_mvp.py --config configs/dataset_mvp_v0.yaml

# 验证完整性(100% 门 + 回读 + hash)
python scripts/validate_dataset.py --dataset datasets/processed/MeshUV-TD-PseudoGT-MVP-v0

# 生成拆分(192/32/32, geometry/content hash 去重分组)
python scripts/make_splits.py --dataset datasets/processed/MeshUV-TD-PseudoGT-MVP-v0

# 训练读取 smoke test(8-object overfit)
python scripts/smoke_overfit.py --config configs/student_v0.yaml
```

全部脚本可从任意工作目录启动（内部以本文件所在目录为项目根定位）。

## 大文件不入 git

`datasets/cache`、`datasets/processed`、`runs`、checkpoint、渲染图均被
`.gitignore` 排除；configs / manifests / schema / splits / dataset card /
代码 / 小型报告保留。
