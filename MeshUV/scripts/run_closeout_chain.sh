#!/bin/bash
# 固定收口顺序(抗会话断连: 每步幂等, 重跑本脚本即续)
set -x
P=/root/miniconda3/envs/geomae/bin/python
M=/root/youjiaZhang/PartUV-clean-v1/MeshUV
until grep -q "BUILD: DONE" $M/datasets/build256_clean.log; do sleep 120; done
$P $M/scripts/audit_clean_256.py && \
$P $M/scripts/teacher_diff.py && \
$P $M/scripts/closeout_migrate.py && \
$P $M/scripts/build_dataset.py --n-candidates 2000 --target 256 && \
$P $M/scripts/audit_clean_256.py && \
CUDA_VISIBLE_DEVICES=7 $P $M/scripts/overfit.py && \
CUDA_VISIBLE_DEVICES=7 $P $M/scripts/sanity_256.py && \
CUDA_VISIBLE_DEVICES=7 $P $M/scripts/gold_closeout.py && \
$P $M/tests/test_notebooks_smoke.py && \
echo CHAIN_DONE || echo CHAIN_FAIL
