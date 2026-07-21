# MeshUV Notebooks

只读可视化入口。所有逻辑在 `src/meshuv/visualization/`, notebook 仅调用公开函数;
不触发 Teacher 重算, 不修改数据集。

## 环境与启动
```bash
pip install -e /root/youjiaZhang/PartUV/MeshUV   # 或已装 geomae 环境
cd /root/youjiaZhang/PartUV/MeshUV/notebooks
jupyter lab
```

## 数据路径配置
每本 notebook 首个 cell 的 `DATASET_ROOT`(相对 MeshUV 根):
- 正式: `datasets/processed/MeshUV-TD-PseudoGT-MVP-v0`
- pilot: `datasets/pilot/TexVerse-1K-16`
- 未来 model run: 指向 `runs/...` 下的评测输出

## 约定
- 提交前清空全部 cell outputs(纹理/二进制不入 Git)
- `10_student_overfit` / `11_student_evaluation` 等待 Student-v0 输出格式确定后实现
