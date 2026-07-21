# MeshUV Clean V1

数据 → 模型 → 可视化最小闭环。核心: baseline charts → texture-demand
estimation → chart texel allocation(β=0.25, 线性 TD 语义)。PartUV 只是可替换
的 baseline chart generator(src/meshuv/baseline)。

```bash
# 数据构建(TexVerse-1K, UID 断点, 4 workers)
python MeshUV/scripts/build_dataset.py --n-candidates 200 --target 20
# 8-object overfit 验收
python MeshUV/scripts/overfit.py
# 测试
python MeshUV/tests/test_canonicalizer.py && python MeshUV/tests/test_density.py \
  && python MeshUV/tests/test_data_model.py
```

环境: geomae; `PARTUV_ROOT` 指向 teacher checkout(默认
/root/youjiaZhang/PartUV/code); `MESHUV_DATA_ROOT` 覆盖数据根。
Notebooks: 01_data_browser / 02_uv_comparison(Gold QA) / 03_student_overfit。
