# P1a 公平性核查报告

- 执行日期：2026-07-14；范围：用户指定六项，未启动信号阶梯/其余冻结项
- 脚本：`code/scripts/run_p1a.py`（`tdlib/rd.py` 扩展：`bake_atlas_masks` /
  `hull_curves` / `ref_gradient_at_samples`）
- 输出：`code/notebook/outputs/p1a/{shoe_22b822,wheel_92ff6}/metrics.json`、`summary.json`

---

## 六项核查逐项结果

**1. 预算核算（B_raw/B_signal/B_pad/B_empty，逐资产×预算×方法）** ✅
全部写入 metrics.json；措辞修正：固定 R 只固定 B_raw，不是固定有效预算——
实测同 R 下各方法 B_signal 确有差异（如车轮 4M 点：L1 sig=0.55M，L2 sig=0.89M，
oracle sig=0.65M；差异来自 chart 缩放改变 shelf 装箱形状）。该差异现已透明报告，
后续比较可按 B_signal 为 x 轴复核（本轮曲线仍按名义 B，另附实测值）。

**2. 采样分离 + hash** ✅
R-D 曲线用 seed=1 采样，质量评价用 seed=2 独立采样；sha1 已存
（鞋 curve=ea8c56f9… / eval=c7209234…；车轮 curve=37c276c5… / eval=995534c5…）。
分离后结论未变，排除了"oracle 在自己的评价集上过拟合"的嫌疑。

**3. reference 饱和检查** ✅
判据：L1 的 B_signal ≥ 原资产被引用纹素（鞋 0.66M / 车轮 1.39M）。
结果：**鞋的 4M 点饱和，剔出主 AUC**；车轮无饱和点。主 AUC 基于非饱和预算点
（鞋 0.5/1/2M，车轮全部四点）。

**4. 贪心递减性前提** ✅（发现显著违规，已处理）
原始 R-D 曲线中：鞋 51 个非单调、65 个非凸（/150 有效 chart）；车轮 21/22（/76）。
已做单调化 + 下凸包预处理，方法更名 **RD_oracle_hull（approximate oracle）**。
凸包版 oracle 在高预算点的表现同步改善（车轮 4M：0.000108→0.000066），
说明此前 4M 异常部分来自曲线噪声/非凸导致的贪心早停。

**5. 结论措辞** ✅
p1_quality_report.md 已改为："shoe 首先暴露了内容信号问题；当前实验尚不能排除
chart 粒度也是次级瓶颈。"

**6. 高频区误差（防止平坦区掩盖细节损失）** ✅
子集定义与分配信号无关：reference 纹理自身亮度梯度 top-10% 采样点。

## 核查后的主结果（非饱和预算点，独立评价采样）

| 资产 | 指标 | L1 | L2 heuristic | RD_oracle_hull |
|---|---|---|---|---|
| 鞋 | 主 AUC（全局 MSE） | 0.0007 | 0.0013 ✗ | **0.0004** |
| 鞋 | 高频区 AUC | 0.0070 | **0.0031** | 0.0036 |
| 车轮 | 主 AUC | 0.0017 | 0.0007 ✓ | **0.0005** |
| 车轮 | 高频区 AUC | 0.0146 | 0.0067 | **0.0038** |

`QUALITY_GATE_main`：鞋 FAIL / 车轮 PASS（与 P1 一致，核查未翻转 gate）。

**新信息（第 6 项带来的）**：鞋上 heuristic 的高频区误差其实**优于 L1 2.3 倍**
（0.0031 vs 0.0070）——它确实保住了细节区；其全局 FAIL 来自**对中低频区域的
过度抽血**（把"看似平坦实则有内容"的区域降得过狠）。这把"信号问题"的定位又
细化了一步：不是"没找到细节"，而是**对非细节区的需求低估**。surface-domain
gradient（下一轮）恰好主要改善这一点。

## 结论与下一步

- 六项核查全部完成；主结论在更严格协议下保持：**oracle(hull) 双资产优于 uniform，
  heuristic 仅车轮优于 uniform**；shoe 首先暴露内容信号问题，chart 粒度作为
  次级瓶颈的可能性保留待验。
- 下一轮（已获批范围）：**仅实现 surface-domain gradient 信号**，与 L1 /
  luminance-std / RD_oracle_hull 同协议比较后暂停汇报。
- 记录：GT-v0 的 teacher label 可直接采用 R-D oracle 的 per-chart 分配
  （E_c 曲线 + 选档），无需先训练信号预测器。
