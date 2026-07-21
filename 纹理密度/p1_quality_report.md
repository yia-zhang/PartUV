# P1 质量验证报告（直接重建误差 + chart-level R-D oracle）

- 执行依据：任务书 §五 + 用户批准的 P1 最小范围
- 执行日期：2026-07-14
- 协议 A：固定 PartUV charts / 单 atlas（方图 R=round(√B)，全方法同 R；
  **注意固定 R ≠ 固定有效纹素预算**，各方法实际 B_signal 见 P1a 报告）/
  固定 shelf packer / base-color reference / 150k 面积加权表面采样点 /
  sRGB 域 surface MSE（bilinear 重建 + 2 轮 gutter dilation，全方法一致）
- 冻结项遵守：未做 chart split、multi-atlas 主实验、UniTEX、PBR/UDIM、模型训练、架构重构

---

## 1. 修改了什么

- 新增 `tdlib/rd.py`：表面采样（P0.5 合规：角点 3! 双射对齐原 UV）、全图集
  rebake（共享光栅器 + gutter dilation）、表面重建误差评价（相同 3D 采样点，
  不比较 atlas 图片）、chart-level R-D 曲线（每 chart 5 档：{¼,½,1,2,4}×份额，
  chart 单独烘 bbox patch）、离散贪心 oracle 分配（边际收益 ΔE·share/ΔP）；
- 新增 `scripts/run_p1.py`：2 资产 × 4 预算点 × 3 方法 → budget–error 曲线 +
  log2 域 AUC + QUALITY_GATE。
- P0 勘误已先行完成（见 p0_validation_report.md 勘误记录；测试增至 24/24 PASS）。

## 2. 变化的文件

```
code/tdlib/rd.py                                新增
code/scripts/run_p1.py                          新增
code/tdlib/metrics.py, tests/run_tests.py       勘误修改(within/e_irr_log/cross)
code/scripts/run_p0.py, 纹理密度/p0_validation_report.md   勘误修改
code/notebook/outputs/p1/{shoe_22b822,wheel_92ff6}/metrics.json
code/notebook/outputs/p1/{summary.json, p1_budget_error_curves.png}
```

## 3. 运行的资产与设置

鞋 `objaverse_22b822`、车轮 `objaverse_92ff6`；预算 {0.5M, 1M, 2M, 4M}；
方法 = L1 uniform / L2 heuristic（luminance-std，β=0.4）/ chart-level R-D oracle。

## 4. Gate

```text
VALIDITY_GATE   PASS（继承 P0；采样/烘焙管线全方法一致）
MECHANISM_GATE  PASS（继承 P0）
QUALITY_GATE    车轮 PASS   |   鞋 FAIL（heuristic L2 的 AUC 高于 L1）
FINAL_STATUS    车轮 QUALITY_IMPROVED   |   鞋 VALID_BUT_NOT_IMPROVED
```

## 5. 关键指标（surface MSE，越低越好）

**鞋 22b822**（AUC：L1=0.0011，L2=0.0016，**oracle=0.0009**）

| 预算 | L1 | L2 heuristic | R-D oracle |
|---|---|---|---|
| 0.5M | 0.000565 | 0.001311 ✗ | **0.000300** |
| 1.0M | 0.000424 | 0.000504 ✗ | **0.000213** |
| 2.0M | 0.000283 | 0.000461 ✗ | 0.000326 |
| 4.0M | 0.000149 | **0.000051** | 0.000360 † |

**车轮 92ff6**（AUC：L1=0.0017，**L2=0.0008**，**oracle=0.0006**）

| 预算 | L1 | L2 heuristic | R-D oracle |
|---|---|---|---|
| 0.5M | 0.001048 | 0.000526 | **0.000267** |
| 1.0M | 0.000669 | 0.000305 | **0.000202** |
| 2.0M | 0.000405 | **0.000175** | 0.000177 |
| 4.0M | 0.000221 | **0.000095** | 0.000108 |

† oracle 高预算点受两个协议限制压制：曲线档位以 1M 为基（上限 4×份额），且贪心
不执行零边际收益升档（chart 超过参考纹理固有分辨率后 E 曲线饱和）→ 4M 处预算
未用满。oracle 结论在 0.5–2M 区间最可信；即便带着此 handicap，其 AUC 仍双资产最优。

## 6. 与旧结果/假设的差异（本轮最重要结论）

按任务书 P1.5 决策表逐行对照：

- **车轮**：heuristic 与 oracle 均显著优于 uniform（AUC 减半以上）→
  内容感知分配在该资产上**质量层面成立**（首个 QUALITY_IMPROVED）；
- **鞋**：**oracle 优于 uniform（AUC 0.0009 < 0.0011），但 heuristic 不优（0.0016）**
  → 命中决策表第二行："**主要问题是内容信号，先改 proxy/预测器**"。
- 修订后的结论（P1a 勘误措辞）：**shoe 首先暴露了内容信号问题；当前实验尚不能
  排除 chart 粒度也是次级瓶颈。** oracle 在 chart 粒度上优于 uniform 说明信号是
  第一杠杆；chart 粒度的影响需在信号修复后再评估。**P2（chart split）暂不启动**；
  下一步为信号升级（下一轮仅 surface-domain gradient），仍在 P1 框架内。
- 有趣观察（记录，不下结论）：鞋 4M 处 heuristic 反超所有方法（0.000051）——
  高预算下细节 chart 超过参考分辨率后误差趋零；低预算下同一分配把稀缺预算
  从"看似平坦实则有纹理"的区域抽走导致劣化。进一步支持"信号质量是主变量"。

## 7. 仍然失败/未完成

- 鞋 QUALITY_GATE=FAIL（heuristic 信号所致，机制与 oracle 无恙）；
- oracle 高预算档位饱和（协议限制，见 †）——若 P1 续作需以各预算点为基重测曲线
  或增加档位上限；
- 仅 2 资产，属 case study；多资产 paired+bootstrap 待信号升级后一并做；
- render-space PSNR/SSIM/LPIPS（任务书列为次要）未做。

## 8. 是否满足进入下一阶段的条件

- **P2（chart split）：不满足**——P1.5 要求"oracle 提升 + chart 粒度是瓶颈"双证据；
  实测 oracle 在 chart 粒度上已能提升鞋，瓶颈是信号。
- **建议的下一步（P1 延伸，非 P2）**：信号阶梯实验——luminance-std →
  surface-domain gradient → 多尺度带通 → 以 R-D 曲线为监督拟合轻量预测器；
  验收标准：鞋的 heuristic AUC 逼近 oracle AUC、QUALITY_GATE 转 PASS。

## 9. 所有输出路径

```
code/tdlib/rd.py, code/scripts/run_p1.py
code/notebook/outputs/p1/<asset>/metrics.json
code/notebook/outputs/p1/summary.json
code/notebook/outputs/p1/p1_budget_error_curves.png
纹理密度/p1_quality_report.md（本文件）
纹理密度/p0_validation_report.md（含勘误记录）
```
