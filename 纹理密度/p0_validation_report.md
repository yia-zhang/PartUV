# P0 验证报告（实验完整性与因果诊断）

- 执行依据：`交给ClaudeCode_PartUV_TD下一阶段执行任务书.md` §四 / §十
- 执行日期：2026-07-14
- 环境：conda `geomae`；PartField ckpt `/root/zhaotianhao/PartField/model/model_objaverse.ckpt`；
  PartUV pip v0.1.2；τ=1.25；β=0.4, z_max=2.5, q∈[0.5,2.83]；gutter=4px；seed=0
- 结论先行：**P0 PASS（含 2026-07-14 勘误）。在当前固定 charts、当前权重与
  目标匹配指标下，chart 内异质性是主要残差来源（e_face≈within，交叉项≈0）；
  实现本身按 chart 正常工作（e_chart 三资产均大幅下降）。**
  纹理质量层面的病因仍待 P1 的直接重建误差验证。建议进入 P1。
- 勘误记录：①删除"因果隔离确认/100%"表述；②不再称 e_irreducible 为已验证的
  失效预测器（它目前只是与观察排序一致的候选预测量，待 P1 用质量数据检验）；
  ③指标重命名与补全——原 e_irreducible（相对交付目标 w̄_arith）更名
  **within_chart_heterogeneity**，新增 **e_irreducible_log**（log 空间逐 chart
  最优尺度=真表达下界）与 **cross_term**（分解恒等式
  e_face²=e_chart²+within²+2·cross，实测 |cross|<0.002）；
  ④ TARGET_MATCH_IMPROVED 仅表示 chart target matching 改善，不可解读为质量改善。

---

## 1. 修改了什么

按任务书要求，公共逻辑从 notebook 抽出为可测试模块（notebook 未改动，保留为
V0 chart-level scaling baseline）：

| 任务书条目 | 实现 |
|---|---|
| P0.1 预算定义与吸附 | `tdlib/budget.py`：rasterized mask 版 B_raw/B_signal/B_pad/B_empty；`choose_resolution` 拆分 `preserve_at_least` / `hard_cap` 两政策（覆盖 R_exact=1081 反例：前者取 2048×1024，后者取 1024² 并报告 −10.3% 缺口） |
| P0.2 multi-atlas 预算 | `choose_multi` 联合离散选择（POT 方图+2:1 矩形，K≤3）：2M 反例下 hard_cap 选 512²+1024×512+1024²（−8.2%，旧 −21.4%），preserve_at_least 选 2048×1024（+4.9%）。**P0 主实验按任务书暂禁 multi-atlas** |
| P0.3 投影天花板 | `tdlib/metrics.py`：e_face / e_chart / within_chart_heterogeneity / e_irreducible_log（等交付预算归一：TD² 与 w 均归一 mean_A=1，全局缩放不变）；CVw/corr 降级为辅助诊断 |
| P0.4 指标口径 | top-10% 改面积加权 quantile；`cw` 更名 `luminance_std_heuristic`；基线命名 `PartUV decomposition + L1 scaling baseline` |
| P0.5 可靠性 | `FaceMatcher.match_charts` 全局唯一性断言；`best_corner_perm` 3! 最小代价双射；NaN/零面积 UV/overlap（rasterized）检查入 gate；记录种子与配置 |
| P0.6 β=0 一致性 | `demand_weights(β=0)` 精确返回全 1；驱动内逐资产断言 per-chart scale 与 L1 逐位相等 |
| §3.2 结果 gate | `tdlib/gates.py` 四级机器判定（VALIDITY/MECHANISM/QUALITY/FINAL_STATUS） |

## 2. 变化的文件

新增（未触碰现有 notebook）：
```
code/tdlib/{__init__,geometry,signal,layout,budget,metrics,gates,pipeline}.py
code/tests/run_tests.py                 （24 项自动化测试，含勘误新增的分解恒等式/下界断言）
code/scripts/run_p0.py                  （P0 驱动）
code/notebook/outputs/p0/{shoe_22b822,wheel_92ff6,synthetic_halfchecker}/metrics.json
code/notebook/outputs/p0/{summary.json, p0_e_metrics.png}
纹理密度/p0_validation_report.md        （本文件）
```

## 3. 运行的资产与测试

- 自动化测试：**24/24 PASS**（预算吸附两反例、multi 联合选择、β=0 精确性、
  孪生面唯一性+法线判别、角点 3! 双射、e 指标合成解析验证、raster 预算核算）；
- 资产：鞋 `objaverse_22b822`（已知失效案例）、车轮 `objaverse_92ff6`（已知成功案例）、
  `synthetic_freq` + **半平坦半棋盘合成信号**（任务书要求的合成信号）。

## 4. Gate 状态（三资产一致）

```text
VALIDITY_GATE   PASS   （覆盖率 99.96–100%、overlap=0、NaN=0、预算政策符合、β=0 PASS、matcher 唯一性 PASS）
MECHANISM_GATE  PASS   （e_chart(L2) < e_chart(L1)，三资产全部成立）
QUALITY_GATE    NOT_EVALUATED（按任务书：P0 无直接重建误差）
FINAL_STATUS    TARGET_MATCH_IMPROVED
```

## 5. 关键指标表

| 资产 | e_chart L1→L2 | e_face(L2) | within | e_irr_log(真下界) | cross | h_c>0.25 | 预算(preserve_at_least) |
|---|---|---|---|---|---|---|---|
| 鞋 22b822（14458F/223C） | 0.437 → **0.055** | 0.833 | 0.832 | 0.779 | −0.001 | 86/223 | 0.66M → 1024²（+58.7%） |
| 车轮 92ff6（7012F/126C） | 1.072 → **0.104** | 0.185 | 0.157 | 0.154 | −0.001 | 15/126 | 1.39M → 2048×1024（+50.4%） |
| 合成半棋盘（432F/6C） | 0.874 → **0.000** | 0.622 | 0.622 | 0.571 | +0.000 | 2/6 | 0.49M → 1024×512（+6.6%） |

**读法（P0.3 判定逻辑，勘误后）**：三资产 e_chart 全部大幅下降 ⇒ chart 级分配
实现正确；e_face ≈ within 且交叉项≈0 ⇒ 在**当前固定 charts、当前权重和目标匹配
指标下**，残差主要来自 chart 内异质性，packing/尺度传播/对应可排除为主要因素；
within 与真下界 e_irr_log 接近（鞋 0.832/0.779）⇒ 交付目标（算术均值）离 log 最优
不远。e_irr_log 排序（0.154 < 0.571 < 0.779）与观察到的 L2 目标匹配效果排序一致，
**作为候选预测量记录，其对纹理质量失效的预测力留待 P1 检验**。

## 6. 与旧结果的差异

- 旧指标 `CVw(TD²/w)` 在鞋上显示"恶化"（0.749→0.809），曾被解读为"L2 失效"。
  新分解表明该指标把"实现误差"与"表达天花板"混在一起：实现层面 e_chart 0.437→0.056
  是**大幅改善**，恶化感来自不可约分量。CVw/corr 按任务书降级为辅助诊断；
- 病因定位（限定范围）：在目标匹配层面，chart 内需求异质性是主要残差来源且可
  预先量化；**纹理质量层面的病因结论仍待 P1**。

## 7. 仍然失败/未完成的项

- **QUALITY 未评**（by design）：无直接重建误差，P1 建立 budget–error 曲线与 R-D oracle；
- preserve_at_least 在方图/2:1 规格集下预算超配明显（+50~59%）——P1 若纳入成本敏感
  比较需扩充规格集或用 hard_cap+缺口报告；
- multi-atlas planner 已实现并通过单元测试，但主实验按任务书**暂禁**，待 P1 协议 B 启用；
- rebake 层面的 P0.5 项（supersampling、gutter、sRGB/linear、wrap 路由、孤儿面 salvage
  的**执行**——目前是检测+quarantine 报告）留待 P1 重建误差管线时一并实现；
- 单资产结论仍属 case study，多资产统计（paired+bootstrap）在 P1。

## 8. 是否满足进入下一阶段的条件

**满足。** P0.7 的 VALIDITY 全项 PASS；MECHANISM 在失效案例（鞋）、成功案例（车轮）
与合成信号上均 PASS。按任务书 §五，建议进入 **P1：建立直接质量尺子与 chart-level
R-D oracle**；P2 的三条候选路线中，鞋的 within=0.832（e_irr_log=0.779）+ 86 个高 h_c chart
为方案 A/B 提供了现成实验对象，但**在 P1 quality 证据之前不启动 P2**。

## 9. 所有输出路径

```
code/tdlib/                                    可测试模块（8 文件）
code/tests/run_tests.py                        22 项测试（22/22 PASS）
code/scripts/run_p0.py                         驱动
code/notebook/outputs/p0/<asset>/metrics.json  逐资产指标+gate
code/notebook/outputs/p0/summary.json          汇总
code/notebook/outputs/p0/p0_e_metrics.png      e 三指标对比图
纹理密度/p0_validation_report.md               本报告
```
