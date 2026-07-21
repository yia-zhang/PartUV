# PartUV + Texel Density 下一阶段执行任务书

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-14
- Verification Status: ANALYZED（基于现有 notebook 代码与已保存输出；尚未在本环境重跑）
- Version Label: partuv_td_claude_code_execution_v1

---

## 给 Claude Code 的执行指令

请基于现有 `partuv_td_test.ipynb` 继续开发，但不要直接跳到“修改 PartUV 分解”。严格按本文的 P0 → P1 → P2 gate 顺序推进。

本轮首先完成 **P0：实验完整性与因果诊断**。P0 未通过前，不要声称方法有效，也不要开始大规模修改 PartUV C++ 核心。完成 P0 后提交结果报告，再依据 P1 的直接质量结果决定 P2 应走哪条技术路线。

不要覆盖或破坏现有 notebook；保留其作为 `V0 chart-level scaling baseline`。公共逻辑应逐步抽出为可测试函数，notebook 只承担配置、执行和可视化。

---

## 一、项目目标

当前目标不是训练新模型，而是验证以下功能：

> 给定 mesh、参考表面纹理信号和固定纹理预算，利用 PartUV 产生语义 chart，并根据纹理内容需要重新分配 texel，使最终烘焙纹理在相同预算下比 uniform texel density 保存更多有效细节。

当前 notebook 已经跑通：

```text
mesh
→ PartUV decomposition / ABF
→ chart packing
→ 原纹理 rebake
→ 内容权重 w(f)
→ chart-level scaling
→ single/multi-atlas 演示
```

但它目前只能称为机制原型，不能称为 L2 已验证方法。

---

## 二、当前已知事实

以 notebook 当前保存的鞋模型执行结果为准：

```text
faces                         14,458
parts                         30
charts                        223
face mapping mismatches       0
coverage                      99.96%
未覆盖面积                    0.02%
L1 geometry TD CVw            0.028
mean_A(q²)                    1.000
CVw(TD²/w)                    0.750 → 0.810（恶化）
corr(logTD², logw)            0.516
top-10% face TD gain          ×1.43
h_c > 0.25                    86 / 223 charts
multi-atlas target budget     2,000,000 texels
multi-atlas actual POT budget 1,572,864 texels（少 21.4%）
```

局部鞋标 chart 从约 `82×83 px` 放大到 `207×202 px`，视觉上更清楚；但这是局部机制证据，不是全局重建质量证据。

当前最强病因假设是：鞋的 chart 内需求异质性高，而当前每 chart 只能使用一个缩放倍率。但该因果关系尚未被隔离，因为仍可能受到以下因素影响：

1. luminance std 不是准确的纹理需求；
2. packing 后实际有效预算漂移；
3. 评价使用 face-level `w_f`，而方法只能表达 chart-level 常数；
4. crop bounding box、未加权 quantile 等指标口径存在问题。

---

## 三、全局开发纪律

### 3.1 不允许继续使用的结论

在 P1 直接质量评测通过前，notebook 和报告不得输出：

- “感知无损”；
- “内容感知布局已经提高总体纹理质量”；
- “车轮上效果很好，因此方法已成立”；
- “鞋失效已确定由 chart 异质性造成”；
- “拆 chart 后将普遍适用于所有物体”。

可使用的准确表述：

> 当前机制可以把 texel 转移到高权重 chart；局部区域显示积极结果，但全局质量尚未验证。chart 内异质性是领先病因假设。

### 3.2 结果状态必须机器判定

每次运行必须输出以下四级状态，不能靠人工看图判断：

```text
VALIDITY_GATE     PASS / FAIL
MECHANISM_GATE    PASS / FAIL / NOT_EVALUATED
QUALITY_GATE      PASS / FAIL / NOT_EVALUATED
FINAL_STATUS      INVALID
                  VALID_BUT_NOT_IMPROVED
                  TARGET_MATCH_IMPROVED
                  QUALITY_IMPROVED
```

P0 阶段 `QUALITY_GATE` 必须为 `NOT_EVALUATED`，因为尚无直接重建误差。

若 L2 未改善，产品 fallback 必须返回 L1，而不是输出更差的 L2。

---

## 四、P0：实验完整性与因果诊断（立即执行）

### P0.1 修正预算定义和吸附逻辑

至少同时统计：

\[
B_{raw}=\sum_k W_kH_k
\]

\[
B_{signal}=\left|\bigcup_c M_c\right|
\]

\[
B_{pad}=\left|\bigcup_c\operatorname{dilate}(M_c,g)\right|-B_{signal}
\]

\[
B_{empty}=B_{raw}-B_{signal}-B_{pad}
\]

其中 `M_c` 必须是最终分辨率下的 rasterized chart mask，不能用连续三角形 UV 面积之和冒充像素 union。

把当前 `snap2(round(log2 R))` 拆成两种明确政策：

```text
preserve_at_least:
    选择满足实际预算 >= target 的最小允许规格

hard_cap:
    在实际预算 <= target 的组合中最大化可用 signal texels
```

必须覆盖当前反例：

```text
R_exact = 1081
```

不能再自动向下取 1024 后报告“预算守恒”。如果因 hard cap 选择 1024，必须报告实际缺口。

### P0.2 修正 multi-atlas 预算

禁止每张 atlas 独立四舍五入到 POT 后不重新检查总预算。

当前反例必须成为单元测试：

```text
target = 2,000,000
chosen = 512² + 1024² + 512²
actual = 1,572,864
error  = -21.4%
```

实现联合离散选择：

- 输入：允许的 `(W,H)` 规格、`K_max`、总预算政策；
- 输出：满足政策的规格组合；
- objective 至少包含需求误差、packing waste 和 atlas 数惩罚；
- 若尚未实现可靠 planner，P0/P1 主实验暂时禁用 multi-atlas，只在单 atlas 下验证核心假设。

### P0.3 加入 chart-level projection ceiling

对每个 chart 定义面积加权平均目标：

\[
\bar w_c=
\frac{\sum_{f\in c}A_f w_f}
     {\sum_{f\in c}A_f}
\]

新增三个指标：

\[
e_{face}=\operatorname{RMS}_A
\left[\log\frac{TD_f^2}{w_f}\right]
\]

\[
e_{chart}=\operatorname{RMS}_A
\left[\log\frac{TD_f^2}{\bar w_{c(f)}}\right]
\]

\[
e_{irreducible}=\operatorname{RMS}_A
\left[\log\frac{\bar w_{c(f)}}{w_f}\right]
\]

比较 PartUV+L1 与当前 L2：

- 若 `e_chart` 显著改善、`e_face` 不改善，说明实现按 chart 工作正常，瓶颈确实是 chart 表达能力；
- 若 `e_chart` 也不改善，优先检查 packing、尺度传播或 face/chart 对应，而不是立即拆 chart；
- `e_irreducible` 是当前一-chart-一标量方法的表达上限，应比较车轮与鞋，并在多资产上验证其是否预测失败。

保留原 `CVw(TD²/w)` 和 correlation 作为辅助诊断，但不再作为质量指标。

### P0.4 修正现有指标口径

1. top-10% 内容面必须使用面积加权 quantile；
2. “chart 像素增益”必须统计 rasterized interior mask texels，不能使用 bounding-box 面积；
3. `PartUV原版` 改名为 `PartUV decomposition + L1 scaling baseline`；
4. `cw` 改名或标注为 `luminance_std_heuristic`，不得称为 oracle/frequency；
5. multi-atlas 跨图集密度比较应使用 `TD/q` 或 `TD²/w`，并按 3D 面积加权；
6. 所有图和 JSON 同时报告 raw、signal、padding、empty budget。

### P0.5 修正可靠性问题

1. `FaceMatcher` KD fallback 必须检查全局一对一；增加 `len(unique(matches)) == expected`；
2. 三角形三个角点映射使用 3! 全排列的最小总代价双射，禁止三个独立 `argmin`；
3. `sample_bilinear` 的 UV clamp 不得用于 repeat/UDIM 资产；先识别 wrap mode，不支持时明确路由或拒绝；
4. 未覆盖有效面必须 salvage 或 quarantine，并进入 `VALIDITY_GATE`；
5. 检查 NaN、零面积 UV、翻转、非法 overlap、越界和最终 raster coverage；
6. padding 必须以像素和 mip 规则定义，不能只使用固定 normalized margin；
7. 保存实际依赖版本、随机种子、PartUV 配置、资产 ID 和 git commit。

### P0.6 β=0 一致性测试

必须验证：

```text
BETA = 0
```

时 L2 路径与 L1 在以下量上等价（允许数值/packer 容差）：

- per-chart target scale；
- final TD distribution；
- raw/signal budget；
- packing output 或可解释的全局刚性差异；
- rebake reconstruction。

若 β=0 不退化到 L1，停止后续实验并修复。

### P0.7 P0 gate

`VALIDITY_GATE=PASS` 至少要求：

```text
有效表面：100% 输出或有明确 quarantine 清单
非法 overlap：0
NaN / zero-area UV：0
预算误差：<= 1%，或符合明确的 hard_cap/preserve_at_least 政策
所有预算项可复算
β=0 consistency：PASS
FaceMatcher uniqueness：PASS
```

`MECHANISM_GATE=PASS` 要求：

- `e_chart(L2) < e_chart(L1)`；
- 若 `e_face` 仍失败，报告 `e_irreducible`，不得把它当代码失败；
- 当前鞋模型和已知车轮模型都要运行；
- 至少加入一个 constant/gradient/checker 合成资产或表面信号。

### P0 交付物

1. 更新后的 notebook；
2. 从 notebook 抽出的可测试模块；
3. 自动化测试；
4. `p0_validation_report.md`；
5. 每资产 `metrics.json`；
6. 一张车轮/鞋的 `e_face、e_chart、e_irreducible` 对比图；
7. 一张预算核算表；
8. 明确的 P0 PASS/FAIL 结论。

完成 P0 后暂停，先汇报结果。不要根据单张视觉图直接进入 P2。

---

## 五、P1：建立直接质量尺子与 R-D oracle

仅在 P0 通过后实施。

### P1.1 固定的 reference 与采样域

使用高分辨率原纹理、连续 Texture Function 或高质量表面信号作为 reference。

评价必须在相同的 3D 表面采样点或相同相机渲染上进行，不能直接比较两张 UV atlas 图，因为不同布局的像素没有位置对应关系。

基础表面误差：

\[
E_{surface}=
\mathbb E_{x\sim S}
\left[\lVert F_{ref}(x)-\hat F_U(x)\rVert^2\right]
\]

需要统一：

- linear RGB / sRGB 处理；
- bilinear、mipmap 和 padding；
- surface sample set；
- 相机、光照与背景；
- 可见性和语义权重是否开启。

### P1.2 两套公平协议

#### 协议 A：分配机制隔离实验

- 固定 chart decomposition；
- 固定 packer；
- 固定有效 surface texels；
- 只改变 chart allocation；
- 用于证明“预算分配是否更好”。

#### 协议 B：端到端产品实验

- 固定 `B_raw` 或实际 memory；
- padding、empty、packing waste 全部计入方法成本；
- 允许 single/multi atlas planner 自行决定布局；
- 用于证明“整个系统是否更好”。

两种协议不能混为一个结果。

### P1.3 必须比较的 baseline

至少包括：

1. 原始资产 UV（仅作为 reference/生产对照）；
2. 官方 PartUV 输出；
3. PartUV + L1 uniform；
4. 当前 luminance-std L2；
5. 同预算随机 chart weight；
6. surface gradient 或 multiscale proxy；
7. chart-level discrete R-D oracle。

### P1.4 Budget–error curve

建议先使用小规模离散点：

```text
0.5M, 1M, 2M, 4M texels
```

对每个 chart 测试若干预算档：

```text
P0/4, P0/2, P0, 2P0, 4P0
```

得到：

\[
E_c(P_c)
\]

以及边际收益：

\[
U_c(P)=-\frac{\Delta E_c}{\Delta P}
\]

用 discrete greedy 或 multiple-choice knapsack 分配总预算。

主结果：

- surface MSE / PSNR；
- budget–error AUC；
- 固定视角 render PSNR/SSIM/LPIPS（次要但必要）；
- 高频/文字区域误差；
- seam、padding、fill、distortion 和 runtime。

### P1.5 P1 决策表

| 结果 | 下一步 |
|---|---|
| R-D oracle 不优于 uniform | 暂停 P2；检查 reference、预算区间、distortion 定义或研究假设 |
| oracle 优于 uniform，但 heuristic 不优 | 主要问题是内容信号，先改 proxy/预测器 |
| chart-level target 成功、face-level/R-D 失败，且 `e_irreducible` 高 | chart 粒度是主要瓶颈，进入 P2 |
| allocation-only 成功、端到端失败 | packing/padding/atlas planner 是主要瓶颈 |
| wheel 成功、shoe 失败，且失败随 `e_irreducible` 增长 | 支持 chart 内异质性病因，但需要多资产统计 |

`QUALITY_GATE=PASS` 的正式标准：相同预算协议下，ours 的 error–budget AUC 低于 L1。多资产阶段使用 paired comparison 与 bootstrap confidence interval；单资产结果只能标为 case study。

---

## 六、P2：处理 chart 内需求异质性

仅当 P1 证明“oracle 可以提升，chart 粒度是瓶颈”后实施。

不要预设“沿 PartUV tree 把 logo 切出来”一定正确。PartUV hierarchy 主要来自 part/geometry feature，不保证与纹理内容边界重合。

至少比较以下三种候选：

### 方案 A：PartUV hierarchy split

适用于内容差异与 part/geometry 边界一致的情况。

- `h_c` 或 `e_irreducible` 仅作为候选筛选；
- 枚举父节点 keep 与子节点 split；
- 重新 parameterize、pack、测量真实成本。

### 方案 B：content-aware surface graph cut

适用于 logo、文字等内容边界不在 PartUV tree 中的情况。

- 在 mesh adjacency graph 上使用 demand/feature discontinuity；
- 要求连通、最小面积、边界平滑；
- 避免沿细碎三角形产生锯齿 seam。

### 方案 C：chart 内 signal-specialized parameterization

适用于不希望增加 seam 的连续表面。

- 在 chart 内使用受限 metric/IMT/目标 Jacobian 调整局部面积；
- 显式约束 angular distortion、anisotropy 和 artist editability；
- 不允许为了局部 logo 产生不可编辑的极端 UV 形变。

生产中特殊的 decal/trim/shared texture 应走 preservation route，不强制 unique split。

### P2 keep/split objective

\[
C_{keep}=E_n(P)+\lambda_DD_n
\]

\[
C_{split}=\min_{P_l+P_r+G\le P}
E_l(P_l)+E_r(P_r)
+\lambda_D(D_l+D_r)
+\lambda_SL_{new}
+\lambda_GG
\]

仅在：

\[
C_{split}<C_{keep}
\]

时拆分。

第一版允许 greedy proof of concept；不要在没有收益证据时直接重写完整 PartUV 搜索。之后再将候选成本放进 tree dynamic programming。

P2 成功只能表述为：

> 方法从 chart 内需求较均匀的资产，扩展到能处理一部分 chart 内异质资产。

不得表述为“解决所有物体”。

---

## 七、Phase 2：接图片 + 白模生成管线

在 P0–P2 形成稳定第二阶段后，再接 UniTEX 或其他生成器。

第二阶段接口应抽象为：

```python
SurfaceSignalProvider.query(surface_points, channel, lod)
```

不要把 UV 优化逻辑硬绑定到 UniTEX 内部实现。

端到端实验必须分开：

1. `rebake`：从同一个连续/高分辨率 reference 烘焙到不同 UV；
2. `regeneration/refinement`：在最终 UV 或表面表示上重新生成。

只有分开，才能判断收益来自存储表示还是生成模型本身。

---

## 八、建议的代码结构

请根据现有仓库实际结构调整路径，但至少拆出以下职责：

```text
budget / atlas planner
    raw/signal/padding/empty accounting
    discrete resolution selection

metrics
    face/chart target match
    rasterized texel counts
    distortion/coverage/overlap

gates
    validity/mechanism/quality status

surface signal
    existing texture adapter
    future Texture Function adapter

rebake
    common rasterizer/filter/padding pipeline

rd oracle
    rate sweep
    marginal utility allocation

chart refinement
    hierarchy candidates
    graph-cut candidates
    keep/split cost

tests
    budget, β=0, matcher uniqueness, synthetic signals
```

避免 single-atlas 与 multi-atlas 各自复制一套 rebake 代码；它们应共享同一个 chart/atlas 数据模型和 rasterizer。

---

## 九、最终汇报格式

完成每个阶段后，请按以下格式汇报：

```text
1. 修改了什么
2. 哪些文件发生变化
3. 运行了哪些资产和测试
4. VALIDITY / MECHANISM / QUALITY gate
5. 关键指标表
6. 与旧结果的差异
7. 仍然失败的案例
8. 是否满足进入下一阶段的条件
9. 所有输出路径
```

若 gate 未通过，请停止并报告真实结果，不要为了符合预期而调整阈值或只展示成功资产。

---

## 十、本轮立即执行范围

本轮只完成 P0，具体顺序：

1. 建立测试与结果 gate；
2. 修复 single/multi-atlas 预算吸附；
3. 加入 rasterized budget accounting；
4. 加入 `e_face / e_chart / e_irreducible`；
5. 修正指标、FaceMatcher uniqueness 与角点双射；
6. 跑鞋、车轮和至少一个合成信号；
7. 生成 `p0_validation_report.md`；
8. 根据 gate 结果建议是否进入 P1。

P0 完成前，不实施 P2，不修改论文结论为“方法已有效”，不接 UniTEX。

