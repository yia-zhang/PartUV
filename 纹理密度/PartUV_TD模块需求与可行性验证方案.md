# PartUV + Texel Density 模块需求与可行性验证方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-14
- Verification Status: ANALYZED（方案与公开原始资料已核对，实验结果尚待执行）
- Version Label: partuv_td_feasibility_v1

## 一、结论先行

当前阶段不应把任务定义成“给 PartUV 输出的 UV islands 乘一个内容权重”，而应定义成：

> **给定白模、参考表面纹理信号和总纹理预算，利用 PartUV 的语义层级产生候选 chart，自适应决定 chart 拆分、每个 chart 的 texel 数、atlas 数量与分辨率，并在不超过预算的前提下最小化最终纹理烘焙或渲染误差。**

这是一个完整、可验证的独立模块：

\[
(M,S,B,\Omega)
\longrightarrow
(C,U,a,\{W_k,H_k\},T,\mathcal R)
\]

其中：

- \(M\)：白模或 mesh；
- \(S\)：定义在表面上的参考纹理信号，可以来自 UniTEX Texture Function、高分辨率纹理、多视图投影或已有 textured mesh；
- \(B\)：总纹理预算与引擎约束；
- \(\Omega\)：可选的相机、语义、用户重要度及材质设置；
- \(C\)：chart 分解；
- \(U\)：UV 坐标；
- \(a\)：chart 到 atlas 的分配；
- \(W_k,H_k\)：每张 atlas 的分辨率；
- \(T\)：最终烘焙纹理；
- \(\mathcal R\)：预算、失真、覆盖率和质量报告。

功能上，最终系统仍然是“图片 + 白模 → 内容感知、artist-friendly 的 UV 资产”。但从架构上，未来的原生模型最好预测**表面需求、seam/chart 结构和 artist prior**，再由确定性参数化与 packing 层保证无重叠、低失真和预算守恒，而不是让神经网络无约束地直接回归一张 UV 图片。

## 二、为什么 PartUV 适合作为当前底座

PartUV 已经提供了最难替代的三个能力：

1. 对 AI 生成、非流形或不规则 mesh 的鲁棒处理；
2. PartField 层级与语义对齐的 chart 候选；
3. 以较少 chart 满足参数化失真约束的递归搜索。

其公开算法本质上是在 PartField tree 上搜索：在面积失真阈值 \(\tau\) 内最小化 chart 数，并不优化纹理内容或固定 texel 预算。论文也明确说明既可 pack 到单 atlas，也可按语义 part 分到多 atlas。因此，“PartUV 只能输出一张图”不是准确的问题定义；准确的缺口是：**它要求用户给定 atlas 数量，却不根据纹理信号、有效预算、padding 和分辨率上限决定 atlas 方案。** [PartUV 论文](https://www.zhaoningwang.com/PartUV/static/partuv.pdf)、[官方实现](https://github.com/EricWang12/PartUV)

当前官方多 atlas 路径使用 UVPackMaster，并由用户指定 `num_atlas`；默认 Blender 路径主要解决单 atlas packing。这说明你们需要实现的是自有的 atlas planner 与 pixel-space packing contract，而不是把“支持多 atlas”本身作为研究贡献。[PartUV PyPI/官方说明](https://pypi.org/project/partuv/)

## 三、核心问题定义

### 3.1 Texel density 必须与 atlas 分辨率一起定义

对位于 atlas \(k\) 的三角形 \(f\)：

\[
P_f=A_{UV}(f)W_kH_k
\]

\[
\rho_f=\sqrt{\frac{P_f}{A_{3D}(f)}}
\]

所以 UV 归一化面积本身没有绝对意义。一个 chart 在 1024² atlas 中占 10%，和在 4096² atlas 中占 10%，实际 texel 数相差 16 倍。

模块必须同时输出 UV、atlas assignment 和 atlas resolution；只输出 UV 坐标无法说明它“符合 texel density”。

### 3.2 优化目标

建议把完整目标写为：

\[
\min_{C,U,a,\{W_k,H_k\},\{P_c\}}
\sum_c \pi_c E_c(P_c)
+\lambda_D E_{dist}
+\lambda_S L_{seam}
+\lambda_G B_{pad}
+\lambda_K K
\]

约束：

\[
\sum_k W_kH_k\le B_{store}
\]

\[
D(c)\le\tau,\quad \forall c
\]

\[
\operatorname{Packable}(C,U,a,\{W_k,H_k\})=1
\]

其中 \(E_c(P_c)\) 是 chart 在分配 \(P_c\) 个有效 texel 后的表面或渲染重建误差，\(\pi_c\) 是可见性、语义或用户重要度。

## 四、四项需求的具体处理

### 4.1 不确定数量的原始纹理贴图

首先要区分“图片文件数”与“独立空间纹理域数量”。输入资产需自动分类：

| 类型 | 例子 | 预算处理 |
|---|---|---|
| 独立 atlas / 材质域 | 5 个材质各有 1024² base color | 可将 5×1024² 作为一个公平的 raw-budget 参考 |
| 同一 UV 域的 PBR 通道 | base color、normal、roughness | 是不同信号通道，不是 3 倍 UV 空间；内存另计 |
| UDIM | 1001、1002… | 每个 tile 是独立空间预算，可保留或重组 |
| tiled texture | UV 超出 [0,1] 重复采样 | 共享存储，不可按表面展开面积直接计预算 |
| trim sheet / mirrored / stacked UV | 多个表面复用同一纹理区域 | 必须保留共享，或显式选择“unique 展开并接受内存膨胀” |
| lightmap 等第二 UV set | UV0 材质、UV1 光照 | 不应与材质 UV 合并，作为独立任务处理 |

需要记录至少三种预算：

\[
B_{store}=\sum_k W_kH_k
\]

\[
B_{signal}=\sum_c |M_c|
\]

\[
B_{pad}=\sum_c |\operatorname{dilate}(M_c,g)\setminus M_c|
\]

以及：

\[
B_{empty}=B_{store}-B_{signal}-B_{pad}
\]

这里 \(M_c\) 是 chart 的实际像素 mask。连续 UV 面积适合优化；rasterized mask 更适合最终预算验收。

例子：5×1024² 的 raw budget 是 5,242,880 texels。一张等面积正方形的理论边长约为 2289；标准规格中 2048²+1024² 正好等于这个预算，单张 2048² 则少 20%。但如果原五张图的有效占用率很低，一张 2048² 也可能足够。因此，**目标 atlas 数不应复制源贴图数，而应由目标预算、实际占用、padding、最大尺寸和引擎约束共同决定。**

对未来“图片 + 白模”管线而言，本来就不存在原始贴图数量；因此源图数量只是当前 rebake benchmark 的公平性问题，不应进入最终方法的核心定义。

### 4.2 更精确地测量纹理内容细节

亮度标准差只能做 contrast baseline，不能作为最终方法，因为它不能区分相同振幅、不同频率的信号，也会随 mesh 三角化改变。

推荐建立两层体系。

#### 金标准：chart-level rate–distortion oracle

对每个 chart \(c\)，测试若干像素预算：

\[
P_c\in\{P_0/4,P_0/2,P_0,2P_0,4P_0\}
\]

每个预算下都使用与最终运行时一致的采样、bilinear、mipmap、padding 和渲染过程，得到：

\[
E_c(P_c)
\]

边际纹素收益为：

\[
U_c(P)=-\frac{\partial E_c(P)}{\partial P}
\]

离散实现可反复把下一块预算分给 \(-\Delta E/\Delta P\) 最大的 chart。最优附近，各 chart 的最后一单位 texel 应有近似相同的边际收益。

这与经典 Signal-Specialized Parametrization 的出发点一致：在有限纹理样本下，按表面信号变化最小化重建误差；Microsoft UVAtlas 的 IMT 也允许按每三角形信号分配纹理空间。因此，不能把“高频区域获得更多 UV 面积”本身声称为新贡献。[Sander 等，Signal-Specialized Parametrization](https://www.microsoft.com/en-us/research/?p=152595)、[UVAtlas IMT 文档](https://learn.microsoft.com/en-us/windows/win32/direct3d9/using-uvatlas)

你们的新意应落在：AI 生成 mesh、PartUV 语义层级、image-conditioned 纹理函数、adaptive tree cut、实际 padding/multi-atlas 约束和端到端生成验证的组合。

#### 生产 proxy：逼近 oracle，而不是凭直觉组合滤波器

建议依次比较：

1. 亮度标准差；
2. surface-domain gradient；
3. 多尺度 Laplacian / Hessian / wavelet band energy；
4. 语义、文字/边缘 saliency 与相机可见性；
5. 由上述特征预测 \(E_c(P)\) 或边际收益的轻量模型。

计算必须在固定物理尺度或归一化表面邻域上进行，再聚合到 face/chart，不能以每个三角形自身大小作为支持域。这样才能通过重三角化不变性测试。

误差需要按通道定义：

- base color：线性 RGB 或感知加权颜色误差；
- normal：解码归一化后的角度误差；
- roughness：BRDF/渲染域误差；
- opacity、文字、mask：边缘加权误差；
- 多通道共享 UV 时：加权和、soft-max，或最终多光照 render-space loss。

还需明确“什么值得保留”：如果只用像素 MSE，随机噪声也会获得大量预算；若目标是人感知质量，需要在 distortion 中加入语义、可见性或感知权重。R-D 不会自动替你决定价值，**distortion 的定义就是产品价值函数。**

### 4.3 PartUV 的自适应调整

当前 chart 级统一缩放只能改变 chart 之间的密度。若同一车门 chart 内只有一个小 logo 高频，其余为纯色，放大整个车门会浪费预算。

建议三层递进：

#### V0：固定 PartUV charts，只做预算守恒缩放

这是现有 MVP，适合验证接口和公平预算：

\[
P_c=\sum_{f\in c}P_f,
\qquad
s_c=\sqrt{P_c/A_{UV,c}}
\]

它只能证明 chart-level content allocation 可行，不能声称 per-face adaptive density。

#### V1：density-aware PartUV tree cut

利用 PartField/PartUV hierarchy，对内容异质 chart 继续探索子节点。现有 \(h_c\) 可作为便宜的候选触发器，但不能作为最终 split 判据；方差高不代表需求在空间上可分，也不代表拆分收益超过新增 seam/padding。

正式判据应比较：

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

仅当 \(C_{split}<C_{keep}\) 时拆分。可在 PartUV tree 上用 dynamic programming 或离散预算 greedy 完成。第一版需要暴露候选 tree nodes 和参数化结果，不必立即重写 ABF。

#### V2：chart 内 signal-specialized parameterization

对于不能通过语义 tree 合理切开的连续密度变化，只有两条路：

1. 沿内容边界增加 seam；
2. 在 chart 内使用 IMT/目标 Jacobian 做非均匀参数化。

后者能减少 seam，但会增加局部形变，可能降低 artist-editability。最终建议采用混合策略：大尺度差异通过 tree cut，chart 间通过 scalar allocation，chart 内仅允许受限的各向同性/各向异性变形。

### 4.4 它是否类似压缩算法

是，而且最准确的表述是：

> **这是一种 surface-signal rate–distortion optimization，也就是在有限纹理存储率下做非均匀采样与预算分配。**

类比关系如下：

| 你们的 UV 系统 | 压缩中的对应概念 |
|---|---|
| 总 atlas texel / 内存 | rate |
| 烘焙或渲染误差 | distortion |
| 每个 chart 的 texel 数 | block bit allocation |
| PartUV tree cut | 自适应 block partition |
| seam、gutter、padding | block/header/边界开销 |
| 单/多 atlas | 容器或 tile 组织 |
| 内容 proxy | rate-control predictor |

但它不是传统意义上的熵编码：如果最终目标仍按未压缩 texel 数计费，你们优化的是**采样位置和空间分辨率**；只有进一步加入 BC/ASTC/KTX 等格式、bits-per-texel 和 mip chain 后，才是完整的 bit-rate optimization。

## 五、单 atlas 与多 atlas 的正式结论

在总 texel 数、有效占用率、padding、mip 规则和采样过滤完全相同时，单 atlas 不会因为“只有一张”而天然更差。质量差异只能来自：

- 总像素预算不同；
- 可选分辨率量化误差；
- packing occupancy 不同；
- padding/gutter 开销不同；
- 最大纹理边长约束；
- material、streaming 或编辑组织不同；
- packer 对各 atlas 独立缩放，破坏了跨 atlas 的目标 pixel scale。

因此 atlas planner 应输入允许的规格集合，例如：

\[
(W_k,H_k)\in
\{512^2,1024^2,2048^2,4096^2,2048\!\times\!1024,\ldots\}
\]

并选择满足预算与最大尺寸的可打包方案。packer 必须工作在固定 pixel-space bins 中，保持 chart 的目标像素尺度；禁止将每张 atlas 独立归一化到满框后却不记录其全局缩放。

## 六、“尽量覆盖所有物体”的正确实现方式

没有一个 unique UV rewrap 能无条件适用于所有生产资产。可靠系统应做**类型识别 + 路由 + fallback**：

| 资产类别 | v1 策略 |
|---|---|
| 单/多材质 unique texture | 完整支持：统一表面信号、重新分配、rebake |
| UDIM unique texture | 支持：保留 UDIM 或由 planner 重新选 tiles |
| mirrored / stacked UV | 默认保留共享；仅在用户选择 unique 模式时展开 |
| trim sheet / repeated tile | preservation mode，不做 naive unique rebake |
| vertex color / procedural / triplanar | 若能查询连续表面信号，可直接 bake 到新 UV |
| 小碎片、孤儿面、非流形、重合孪生面 | FaceMatcher + repair/salvage；所有有效面必须有输出或被显式隔离 |
| hair cards / foliage / alpha cutout | 保留重叠和卡片布局，单独策略 |
| lightmap UV、动画专用 UV | 与材质 UV 分离，暂不纳入 v1 联合优化 |

“通用”不是强行改写每个资产，而是系统能判断：优化、保留、转换还是拒绝，并给出原因与预算后果。

## 七、推荐的模块接口

### 输入

1. `MeshAsset`
   - geometry、face/material IDs、全部 UV sets、world scale（可选）；
2. `SurfaceSignalProvider`
   - `query(x, channel, lod)`；
   - 可由 UniTEX TF、已有纹理、multi-view 或 procedural material 实现；
3. `BudgetSpec`
   - `total_texels` 或 `memory_bits`；
   - `allowed_resolutions`、`max_texture_size`、`max_atlases`；
   - padding、mip、压缩 block 对齐；
4. `ImportanceSpec`（可选）
   - camera distribution、semantic/user mask、最低 TD floor；
5. `Policy`
   - L1 uniform、L2 content-aware、L3 user-defined；
   - preserve-shared / unique-rebake。

### 输出

1. mesh + UV coordinates；
2. chart IDs、part hierarchy 与 seam；
3. atlas assignment + 每张分辨率；
4. baked texture maps；
5. TD heatmap、content demand、R-D curves；
6. coverage、overlap、distortion、anisotropy、packing、padding 和预算报告；
7. 未来训练用标签：per-face demand、candidate tree cut、selected cut、\(E_c(P)\)、atlas plan。

## 八、可行性实验计划

### 8.1 研究问题

- RQ1：单/多 atlas 在严格同预算、同有效占用下是否还有质量差异？
- RQ2：内容感知分配能否在固定预算下优于 uniform TD？
- RQ3：哪些低成本 proxy 最接近 R-D oracle？
- RQ4：density-aware tree cut 是否优于只做 chart scaling？
- RQ5：方法是否对重三角化、源 UV 改写和模型尺度稳定？

### 8.2 对照方法

1. PartUV 默认 1×1024²（仅作为“不公平默认设置”的诊断）；
2. PartUV uniform、严格等预算；
3. luminance std；
4. surface gradient；
5. multiscale band energy；
6. chart-level R-D oracle；
7. ours：scaling；
8. ours：tree cut + scaling；
9. ours：tree cut + adaptive atlas。

### 8.3 预算曲线

至少测试：

\[
0.5M,1M,2M,4M,8M\text{ texels}
\]

每个点固定：同一 reference、同一过滤、同一 padding/mip、同一 packer 设置。不要只比较一个 1024² 点。

### 8.4 测试数据

需要同时包含：

- 合成表面信号：常量、渐变、不同频率/振幅正弦、方向条纹、checker、sharp edge、等亮度色边界、随机噪声、normal 高频；
- artist-created single-atlas；
- 多材质、多分辨率、UDIM；
- AI-generated messy meshes；
- mirrored/stacked、tiling、trim-sheet 负例；
- 同一 mesh 的多种纹理内容，用于证明 UV 由图片/内容条件化，而非只由几何决定。

### 8.5 主指标与有效性门槛

主结果：

\[
\text{surface/render error versus actual texture budget}
\]

比较整条 budget-error curve 及其 AUC，并对每个资产做 paired comparison 和 bootstrap confidence interval。

有效性 gate：

- 所有有效表面覆盖；无法处理的面被显式 quarantine，不能静默丢失；
- UV overlap = 0（明确允许共享的 preservation mode 除外）；
- 实际 raw budget 与目标误差 ≤1%；
- 参数化失真不超过设定阈值或相对 baseline 明确报告；
- padding、mip 和 block alignment 计入真实成本；
- content-aware 的 error–budget AUC 优于 uniform，并且不是只提高 `corr(TD, demand)`；
- 重三角化、源 UV 旋转/缩放/重切、模型统一缩放三项不变性通过。

### 8.6 结果诊断逻辑

| 结果 | 含义 |
|---|---|
| oracle 优于 uniform，proxy 不优 | 内容信号估计失败，主框架仍成立 |
| scaling 无提升，tree cut 有提升 | PartUV chart 粒度是瓶颈 |
| oracle 也不优于 uniform | 预算范围、误差定义或研究假设有问题 |
| single/multi 等预算后持平 | atlas 数只是工程选择，不是质量贡献 |
| multi 只因 occupancy/padding 获胜 | 贡献应写 adaptive atlas planning |
| rebake 提升、重新生成不提升 | 表示层有效，但生成模型未利用新布局 |
| 重新生成提升更明显 | UV 连续性/语义布局也帮助了生成模型 |

## 九、实施顺序

### P0：先把比较做公平

1. `TextureDomainInventory`：识别 atlas/channel/UDIM/tile/trim/overlap/UV sets；
2. 多 atlas、非方形 atlas 的统一 TD 和预算统计；
3. pixel-space packing，禁止跨 atlas 密度漂移；
4. strict equal-budget single-vs-multi 实验；
5. coverage salvage 与资产路由报告。

### P1：建立论文金标准

1. chart-level R-D oracle；
2. surface/render error budget curves；
3. std、gradient、multiscale proxy 阶梯比较；
4. 合成纹理与三项不变性实验。

### P2：把密度真正放进 PartUV 搜索

1. 暴露 PartUV hierarchy/candidate charts；
2. 用 \(h_c\) 只做候选触发；
3. 实现 R-D + seam + gutter-aware tree cut；
4. 联合 atlas resolution selection。

### P3：接 UniTEX 做端到端

1. `SurfaceSignalProvider` 适配 Texture Function；
2. 粗生成 → 需求估计 → UV 优化 → 从 TF 直接 bake；
3. 若需要，最终 UV 上再 refinement/regeneration；
4. 比较 rebake 与 regeneration 两条实验。

UniTEX 论文确实提供“图片 + 无纹理 mesh → 完整 Texture Function”的正确上游抽象。[UniTEX CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_UniTEX_Universal_High_Fidelity_Generative_Texturing_for_3D_Shapes_CVPR_2026_paper.pdf) 但截至 2026-07-14，官方仓库同时提供了推理示例，又仍把部分 basic code/LTM checkpoints 列在 TODO 中，因此模块接口不应硬绑定其内部实现；先用高分辨率 reference 或现有 textured mesh 完成第二步验证。[UniTEX 官方仓库](https://github.com/YixunLiang/UniTEX)

## 十、它如何服务未来的原生模型

当前优化器不是一次性平替，而是未来训练数据生成器。每个样本可以自动产生：

- 图片/表面信号条件下的 per-face demand；
- PartUV hierarchy 中所有候选 chart；
- 每个 chart 的 R-D 曲线；
- 最优或近优 tree cut；
- chart texel allocation；
- atlas 数量、分辨率与 packing；
- 有效性和 artist-friendly 约束标签。

未来模型可学习：

\[
(I,M,B)
\rightarrow
(\widehat E_c(P),\widehat{seam},\widehat{tree\ cut},\widehat P_c,\widehat a)
\]

再由确定性 UV solver 输出合法结果。这样既能实现端到端使用体验，又保留无重叠、预算和失真等硬约束。

这与现有 artist-style 学习方法形成清楚差异：ArtUV 学习语义 seam 与 artist-style 参数化；2026 年 6 月的 DreamUV 用 Flow Matching 学习 artist-authored UV 分布，但其公开版本依赖预切 mesh，且不处理 packing。公开目标主要是边界规整、轴对齐、低失真与 artist preference，而不是“输入图片决定局部 texel demand，并在固定预算下联合 atlas allocation”。[ArtUV](https://chenyg59.github.io/ArtUV/)、[DreamUV](https://arxiv.org/abs/2606.22445)

因此最终研究主线可以表述为：

> **Image-conditioned, part-aware UV discretization under a fixed texture budget：把连续或高分辨率表面纹理信号转换成 production-ready UV 资产时，联合优化 semantic chart selection、texel allocation、atlas planning 与 packing。**

## 十一、对当前 `partuv_td_test.ipynb` 的实装审查

该 notebook 已经跑通了真实的最小闭环，但当前更准确的结论是：

> **局部功能可行；全局内容分配尚未通过；multi-atlas 仍是概念演示。**

### 11.1 已经成立的部分

- PartUV、FaceMatcher、原 mesh 对齐、rebake 与 Blender repack 已串通；
- 当前样例 14,458 faces，PartUV 输出 30 parts / 223 charts；
- 面对应 `mismatches=0`，覆盖率 99.96%，仅 0.02% 面积未覆盖；
- L1 baseline 的几何 TD `CVw=0.028`，说明 uniform chart scaling 基本正确；
- `mean_A(q²)=1.000`，线性倍率 \(q\) 与面积预算 \(q^2\) 的归一逻辑在该样例中成立；
- Converse 鞋标 chart 从约 82×83 px 放大到 207×202 px，局部文字清晰度明显改善；
- `h_c`、Jacobian anisotropy 与 log-stretch 已进入诊断体系，这是下一版 tree cut 的良好基础。

### 11.2 当前执行结果与结论存在冲突

该次执行打印：

```text
CVw(TD²/w): 0.750 -> 0.810
corr(logTD², logw) = 0.516
h_c > 0.25: 86 / 223 charts
```

因此不能把本次运行总结为“信息量均匀度显著下降”或“全部提升成立”。全局目标指标反而变差，说明 face-level demand 与 chart-level 单标量之间存在明显表达鸿沟。局部鞋标成功不等于全资产 R-D 改善。

建议立即加入结果 gate：若主指标未改善，结论单元格必须自动输出 `NOT YET VALIDATED`，不得使用“感知无损”或“提升已归因于纹理密度”的措辞。

### 11.3 预算锚定存在一个明确误报

当前样例中：

```text
R_budget = 1081
snap2(R_budget) = 1024
```

代码使用最近的 2 的幂，因此把 1081 向下吸附到 1024，却输出“满足预算守恒”。按连续 fill 估算，这会少约 10% 的有效占用 texel。应区分两种政策：

- `preserve_at_least`：向上取整或选混合/非方形规格；
- `hard_cap`：允许向下，但必须报告未使用或损失预算并优化其余 atlas 组合。

不能再用 `round(log2 R)` 同时代表这两种语义。

### 11.4 multi-atlas 预算并未守恒

附录设置 `TOTAL_TEXELS=2,000,000`，独立吸附后得到：

```text
512² + 1024² + 512² = 1,572,864 texels
```

实际比目标少 21.4%。此外三张图集 fill 为约 40% / 13% / 38%，说明简单“按 demand 平衡 part + shelf pack”没有考虑可装箱形状。跨 atlas 报告的 `mean TD` 还是未按面积加权、也未除以 L2 的目标倍率 \(q\)；在内容感知模式下，绝对 TD 本来就应不同，正确比较应是 `TD/q` 或 `TD²/w` 的归一尺度。

因此附录当前证明的是“多图集数据流能画出来”，尚未证明“总预算、跨图集密度或 packing 已正确”。

### 11.5 需优先修正的实现细节

1. STEP 2 的所谓“PartUV 原版”实际是 `PartUV decomposition + 自定义 L1 scaling + Blender packing`，应改名为 `PartUV+L1 baseline`；
2. `cw` 是每面 24 个随机点的 sRGB 亮度 std，且采样使用 nearest，不是 frequency oracle；应标为 heuristic baseline；
3. top-10% 阈值使用未加权 `np.quantile`，与全篇面积加权原则不一致；
4. “同一表面区域像素 ×6.14”使用 chart bounding box 面积，易受旋转和空白影响；应改为 rasterized interior texel count；
5. 源 UV 越界在 `sample_bilinear` 中被 clamp，不等价于 repeat/wrap；tiling 资产当前不应标为支持；
6. `used_frac=sum triangle UV areas` 不是 rasterized union，遇到 overlap/stack/tile 会失真；
7. rebake 没有 supersampling、gutter dilation、mipmap、sRGB/linear 管理或 PBR 通道误差；
8. FaceMatcher 的 KD fallback 不保证一对一，应增加全局唯一性断言；三角形角点映射也应求 3! 最优双射，不能独立 argmin；
9. 未覆盖面目前留空；生产输出必须 salvage 或显式失败；
10. notebook 只保存图和 `metrics.json`，尚未导出可直接消费的 final mesh、material assignment 和 multi-atlas manifest。

### 11.6 基于实装结果修订后的最小下一步

优先级应调整为：

1. 修正结论 gate、预算吸附和 multi-atlas 实际总预算；
2. 同时报告 face-level 与 chart-aggregated target-match，确认问题究竟来自信号还是 chart 粒度；
3. 对 86 个高异质 chart 做最小 hierarchy split 实验；
4. 增加真实 reconstruction error，而不再用 `CVw(TD²/w)` 与 correlation 代替质量；
5. 把单/多 atlas 纳入同一个离散 planner；
6. 最后再替换 luminance std 为 multiscale proxy / R-D oracle。

## 十二、最终决策建议

现阶段应把成功定义为：

> **不训练新模型，仅使用开源上游或高分辨率参考信号，证明 PartUV + TD 模块可以对不同拓扑、不同纹理域和不同内容频率的物体，自动选择合法的 chart/atlas 方案，并在严格相同的真实纹理预算下，比 uniform PartUV 保存更多有效视觉信息。**

最优先的三个工作项是：

1. 完成 texture-domain inventory 与严格等预算 single/multi-atlas 实验；
2. 以 chart-level R-D oracle 取代亮度 std 作为金标准；
3. 在 PartUV hierarchy 上实现 R-D/seam/gutter-aware tree cut。

这三项同时回答了你提出的多贴图、内容细节、PartUV 局限和“是否属于压缩”四个问题，也会直接生成未来原生模型所需的监督数据。
