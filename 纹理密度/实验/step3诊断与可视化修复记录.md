# STEP3 热力图诊断与 notebook 可视化修复记录 (2026-07-16)

## 修复(全部进 source of truth)

| 问题 | 根因 | 修复位置 |
|---|---|---|
| STEP1 输入模型走样(高频贴图呈假棋盘/网格纹) | 逐面 UV 质心 1 次采样平涂 | `build_demo_notebook.py` cell1/cell3: `hi_res_render`(GPU 逐像素 / CPU 64× 细分)共用 |
| STEP3 热力图椒盐(橙底黑斑) | 33.5% 面 cw≤1e-6(17.4% 精确 0 + 浮点碎屑),P5 被拉到 log(1e-6),真平坦面(~1e-3)被顶到色标中段 | cell9: `z = log(max(cw, 1/510))` 显示地板(半个 8-bit 量化步,亚量化=平坦);cell11 chart 均值着色同步 |
| STEP3 右图/STEP5 三联图朝暗面(torus) | top chart 是内赤道条带,法线指向被对面管壁遮挡的方向;任何法线启发式无解 | `tdlib/gpu.py` 新增 `visible_weight()`(z-buffer 实测可见加权面积);`gen_dashboard_assets.facing_view` 有 GPU 时 15° 网格实测选视角(w=cw 加权),无 GPU 回退原启发式 |
| 新 kernel 抓满载 GPU0 OOM | cell1 的 `pick_free_gpu` 曾被删 | cell1 恢复 |

验证:objaverse 鞋 + synth_half_flat_detail torus 两资产 nbconvert 全程无错,
STEP1/3/5 逐图人工核验(平坦=暗、细节=亮、视角可见、三联图一致)。

## 已知但未动的问题(算法层,待评审决策)

1. **UV wrap 接缝面的 cw 是错的**:corner u(或 v)跨 0/1 边界的面(torus 上 286 个),
   24 个采样点沿线性插值横扫整张贴图,得到与内容无关的 std。本资产上碰巧接缝就在
   条纹/平坦边界,数值(0.43)与真条纹面(0.45)相近而无害;但接缝落在均匀区、
   贴图两端内容不同的资产会产生虚假高需求。修法:采样前对跨缝面 unwrap
   (span>0.5 时小坐标 +1,查表 mod 1)。`luminance_std_heuristic`、rebake 采样、
   预览渲染同要检查。
2. **平坦面的 q 分档噪声**:同样平坦的面因 cw=0/碎屑/噪声底的数值偶然,
   demand_weights 里被拆成 q≈0.5 与 q≈0.73 两档(chart 级求和摊平大半,
   实测 q P5/P50/P95 = 0.51/0.73/2.13)。根治需在信号端设量化级下限(如 1/510),
   会改变分配结果,属算法决策。

诊断脚本/数据:`step3_heatmap_diag.py`、`step3_diag_cache.npz`、`facing_dbg.npz`、
`facing_view_check.png`(遮挡感知视角单测:内赤道 chart 从洞口可见)。
