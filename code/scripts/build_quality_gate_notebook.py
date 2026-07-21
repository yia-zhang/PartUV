# -*- coding: utf-8 -*-
"""构建只读 pseudo_gt_quality_gate.ipynb: 不复制算法, 只读取
outputs/pseudo_gt(_quality) 的 JSON/图片。数据由
scripts/run_pseudo_gt_quality_gate.py 生成。"""
import nbformat as nbf

NB = "/root/youjiaZhang/PartUV/code/notebook/pseudo_gt_quality_gate.ipynb"
md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)
cells = []

cells.append(md("""# Pseudo-GT Quality Gate（只读）

判断已通过 **Structural Acceptance** 的 object-level pseudo-GT 是否具有
**TD label quality**。对比仅 Reference / PartUV-Uniform / PseudoGT-TD
（同 chart hash / local UV / packer / padding / baker / 相机 / atlas 分辨率；
主公平轴 = 相同 B_raw，两档 50%/25%）。本 notebook **只读取** 质量门脚本
（`scripts/run_pseudo_gt_quality_gate.py`）产出的 JSON 与图片，不复制算法。"""))

cells.append(code("""import json, os
import matplotlib.pyplot as plt
from matplotlib.image import imread

SAMPLE_ID = "shoe_22b822__partuv_td_teacher_pseudo_gt_v1"
SD  = f"outputs/pseudo_gt/shoe_22b822_v1"
QD  = f"outputs/pseudo_gt_quality/{SAMPLE_ID}"

def jload(p):
    if not os.path.exists(p):
        print(f"[缺失] {p} —— 先运行 scripts/run_pseudo_gt_quality_gate.py")
        return None
    return json.load(open(p))

def show_png(p, figw=14):
    if not os.path.exists(p):
        print(f"[缺失] {p}"); return
    im = imread(p)
    plt.figure(figsize=(figw, figw * im.shape[0] / im.shape[1]))
    plt.imshow(im); plt.axis("off"); plt.show()

manifest = jload(f"{SD}/manifest.json")
metrics  = jload(f"{QD}/metrics.json")
report   = jload(f"{QD}/quality_report.json")
print("加载:", "manifest" if manifest else "-", "| metrics" if metrics else "-",
      "| quality_report" if report else "-")"""))

cells.append(md("""## 1. Structural Acceptance（来自样本 manifest）"""))

cells.append(code("""if manifest:
    print(f"sample_id: {manifest['sample_id']}")
    print(f"structural status: {manifest['status']}   "
          f"label={manifest['label_type']}  artist_gt={manifest['artist_gt']}")
    print(f"scope: {manifest['supervised_scope']}  "
          f"local_uv_refinement: {manifest['local_uv_refinement']}")
    print(f"teacher: {manifest['teacher']['name']} | beta={manifest['teacher']['beta']} "
          f"| packer={manifest['teacher']['packer']} | atlas={manifest['teacher']['atlas_size']}")
    print(f"chart_hash: {manifest['teacher']['chart_hash'][:20]}…")
    g = manifest['geometry']
    print(f"faces={g['n_faces']:,} charts={g['n_charts']} "
          f"coverage={g['train_face_coverage']*100:.3f}% reload_ok={g['reload_ok']}")
    n_gate = len(manifest['gates']); n_pass = sum(map(bool, manifest['gates'].values()))
    print(f"structural gates: {n_pass}/{n_gate} PASS")"""))

cells.append(md("""## 2. 2D Layout 与预算（同 B_raw，两档）"""))

cells.append(code("""if metrics:
    for t, row in metrics["tiers"].items():
        print(f"—— {t}: B_raw={row['B_raw']:,} (两方法相同, 偏差 {row['braw_dev']*100:.2f}%) ——")
        for m, v in row["methods"].items():
            print(f"  {m:16s} B_signal={v['B_signal']:,}  fill={v['packing_fill']*100:.1f}%  "
                  f"overlap={v['overlap']}")
show_png(f"{QD}/layout_comparison.png", figw=11)"""))

cells.append(md("""## 3. Global / HF / Seam 指标（线性 RGB；LPIPS 不可用 → masked SSIM，已标注）"""))

cells.append(code("""if metrics:
    for t, row in metrics["tiers"].items():
        print(f"—— {t} ——")
        for m, v in row["methods"].items():
            print(f"  {m:16s} PSNR={v['psnr_db']}dB  HF MSE={v['mse_hf']:.3e}  "
                  f"seam MSE={v['mse_seam']:.3e}  interior={v['mse_interior']:.3e}  "
                  f"maskedSSIM={v['masked_ssim_mean']}")
        print(f"  G_global={row['G_global']:+.3f}   G_HF={row['G_HF']:+.3f}")
    print(f"\\nsignal_dist(demand vs 面积分布)={metrics['signal_dist']:.3f} "
          f"(< {metrics['protocol']['low_signal_dist']} 判 LOW_SIGNAL)")"""))

cells.append(md("""## 4. 多视角渲染 / 误差热图 / 细节裁剪（50% 档）"""))

cells.append(code("""show_png(f"{QD}/render_comparison.png", figw=13)
show_png(f"{QD}/error_heatmap.png", figw=11)
show_png(f"{QD}/detail_crops.png", figw=13)"""))

cells.append(md("""## 5. 最终 Quality Gate"""))

cells.append(code("""if report:
    for t, gg in report["gates"].items():
        print(f"—— {t} ——")
        for k, v in gg.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\\nstructural_status = {report['structural_status']}")
    print(f"quality_status    = {report['quality_status']}")
    print(f"training_eligible = {report['training_eligible']}")
    print(f"protocol_hash     = {report['protocol_hash'][:20]}…")
    print(f"\\n注: {report['notes']}")"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                             "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
with open(NB, "w") as f:
    nbf.write(nb, f)
print(f"written: {NB} ({len(cells)} cells)")
