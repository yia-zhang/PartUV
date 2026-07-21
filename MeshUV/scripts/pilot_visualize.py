# -*- coding: utf-8 -*-
"""pilot 可视化: 前 N 个 accepted 对象的 reference 纹理 + TD heatmap
(source-UV 域 tripcolor, 颜色=chart_log_density_ratio) + loader 回读验证。
用法: python scripts/pilot_visualize.py --pilot datasets/pilot/TexVerse-1K-16 [--n 3]
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.data.dataset import MeshUVTDDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="datasets/pilot/TexVerse-1K-16")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    root = a.pilot if os.path.isabs(a.pilot) else os.path.join(ROOT, a.pilot)
    ds = MeshUVTDDataset(root)
    print(f"loader 回读: {len(ds)} accepted 对象")
    outd = os.path.join(root, "viz")
    os.makedirs(outd, exist_ok=True)
    report = []
    for i in range(min(a.n, len(ds))):
        it = ds[i]
        mi, tt = it["model_inputs"], it["training_targets"]
        uv = mi["source_uv"].reshape(-1, 2)
        tris = np.arange(len(uv)).reshape(-1, 3)
        logr_face = tt["chart_log_density_ratio"][mi["face_to_chart"]]
        ref = np.asarray(Image.open(it["reference_texture"]))
        fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
        axs[0].imshow(ref)
        axs[0].set_axis_off()
        axs[0].set_title("reference basecolor(1K)", fontsize=10)
        vmax = max(float(np.abs(logr_face).max()), 1e-3)
        tp = axs[1].tripcolor(uv[:, 0], uv[:, 1], tris, facecolors=logr_face,
                              cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axs[1].set_aspect("equal")
        axs[1].set_axis_off()
        axs[1].set_title("TD heatmap: chart_log_density_ratio(线性密度 log 比)",
                         fontsize=10)
        fig.colorbar(tp, ax=axs[1], fraction=0.04)
        p = f"{outd}/{it['object_id']}.png"
        plt.tight_layout()
        plt.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        report.append(dict(object_id=it["object_id"], viz=p,
                           n_charts=int(len(tt["chart_target_scale"])),
                           logr_range=[round(float(logr_face.min()), 3),
                                       round(float(logr_face.max()), 3)]))
        print(f"  viz -> {p}")
    json.dump(report, open(f"{outd}/loader_readback.json", "w"), indent=1,
              ensure_ascii=False)
    print("VIZ: DONE")


if __name__ == "__main__":
    main()
