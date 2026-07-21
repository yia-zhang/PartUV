# -*- coding: utf-8 -*-
"""Sample Gallery v0: 从已 accepted 对象确定性选 5 个, 生成 GitHub 可验收总览图.

选择标准(去重顺延): logr 方差最低/中位/最高、chart 数最多、G_HF_eq 最大。
packed UV 与 rebake 来自冻结 Teacher(β=0.25)的真实 xatlas/texel-center baker
(adapter.quality_check_medium 唯一 canonical 实现; 预算公平轴=等 B_signal±1%)。
source UV 与 target packed UV 严格分开标注。不暂停 256 构建。
"""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from meshuv.teacher_adapter import (pick_free_gpu, quality_check_medium,
                                    tc_from_sample, textured_render,
                                    TEACHER_VERSION)      # noqa: E402
from meshuv.visualization.plots import _t                 # noqa: E402

DS = f"{ROOT}/datasets/processed/MeshUV-TD-PseudoGT-MVP-v0"
OUT = f"{ROOT}/reports/sample_gallery_v0"
os.makedirs(OUT, exist_ok=True)
BETA = 0.25
WHY_EN = {"logr 方差最低": "min logr variance", "logr 方差中位": "median logr variance",
          "logr 方差最高": "max logr variance", "chart 数最多": "most charts",
          "G_HF_eq 最大": "max G_HF_eq"}


def scan_accepted():
    import glob
    rows = []
    for f in sorted(glob.glob(f"{DS}/objects/*/status.json")):
        st = json.load(open(f))
        if st.get("status") != "ACCEPTED":
            continue
        d = os.path.dirname(f)
        z = np.load(f"{d}/arrays.npz")
        m = z["chart_valid_mask"]
        qc = st.get("quality_check") or {}
        rows.append(dict(
            object_id=st["object_id"],
            uid=st.get("uid") or st["object_id"].replace("mv0_", ""), dir=d,
            logr_var=float(np.var(z["chart_log_density_ratio"][m])),
            n_charts=int(m.sum()),
            g_hf=qc.get("G_HF_eq"), g_g=qc.get("G_global_eq"),
            quality=st.get("quality_status")))
    return rows


def pick5(rows):
    by_var = sorted(rows, key=lambda r: (r["logr_var"], r["uid"]))
    picks, used = [], set()

    def take(seq, why):
        for r in seq:
            if r["uid"] not in used:
                used.add(r["uid"])
                picks.append((r, why))
                return

    take(by_var, "logr 方差最低")
    take(by_var[len(by_var) // 2:], "logr 方差中位")
    take(by_var[::-1], "logr 方差最高")
    take(sorted(rows, key=lambda r: (-r["n_charts"], r["uid"])), "chart 数最多")
    take(sorted([r for r in rows if r["g_hf"] is not None],
                key=lambda r: (-r["g_hf"], r["uid"])), "G_HF_eq 最大")
    assert len({u for u, _ in [(r["uid"], w) for r, w in picks]}) == 5
    return picks


def uv_panel(ax, uvs, charts_F, title):
    for ci, uv in enumerate(uvs):
        col = plt.cm.tab20(ci % 20)
        ax.add_collection(PolyCollection(uv[np.asarray(charts_F[ci])],
                                         facecolors=col, edgecolors="none"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def make_fig(rank, rec, why):
    tc, labels, z, texA = tc_from_sample(rec["dir"])
    q = quality_check_medium(tc, labels, return_artifacts=True)
    if q["status"] != "OK":
        print(f"[缺失] {rec['uid']}: 真实 packing 不可恢复({q['status']}), "
              f"跳过 packed/rebake 面板")
        return None
    art = q["artifacts"]
    V, F = z["vertices"], z["faces"]
    f2c = z["face_to_chart"]
    charts_F = [np.asarray(c["F"]) for c in tc["pu"]["charts"]]
    ok = np.ones(len(F), bool)
    uv_src = z["source_uv"].reshape(-1, 2)
    tris = np.arange(len(uv_src)).reshape(-1, 3)

    fig, axs = plt.subplots(3, 4, figsize=(19, 14))
    fig.suptitle(f"#{rank} {rec['uid']} — {WHY_EN[why]} | TexVerse-1K | "
                 f"teacher {TEACHER_VERSION} β={BETA}", fontsize=12)
    # 1 basecolor / 2 白模 / 3 source UV wireframe / 4 chart 分割
    axs[0, 0].imshow(np.asarray(Image.open(f"{rec['dir']}/reference_basecolor.png")))
    axs[0, 0].set_title("source base color(1K)", fontsize=9)
    im = textured_render(V, F, z["source_uv"], z["source_uv_valid"],
                         texA * 0 + 0.78, view=(18, 40))
    axs[0, 1].imshow(im); axs[0, 1].set_title(
        f"white mesh({len(F):,} faces)", fontsize=9)
    axs[0, 2].imshow(np.asarray(Image.open(
        f"{rec['dir']}/reference_basecolor.png")), extent=[0, 1, 0, 1])
    segs = np.concatenate([z["source_uv"][:, [0, 1]], z["source_uv"][:, [1, 2]],
                           z["source_uv"][:, [2, 0]]])[::max(1, len(F) // 20000)]
    from matplotlib.collections import LineCollection
    axs[0, 2].add_collection(LineCollection(segs, colors="#00e5ff",
                                            linewidths=0.3, alpha=0.65))
    axs[0, 2].set_xlim(0, 1); axs[0, 2].set_ylim(0, 1)
    axs[0, 2].set_title("SOURCE UV wireframe (NOT target)", fontsize=9)
    tp = axs[0, 3].tripcolor(uv_src[:, 0], uv_src[:, 1], tris,
                             facecolors=(f2c % 20).astype(float), cmap="tab20")
    axs[0, 3].set_title(f"PartUV charts({int(f2c.max()) + 1})", fontsize=9)
    # 5 content / 6 target 密度
    tp = axs[1, 0].tripcolor(uv_src[:, 0], uv_src[:, 1], tris,
                             facecolors=z["face_content_score"], cmap="magma")
    axs[1, 0].set_title("teacher content score (diagnostic)", fontsize=9)
    plt.colorbar(tp, ax=axs[1, 0], fraction=0.04)
    logr_f = z["chart_log_density_ratio"][f2c]
    vmax = max(float(np.abs(logr_f).max()), 1e-3)
    tp = axs[1, 1].tripcolor(uv_src[:, 0], uv_src[:, 1], tris,
                             facecolors=logr_f, cmap="coolwarm",
                             vmin=-vmax, vmax=vmax)
    axs[1, 1].set_title("target linear texel-density log-ratio", fontsize=9)
    plt.colorbar(tp, ax=axs[1, 1], fraction=0.04)
    # 7/8 真实 packed UV
    from meshuv.teacher_adapter import _ensure_path
    uv_panel(axs[1, 2], art["uniform"].get("uvs") or [], charts_F,
             f"UNIFORM packed UV (real xatlas, R={art['uniform']['R']})")
    uv_panel(axs[1, 3], art["td"].get("uvs") or [], charts_F,
             f"TD TARGET packed UV (beta={BETA}, R={art['td']['R']})")
    # 9/10 rebake / 11 差异 / 12 指标
    axs[2, 0].imshow(np.clip(art["uniform"]["tex"], 0, 1))
    axs[2, 0].set_title(f"UNIFORM rebake (B_signal={art['uniform']['B_signal']:,})",
                        fontsize=9)
    axs[2, 1].imshow(np.clip(art["td"]["tex"], 0, 1))
    axs[2, 1].set_title(f"TD rebake (B_signal={art['td']['B_signal']:,})",
                        fontsize=9)
    r_u = textured_render(V, F, art["uniform"]["nuv"], ok,
                          art["uniform"]["tex"], view=(18, 40))
    r_t = textured_render(V, F, art["td"]["nuv"], ok, art["td"]["tex"],
                          view=(18, 40))
    dif = np.abs(r_u.astype(float) - r_t.astype(float)).mean(-1)
    imd = axs[2, 2].imshow(dif, cmap="inferno")
    axs[2, 2].set_title("render |uniform - TD| (same camera)", fontsize=9)
    plt.colorbar(imd, ax=axs[2, 2], fraction=0.04)
    man = json.load(open(f"{rec['dir']}/manifest.json"))
    fill_u = art["uniform"]["B_signal"] / art["uniform"]["R"] ** 2
    fill_t = art["td"]["B_signal"] / art["td"]["R"] ** 2
    axs[2, 3].text(0.02, 0.96, "\n".join([
        f"faces={len(F):,}  charts={int(f2c.max()) + 1}",
        f"beta={BETA} (FROZEN)  logr_var={rec['logr_var']:.4f}",
        f"budget axis: equal B_signal (dev {abs(q['bsignal_match'] - 1) * 100:.2f}%)",
        f"coverage={man['geometry']['train_face_coverage'] * 100:.2f}%",
        f"occupancy: uniform {fill_u * 100:.1f}% / TD {fill_t * 100:.1f}%",
        f"G_global_eq={q['G_global_eq']:+.4f}",
        f"G_HF_eq={q['G_HF_eq']:+.4f}",
        f"quality={rec['quality']}",
    ]), va="top", fontsize=11, family="monospace",
        transform=axs[2, 3].transAxes)
    axs[2, 3].set_axis_off()
    for ax in axs.ravel():
        if not ax.has_data() and ax is not axs[2, 3]:
            ax.set_axis_off()
    axs[0, 3].set_aspect("equal"); axs[0, 3].set_axis_off()
    axs[1, 0].set_aspect("equal"); axs[1, 0].set_axis_off()
    axs[1, 1].set_aspect("equal"); axs[1, 1].set_axis_off()
    for a in (axs[0, 0], axs[0, 1], axs[0, 2], axs[2, 0], axs[2, 1], axs[2, 2]):
        a.set_axis_off()
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    p = f"{OUT}/{rank:02d}_{rec['uid'][:16]}.png"
    plt.savefig(p, dpi=78)
    plt.close(fig)
    return dict(file=os.path.basename(p), uid=rec["uid"],
                object_id=rec["object_id"], reason=why,
                logr_var=round(rec["logr_var"], 5), n_charts=rec["n_charts"],
                G_global_eq=q["G_global_eq"], G_HF_eq=q["G_HF_eq"],
                bsignal=dict(uniform=art["uniform"]["B_signal"],
                             td=art["td"]["B_signal"]),
                size_kb=round(os.path.getsize(p) / 1024))


def main():
    pick_free_gpu()
    rows = scan_accepted()
    print(f"accepted 池: {len(rows)}")
    picks = pick5(rows)
    entries = []
    for k, (rec, why) in enumerate(picks, 1):
        print(f"[{k}] {rec['uid']} ({why})", flush=True)
        e = make_fig(k, rec, why)
        if e:
            entries.append(e)
    commit = subprocess.run(["git", "rev-parse", "HEAD"],
                            cwd=os.path.dirname(ROOT), capture_output=True,
                            text=True).stdout.strip()
    man = dict(schema="sample_gallery_v0", commit=commit,
               teacher=dict(version=TEACHER_VERSION, beta=BETA,
                            budget="fixed medium B_signal(等交付纹素±1%, "
                                   "frac=0.5, r_cap=2048)"),
               selection=["logr 方差最低", "logr 方差中位", "logr 方差最高",
                          "chart 数最多", "G_HF_eq 最大"],
               accepted_pool=len(rows), entries=entries,
               semantics="packed UV/rebake 为冻结 Teacher 真实 xatlas+"
                         "texel-center baker 产物; source UV 与 target 分开标注")
    json.dump(man, open(f"{OUT}/gallery_manifest.json", "w"),
              indent=1, ensure_ascii=False)
    readme = ["# Sample Gallery v0\n",
              "从 TexVerse-256 构建中的 accepted 对象确定性选 5 个; packed UV 与",
              "rebake 来自冻结 Teacher(β=0.25)真实 packing(等 B_signal 公平轴)。\n"]
    for e in entries:
        readme.append(f"- `{e['file']}` — {e['reason']}: uid `{e['uid']}`, "
                      f"{e['n_charts']} charts, G_HF_eq={e['G_HF_eq']:+.3f}, "
                      f"{e['size_kb']}KB")
    open(f"{OUT}/README.md", "w").write("\n".join(readme) + "\n")
    print(json.dumps([{k: e[k] for k in ('file', 'reason', 'size_kb')}
                      for e in entries], ensure_ascii=False, indent=1))
    print("GALLERY: DONE")


if __name__ == "__main__":
    main()
