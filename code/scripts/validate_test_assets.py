# -*- coding: utf-8 -*-
"""验证新增测试资产: check_asset_support + measure_source_budget 分类入库.
核对合成资产的预期行为(UNSUPPORTED/reuse/岛数), 输出 CATALOG 追加段."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh

from tdlib.api import check_asset_support, measure_source_budget

DATA = "/root/youjiaZhang/PartUV/code/data"
EXPECT_UNSUPPORTED = {"synth_tiled_uv", "synth_udim_like", "synth_no_uv",
                      "synth_vertex_color"}
# 注: synth_mirrored_uv 的镜像重叠发生在单一岛内部, 岛级 reuse 统计天然不可见
# (api 已声明该局限) —— 预期 reuse=1, 该资产保留为"岛内自重叠"局限的展示例。
EXPECT = {"synth_trimsheet_reuse6": ("reuse", 6.0, 0.15),
          "synth_mirrored_uv": ("reuse", 1.0, 0.05),
          "synth_twin_shell": ("reuse", 2.0, 0.05),
          "synth_many_islands_144": ("islands", 144, 0)}

files = sorted(glob.glob(f"{DATA}/sample_*.glb") + glob.glob(f"{DATA}/synth_*.glb"))
rows, fails = [], []
for p in files:
    name = os.path.splitext(os.path.basename(p))[0]
    try:
        s = check_asset_support(p)
    except Exception as e:
        rows.append((name, "读取异常", str(e)[:50], "", "", "", ""))
        fails.append(f"{name}:crash")
        continue
    if not s["supported"]:
        ok_exp = name in EXPECT_UNSUPPORTED
        rows.append((name, "UNSUPPORTED", s["reason"][:46].replace("UNSUPPORTED: ", ""),
                     "", "", "", "✓预期" if ok_exp else "✗意外"))
        if not ok_exp:
            fails.append(f"{name}:unexpected_unsupported")
        continue
    if name in EXPECT_UNSUPPORTED:
        rows.append((name, "supported", "", "", "", "", "✗应为UNSUPPORTED"))
        fails.append(f"{name}:should_be_unsupported")
        continue
    orig = trimesh.load(p, force="mesh")
    b = measure_source_budget(orig)
    note = ""
    if name in EXPECT:
        kind, val, tol = EXPECT[name]
        got = (b["source_reuse_factor"] if kind == "reuse"
               else s.get("n_uv_islands", -1))
        ok = abs(got - val) <= max(tol * val, tol)
        note = f"{'✓' if ok else '✗'}预期{kind}≈{val}(实测{got:.2f})" if kind == "reuse" \
            else f"{'✓' if ok else '✗'}预期{kind}={val}(实测{got})"
        if not ok:
            fails.append(f"{name}:{kind}_mismatch")
    ts = s["tex_shape"]
    rows.append((name, "supported",
                 f"{len(orig.faces):,}面 {ts[1]}x{ts[0]}",
                 f"{s.get('n_uv_islands', '?')}岛",
                 f"{b['source_B_surface'] / 1e6:.2f}M",
                 f"{b['source_reuse_factor']:.2f}x", note))

hdr = f"{'资产':34s} {'状态':12s} {'规格/原因':28s} {'岛':>7s} {'B_surface':>9s} {'reuse':>6s}  备注"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r[0]:34s} {r[1]:12s} {r[2]:28s} {r[3]:>7s} {r[4]:>9s} {r[5]:>6s}  {r[6]}")
print("\nRESULT:", "ALL PASS" if not fails else f"FAILS={fails}")
sys.exit(0 if not fails else 1)
