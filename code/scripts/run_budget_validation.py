# -*- coding: utf-8 -*-
"""Absolute Texel Budget Fix 验证:
1) 鞋 / 车轮 / 8d1b(4096 多材质) auto 模式 e2e -> 逐资产绝对预算报告;
2) 显式 atlas_size=1024 严格尊重 + 诚实报告预算不足(车轮);
3) auto + max_atlas 过低 -> BUDGET_LIMIT_EXCEEDED 且不导出(车轮)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdlib.api import BudgetLimitExceededError, map_partuv_td

DATA = "/root/youjiaZhang/PartUV/code/data"
OUT = "/root/youjiaZhang/PartUV/code/notebook/outputs/budget_fix"

ASSETS = [
    ("shoe", f"{DATA}/objaverse_22b822c6520d4d49.glb"),
    ("wheel", f"{DATA}/objaverse_92ff65712c62408d.glb"),
    ("multi_8d1b", f"{DATA}/8d1b6fc369484f4c4517d1f44d88471fa2083a75a9d4d7ecef9a880e107729ee.glb"),
]

fails, rows = [], []

for tag, path in ASSETS:
    print(f"\n===== auto e2e: {tag} =====", flush=True)
    res = map_partuv_td(path, f"{OUT}/{tag}/")          # atlas_size="auto" 默认
    b = res["budget"]
    rows.append((tag, b))
    print(f"  B_unique={b['source_B_unique']/1e6:.2f}M  "
          f"B_surface={b['source_B_surface']/1e6:.2f}M  "
          f"reuse={b['source_reuse_factor']:.2f}x", flush=True)
    print(f"  selected R={b['selected_atlas_size']}  "
          f"B_signal_out={b['output_B_signal']/1e6:.2f}M  "
          f"fill={b['output_packing_fill']*100:.0f}%  "
          f"budget_ratio={b['budget_ratio']:.2f}  met={b['budget_met']}", flush=True)
    for wmsg in res["warnings"]:
        print(f"  warning: {wmsg}", flush=True)
    if not b["budget_met"]:
        fails.append(f"{tag}:auto_not_met")
    if not os.path.exists(res["glb_path"]):
        fails.append(f"{tag}:no_glb")

print("\n===== 显式 atlas_size=1024 (wheel): 严格尊重 + 诚实报告 =====", flush=True)
res = map_partuv_td(f"{DATA}/objaverse_92ff65712c62408d.glb",
                    f"{OUT}/wheel_fixed1024/", atlas_size=1024)
b = res["budget"]
print(f"  selected R={b['selected_atlas_size']} budget_ratio={b['budget_ratio']:.2f} "
      f"met={b['budget_met']}", flush=True)
warn_hit = any("外观降级" in w and "WARNING" in w for w in res["warnings"])
print(f"  预算不足 WARNING 出现: {warn_hit}")
if b["selected_atlas_size"] != 1024:
    fails.append("fixed1024:not_respected")
if b["budget_met"] or not warn_hit:
    fails.append("fixed1024:dishonest_report")

print("\n===== auto + max_atlas=1024 (wheel): 必须 BUDGET_LIMIT_EXCEEDED 且不导出 =====",
      flush=True)
limit_dir = f"{OUT}/wheel_limit/"
try:
    map_partuv_td(f"{DATA}/objaverse_92ff65712c62408d.glb", limit_dir,
                  atlas_size="auto", max_atlas=1024)
    fails.append("limit:no_raise")
except BudgetLimitExceededError as e:
    msg = str(e)
    print(f"  抛出: {msg[:110]}...")
    leftovers = [f for f in os.listdir(limit_dir)
                 if "_td_aware" in f] if os.path.isdir(limit_dir) else []
    print(f"  残留资产文件: {leftovers}")
    if "BUDGET_LIMIT_EXCEEDED" not in msg:
        fails.append("limit:tag_missing")
    if leftovers:
        fails.append("limit:files_left")

print("\n===== 预算表 =====")
hdr = (f"{'asset':12s} {'B_unique':>9s} {'B_surface':>10s} {'reuse':>6s} "
       f"{'R':>5s} {'B_signal':>9s} {'fill':>5s} {'ratio':>6s} {'met':>5s}")
print(hdr); print("-" * len(hdr))
for tag, b in rows:
    print(f"{tag:12s} {b['source_B_unique']/1e6:>8.2f}M {b['source_B_surface']/1e6:>9.2f}M "
          f"{b['source_reuse_factor']:>5.2f}x {b['selected_atlas_size']:>5d} "
          f"{b['output_B_signal']/1e6:>8.2f}M {b['output_packing_fill']*100:>4.0f}% "
          f"{b['budget_ratio']:>6.2f} {str(b['budget_met']):>5s}")

print("\nRESULT:", "ALL PASS" if not fails else f"FAILS={fails}")
sys.exit(0 if not fails else 1)
