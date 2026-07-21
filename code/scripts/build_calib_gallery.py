# -*- coding: utf-8 -*-
"""生成 50 资产 calibration 可视化总览 HTML(缩略图 base64 内嵌, 快照式).
状态实时读 result/result_driver, 可随运行推进重新生成。"""
import base64
import html as H
import json
import os
import sys
from collections import Counter

CODE = "/root/youjiaZhang/PartUV/code"
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
DEST = sys.argv[1] if len(sys.argv) > 1 else f"{OUTD}/gallery.html"
man = json.load(open(f"{OUTD}/calibration_manifest_v2.json"))

STATUS_UI = {"OK": ("已评价", "ok"), "PACKING_FAILED": ("打包失败", "crit"),
             "PROCESSING_TIMEOUT": ("超时>30min", "warn"),
             "PRECHECK_REJECTED": ("预检拒绝", "warn"),
             "PARTUV_FAILED": ("PartUV失败", "crit"),
             "ERROR": ("异常", "crit"), "PENDING": ("待处理", "mut")}
LBL_UI = {"POSITIVE": "pos", "NEUTRAL": "neu", "MIXED": "mix",
          "NEGATIVE": "neg", "NOT_EVALUATED": "mut", "-": "mut"}
GROUP_CN = {"random": "随机", "low_texture": "低纹理", "local_logo": "局部logo",
            "distributed_hf": "分布高频", "many_charts": "many-chart",
            "multimat_overlap": "多材质", "geometry_misc": "几何复杂"}

cards, stat_cnt = [], Counter()
for a in sorted(man["assets"], key=lambda x: x["selection_rank"]):
    oid = a["object_id"]
    r = {}
    for f in (f"{OUTD}/{oid}/result.json", f"{OUTD}/{oid}/result_driver.json"):
        if os.path.exists(f):
            r = json.load(open(f))
            break
    st = r.get("processing_status", "PENDING")
    if st == "ERROR" and "PackingFailedError" in r.get("reason", ""):
        st = "PACKING_FAILED"
    stat_cnt[st] += 1
    st_txt, st_cls = STATUS_UI.get(st, (st, "mut"))
    labs = {b: r.get("betas", {}).get(b, {}).get("label", "-")
            for b in ("0.125", "0.25")}
    thumb = f"{OUTD}/thumbs/{oid}.jpg"
    b64 = base64.b64encode(open(thumb, "rb").read()).decode() \
        if os.path.exists(thumb) else ""
    d = a["descriptors"]
    lic = a.get("license_id", "UNKNOWN")
    elig = a.get("release_or_training_eligible", False)
    beta_chips = "".join(
        f'<span class="chip l-{LBL_UI[labs[b]]}">β{b.lstrip("0")}: '
        f'{H.escape(labs[b])}</span>' for b in ("0.125", "0.25")) \
        if st == "OK" else f'<span class="chip l-mut">{H.escape(r.get("reason", "")[:46]) or "—"}</span>'
    cards.append(f'''
<div class="card s-{st_cls}" data-st="{st}" data-g="{a['group']}">
 <div class="imgw"><img loading="lazy" src="data:image/jpeg;base64,{b64}" alt="{oid}">
  <span class="pill p-{st_cls}">{st_txt}</span></div>
 <div class="body">
  <div class="row1"><span class="oid">{oid}</span>
   <span class="tag {'t-ch' if a['group'] != 'random' else 't-rd'}">{GROUP_CN.get(a['group'], a['group'])}</span></div>
  <div class="chips">{beta_chips}</div>
  <div class="meta">{d['n_faces']:,} 面 · {d['n_islands']} 岛 · {d['n_geoms']} 材质
   <span class="lic {'lic-ok' if elig else 'lic-no'}" title="license: {H.escape(lic)}">{'✓ 许可' if elig else '✗ ' + H.escape(lic[:14])}</span></div>
 </div>
</div>''')

summary = " · ".join(f"{STATUS_UI.get(k, (k,))[0]} {v}" for k, v in
                     sorted(stat_cnt.items(), key=lambda x: -x[1]))
page = f'''<title>β Calibration V1 · 50 资产总览</title>
<style>
:root {{ --bg:#F5F4F1; --card:#FFFFFF; --ink:#23262B; --sub:#6E7178; --line:#E2E0DB;
  --acc:#2C7A8C; --ok:#3E7D4E; --warn:#B07C2E; --crit:#A8433B; --mut:#8A8D93;
  --pos:#3E7D4E; --neu:#7A7D84; --mix:#B07C2E; --neg:#A8433B; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#17191D; --card:#1F2228;
  --ink:#E8E7E3; --sub:#9A9DA4; --line:#2C3037; --acc:#5FAFC0; --ok:#6FBE81;
  --warn:#D6A254; --crit:#D0716A; --mut:#71747B; --pos:#6FBE81; --neu:#8B8E95;
  --mix:#D6A254; --neg:#D0716A; }} }}
:root[data-theme="dark"] {{ --bg:#17191D; --card:#1F2228; --ink:#E8E7E3;
  --sub:#9A9DA4; --line:#2C3037; --acc:#5FAFC0; --ok:#6FBE81; --warn:#D6A254;
  --crit:#D0716A; --mut:#71747B; --pos:#6FBE81; --neu:#8B8E95; --mix:#D6A254; --neg:#D0716A; }}
:root[data-theme="light"] {{ --bg:#F5F4F1; --card:#FFFFFF; --ink:#23262B;
  --sub:#6E7178; --line:#E2E0DB; --acc:#2C7A8C; --ok:#3E7D4E; --warn:#B07C2E;
  --crit:#A8433B; --mut:#8A8D93; --pos:#3E7D4E; --neu:#7A7D84; --mix:#B07C2E; --neg:#A8433B; }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:15px/1.55 "Avenir Next","Seravek","Segoe UI",system-ui,sans-serif; }}
header {{ padding:26px 28px 14px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 4px; font-size:21px; font-weight:600; letter-spacing:.01em; }}
.sub {{ color:var(--sub); font-size:13px; }}
.sub b {{ color:var(--ink); font-weight:600; }}
.filters {{ display:flex; gap:8px; flex-wrap:wrap; padding:12px 28px; }}
.fbtn {{ border:1px solid var(--line); background:var(--card); color:var(--sub);
  border-radius:3px; padding:4px 12px; font-size:12.5px; cursor:pointer; }}
.fbtn.on {{ border-color:var(--acc); color:var(--acc); font-weight:600; }}
.fbtn:focus-visible {{ outline:2px solid var(--acc); outline-offset:1px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(236px,1fr));
  gap:14px; padding:16px 28px 40px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:4px;
  overflow:hidden; border-top:3px solid var(--mut); }}
.card.s-ok {{ border-top-color:var(--ok); }} .card.s-warn {{ border-top-color:var(--warn); }}
.card.s-crit {{ border-top-color:var(--crit); }}
.imgw {{ position:relative; aspect-ratio:1; background:#2A2E35; }}
.imgw img {{ width:100%; height:100%; object-fit:cover; display:block; }}
.pill {{ position:absolute; top:8px; left:8px; font-size:11px; padding:2px 8px;
  border-radius:2px; color:#fff; letter-spacing:.04em; }}
.p-ok {{ background:var(--ok); }} .p-warn {{ background:var(--warn); }}
.p-crit {{ background:var(--crit); }} .p-mut {{ background:var(--mut); }}
.body {{ padding:10px 12px 12px; }}
.row1 {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; }}
.oid {{ font:12px "SF Mono","Cascadia Code",Consolas,monospace; color:var(--ink); }}
.tag {{ font-size:11px; padding:1px 7px; border-radius:2px; white-space:nowrap; }}
.t-ch {{ background:color-mix(in srgb, var(--acc) 14%, transparent); color:var(--acc); }}
.t-rd {{ background:color-mix(in srgb, var(--mut) 16%, transparent); color:var(--sub); }}
.chips {{ display:flex; gap:6px; flex-wrap:wrap; margin:8px 0 6px; }}
.chip {{ font-size:11.5px; padding:2px 7px; border-radius:2px; border:1px solid var(--line); }}
.l-pos {{ color:var(--pos); border-color:var(--pos); }}
.l-neu {{ color:var(--neu); }} .l-mix {{ color:var(--mix); border-color:var(--mix); }}
.l-neg {{ color:var(--neg); border-color:var(--neg); }} .l-mut {{ color:var(--mut); }}
.meta {{ font-size:12px; color:var(--sub); font-variant-numeric:tabular-nums;
  display:flex; justify-content:space-between; gap:6px; flex-wrap:wrap; }}
.lic-ok {{ color:var(--ok); }} .lic-no {{ color:var(--warn); }}
</style>
<header>
 <h1>Global β Calibration V1 — 50 资产总览</h1>
 <div class="sub"><b>public-source Objaverse calibration</b>（不代表 Meshy 目标域） ·
  30 随机 + 20 challenge · β∈{{0, 0.125, 0.25}} · 状态快照：{summary} ·
  许可合规 {sum(1 for a in man['assets'] if a.get('release_or_training_eligible'))}/50</div>
</header>
<div class="filters">
 <button class="fbtn on" data-f="*">全部 50</button>
 <button class="fbtn" data-f="st:OK">已评价</button>
 <button class="fbtn" data-f="st:PACKING_FAILED">打包失败</button>
 <button class="fbtn" data-f="st:PROCESSING_TIMEOUT">超时</button>
 <button class="fbtn" data-f="g:random">随机组</button>
 <button class="fbtn" data-f="ch">challenge 组</button>
</div>
<div class="grid">{''.join(cards)}</div>
<script>
document.querySelectorAll('.fbtn').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.fbtn').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  const f = b.dataset.f;
  document.querySelectorAll('.card').forEach(c => {{
    let show = f === '*';
    if (f === 'ch') show = c.dataset.g !== 'random';
    else if (f.startsWith('st:')) show = c.dataset.st === f.slice(3);
    else if (f.startsWith('g:')) show = c.dataset.g === f.slice(2);
    c.style.display = show ? '' : 'none';
  }});
}});
</script>'''
open(DEST, "w").write(page)
print(f"gallery -> {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")
