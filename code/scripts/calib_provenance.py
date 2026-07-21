# -*- coding: utf-8 -*-
"""Calibration 资产溯源 —— public-source Objaverse calibration 标注.

为冻结 50 资产补: uid / source URL / creator / license_id /
metadata snapshot time / raw SHA-256 / random|challenge / selection_rank。
license 经 Sketchfab API(objaverse uid=Sketchfab model id)获取; 拉取失败或
许可未知/不兼容 -> release_or_training_eligible=false(不影响本轮工程诊断,
但不得静默进入未来发布数据集)。
输出: calibration_manifest_v2.json(原 manifest 保留不覆盖)。
"""
import gzip
import hashlib
import json
import os
import time
import urllib.request

CODE = "/root/youjiaZhang/PartUV/code"
OUTD = f"{CODE}/notebook/outputs/calibration_v1"
OPATH = os.path.expanduser("~/.objaverse/hf-objaverse-v1/object-paths.json.gz")

OK_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "cc0", "by", "by-sa",
               "CC Attribution", "CC0 Public Domain"}

man = json.load(open(f"{OUTD}/calibration_manifest.json"))
obj_paths = json.load(gzip.open(OPATH, "rt"))
snap = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

for rank, a in enumerate(man["assets"]):
    uid = a["uid"]
    a["selection_rank"] = rank
    a["raw_sha256"] = hashlib.sha256(open(a["glb"], "rb").read()).hexdigest()
    rel = obj_paths.get(uid, "")
    a["source_url"] = (f"https://huggingface.co/datasets/allenai/objaverse/"
                       f"blob/main/{rel}" if rel else "")
    a["sketchfab_url"] = f"https://sketchfab.com/3d-models/{uid}"
    a["metadata_snapshot_time"] = snap
    a["creator"] = a["license_id"] = "UNKNOWN"
    try:
        req = urllib.request.Request(
            f"https://api.sketchfab.com/v3/models/{uid}",
            headers={"User-Agent": "calibration-provenance/1.0"})
        with urllib.request.urlopen(req, timeout=15) as fp:
            meta = json.load(fp)
        a["creator"] = (meta.get("user", {}) or {}).get("username", "UNKNOWN")
        lic = meta.get("license", {}) or {}
        a["license_id"] = lic.get("slug") or lic.get("label") or "UNKNOWN"
    except Exception as e:
        a["provenance_error"] = f"{type(e).__name__}"
    a["release_or_training_eligible"] = a["license_id"] in OK_LICENSES
    print(f"[{rank:02d}] {uid[:12]} license={a['license_id']:24s} "
          f"creator={a['creator'][:20]:20s} "
          f"eligible={a['release_or_training_eligible']}", flush=True)
    time.sleep(0.4)                                # API 限速礼貌间隔

man["domain_label"] = ("PUBLIC-SOURCE OBJAVERSE CALIBRATION; 不能代表 Meshy "
                       "target-domain performance(内部数据本机不可访问)")
man["provenance"] = dict(
    snapshot_time=snap,
    license_policy="未知/不兼容许可 -> release_or_training_eligible=false, "
                   "不影响本轮工程诊断, 不得静默进入发布数据集",
    ok_licenses=sorted(OK_LICENSES))
with open(f"{OUTD}/calibration_manifest_v2.json", "w") as fp:
    json.dump(man, fp, indent=1, ensure_ascii=False)
n_ok = sum(a["release_or_training_eligible"] for a in man["assets"])
print(f"\nrelease_or_training_eligible: {n_ok}/50")
print("PROVENANCE: DONE")
