# -*- coding: utf-8 -*-
"""TexVerse-1K 数据源(clean 版: 枚举/增量 manifest/原子下载/UID 断点).
复用旧 checkout 的下载缓存(只读+可写新增), 不重复下载。"""
import json
import os
import time
import urllib.request

REPO = "YiboZhang2001/TexVerse-1K"
API = "https://huggingface.co/api/datasets/{r}/tree/main/{p}"
RES = "https://huggingface.co/datasets/{r}/resolve/main/{p}"
GLB_ROOT = "glbs/glbs_1k"


class TexVerse:
    def __init__(self, cache_dir):
        self.cache = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(self.cache, exist_ok=True)
        self.manifest = f"{self.cache}/candidates.jsonl"
        self._rows = None

    def _api(self, path, tries=4):
        req = urllib.request.Request(API.format(r=REPO, p=path),
                                     headers={"User-Agent": "meshuv-clean/1"})
        for k in range(tries):
            try:
                return json.load(urllib.request.urlopen(req, timeout=30))
            except Exception:
                if k == tries - 1:
                    raise
                time.sleep(20 * (2 ** k))

    def candidates(self, n):
        if self._rows is None:
            self._rows, seen = [], set()
            if os.path.exists(self.manifest):
                for ln in open(self.manifest):
                    if ln.strip():
                        r = json.loads(ln)
                        if r["uid"] not in seen:
                            seen.add(r["uid"])
                            self._rows.append(r)
        done_shards = {r["shard"] for r in self._rows if r.get("shard_complete")}
        if len(self._rows) < n:
            shards = sorted(e["path"].rsplit("/", 1)[-1]
                            for e in self._api(GLB_ROOT)
                            if e["type"] == "directory")
            with open(self.manifest, "a") as fp:
                for sh in shards:
                    if len(self._rows) >= n:
                        break
                    if sh in done_shards:
                        continue
                    files = sorted(e["path"] for e in self._api(f"{GLB_ROOT}/{sh}")
                                   if e["type"] == "file"
                                   and e["path"].endswith(".glb"))
                    have = {r["uid"] for r in self._rows}
                    for i, rel in enumerate(files):
                        uid = os.path.basename(rel)[:-len("_1024.glb")]
                        if uid in have:
                            continue
                        row = dict(uid=uid, rel=rel, shard=sh,
                                   shard_complete=(i == len(files) - 1))
                        self._rows.append(row)
                        fp.write(json.dumps(row) + "\n")
                    fp.flush()
        return self._rows[:n]

    def fetch(self, row, timeout=90, retries=2):
        dst = f"{self.cache}/{row['uid']}.glb"
        if os.path.exists(dst) and os.path.getsize(dst) > 64:
            return dst
        tmp = dst + ".part"
        for k in range(retries + 1):
            try:
                req = urllib.request.Request(RES.format(r=REPO, p=row["rel"]),
                                             headers={"User-Agent": "meshuv/1"})
                with urllib.request.urlopen(req, timeout=timeout) as r, \
                        open(tmp, "wb") as fp:
                    while True:
                        b = r.read(1 << 20)
                        if not b:
                            break
                        fp.write(b)
                if os.path.getsize(tmp) < 64:
                    raise IOError("体积异常")
                os.replace(tmp, dst)
                return dst
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                if k < retries:
                    time.sleep(10)
        return None
