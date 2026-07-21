# -*- coding: utf-8 -*-
"""TexVerse-1K 数据源(Hugging Face YiboZhang2001/TexVerse-1K).

布局: glbs/glbs_1k/<shard>/<uid>_1024.glb (1K basecolor 版本)。
metadata-first: 逐 shard 调 HF tree API 枚举文件, 候选按 (shard, 文件名) 排序
确定性展开; 枚举结果增量追加到 <cache>/candidates.jsonl(断连后续跑不重列)。
lazy download: 只在 ensure_local(uid) 时下载单个 GLB。
"""
import json
import os
import urllib.request

from .base import DataSource

REPO = "YiboZhang2001/TexVerse-1K"
API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
GLB_ROOT = "glbs/glbs_1k"


class TexVerse1K(DataSource):
    name = "texverse_1k"

    def __init__(self, cache_dir, repo=REPO):
        super().__init__(cache_dir)
        self.repo = repo
        self.manifest_path = f"{self.cache}/candidates.jsonl"
        self._rows = None

    # ---- 枚举(增量落盘) ----
    def _api_json(self, path):
        req = urllib.request.Request(
            API_TREE.format(repo=self.repo, path=path),
            headers={"User-Agent": "meshuv-texverse/0.1"})
        return json.load(urllib.request.urlopen(req, timeout=30))

    def _load_manifest(self):
        rows = []
        if os.path.exists(self.manifest_path):
            for ln in open(self.manifest_path):
                if ln.strip():
                    rows.append(json.loads(ln))
        return rows

    def list_candidates(self, n):
        """前 n 个候选(uid, rel_path). 不足则继续枚举下一 shard 并增量追加."""
        if self._rows is None:
            self._rows = self._load_manifest()
        shards_done = {r["shard"] for r in self._rows if r.get("shard_complete")}
        if len(self._rows) < n:
            shards = sorted(e["path"].rsplit("/", 1)[-1]
                            for e in self._api_json(GLB_ROOT)
                            if e["type"] == "directory")
            with open(self.manifest_path, "a") as fp:
                for sh in shards:
                    if len(self._rows) >= n:
                        break
                    if sh in shards_done:
                        continue
                    files = sorted(e["path"] for e in
                                   self._api_json(f"{GLB_ROOT}/{sh}")
                                   if e["type"] == "file"
                                   and e["path"].endswith(".glb"))
                    for i, rel in enumerate(files):
                        uid = os.path.basename(rel).replace("_1024.glb", "")
                        row = dict(uid=uid, rel_path=rel, shard=sh,
                                   shard_complete=(i == len(files) - 1))
                        self._rows.append(row)
                        fp.write(json.dumps(row) + "\n")   # 增量, 每 shard 落盘
                    fp.flush()
        return self._rows[:n]

    def resolve_url(self, uid):
        if self._rows is None:
            self._rows = self._load_manifest()
        rel = next((r["rel_path"] for r in self._rows if r["uid"] == uid), None)
        if rel is None:                       # 未枚举到 -> 按布局规则构造
            raise KeyError(f"uid 不在候选清单: {uid}")
        return RESOLVE.format(repo=self.repo, path=rel)
