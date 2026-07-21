# -*- coding: utf-8 -*-
"""数据源抽象: 枚举候选 UID -> 解析远程 GLB -> 按需原子下载 -> 稳定本地路径.

断点恢复语义:
- 每对象独立状态文件 <cache>/status/<uid>.json;
- 下载写 <dst>.part 临时文件, 成功后原子 os.replace -> 失败绝不伪装成完成;
- 已完成文件按 size(+可选 sha256)校验后复用, 不重复下载;
- candidate manifest 由子类增量落盘(JSONL append), 不依赖一次性全量写。
"""
import hashlib
import json
import os
import time
import urllib.request

# 对象生命周期状态(下载层产生前 4 种, 管线层产生其余)
DownloadStatus = ("PENDING", "DOWNLOADING", "DOWNLOADED", "ERROR")
PIPELINE_STATUS = ("PRECHECK_REJECTED", "PARTUV_FAILED", "PACKING_FAILED",
                   "TIMEOUT", "QA_REJECTED", "ACCEPTED", "ERROR")


class DataSource:
    """子类实现 list_candidates() 与 resolve_url()."""

    name = "base"

    def __init__(self, cache_dir):
        self.cache = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(f"{self.cache}/status", exist_ok=True)

    # ---- 子类接口 ----
    def list_candidates(self, n):
        """确定性顺序返回前 n 个候选 uid(增量 manifest 由子类维护)."""
        raise NotImplementedError

    def resolve_url(self, uid):
        raise NotImplementedError

    def local_path(self, uid):
        return f"{self.cache}/{uid}.glb"

    # ---- 通用下载(原子/可恢复/状态记录) ----
    def _status_path(self, uid):
        return f"{self.cache}/status/{uid}.json"

    def _write_status(self, uid, **kw):
        kw.setdefault("uid", uid)
        kw.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with open(self._status_path(uid), "w") as fp:
            json.dump(kw, fp, ensure_ascii=False)
        return kw

    def read_status(self, uid):
        p = self._status_path(uid)
        return json.load(open(p)) if os.path.exists(p) else dict(
            uid=uid, status="PENDING")

    def ensure_local(self, uid, timeout=120):
        """返回本地 GLB 路径; 失败返回 None 并记录原因. 幂等可断点."""
        dst = self.local_path(uid)
        st = self.read_status(uid)
        if st.get("status") == "DOWNLOADED" and os.path.exists(dst):
            if os.path.getsize(dst) == st.get("size", -1):
                return dst                       # 校验通过, 复用
        tmp = dst + ".part"
        self._write_status(uid, status="DOWNLOADING")
        try:
            url = self.resolve_url(uid)
            t0 = time.time()
            urllib.request.urlretrieve(url, tmp)
            size = os.path.getsize(tmp)
            h = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
            os.replace(tmp, dst)                 # 原子重命名: 完成才可见
            self._write_status(uid, status="DOWNLOADED", size=size,
                               sha256=h, url=url,
                               seconds=round(time.time() - t0, 2))
            return dst
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)                   # 半成品绝不留作完成文件
            self._write_status(uid, status="ERROR",
                               error=f"{type(e).__name__}: {str(e)[:120]}")
            return None
