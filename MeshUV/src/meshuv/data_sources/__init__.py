# -*- coding: utf-8 -*-
"""数据源注册表: builder 通过配置名选择数据源, 不在 builder 内硬编码 URL."""
from .base import DataSource, DownloadStatus  # noqa: F401


def get_source(name, **kw):
    if name == "texverse_1k":
        from .texverse import TexVerse1K
        return TexVerse1K(**kw)
    raise KeyError(f"未知数据源: {name}")
