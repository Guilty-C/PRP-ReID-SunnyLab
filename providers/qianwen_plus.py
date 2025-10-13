
from __future__ import annotations
from .http_openai_compat import OpenAICompatEmbedTextProvider

class QianwenPlusProvider(OpenAICompatEmbedTextProvider):
    """DashScope 兼容（OpenAI embeddings 接口）。环境变量前缀：QIANWEN_PLUS_*"""
    name = "qianwen-plus"
    def __init__(self):
        super().__init__(prefix="QIANWEN_PLUS")
        # extra header for DashScope
        self.session.headers['X-DashScope-API-Key']=self.api_key
        if not self.api_base:
            self.api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
