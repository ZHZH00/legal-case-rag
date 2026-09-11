"""提供统一的案件向量生成服务。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openrouter import OpenRouter


# 加载项目级环境变量。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# 建库与查询共用同一向量配置。
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIMENSIONS = 1024

# 查询指令限定刑事类案语义。
QUERY_INSTRUCTION = (
     "根据一段中文刑事案件事实描述，"
    "检索事实和情节相似的历史刑事案件。"
)


def _get_api_key():
    """读取OpenRouter密钥。"""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "没有找到 OPENROUTER_API_KEY。请把 .env.example 复制为 .env，"
            "并填写你的 OpenRouter API Key。"
        )
    return api_key


class OpenRouterEmbeddings(Embeddings):
    """适配OpenRouter与LangChain Embeddings接口。"""

    def __init__(self):
        # 复用统一的OpenRouter客户端。
        self.client = OpenRouter(api_key=_get_api_key, timeout_ms=90_000)

    def _embed(self, texts, input_type):
        """批量生成并校验向量。"""
        # 空批次直接返回。
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("生成向量的文本不能为空")

        # 调用指定输入类型的Embedding接口。
        try:
            response = self.client.embeddings.generate(
                model=EMBEDDING_MODEL,
                input=texts,
                dimensions=EMBEDDING_DIMENSIONS,
                encoding_format="float",
                input_type=input_type,
            )
        except Exception as error:
            raise RuntimeError(f"OpenRouter Embedding 请求失败：{error}") from error

        if isinstance(response, str):
            raise RuntimeError(f"OpenRouter 返回了无法解析的结果：{response[:200]}")

        # 按响应下标恢复输入顺序。
        ordered_items = sorted(
            response.data,
            key=lambda item: item.index if item.index is not None else 0,
        )
        embeddings = [list(item.embedding) for item in ordered_items]

        # 拒绝数量或维度异常的响应。
        if len(embeddings) != len(texts):
            raise RuntimeError("OpenRouter 返回的向量数量与输入文本数量不一致")
        if any(len(embedding) != EMBEDDING_DIMENSIONS for embedding in embeddings):
            raise RuntimeError(f"OpenRouter 返回的向量不是 {EMBEDDING_DIMENSIONS} 维")

        return embeddings

    def embed_documents(self, texts):
        """生成案件文档向量。"""
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text):
        """生成带检索指令的查询向量。"""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("问题不能为空")

        instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {clean_text}"
        return self._embed([instructed_query], input_type="search_query")[0]


# 全局实例供入库与检索复用。
embedding_model = OpenRouterEmbeddings()
