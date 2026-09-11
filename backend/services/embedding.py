"""创建统一的 LangChain Embedding 对象供建库和查询使用。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openrouter import OpenRouter


# 项目根目录用于定位本地 .env 配置文件。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# 历史案件和用户案情必须使用相同的模型与向量维度。
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIMENSIONS = 1024

# 查询指令告诉模型要寻找事实和情节相似的中文刑事案件。
QUERY_INSTRUCTION = (
     "根据一段中文刑事案件事实描述，"
    "检索事实和情节相似的历史刑事案件。"
)


def _get_api_key():
    """从环境变量读取 OpenRouter API Key。"""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "没有找到 OPENROUTER_API_KEY。请把 .env.example 复制为 .env，"
            "并填写你的 OpenRouter API Key。"
        )
    return api_key


class OpenRouterEmbeddings(Embeddings):
    """把 OpenRouter Embedding 接口适配成 LangChain 的标准调用形式。"""

    def __init__(self):
        # SDK 负责发送请求、解析响应和处理临时网络错误。
        self.client = OpenRouter(api_key=_get_api_key, timeout_ms=90_000)

    def _embed(self, texts, input_type):
        """提交一组文本并按原始顺序返回对应向量。"""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("生成向量的文本不能为空")

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

        # index 表示每个向量对应输入列表中的哪一段文本。
        ordered_items = sorted(
            response.data,
            key=lambda item: item.index if item.index is not None else 0,
        )
        embeddings = [list(item.embedding) for item in ordered_items]

        # 数量和维度不一致时停止入库，避免案件与向量错位。
        if len(embeddings) != len(texts):
            raise RuntimeError("OpenRouter 返回的向量数量与输入文本数量不一致")
        if any(len(embedding) != EMBEDDING_DIMENSIONS for embedding in embeddings):
            raise RuntimeError(f"OpenRouter 返回的向量不是 {EMBEDDING_DIMENSIONS} 维")

        return embeddings

    def embed_documents(self, texts):
        """把历史案件 fact 列表转换成文档向量列表。"""
        return self._embed(texts, input_type="search_document")

    def embed_query(self, text):
        """把一段用户案情转换成查询向量。"""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("问题不能为空")

        instructed_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {clean_text}"
        return self._embed([instructed_query], input_type="search_query")[0]


# 其他文件直接使用这个对象调用 embed_documents() 或 embed_query()。
embedding_model = OpenRouterEmbeddings()
