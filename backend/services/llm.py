"""创建 LangChain 的 OpenRouter 聊天模型，集中保存模型调用配置。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


# 系统环境变量优先；项目根目录中的 .env 只作为本地开发时的可选补充。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# 默认使用快速版 DeepSeek 生成类案对比，并启用中等强度推理。
CHAT_MODEL = "deepseek/deepseek-v4-flash-0731"


def _get_api_key():
    """读取 OpenRouter API Key，并在真正创建模型前给出清楚的缺失提示。"""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "没有找到 OPENROUTER_API_KEY。请确认当前终端能够读取该系统环境变量。"
        )
    return api_key


def create_chat_model():
    """返回配置好的 ChatOpenRouter，后续可以直接放入 LangChain Chain。"""
    # 所有聊天模型参数集中在这里，rag.py 不需要关心具体服务配置。
    return ChatOpenRouter(
        model=CHAT_MODEL,
        api_key=_get_api_key(),
        # 类案比较强调忠实于历史案件，低温度可以减少模型随意发挥。
        temperature=0,
        # 使用中等推理强度分析类案，但不把模型的内部推理过程返回给前端。
        reasoning={"effort": "medium", "exclude": True},
        # 中等推理会占用部分输出预算，因此预留六千 Token 防止正文中途截断。
        max_tokens=6000,
        # ChatOpenRouter 的 timeout 单位是毫秒，60_000 表示单次最多等待 60 秒。
        timeout=60_000,
        # 交互式问答只额外重试一次，避免网络异常时让用户等待数分钟。
        max_retries=1,
    )
