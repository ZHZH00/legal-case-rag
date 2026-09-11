"""集中创建OpenRouter聊天模型。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


# 加载本地开发环境变量。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# 默认类案分析模型。
CHAT_MODEL = "deepseek/deepseek-v4-flash-0731"


def _get_api_key():
    """读取OpenRouter密钥。"""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "没有找到 OPENROUTER_API_KEY。请确认当前终端能够读取该系统环境变量。"
        )
    return api_key


def create_chat_model():
    """返回统一配置的ChatOpenRouter。"""
    # 集中维护模型调用参数。
    return ChatOpenRouter(
        model=CHAT_MODEL,
        api_key=_get_api_key(),
        # 零温度降低事实扩写。
        temperature=0,
        # 使用中等推理且隐藏推理文本。
        reasoning={"effort": "medium", "exclude": True},
        # 预留完整类案分析的输出空间。
        max_tokens=6000,
        # 单次请求超时60秒。
        timeout=60_000,
        # 网络失败最多重试一次。
        max_retries=1,
    )
