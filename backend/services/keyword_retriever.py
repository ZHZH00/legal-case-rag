"""使用中文分词和 BM25，从已入库案件中执行关键词检索。"""

import logging
import pickle
import re
from functools import lru_cache
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from backend.services.vector_store import (
    get_all_cases,
    get_cases_by_ids,
    get_legal_case_collection,
)


# 关闭结巴分词的初始化日志，避免启动时输出无关信息。
jieba.setLogLevel(logging.WARNING)
# 正则只保留中文、英文、数字和常见技术符号作为检索词。
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9._+-]*", re.IGNORECASE)
# BM25 索引保存在项目数据目录，后端重启后仍可直接加载。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BM25_INDEX_PATH = PROJECT_ROOT / "backend" / "data" / "bm25" / "legal_cases_bm25.pkl"


def tokenize_text(text):
    """把案件事实转换成 BM25 可以比较的关键词列表。"""
    # 结巴先切分中文句子，正则再清除标点和空白。
    tokens = []
    for word in jieba.lcut(text.lower()):
        tokens.extend(TOKEN_PATTERN.findall(word))
    return tokens


def build_keyword_index(progress_callback=None):
    """根据 Chroma 中的全部案件建立 BM25 索引并保存到本地。"""
    cases = get_all_cases()
    if not cases:
        raise RuntimeError("Chroma 中没有案件，无法建立 BM25 索引")

    tokenized_corpus = []
    total = len(cases)
    for processed, case in enumerate(cases, start=1):
        tokenized_corpus.append(tokenize_text(case["content"]))
        if progress_callback and (processed % 1000 == 0 or processed == total):
            progress_callback(processed, total)

    # 索引只保存案件 ID，命中后再从 Chroma 读取正文，避免重复保存完整案件。
    payload = {
        "case_count": total,
        "case_ids": [case["id"] for case in cases],
        "bm25_index": BM25Okapi(tokenized_corpus),
    }
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = BM25_INDEX_PATH.with_suffix(".tmp")
    with temporary_path.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(BM25_INDEX_PATH)
    load_keyword_index.cache_clear()

    return {
        "case_count": total,
        "index_path": str(BM25_INDEX_PATH),
        "file_size": BM25_INDEX_PATH.stat().st_size,
    }


@lru_cache(maxsize=1)
def load_keyword_index():
    """从本地文件加载 BM25 索引，并确认它与当前 Chroma 数据一致。"""
    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(
            "没有找到 BM25 索引，请先运行："
            "python -m backend.scripts.build_keyword_index"
        )

    try:
        with BM25_INDEX_PATH.open("rb") as file:
            payload = pickle.load(file)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise RuntimeError("BM25 索引文件无法读取，请重新建立索引") from error

    case_ids = payload.get("case_ids", [])
    bm25_index = payload.get("bm25_index")
    saved_count = payload.get("case_count")
    current_count = get_legal_case_collection().count()
    if (
        bm25_index is None
        or saved_count != len(case_ids)
        or saved_count != current_count
    ):
        raise RuntimeError(
            "BM25 索引与当前 Chroma 案件数量不一致，请重新建立索引："
            "python -m backend.scripts.build_keyword_index"
        )
    return case_ids, bm25_index


def retrieve_keyword_cases(case_fact, top_k=5):
    """使用 BM25 返回关键词最相关的历史案件。"""
    # 案情和 top_k 在建立索引前校验，避免无效计算。
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 第一次查询从本地读取索引，后续查询直接复用内存缓存。
    case_ids, bm25_index = load_keyword_index()

    # 用户案情使用与案件语料相同的分词规则。
    query_tokens = tokenize_text(clean_case_fact)
    if not query_tokens:
        return []

    # BM25 为知识库中的每条案件计算关键词相关性分数。
    scores = bm25_index.get_scores(query_tokens)
    ranked_indexes = sorted(
        range(len(case_ids)),
        key=lambda index: scores[index],
        reverse=True,
    )
    # 只保留得分大于零的前 top_k 条，避免返回完全没有关键词交集的案件。
    ranked_indexes = [
        index for index in ranked_indexes if scores[index] > 0
    ][: min(top_k, len(case_ids))]

    # 命中后只从 Chroma 读取最终案件，并按照 BM25 排名恢复顺序。
    ranked_case_ids = [case_ids[index] for index in ranked_indexes]
    ranked_scores = {
        case_ids[index]: float(scores[index])
        for index in ranked_indexes
    }
    ranked_cases = get_cases_by_ids(ranked_case_ids)
    return [
        {
            **case,
            "score": ranked_scores[case["id"]],
        }
        for case in ranked_cases
    ]
