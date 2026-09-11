"""提供基于中文分词的BM25检索。"""

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


# 静默jieba初始化日志。
jieba.setLogLevel(logging.WARNING)
# 仅保留可检索字符。
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9._+-]*", re.IGNORECASE)
# BM25索引持久化路径。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BM25_INDEX_PATH = PROJECT_ROOT / "backend" / "data" / "bm25" / "legal_cases_bm25.pkl"


def tokenize_text(text):
    """将案件文本转换为BM25词项。"""
    # 分词后过滤标点与空白。
    tokens = []
    for word in jieba.lcut(text.lower()):
        tokens.extend(TOKEN_PATTERN.findall(word))
    return tokens


def build_keyword_index(progress_callback=None):
    """从Chroma全量案件构建并保存BM25索引。"""
    # 读取已入库案件作为语料。
    cases = get_all_cases()
    if not cases:
        raise RuntimeError("Chroma 中没有案件，无法建立 BM25 索引")

    # 逐案分词并报告进度。
    tokenized_corpus = []
    total = len(cases)
    for processed, case in enumerate(cases, start=1):
        tokenized_corpus.append(tokenize_text(case["content"]))
        if progress_callback and (processed % 1000 == 0 or processed == total):
            progress_callback(processed, total)

    # 索引仅保存案件ID和词项统计。
    payload = {
        "case_count": total,
        "case_ids": [case["id"] for case in cases],
        "bm25_index": BM25Okapi(tokenized_corpus),
    }
    # 临时文件写完后原子替换正式索引。
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
    """加载并校验本地BM25索引。"""
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

    # 案件数量必须与Chroma一致。
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
    """返回BM25得分最高的案件。"""
    # 校验查询和返回数量。
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 索引首次加载后驻留内存。
    case_ids, bm25_index = load_keyword_index()

    # 查询与语料使用相同分词规则。
    query_tokens = tokenize_text(clean_case_fact)
    if not query_tokens:
        return []

    # 计算全库BM25分数。
    scores = bm25_index.get_scores(query_tokens)
    ranked_indexes = sorted(
        range(len(case_ids)),
        key=lambda index: scores[index],
        reverse=True,
    )
    # 过滤零分结果并截取Top K。
    ranked_indexes = [
        index for index in ranked_indexes if scores[index] > 0
    ][: min(top_k, len(case_ids))]

    # 按命中ID回查案件正文。
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
