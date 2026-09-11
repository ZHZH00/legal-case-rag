"""使用专门的重排模型筛选 Hybrid 检索产生的候选案件。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openrouter import OpenRouter


# 重排模型适合直接比较“用户案情 + 候选案件”，比向量距离更精细。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
RERANK_MODEL = "qwen/qwen3-reranker-8b"
# 绝对阈值过滤模型明确判断为低相关的案件。
MIN_RELEVANCE_SCORE = 0.01
# 相对阈值过滤与第一名分数差距过大的案件。
RELATIVE_SCORE_RATIO = 0.1
MAX_DOCUMENT_CHARACTERS = 3000
# 评测表明保留 Hybrid 前两名可以减少 Reranker 打乱头部结果的风险。
PROTECTED_HYBRID_COUNT = 2
# Hybrid 头部案件的分数达到最高分的 70% 时才继续保护其位置。
HYBRID_PROTECTION_SCORE_RATIO = 0.7

# 这段指令要求模型优先比较核心犯罪行为，避免被通用量刑词干扰。
RERANK_INSTRUCTION = (
    "请判断历史刑事案件与用户案情在核心行为方式、行为对象、危害结果、"
    "争议焦点和关键情节上的事实相似度。罪名或核心行为明显不同的案件应降低分数；"
    "不要仅因如实供述、退赔、谅解等常见量刑词相同而判为高度相关。"
)


def _get_api_key():
    """读取重排接口使用的 OpenRouter API Key。"""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("没有找到 OPENROUTER_API_KEY，无法执行案件重排")
    return api_key


def build_rerank_document(case):
    """把罪名和案件事实整理成重排模型需要比较的一段文本。"""
    # 限制超长案件可以减少请求耗时，并降低整篇判决书中通用表述的干扰。
    fact = case["content"][:MAX_DOCUMENT_CHARACTERS]
    charge = case["metadata"].get("charge") or "未提供"
    return f"罪名：{charge}\n案件事实：{fact}"


def _read_results(response):
    """兼容 SDK 对象和普通字典两种重排响应形式。"""
    if isinstance(response, dict):
        return response.get("results", [])
    return getattr(response, "results", [])


def _read_result_value(result, field_name):
    """从一条重排结果中读取下标或相关性分数。"""
    if isinstance(result, dict):
        return result.get(field_name)
    return getattr(result, field_name, None)


def rerank_cases(
    case_fact,
    candidate_cases,
    top_k=3,
    min_score=MIN_RELEVANCE_SCORE,
    relative_score_ratio=RELATIVE_SCORE_RATIO,
):
    """重新计算候选案件相关性，过滤低分案件并返回最终 Top K。"""
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not 0 <= min_score <= 1:
        raise ValueError("min_score 必须在 0 到 1 之间")
    if not 0 <= relative_score_ratio <= 1:
        raise ValueError("relative_score_ratio 必须在 0 到 1 之间")
    if not candidate_cases:
        return []

    # 查询中加入法律类案判断标准，让重排优先关注核心事实而非通用措辞。
    rerank_query = f"{RERANK_INSTRUCTION}\n\n用户案情：{clean_case_fact}"
    documents = [build_rerank_document(case) for case in candidate_cases]

    try:
        client = OpenRouter(api_key=_get_api_key(), timeout_ms=60_000)
        response = client.rerank.rerank(
            model=RERANK_MODEL,
            query=rerank_query,
            documents=documents,
            top_n=len(documents),
        )
    except Exception as error:
        raise RuntimeError(f"OpenRouter 案件重排请求失败：{error}") from error

    # 最终 score 使用重排相关性分数，同时保留 Hybrid 分数便于后续调试。
    reranked_cases = []
    seen_indexes = set()
    for result in _read_results(response):
        index = _read_result_value(result, "index")
        relevance_score = _read_result_value(result, "relevance_score")
        if not isinstance(index, int) or index in seen_indexes:
            continue
        if not 0 <= index < len(candidate_cases) or relevance_score is None:
            continue

        seen_indexes.add(index)
        score = float(relevance_score)
        candidate = candidate_cases[index]
        reranked_cases.append(
            {
                **candidate,
                "hybrid_score": float(candidate["score"]),
                "rerank_score": score,
                "score": score,
            }
        )

    # Reranker 分数相同时使用 Hybrid 分数区分先后，避免退化为按案件编号排序。
    reranked_cases.sort(
        key=lambda case: (
            -case["rerank_score"],
            -case["hybrid_score"],
            case["id"],
        )
    )
    if not reranked_cases:
        return []

    # 同时使用绝对阈值和相对最高分阈值，避免低分案件被强行凑进 Top K。
    dynamic_min_score = max(
        min_score,
        reranked_cases[0]["rerank_score"] * relative_score_ratio,
    )
    relevant_cases = [
        case for case in reranked_cases if case["rerank_score"] >= dynamic_min_score
    ]
    return relevant_cases[:top_k]


def rerank_cases_with_hybrid_protection(
    case_fact,
    candidate_cases,
    top_k=3,
    protected_count=PROTECTED_HYBRID_COUNT,
):
    """保留Hybrid头部案件，再使用Reranker排列其余候选。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if protected_count < 0:
        raise ValueError("protected_count 不能小于 0")
    if not candidate_cases:
        return []

    # 需要返回的案件都属于保护范围时，直接沿用Hybrid顺序即可。
    actual_protected_count = min(protected_count, top_k, len(candidate_cases))
    if top_k <= actual_protected_count:
        return candidate_cases[:top_k]

    # 关闭过滤并读取全部候选排名，确保保护后的空位仍能由Reranker补足。
    reranked_cases = rerank_cases(
        case_fact,
        candidate_cases,
        top_k=len(candidate_cases),
        min_score=0,
        relative_score_ratio=0,
    )
    reranked_by_id = {case["id"]: case for case in reranked_cases}

    # Hybrid 前两名只有与最高分差距不大时才保留，低分案件回到重排顺序。
    best_rerank_score = reranked_cases[0]["rerank_score"]
    protected_cases = []
    for candidate in candidate_cases[:actual_protected_count]:
        reranked_candidate = reranked_by_id.get(candidate["id"])
        if (
            reranked_candidate
            and reranked_candidate["rerank_score"]
            >= best_rerank_score * HYBRID_PROTECTION_SCORE_RATIO
        ):
            protected_cases.append(reranked_candidate)
    protected_ids = {case["id"] for case in protected_cases}

    # 删除与保护案件重复的结果后，按Reranker顺序补齐最终Top K。
    remaining_cases = [
        case for case in reranked_cases if case["id"] not in protected_ids
    ]
    return (protected_cases + remaining_cases)[:top_k]
