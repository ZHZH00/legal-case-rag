"""使用专用模型重排Hybrid候选案件。"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openrouter import OpenRouter


# 加载重排模型配置。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
RERANK_MODEL = "qwen/qwen3-reranker-8b"
# 绝对相关性下限。
MIN_RELEVANCE_SCORE = 0.01
# 相对最高分下限。
RELATIVE_SCORE_RATIO = 0.1
MAX_DOCUMENT_CHARACTERS = 3000
# 固定保护的Hybrid头部数量。
PROTECTED_HYBRID_COUNT = 2

# 重排指令聚焦核心犯罪事实。
RERANK_INSTRUCTION = (
    "请判断历史刑事案件与用户案情在核心行为方式、行为对象、危害结果、"
    "争议焦点和关键情节上的事实相似度。罪名或核心行为明显不同的案件应降低分数；"
    "不要仅因如实供述、退赔、谅解等常见量刑词相同而判为高度相关。"
)


def _get_api_key():
    """读取OpenRouter密钥。"""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("没有找到 OPENROUTER_API_KEY，无法执行案件重排")
    return api_key


def build_rerank_document(case):
    """构建受长度限制的重排文档。"""
    # 截断长文以控制噪声和请求体积。
    fact = case["content"][:MAX_DOCUMENT_CHARACTERS]
    charge = case["metadata"].get("charge") or "未提供"
    return f"罪名：{charge}\n案件事实：{fact}"


def _read_results(response):
    """读取字典或SDK形式的结果列表。"""
    if isinstance(response, dict):
        return response.get("results", [])
    return getattr(response, "results", [])


def _read_result_value(result, field_name):
    """读取字典或SDK对象字段。"""
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
    """重排候选并过滤低分结果。"""
    # 校验查询、数量和过滤阈值。
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

    # 将类案判断标准加入查询。
    rerank_query = f"{RERANK_INSTRUCTION}\n\n用户案情：{clean_case_fact}"
    documents = [build_rerank_document(case) for case in candidate_cases]

    # 请求全部候选的重排分数。
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

    # 解析结果并保留两阶段分数。
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

    # 同分时优先Hybrid分数。
    reranked_cases.sort(
        key=lambda case: (
            -case["rerank_score"],
            -case["hybrid_score"],
            case["id"],
        )
    )
    if not reranked_cases:
        return []

    # 合并绝对与相对过滤阈值。
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
    """固定保护Hybrid头部并补入重排结果。"""
    # 校验返回数量和保护数量。
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if protected_count < 0:
        raise ValueError("protected_count 不能小于 0")
    if not candidate_cases:
        return []

    # Top K未超出保护范围时直接返回。
    actual_protected_count = min(protected_count, top_k, len(candidate_cases))
    if top_k <= actual_protected_count:
        return candidate_cases[:top_k]

    # 关闭过滤以保留完整重排候选。
    reranked_cases = rerank_cases(
        case_fact,
        candidate_cases,
        top_k=len(candidate_cases),
        min_score=0,
        relative_score_ratio=0,
    )
    reranked_by_id = {case["id"]: case for case in reranked_cases}

    # 取回带重排分数的受保护案件。
    protected_cases = [
        reranked_by_id[candidate["id"]]
        for candidate in candidate_cases[:actual_protected_count]
        if candidate["id"] in reranked_by_id
    ]
    protected_ids = {case["id"] for case in protected_cases}

    # 去重后按重排顺序补足Top K。
    remaining_cases = [
        case for case in reranked_cases if case["id"] not in protected_ids
    ]
    return (protected_cases + remaining_cases)[:top_k]
