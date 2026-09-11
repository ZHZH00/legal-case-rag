"""使用RRF融合Dense与BM25结果。"""

from backend.services.keyword_retriever import retrieve_keyword_cases
from backend.services.retriever import DEFAULT_TOP_K, retrieve_relevant_cases


# 配置RRF平滑常数和单路候选量。
RRF_RANK_CONSTANT = 60
DEFAULT_CANDIDATE_K = 20


def add_ranked_results(fused_results, results, source_name, rank_constant):
    """累加单路结果的RRF分数。"""
    # 排名从1开始计算RRF分值。
    for rank, result in enumerate(results, start=1):
        case_id = result["id"]
        # 首次命中时创建统一候选结构。
        if case_id not in fused_results:
            fused_results[case_id] = {
                "id": case_id,
                "content": result["content"],
                "metadata": result["metadata"],
                "score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
            }

        # 多路命中共享同一条累计记录。
        fused_result = fused_results[case_id]
        fused_result["score"] += 1 / (rank_constant + rank)
        fused_result[f"{source_name}_rank"] = rank


def retrieve_hybrid_cases(
    case_fact,
    top_k=DEFAULT_TOP_K,
    candidate_k=DEFAULT_CANDIDATE_K,
    rank_constant=RRF_RANK_CONSTANT,
):
    """执行双路检索并返回RRF排序。"""
    # 校验查询与融合参数。
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0")
    if rank_constant < 0:
        raise ValueError("rank_constant 不能小于 0")

    # 每路先召回足量候选。
    candidate_limit = max(top_k, candidate_k)
    # 分别获取语义和关键词结果。
    vector_results = retrieve_relevant_cases(
        clean_case_fact,
        top_k=candidate_limit,
    )
    keyword_results = retrieve_keyword_cases(
        clean_case_fact,
        top_k=candidate_limit,
    )

    # 按案件ID合并两路排名。
    fused_results = {}
    add_ranked_results(fused_results, vector_results, "vector", rank_constant)
    add_ranked_results(fused_results, keyword_results, "bm25", rank_constant)

    # 同分时按案件ID稳定排序。
    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: (-result["score"], result["id"]),
    )
    return ranked_results[:top_k]
