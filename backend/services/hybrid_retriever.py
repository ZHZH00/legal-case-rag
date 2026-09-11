"""使用 RRF 合并类案向量检索和 BM25 检索结果。"""

from backend.services.keyword_retriever import retrieve_keyword_cases
from backend.services.retriever import DEFAULT_TOP_K, retrieve_relevant_cases


# RRF 常数控制名次差距的影响，候选数量控制每路检索参与融合的范围。
RRF_RANK_CONSTANT = 60
DEFAULT_CANDIDATE_K = 20


def add_ranked_results(fused_results, results, source_name, rank_constant):
    """把一组有序结果的 RRF 分数累加到统一候选案件中。"""
    # rank 从 1 开始，排名越靠前的案件获得越高的 RRF 分数。
    for rank, result in enumerate(results, start=1):
        case_id = result["id"]
        # 第一次遇到案件时先保存它的正文、Metadata 和两路排名位置。
        if case_id not in fused_results:
            fused_results[case_id] = {
                "id": case_id,
                "content": result["content"],
                "metadata": result["metadata"],
                "score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
            }

        # 同一案件若同时出现在两路结果中，它会获得两次分数累加。
        fused_result = fused_results[case_id]
        fused_result["score"] += 1 / (rank_constant + rank)
        fused_result[f"{source_name}_rank"] = rank


def retrieve_hybrid_cases(
    case_fact,
    top_k=DEFAULT_TOP_K,
    candidate_k=DEFAULT_CANDIDATE_K,
    rank_constant=RRF_RANK_CONSTANT,
):
    """分别执行向量和 BM25 检索，再按 RRF 返回最终类案。"""
    # 统一清理用户案情并检查三个检索参数是否合法。
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if candidate_k <= 0:
        raise ValueError("candidate_k 必须大于 0")
    if rank_constant < 0:
        raise ValueError("rank_constant 不能小于 0")

    # 每路先多取一些候选，融合后再截取最终 top_k。
    candidate_limit = max(top_k, candidate_k)
    # Dense 负责语义相似，BM25 负责关键词重合。
    vector_results = retrieve_relevant_cases(
        clean_case_fact,
        top_k=candidate_limit,
    )
    keyword_results = retrieve_keyword_cases(
        clean_case_fact,
        top_k=candidate_limit,
    )

    # 两路结果使用案件 pid 去重，并分别记录原始排名。
    fused_results = {}
    add_ranked_results(fused_results, vector_results, "vector", rank_constant)
    add_ranked_results(fused_results, keyword_results, "bm25", rank_constant)

    # 最终按照融合分数降序排列，分数相同时用案件 ID 保证顺序稳定。
    ranked_results = sorted(
        fused_results.values(),
        key=lambda result: (-result["score"], result["id"]),
    )
    return ranked_results[:top_k]
