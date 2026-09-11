"""把用户案情向量化，并从 Chroma 返回事实最相似的案件。"""

from backend.services.embedding import embedding_model
from backend.services.vector_store import query_cases


# Dense 检索默认返回最相似的五个历史案件。
DEFAULT_TOP_K = 5


def retrieve_relevant_cases(case_fact, top_k=DEFAULT_TOP_K):
    """完成“用户案情 → Embedding → Chroma 查询”的向量检索。"""
    # 去除首尾空白后再校验，避免空案情调用远程 Embedding 服务。
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")

    # 用户案情先转换为查询向量，再交给 Chroma 计算语义相似度。
    query_embedding = embedding_model.embed_query(clean_case_fact)
    # 返回值包含案件事实、Metadata、距离和相似度分数。
    return query_cases(query_embedding, top_k=top_k)
