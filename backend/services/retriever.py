"""提供Chroma Dense案件检索。"""

from backend.services.embedding import embedding_model
from backend.services.vector_store import query_cases


# 默认返回数量。
DEFAULT_TOP_K = 5


def retrieve_relevant_cases(case_fact, top_k=DEFAULT_TOP_K):
    """向量化案情并查询相似案件。"""
    # 空案情不调用Embedding服务。
    clean_case_fact = case_fact.strip()
    if not clean_case_fact:
        raise ValueError("用户案情不能为空")

    # 生成查询向量。
    query_embedding = embedding_model.embed_query(clean_case_fact)
    # 在Chroma中执行余弦检索。
    return query_cases(query_embedding, top_k=top_k)
