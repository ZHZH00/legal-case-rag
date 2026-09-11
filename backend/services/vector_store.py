"""封装本地Chroma案件存储。"""

from pathlib import Path

import chromadb

from backend.services.embedding import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL


# Chroma目录与Collection配置。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "chroma"
COLLECTION_NAME = "legal_cases"
# ID查询批量上限。
ID_LOOKUP_BATCH_SIZE = 500
# 全库读取分页大小。
CASE_READ_BATCH_SIZE = 500


def get_legal_case_collection():
    """创建或复用法律案件Collection。"""
    # 初始化持久化客户端。
    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
    # 复用同名Collection。
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "LeCaRDv2 中文刑事案件事实",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )

    # 校验存量向量配置。
    stored_metadata = collection.metadata or {}
    if (
        stored_metadata.get("embedding_model") != EMBEDDING_MODEL
        or stored_metadata.get("embedding_dimensions") != EMBEDDING_DIMENSIONS
    ):
        raise RuntimeError("现有法律案件 Collection 的 Embedding 配置与当前代码不一致")

    return collection


def build_case_metadata(case):
    """构建Chroma案件Metadata。"""
    # 列表字段序列化为字符串。
    return {
        "pid": case["pid"],
        "reason": case["reason"],
        "result": case["result"],
        "charge": "、".join(case["charge"]),
        "article": ",".join(case["article"]),
        "source_file": case["source_file"],
    }


def get_existing_case_ids(collection, cases):
    """批量查询已存在的案件ID。"""
    # 空输入不访问数据库。
    if not cases:
        return set()

    # 分批查询并汇总ID集合。
    candidate_ids = [case["pid"] for case in cases]
    existing_ids = set()
    for start in range(0, len(candidate_ids), ID_LOOKUP_BATCH_SIZE):
        batch_ids = candidate_ids[start : start + ID_LOOKUP_BATCH_SIZE]
        result = collection.get(ids=batch_ids, include=[])
        existing_ids.update(result["ids"])
    return existing_ids


def upsert_cases(collection, cases, embeddings):
    """批量写入案件、向量和Metadata。"""
    # 案件与向量必须一一对应。
    if len(cases) != len(embeddings):
        raise ValueError("案件数量与向量数量不一致")
    # 空批次直接返回。
    if not cases:
        return

    # 各字段按相同顺序写入。
    collection.upsert(
        ids=[case["pid"] for case in cases],
        documents=[case["fact"] for case in cases],
        embeddings=embeddings,
        metadatas=[build_case_metadata(case) for case in cases],
    )


def get_all_cases():
    """分页读取全部案件供BM25建索引。"""
    collection = get_legal_case_collection()
    cases = []
    collection_count = collection.count()

    # 仅读取BM25所需字段。
    for offset in range(0, collection_count, CASE_READ_BATCH_SIZE):
        result = collection.get(
            limit=min(CASE_READ_BATCH_SIZE, collection_count - offset),
            offset=offset,
            include=["documents", "metadatas"],
        )
        cases.extend(
            {
                "id": case_id,
                "content": fact,
                "metadata": metadata,
            }
            for case_id, fact, metadata in zip(
                result["ids"],
                result["documents"],
                result["metadatas"],
            )
        )
    return cases


def get_cases_by_ids(case_ids):
    """按ID批量读取案件。"""
    if not case_ids:
        return []

    collection = get_legal_case_collection()
    result = collection.get(ids=case_ids, include=["documents", "metadatas"])
    cases_by_id = {
        case_id: {
            "id": case_id,
            "content": fact,
            "metadata": metadata,
        }
        for case_id, fact, metadata in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
        )
    }

    # 按请求ID恢复结果顺序。
    return [cases_by_id[case_id] for case_id in case_ids if case_id in cases_by_id]


def query_cases(query_embedding, top_k=5):
    """按查询向量返回Top K案件。"""
    # 校验查询向量维度。
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"查询向量应为 {EMBEDDING_DIMENSIONS} 维，实际为 {len(query_embedding)} 维"
        )
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 空Collection直接返回。
    collection = get_legal_case_collection()
    collection_count = collection.count()
    if collection_count == 0:
        return []

    # 使用余弦距离执行向量查询。
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )

    # 转换为统一案件结果结构。
    return [
        {
            "id": case_id,
            "content": fact,
            "metadata": metadata,
            "distance": float(distance),
            "score": 1.0 - float(distance),
        }
        for case_id, fact, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]
