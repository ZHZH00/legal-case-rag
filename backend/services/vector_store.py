"""封装法律案件在本地 Chroma 中的创建、写入和查询。"""

from pathlib import Path

import chromadb

from backend.services.embedding import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL


# 所有案件向量都持久化到项目内的 legal_cases Collection。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIRECTORY = PROJECT_ROOT / "backend" / "data" / "chroma"
COLLECTION_NAME = "legal_cases"
# 批量检查案件 ID，避免一次查询过多 ID 超过 SQLite 参数上限。
ID_LOOKUP_BATCH_SIZE = 500
# 分页读取案件，避免一次返回全部数据超过 SQLite 参数上限。
CASE_READ_BATCH_SIZE = 500


def get_legal_case_collection():
    """取得当前系统唯一使用的法律案件 Collection。"""
    # 首次运行时自动创建 Chroma 目录，并使用该目录连接本地数据库。
    CHROMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIRECTORY))
    # Collection 不存在时创建，存在时直接复用原有案件数据。
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "LeCaRDv2 中文刑事案件事实",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )

    # 旧向量只有在模型和维度相同时才能继续与新查询比较。
    stored_metadata = collection.metadata or {}
    if (
        stored_metadata.get("embedding_model") != EMBEDDING_MODEL
        or stored_metadata.get("embedding_dimensions") != EMBEDDING_DIMENSIONS
    ):
        raise RuntimeError("现有法律案件 Collection 的 Embedding 配置与当前代码不一致")

    return collection


def build_case_metadata(case):
    """把不参与向量相似度计算的案件资料保存为 Metadata。"""
    # Chroma Metadata 不保存列表，因此罪名和法条先拼接成字符串。
    return {
        "pid": case["pid"],
        "reason": case["reason"],
        "result": case["result"],
        "charge": "、".join(case["charge"]),
        "article": ",".join(case["article"]),
        "source_file": case["source_file"],
    }


def get_existing_case_ids(collection, cases):
    """找出已经入库的案件，避免重复生成向量。"""
    # 没有待检查案件时直接返回空集合，避免无意义的数据库查询。
    if not cases:
        return set()

    # 分批查询 pid，并用集合提高后续判断速度。
    candidate_ids = [case["pid"] for case in cases]
    existing_ids = set()
    for start in range(0, len(candidate_ids), ID_LOOKUP_BATCH_SIZE):
        batch_ids = candidate_ids[start : start + ID_LOOKUP_BATCH_SIZE]
        result = collection.get(ids=batch_ids, include=[])
        existing_ids.update(result["ids"])
    return existing_ids


def upsert_cases(collection, cases, embeddings):
    """把案件编号、事实原文、事实向量和其他案件资料写入 Chroma。"""
    # 每条案件必须恰好对应一个向量，防止内容和向量错位。
    if len(cases) != len(embeddings):
        raise ValueError("案件数量与向量数量不一致")
    # 空批次无需调用 Chroma。
    if not cases:
        return

    # pid、fact、向量和 Metadata 按相同顺序一次写入。
    collection.upsert(
        ids=[case["pid"] for case in cases],
        documents=[case["fact"] for case in cases],
        embeddings=embeddings,
        metadatas=[build_case_metadata(case) for case in cases],
    )


def get_all_cases():
    """读取全部案件事实和 Metadata，供 BM25 关键词索引使用。"""
    collection = get_legal_case_collection()
    cases = []
    collection_count = collection.count()

    # BM25 不读取向量，只分页读取案件 ID、事实正文和 Metadata。
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
    """按照案件 ID 列表从 Chroma 取回案件事实和 Metadata。"""
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

    # Chroma 不保证返回顺序，因此按调用者传入的 ID 顺序重新排列。
    return [cases_by_id[case_id] for case_id in case_ids if case_id in cases_by_id]


def query_cases(query_embedding, top_k=5):
    """使用用户案情向量查询 Chroma，返回事实最相似的历史案件。"""
    # 查询向量必须和建库向量维度一致。
    if len(query_embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"查询向量应为 {EMBEDDING_DIMENSIONS} 维，实际为 {len(query_embedding)} 维"
        )
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 空知识库没有可检索内容，因此直接返回空列表。
    collection = get_legal_case_collection()
    collection_count = collection.count()
    if collection_count == 0:
        return []

    # Chroma 使用余弦距离取回最接近用户案情的 fact。
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )

    # 将距离转换成直观分数，并恢复统一的案件结果结构。
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
