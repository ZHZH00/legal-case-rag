"""读取LeCaRDv2案件并写入Chroma。"""

import json
from pathlib import Path

from backend.services.embedding import embedding_model
from backend.services.vector_store import (
    CHROMA_DIRECTORY,
    get_existing_case_ids,
    get_legal_case_collection,
    upsert_cases,
)


# 默认读取LeCaRDv2候选案件目录。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_DIRECTORY = (
    PROJECT_ROOT / "backend" / "data" / "lecardv2" / "candidate_55192"
)
# 默认批量和试运行数量。
DEFAULT_BATCH_SIZE = 32
DEFAULT_CASE_LIMIT = 100


def normalize_list(value):
    """将可选字段规范为字符串列表。"""
    # 缺失值统一为空列表。
    if value is None:
        return []
    # 列表元素统一转为字符串。
    if isinstance(value, list):
        return [str(item) for item in value]
    # 标量统一包装为列表。
    return [str(value)]


def load_case_file(file_path):
    """读取并校验单个案件文件。"""
    try:
        # 每个JSON文件对应一条案件。
        raw_case = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取案件文件 {file_path.name}：{error}") from error

    # 提取唯一ID和检索正文。
    pid = raw_case.get("pid")
    fact = str(raw_case.get("fact") or "").strip()
    # 拒绝缺少必要字段的案件。
    if pid is None:
        raise ValueError(f"案件文件 {file_path.name} 缺少 pid")
    if not fact:
        raise ValueError(f"案件文件 {file_path.name} 缺少 fact")

    # 返回统一入库结构。
    return {
        "pid": str(pid),
        "fact": fact,
        "reason": str(raw_case.get("reason") or "").strip(),
        "result": str(raw_case.get("result") or "").strip(),
        "charge": normalize_list(raw_case.get("charge")),
        "article": normalize_list(raw_case.get("article")),
        "source_file": file_path.name,
    }


def case_file_sort_key(file_path):
    """生成稳定的案件文件排序键。"""
    # 数字文件名按整数排序。
    if file_path.stem.isdigit():
        return (0, int(file_path.stem))
    # 其他文件名按文本排序。
    return (1, file_path.stem)


def load_legal_cases(case_directory=DEFAULT_CASE_DIRECTORY, limit=DEFAULT_CASE_LIMIT):
    """按稳定顺序读取限定数量的案件。"""
    # 路径统一转为绝对路径。
    directory = Path(case_directory).resolve()
    # 读取前校验目录和数量。
    if not directory.is_dir():
        raise FileNotFoundError(f"没有找到案件目录：{directory}")
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    # 仅收集当前目录的JSON文件。
    case_files = sorted(directory.glob("*.json"), key=case_file_sort_key)
    if not case_files:
        raise ValueError(f"案件目录中没有 JSON 文件：{directory}")

    # 按limit截取输入文件。
    return [load_case_file(file_path) for file_path in case_files[:limit]]


def build_knowledge_base(
    case_directory=DEFAULT_CASE_DIRECTORY,
    limit=DEFAULT_CASE_LIMIT,
    batch_size=DEFAULT_BATCH_SIZE,
    progress_callback=None,
):
    """分批向量化并写入案件知识库。"""
    # 批量大小必须为正数。
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    # 读取案件并连接Collection。
    cases = load_legal_cases(case_directory=case_directory, limit=limit)
    collection = get_legal_case_collection()
    # 跳过已存在的案件ID。
    existing_ids = get_existing_case_ids(collection, cases)
    pending_cases = [case for case in cases if case["pid"] not in existing_ids]

    # 新案件分批向量化入库。
    for start in range(0, len(pending_cases), batch_size):
        batch_cases = pending_cases[start : start + batch_size]
        # 仅对fact生成语义向量。
        batch_embeddings = embedding_model.embed_documents(
            [case["fact"] for case in batch_cases]
        )
        upsert_cases(collection, batch_cases, batch_embeddings)

        # 每批完成后回调进度。
        if progress_callback:
            processed = len(existing_ids) + min(
                start + batch_size,
                len(pending_cases),
            )
            progress_callback(processed, len(cases))

    # 返回本次构建统计。
    return {
        "loaded_case_count": len(cases),
        "existing_case_count": len(existing_ids),
        "written_case_count": len(pending_cases),
        "collection_count": collection.count(),
        "chroma_directory": CHROMA_DIRECTORY,
    }
