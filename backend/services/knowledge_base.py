"""读取 LeCaRDv2 案件 JSON，并把案件事实向量写入 Chroma。"""

import json
from pathlib import Path

from backend.services.embedding import embedding_model
from backend.services.vector_store import (
    CHROMA_DIRECTORY,
    get_existing_case_ids,
    get_legal_case_collection,
    upsert_cases,
)


# 默认数据目录指向已经解压完成的 LeCaRDv2 候选案件文件夹。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_DIRECTORY = (
    PROJECT_ROOT / "backend" / "data" / "lecardv2" / "candidate_55192"
)
# 小批量请求可以降低第一次测试时的等待时间和接口压力。
DEFAULT_BATCH_SIZE = 32
DEFAULT_CASE_LIMIT = 100


def normalize_list(value):
    """把罪名或法条统一整理成字符串列表。"""
    # 缺失字段统一使用空列表，方便后续拼接成 Chroma Metadata。
    if value is None:
        return []
    # 列表中的数字法条也转为字符串，保证每项类型一致。
    if isinstance(value, list):
        return [str(item) for item in value]
    # 单个罪名或法条也包装成列表，避免后续分别处理多种格式。
    return [str(value)]


def load_case_file(file_path):
    """读取一条案件，并检查检索必需的 pid 和 fact。"""
    try:
        # 每个 JSON 文件对应一条完整案件，读取后转换成 Python 字典。
        raw_case = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取案件文件 {file_path.name}：{error}") from error

    # pid 用作案件唯一 ID，fact 用作向量检索正文。
    pid = raw_case.get("pid")
    fact = str(raw_case.get("fact") or "").strip()
    # 缺少唯一 ID 或案件事实的数据不能进入向量库。
    if pid is None:
        raise ValueError(f"案件文件 {file_path.name} 缺少 pid")
    if not fact:
        raise ValueError(f"案件文件 {file_path.name} 缺少 fact")

    # 这里把不同字段整理成后续入库统一使用的案件结构。
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
    """数字文件名按数值排序，让限制条数时每次读取相同案件。"""
    # 纯数字文件名优先按整数比较，避免 10.json 排在 2.json 前面。
    if file_path.stem.isdigit():
        return (0, int(file_path.stem))
    # 非数字文件名放在数字文件之后并按普通文本排序。
    return (1, file_path.stem)


def load_legal_cases(case_directory=DEFAULT_CASE_DIRECTORY, limit=DEFAULT_CASE_LIMIT):
    """从候选案件目录读取指定数量的数据，默认先处理 100 条。"""
    # resolve 将传入路径统一转换成便于检查的绝对路径。
    directory = Path(case_directory).resolve()
    # 目录和数量在读取文件前校验，错误可以尽早反馈。
    if not directory.is_dir():
        raise FileNotFoundError(f"没有找到案件目录：{directory}")
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    # 这里只收集当前目录中的 JSON，并用固定规则保证读取顺序稳定。
    case_files = sorted(directory.glob("*.json"), key=case_file_sort_key)
    if not case_files:
        raise ValueError(f"案件目录中没有 JSON 文件：{directory}")

    # limit 控制本次最多读取多少条，便于先用小规模数据验证流程。
    return [load_case_file(file_path) for file_path in case_files[:limit]]


def build_knowledge_base(
    case_directory=DEFAULT_CASE_DIRECTORY,
    limit=DEFAULT_CASE_LIMIT,
    batch_size=DEFAULT_BATCH_SIZE,
    progress_callback=None,
):
    """把指定数量案件的 fact 向量和其他字段保存到 Chroma。"""
    # 批次必须为正数，否则分批循环无法正常推进。
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    # 先读取规范化案件，再取得当前项目使用的法律案件 Collection。
    cases = load_legal_cases(case_directory=case_directory, limit=limit)
    collection = get_legal_case_collection()
    # 根据 pid 找出已入库案件，只为新增案件调用 Embedding 接口。
    existing_ids = get_existing_case_ids(collection, cases)
    pending_cases = [case for case in cases if case["pid"] not in existing_ids]

    # 新案件按 batch_size 分批生成向量并立即写入 Chroma。
    for start in range(0, len(pending_cases), batch_size):
        batch_cases = pending_cases[start : start + batch_size]
        # 只有 fact 参与语义向量计算，其他字段仍随案件一起保存。
        batch_embeddings = embedding_model.embed_documents(
            [case["fact"] for case in batch_cases]
        )
        upsert_cases(collection, batch_cases, batch_embeddings)

        # 调用入口脚本传入的函数，在每批完成后显示处理进度。
        if progress_callback:
            processed = len(existing_ids) + min(
                start + batch_size,
                len(pending_cases),
            )
            progress_callback(processed, len(cases))

    # 返回统计信息供命令行脚本打印本次入库结果。
    return {
        "loaded_case_count": len(cases),
        "existing_case_count": len(existing_ids),
        "written_case_count": len(pending_cases),
        "collection_count": collection.count(),
        "chroma_directory": CHROMA_DIRECTORY,
    }
