"""从命令行构建Chroma案件知识库。"""

import argparse
import sys
from pathlib import Path

# 复用知识库服务配置。
from backend.services.knowledge_base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CASE_DIRECTORY,
    DEFAULT_CASE_LIMIT,
    build_knowledge_base,
)


# Windows终端强制使用UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def print_progress(processed, total):
    """输出案件入库进度。"""
    print(f"正在生成 fact 向量并写入 Chroma：{processed}/{total}")


def parse_arguments():
    """解析并校验入库参数。"""
    # 定义命令行参数。
    parser = argparse.ArgumentParser(description="把 LeCaRDv2 案件写入 Chroma")
    # 案件JSON目录。
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_CASE_DIRECTORY,
        help=f"候选案件目录，默认 {DEFAULT_CASE_DIRECTORY}",
    )
    # 本次案件上限。
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CASE_LIMIT,
        help=f"本次读取的案件数量，默认 {DEFAULT_CASE_LIMIT}",
    )
    # Embedding批量大小。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每批生成向量的案件数量，默认 {DEFAULT_BATCH_SIZE}",
    )
    # 读取实际参数。
    arguments = parser.parse_args()

    # 拒绝非正数配置。
    if arguments.limit <= 0:
        parser.error("--limit 必须大于 0")
    if arguments.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    return arguments


def main():
    """执行一次知识库构建。"""
    # 解析参数并规范路径。
    arguments = parse_arguments()
    case_directory = arguments.directory.resolve()

    print(f"案件目录：{case_directory}")
    print(f"本次最多读取：{arguments.limit} 条")
    print("目标 Collection：legal_cases")

    # 调用分批入库服务。
    result = build_knowledge_base(
        case_directory=case_directory,
        limit=arguments.limit,
        batch_size=arguments.batch_size,
        progress_callback=print_progress,
    )

    # 输出构建统计。
    print("\n入库完成")
    print(f"本次读取案件：{result['loaded_case_count']}")
    print(f"已存在并跳过：{result['existing_case_count']}")
    print(f"本次实际写入：{result['written_case_count']}")
    print(f"Collection 当前案件：{result['collection_count']}")
    print(f"Chroma 保存位置：{result['chroma_directory']}")


if __name__ == "__main__":
    # 将预期异常转换为命令行错误。
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"案件入库失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
