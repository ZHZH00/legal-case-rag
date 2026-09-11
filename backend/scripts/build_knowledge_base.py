"""默认把前 100 条 LeCaRDv2 候选案件写入 Chroma。"""

import argparse
import sys
from pathlib import Path

# 入库脚本复用 services 中的默认配置和核心构建函数。
from backend.services.knowledge_base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CASE_DIRECTORY,
    DEFAULT_CASE_LIMIT,
    build_knowledge_base,
)


# Windows 终端统一使用 UTF-8，避免中文进度信息显示乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def print_progress(processed, total):
    """每完成一批案件就在终端显示已经写入的数量。"""
    print(f"正在生成 fact 向量并写入 Chroma：{processed}/{total}")


def parse_arguments():
    """允许修改案件目录、测试条数和每批处理数量。"""
    # argparse 将命令行选项转换成后续可以直接访问的参数对象。
    parser = argparse.ArgumentParser(description="把 LeCaRDv2 案件写入 Chroma")
    # directory 指定包含候选案件 JSON 的文件夹。
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_CASE_DIRECTORY,
        help=f"候选案件目录，默认 {DEFAULT_CASE_DIRECTORY}",
    )
    # limit 控制本次最多读取多少个案件文件。
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_CASE_LIMIT,
        help=f"本次读取的案件数量，默认 {DEFAULT_CASE_LIMIT}",
    )
    # batch-size 控制每次向 Embedding 接口提交多少条 fact。
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"每批生成向量的案件数量，默认 {DEFAULT_BATCH_SIZE}",
    )
    # parse_args 读取用户本次在命令行中提供的实际选项。
    arguments = parser.parse_args()

    # 非正数参数没有实际意义，因此在开始读取案件前直接终止。
    if arguments.limit <= 0:
        parser.error("--limit 必须大于 0")
    if arguments.batch_size <= 0:
        parser.error("--batch-size 必须大于 0")
    return arguments


def main():
    """读取命令行参数并启动一次案件知识库构建任务。"""
    # 入口函数先取得参数，再把案件目录转换成绝对路径。
    arguments = parse_arguments()
    case_directory = arguments.directory.resolve()

    print(f"案件目录：{case_directory}")
    print(f"本次最多读取：{arguments.limit} 条")
    print("目标 Collection：legal_cases")

    # 核心函数负责读取 JSON、生成 fact 向量并写入 Chroma。
    result = build_knowledge_base(
        case_directory=case_directory,
        limit=arguments.limit,
        batch_size=arguments.batch_size,
        progress_callback=print_progress,
    )

    # 构建完成后集中显示读取、跳过、写入和数据库总数。
    print("\n入库完成")
    print(f"本次读取案件：{result['loaded_case_count']}")
    print(f"已存在并跳过：{result['existing_case_count']}")
    print(f"本次实际写入：{result['written_case_count']}")
    print(f"Collection 当前案件：{result['collection_count']}")
    print(f"Chroma 保存位置：{result['chroma_directory']}")


if __name__ == "__main__":
    # 直接运行模块时启动任务，并把可预期错误转换成清楚的终端提示。
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"案件入库失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
