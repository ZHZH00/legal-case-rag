"""提前建立并保存 BM25 关键词索引。"""

import sys

from backend.services.keyword_retriever import build_keyword_index


# Windows 终端统一使用 UTF-8，避免中文进度信息显示乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def print_progress(processed, total):
    """每处理一千条案件就显示一次分词进度。"""
    print(f"正在为案件事实分词：{processed}/{total}", flush=True)


def main():
    """从 Chroma 读取全部案件并生成本地 BM25 索引文件。"""
    print("开始建立 BM25 索引……", flush=True)
    result = build_keyword_index(progress_callback=print_progress)

    print("\nBM25 索引建立完成")
    print(f"案件数量：{result['case_count']}")
    print(f"索引位置：{result['index_path']}")
    print(f"文件大小：{result['file_size'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"BM25 索引建立失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
