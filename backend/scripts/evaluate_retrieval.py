"""使用 LeCaRDv2 专家标签评测 BM25、Dense 或 Hybrid 类案检索。"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from backend.services.hybrid_retriever import RRF_RANK_CONSTANT, retrieve_hybrid_cases
from backend.services.keyword_retriever import retrieve_keyword_cases
from backend.services.reranker import rerank_cases
from backend.services.retriever import retrieve_relevant_cases
from backend.services.vector_store import get_cases_by_ids


# 保证 Windows 终端可以正常显示中文评测结果。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "backend" / "data" / "lecardv2"
EVALUATION_ROOT = DATA_ROOT / "evaluation"
RANKING_POOL_PATH = DATA_ROOT / "label" / "ranking_pool.json"
# 批量评测时主动放慢 Reranker 请求，避免触发服务商速率限制。
RERANK_DELAY_SECONDS = 20
RERANK_RETRY_WAITS = [30, 60, 120]
SPLIT_FILES = {
    "test": {
        "query": DATA_ROOT / "query" / "test_query.json",
        "qrels": DATA_ROOT / "label" / "test_relevence.trec",
        "gold": DATA_ROOT / "label" / "test_relevence_gold.trec",
    },
    "all": {
        "query": DATA_ROOT / "query" / "query.json",
        "qrels": DATA_ROOT / "label" / "relevence.trec",
        "gold": DATA_ROOT / "label" / "relevence_gold.trec",
    },
}


def parse_arguments():
    """读取检索方式、数据范围和 Top K 等评测参数。"""
    parser = argparse.ArgumentParser(description="评测 LeCaRDv2 类案检索效果")
    parser.add_argument(
        "--method",
        choices=[
            "bm25",
            "vector",
            "hybrid",
            "hybrid_rerank",
            "hybrid_rerank_fusion",
            "rerank_pool",
        ],
        default="bm25",
        help="需要评测的检索方式，默认 bm25",
    )
    parser.add_argument(
        "--split",
        choices=["test", "all"],
        default="test",
        help="test 使用 160 条测试集，all 使用全部 800 条查询",
    )
    parser.add_argument("--top-k", type=int, default=10, help="检查前 K 个结果")
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="只评测前 N 条查询，传入 0 表示评测所选数据集的全部查询",
    )
    arguments = parser.parse_args()
    if arguments.top_k <= 0:
        parser.error("--top-k 必须大于 0")
    if arguments.limit < 0:
        parser.error("--limit 不能小于 0")
    return arguments


def load_queries(path):
    """从 JSONL 文件读取查询编号和用于检索的案件事实。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到查询文件：{path}")

    queries = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"查询文件第 {line_number} 行不是有效 JSON") from error

            query_id = record.get("id")
            fact = str(record.get("fact") or "").strip()
            if query_id is None or not fact:
                raise ValueError(f"查询文件第 {line_number} 行缺少 id 或 fact")
            queries.append({"id": str(query_id), "fact": fact})
    return queries


def load_qrels(path):
    """读取 TREC 标签并整理为“查询 ID → 案件 ID → 相关等级”。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到标签文件：{path}")

    qrels = defaultdict(dict)
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"标签文件第 {line_number} 行不是四列 TREC 格式")

            query_id, _, case_id, relevance = parts
            try:
                relevance = int(relevance)
            except ValueError as error:
                raise ValueError(f"标签文件第 {line_number} 行相关等级不是整数") from error
            qrels[query_id][case_id] = max(
                relevance,
                qrels[query_id].get(case_id, 0),
            )
    return dict(qrels)


def load_ranking_pools(path):
    """读取每条查询对应的100个官方重排候选案件。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到候选池文件：{path}")

    ranking_pools = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"候选池第 {line_number} 行不是有效 JSON") from error

            query_id = record.get("qid")
            case_ids = record.get("rank_doc_id")
            if query_id is None or not isinstance(case_ids, list) or not case_ids:
                raise ValueError(f"候选池第 {line_number} 行缺少 qid 或 rank_doc_id")
            ranking_pools[str(query_id)] = [str(case_id) for case_id in case_ids]
    return ranking_pools


def recall_at_k(retrieved_ids, relevant_ids, top_k):
    """计算前 K 条结果找回了多少专家认定的相关案件。"""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:top_k]) & relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids, relevance_by_id, top_k):
    """根据多级相关性计算前 K 条结果的排序质量。"""
    dcg = sum(
        (2 ** relevance_by_id.get(case_id, 0) - 1) / math.log2(rank + 1)
        for rank, case_id in enumerate(retrieved_ids[:top_k], start=1)
    )
    ideal_relevances = sorted(relevance_by_id.values(), reverse=True)[:top_k]
    ideal_dcg = sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids, top_k):
    """计算第一个专家相关案件在前 K 条结果中的倒数排名。"""
    for rank, case_id in enumerate(retrieved_ids[:top_k], start=1):
        if case_id in relevant_ids:
            return 1 / rank
    return 0.0


def rerank_cases_with_retry(
    fact,
    candidates,
    top_k,
    min_score=None,
    relative_score_ratio=None,
):
    """Reranker 触发429限流时，等待后自动重试。"""
    for attempt in range(len(RERANK_RETRY_WAITS) + 1):
        try:
            rerank_arguments = {"top_k": top_k}
            if min_score is not None:
                rerank_arguments["min_score"] = min_score
            if relative_score_ratio is not None:
                rerank_arguments["relative_score_ratio"] = relative_score_ratio
            return rerank_cases(fact, candidates, **rerank_arguments)
        except RuntimeError as error:
            is_rate_limit = (
                "429" in str(error)
                or "RATE_LIMIT_EXCEEDED" in str(error)
            )
            if not is_rate_limit or attempt == len(RERANK_RETRY_WAITS):
                raise

            wait_seconds = RERANK_RETRY_WAITS[attempt]
            print(
                f"Reranker触发限流，等待{wait_seconds}秒后重试……",
                flush=True,
            )
            time.sleep(wait_seconds)


def fuse_hybrid_and_reranker(fact, hybrid_candidates):
    """用RRF融合Hybrid名次与Reranker名次，并保留全部候选供离线分析。"""
    reranked_candidates = rerank_cases_with_retry(
        fact,
        hybrid_candidates,
        top_k=len(hybrid_candidates),
        min_score=0,
        relative_score_ratio=0,
    )
    if len(reranked_candidates) != len(hybrid_candidates):
        raise RuntimeError(
            "Reranker没有返回全部Hybrid候选，无法进行完整的排名融合"
        )

    hybrid_rank_by_id = {
        candidate["id"]: rank
        for rank, candidate in enumerate(hybrid_candidates, start=1)
    }
    rerank_rank_by_id = {
        candidate["id"]: rank
        for rank, candidate in enumerate(reranked_candidates, start=1)
    }

    fused_candidates = []
    for candidate in reranked_candidates:
        case_id = candidate["id"]
        hybrid_rank = hybrid_rank_by_id[case_id]
        rerank_rank = rerank_rank_by_id[case_id]
        fusion_score = (
            1 / (RRF_RANK_CONSTANT + hybrid_rank)
            + 1 / (RRF_RANK_CONSTANT + rerank_rank)
        )
        fused_candidates.append(
            {
                **candidate,
                "hybrid_rank": hybrid_rank,
                "rerank_rank": rerank_rank,
                "fusion_score": fusion_score,
                "score": fusion_score,
            }
        )

    fused_candidates.sort(
        key=lambda candidate: (
            -candidate["fusion_score"],
            candidate["hybrid_rank"],
            candidate["id"],
        )
    )
    return fused_candidates


def load_checkpoint(path):
    """读取已经完成的逐查询评测结果，没有文件时返回空记录。"""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"评测断点文件无法读取：{path}") from error

    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise RuntimeError(f"评测断点文件格式错误：{path}")
    return results


def save_checkpoint(path, method, split, top_k, results):
    """每完成一条查询就原子保存结果，避免限流后丢失进度。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "split": split,
        "top_k": top_k,
        "results": results,
    }
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def retrieve_cases(method, fact, top_k, query_id=None, ranking_pools=None):
    """按照命令行选择调用项目现有的检索函数。"""
    if method == "bm25":
        return retrieve_keyword_cases(fact, top_k=top_k)
    if method == "vector":
        return retrieve_relevant_cases(fact, top_k=top_k)
    if method == "hybrid":
        return retrieve_hybrid_cases(fact, top_k=top_k)

    if method == "rerank_pool":
        if ranking_pools is None or query_id not in ranking_pools:
            raise ValueError(f"查询 {query_id} 没有对应的官方候选池")

        pool_ids = ranking_pools[query_id]
        pool_cases = get_cases_by_ids(pool_ids)
        if len(pool_cases) != len(pool_ids):
            found_ids = {case["id"] for case in pool_cases}
            missing_ids = [case_id for case_id in pool_ids if case_id not in found_ids]
            raise ValueError(
                f"查询 {query_id} 的候选池有 {len(missing_ids)} 个案件不在 Chroma 中"
            )

        # 官方候选池顺序仅用于解决 Reranker 同分，不参与主要相关性判断。
        ranked_pool_cases = [
            {
                **case,
                "score": 1.0 / rank,
            }
            for rank, case in enumerate(pool_cases, start=1)
        ]
        return rerank_cases_with_retry(fact, ranked_pool_cases, top_k)

    if method == "hybrid_rerank_fusion":
        # 先取得20个Hybrid候选，再保留两套名次进行等权RRF融合。
        candidate_count = max(20, top_k)
        hybrid_candidates = retrieve_hybrid_cases(
            fact,
            top_k=candidate_count,
        )
        return fuse_hybrid_and_reranker(fact, hybrid_candidates)

    # 完整流程先让 Hybrid 召回20个候选，再由 Reranker 精排出最终 Top K。
    candidate_count = max(20, top_k)
    candidates = retrieve_hybrid_cases(fact, top_k=candidate_count)
    return rerank_cases_with_retry(fact, candidates, top_k)


def evaluate(
    method,
    queries,
    qrels,
    gold_qrels,
    top_k,
    split,
    checkpoint_path=None,
    ranking_pools=None,
):
    """逐条执行检索并返回三个指标的平均值。"""
    metric_totals = {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0}
    started_at = time.perf_counter()
    total = len(queries)
    checkpoint_results = (
        load_checkpoint(checkpoint_path)
        if checkpoint_path is not None
        else {}
    )
    cached_count = sum(
        query["id"] in checkpoint_results
        for query in queries
    )
    if cached_count:
        print(
            f"已加载断点：{cached_count}/{total} 条查询已经完成",
            flush=True,
        )

    for position, query in enumerate(queries, start=1):
        query_id = query["id"]
        query_metrics = checkpoint_results.get(query_id)
        was_cached = query_metrics is not None
        if query_metrics is None:
            try:
                results = retrieve_cases(
                    method,
                    query["fact"],
                    top_k,
                    query_id=query_id,
                    ranking_pools=ranking_pools,
                )
            except RuntimeError as error:
                if checkpoint_path is not None:
                    print(
                        f"已保存 {len(checkpoint_results)}/{total} 条断点结果，"
                        "下次运行相同命令将自动继续。",
                        flush=True,
                    )
                raise

            retrieved_ids = [str(result["id"]) for result in results]
            relevance_by_id = qrels.get(query_id, {})
            relevant_ids = set(gold_qrels.get(query_id, {}))
            query_metrics = {
                "recall": recall_at_k(retrieved_ids, relevant_ids, top_k),
                "ndcg": ndcg_at_k(retrieved_ids, relevance_by_id, top_k),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids, top_k),
                "retrieved_ids": retrieved_ids,
            }

            if method == "rerank_pool":
                pool_top_ids = ranking_pools[query_id][:top_k]
                query_metrics.update(
                    {
                        "pool_top_ids": pool_top_ids,
                        "pool_top_relevance": [
                            relevance_by_id.get(case_id, 0)
                            for case_id in pool_top_ids
                        ],
                        "reranked_relevance": [
                            relevance_by_id.get(case_id, 0)
                            for case_id in retrieved_ids
                        ],
                        "rerank_scores": [
                            float(result.get("rerank_score", 0.0))
                            for result in results
                        ],
                    }
                )

            if method == "hybrid_rerank_fusion":
                hybrid_order = sorted(
                    results,
                    key=lambda result: result["hybrid_rank"],
                )
                reranker_order = sorted(
                    results,
                    key=lambda result: result["rerank_rank"],
                )
                hybrid_top_ids = [
                    str(result["id"])
                    for result in hybrid_order[:top_k]
                ]
                reranker_top_ids = [
                    str(result["id"])
                    for result in reranker_order[:top_k]
                ]
                query_metrics.update(
                    {
                        "hybrid_top_ids": hybrid_top_ids,
                        "reranker_top_ids": reranker_top_ids,
                        "hybrid_top_relevance": [
                            relevance_by_id.get(case_id, 0)
                            for case_id in hybrid_top_ids
                        ],
                        "reranker_top_relevance": [
                            relevance_by_id.get(case_id, 0)
                            for case_id in reranker_top_ids
                        ],
                        "fused_top_relevance": [
                            relevance_by_id.get(case_id, 0)
                            for case_id in retrieved_ids[:top_k]
                        ],
                        "ranking_details": [
                            {
                                "id": str(result["id"]),
                                "hybrid_rank": result["hybrid_rank"],
                                "rerank_rank": result["rerank_rank"],
                                "rerank_score": float(result["rerank_score"]),
                                "fusion_score": float(result["fusion_score"]),
                            }
                            for result in results
                        ],
                    }
                )

            if checkpoint_path is not None:
                checkpoint_results[query_id] = query_metrics
                save_checkpoint(
                    checkpoint_path,
                    method,
                    split,
                    top_k,
                    checkpoint_results,
                )

        metric_totals["recall"] += query_metrics["recall"]
        metric_totals["ndcg"] += query_metrics["ndcg"]
        metric_totals["mrr"] += query_metrics["mrr"]

        if method == "rerank_pool":
            print(
                f"查询 {query_id}：候选池前{top_k}等级 "
                f"{query_metrics['pool_top_relevance']} → "
                f"Reranker前{top_k}等级 {query_metrics['reranked_relevance']}",
                flush=True,
            )

        if method == "hybrid_rerank_fusion":
            print(
                f"查询 {query_id}：Hybrid前{top_k}等级 "
                f"{query_metrics['hybrid_top_relevance']}；"
                f"Reranker前{top_k}等级 "
                f"{query_metrics['reranker_top_relevance']}；"
                f"融合前{top_k}等级 {query_metrics['fused_top_relevance']}",
                flush=True,
            )

        if position % 10 == 0 or position == total:
            elapsed = time.perf_counter() - started_at
            print(f"进度：{position}/{total}，已用时 {elapsed:.1f} 秒", flush=True)

        # 已有断点的查询不发起网络请求，因此也不需要等待。
        has_unfinished_query = any(
            later_query["id"] not in checkpoint_results
            for later_query in queries[position:]
        )
        if (
            method in {
                "hybrid_rerank",
                "hybrid_rerank_fusion",
                "rerank_pool",
            }
            and not was_cached
            and has_unfinished_query
        ):
            print(
                f"等待{RERANK_DELAY_SECONDS}秒后处理下一条……",
                flush=True,
            )
            time.sleep(RERANK_DELAY_SECONDS)

    return {
        name: total_value / total
        for name, total_value in metric_totals.items()
    }


def main():
    arguments = parse_arguments()
    files = SPLIT_FILES[arguments.split]
    queries = load_queries(files["query"])
    if arguments.limit:
        queries = queries[: arguments.limit]
    if not queries:
        raise ValueError("没有可供评测的查询案件")

    qrels = load_qrels(files["qrels"])
    gold_qrels = load_qrels(files["gold"])
    ranking_pools = (
        load_ranking_pools(RANKING_POOL_PATH)
        if arguments.method == "rerank_pool"
        else None
    )
    print(f"检索方式：{arguments.method}")
    print(f"数据范围：{arguments.split}")
    print(f"查询数量：{len(queries)}")
    print(f"评测范围：Top {arguments.top_k}")

    checkpoint_path = None
    if arguments.method in {
        "hybrid_rerank",
        "hybrid_rerank_fusion",
        "rerank_pool",
    }:
        checkpoint_path = EVALUATION_ROOT / (
            f"{arguments.method}_{arguments.split}_top{arguments.top_k}.json"
        )
        print(f"断点文件：{checkpoint_path}")

    metrics = evaluate(
        arguments.method,
        queries,
        qrels,
        gold_qrels,
        arguments.top_k,
        arguments.split,
        checkpoint_path,
        ranking_pools,
    )
    print("\n评测结果：")
    print(f"Recall@{arguments.top_k}: {metrics['recall']:.4f}")
    print(f"NDCG@{arguments.top_k}: {metrics['ndcg']:.4f}")
    print(f"MRR@{arguments.top_k}: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"评测失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
