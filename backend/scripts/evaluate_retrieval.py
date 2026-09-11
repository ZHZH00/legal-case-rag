"""使用LeCaRDv2标签评测检索与重排策略。"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from backend.services.hybrid_retriever import RRF_RANK_CONSTANT, retrieve_hybrid_cases
from backend.services.keyword_retriever import retrieve_keyword_cases
from backend.services.reranker import (
    PROTECTED_HYBRID_COUNT,
    rerank_cases,
)
from backend.services.retriever import retrieve_relevant_cases
from backend.services.vector_store import get_cases_by_ids


# Windows终端强制使用UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "backend" / "data" / "lecardv2"
EVALUATION_ROOT = DATA_ROOT / "evaluation"
RANKING_POOL_PATH = DATA_ROOT / "label" / "ranking_pool.json"
# Reranker限流与重试配置。
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
    """解析并校验评测参数。"""
    # 定义评测方式、数据范围和数量参数。
    parser = argparse.ArgumentParser(description="评测 LeCaRDv2 类案检索效果")
    parser.add_argument(
        "--method",
        choices=[
            "bm25",
            "vector",
            "hybrid",
            "hybrid_rerank",
            "hybrid_rerank_fusion",
            "rerank_ablation",
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
    # 拒绝无效数量参数。
    arguments = parser.parse_args()
    if arguments.top_k <= 0:
        parser.error("--top-k 必须大于 0")
    if arguments.limit < 0:
        parser.error("--limit 不能小于 0")
    return arguments


def load_queries(path):
    """读取JSONL格式的评测查询。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到查询文件：{path}")

    # 逐行解析查询ID和案件事实。
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
    """读取TREC标签并按查询ID分组。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到标签文件：{path}")

    # 重复标签保留最高相关等级。
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
    """读取官方Reranker候选池。"""
    if not path.exists():
        raise FileNotFoundError(f"没有找到候选池文件：{path}")

    # 建立查询ID到候选ID列表的映射。
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
    """计算Top K相关案件覆盖率。"""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:top_k]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, top_k):
    """计算Top K相关案件占比。"""
    hits = sum(
        case_id in relevant_ids
        for case_id in retrieved_ids[:top_k]
    )
    return hits / top_k


def hit_rate_at_k(retrieved_ids, relevant_ids, top_k):
    """计算Top K是否至少命中一次。"""
    return float(
        any(
            case_id in relevant_ids
            for case_id in retrieved_ids[:top_k]
        )
    )


def ndcg_at_k(retrieved_ids, relevance_by_id, top_k):
    """计算Top K多级相关性排序质量。"""
    dcg = sum(
        (2 ** relevance_by_id.get(case_id, 0) - 1) / math.log2(rank + 1)
        for rank, case_id in enumerate(retrieved_ids[:top_k], start=1)
    )
    # 使用理想排序归一化DCG。
    ideal_relevances = sorted(relevance_by_id.values(), reverse=True)[:top_k]
    ideal_dcg = sum(
        (2**relevance - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def reciprocal_rank(retrieved_ids, relevant_ids, top_k):
    """计算Top K首个相关结果的倒数排名。"""
    for rank, case_id in enumerate(retrieved_ids[:top_k], start=1):
        if case_id in relevant_ids:
            return 1 / rank
    return 0.0


def calculate_metrics(retrieved_ids, relevance_by_id, relevant_ids, top_k):
    """计算单条查询的全部指标。"""
    return {
        "precision": precision_at_k(retrieved_ids, relevant_ids, top_k),
        "recall": recall_at_k(retrieved_ids, relevant_ids, top_k),
        "hit_rate": hit_rate_at_k(retrieved_ids, relevant_ids, top_k),
        "ndcg": ndcg_at_k(retrieved_ids, relevance_by_id, top_k),
        "mrr": reciprocal_rank(retrieved_ids, relevant_ids, top_k),
    }


def rerank_cases_with_retry(
    fact,
    candidates,
    top_k,
    min_score=None,
    relative_score_ratio=None,
):
    """在Reranker限流时退避重试。"""
    for attempt in range(len(RERANK_RETRY_WAITS) + 1):
        try:
            rerank_arguments = {"top_k": top_k}
            if min_score is not None:
                rerank_arguments["min_score"] = min_score
            if relative_score_ratio is not None:
                rerank_arguments["relative_score_ratio"] = relative_score_ratio
            return rerank_cases(fact, candidates, **rerank_arguments)
        except RuntimeError as error:
            # 非限流异常不进入退避流程。
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
    """融合Hybrid与Reranker名次。"""
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

    # 建立Hybrid与Reranker名次映射。
    hybrid_rank_by_id = {
        candidate["id"]: rank
        for rank, candidate in enumerate(hybrid_candidates, start=1)
    }
    rerank_rank_by_id = {
        candidate["id"]: rank
        for rank, candidate in enumerate(reranked_candidates, start=1)
    }

    # 累加双路RRF分数。
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

    # 同分时优先原Hybrid名次。
    fused_candidates.sort(
        key=lambda candidate: (
            -candidate["fusion_score"],
            candidate["hybrid_rank"],
            candidate["id"],
        )
    )
    return fused_candidates


def retrieve_rerank_ablation_data(fact, top_k):
    """获取消融实验共用的原始排序。"""
    candidate_count = max(20, top_k)
    hybrid_candidates = retrieve_hybrid_cases(
        fact,
        top_k=candidate_count,
    )
    reranked_candidates = rerank_cases_with_retry(
        fact,
        hybrid_candidates,
        top_k=len(hybrid_candidates),
        min_score=0,
        relative_score_ratio=0,
    )
    if len(reranked_candidates) != len(hybrid_candidates):
        raise RuntimeError(
            "Reranker没有返回全部Hybrid候选，无法进行统一策略比较"
        )

    # 断点仅保存ID与分数。
    return {
        "hybrid_ids": [str(case["id"]) for case in hybrid_candidates],
        "reranker_results": [
            {
                "id": str(case["id"]),
                "rerank_score": float(case["rerank_score"]),
            }
            for case in reranked_candidates
        ],
    }


def prepend_protected_cases(protected_ids, reranker_ids, top_k):
    """拼接受保护案件与剩余重排结果。"""
    protected_id_set = set(protected_ids)
    remaining_ids = [
        case_id
        for case_id in reranker_ids
        if case_id not in protected_id_set
    ]
    return (protected_ids + remaining_ids)[:top_k]


def build_rerank_ablation_rankings(raw_result, top_k):
    """从共用原始排名生成三种Top K策略。"""
    hybrid_ids = raw_result["hybrid_ids"]
    reranker_results = raw_result["reranker_results"]
    reranker_ids = [result["id"] for result in reranker_results]
    if not hybrid_ids or not reranker_ids:
        return {
            "hybrid": [],
            "reranker": [],
            "fixed_protection": [],
        }

    # 保护数受Top K和候选数约束。
    actual_protected_count = min(
        PROTECTED_HYBRID_COUNT,
        top_k,
        len(hybrid_ids),
    )
    hybrid_head_ids = hybrid_ids[:actual_protected_count]
    fixed_ids = prepend_protected_cases(
        hybrid_head_ids,
        reranker_ids,
        top_k,
    )

    return {
        "hybrid": hybrid_ids[:top_k],
        "reranker": reranker_ids[:top_k],
        "fixed_protection": fixed_ids,
    }


def load_checkpoint(path):
    """读取评测断点。"""
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"评测断点文件无法读取：{path}") from error

    # 断点主体必须是查询结果映射。
    results = payload.get("results", {})
    if not isinstance(results, dict):
        raise RuntimeError(f"评测断点文件格式错误：{path}")
    return results


def save_checkpoint(path, method, split, top_k, results):
    """原子保存逐查询评测断点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "split": split,
        "top_k": top_k,
        "results": results,
    }
    # 临时文件写完后原子替换。
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def retrieve_cases(method, fact, top_k, query_id=None, ranking_pools=None):
    """按method分派检索流程。"""
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

        # 官方候选顺序仅作为同分依据。
        ranked_pool_cases = [
            {
                **case,
                "score": 1.0 / rank,
            }
            for rank, case in enumerate(pool_cases, start=1)
        ]
        return rerank_cases_with_retry(fact, ranked_pool_cases, top_k)

    if method == "hybrid_rerank_fusion":
        # 融合Hybrid与Reranker名次。
        candidate_count = max(20, top_k)
        hybrid_candidates = retrieve_hybrid_cases(
            fact,
            top_k=candidate_count,
        )
        return fuse_hybrid_and_reranker(fact, hybrid_candidates)

    # 默认重排20个Hybrid候选。
    candidate_count = max(20, top_k)
    candidates = retrieve_hybrid_cases(fact, top_k=candidate_count)
    return rerank_cases_with_retry(fact, candidates, top_k)


RERANK_ABLATION_LABELS = {
    "hybrid": "Hybrid直接取前K名",
    "reranker": "Reranker完全重排",
    "fixed_protection": f"固定保护Hybrid前{PROTECTED_HYBRID_COUNT}名",
}


def evaluate_rerank_ablation(
    queries,
    qrels,
    gold_qrels,
    top_k,
    split,
    checkpoint_path,
):
    """比较三种共享原始结果的排序策略。"""
    metric_names = ("precision", "recall", "hit_rate", "ndcg", "mrr")
    metric_totals = {
        strategy: {metric_name: 0.0 for metric_name in metric_names}
        for strategy in RERANK_ABLATION_LABELS
    }
    # 复用已完成查询的原始排序。
    checkpoint_results = load_checkpoint(checkpoint_path)
    total = len(queries)
    started_at = time.perf_counter()
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
        raw_result = checkpoint_results.get(query_id)
        was_cached = raw_result is not None
        # 仅为未缓存查询请求Reranker。
        if raw_result is None:
            try:
                raw_result = retrieve_rerank_ablation_data(
                    query["fact"],
                    top_k,
                )
            except RuntimeError:
                print(
                    f"已保存 {len(checkpoint_results)}/{total} 条断点结果，"
                    "下次运行相同命令将自动继续。",
                    flush=True,
                )
                raise
            checkpoint_results[query_id] = raw_result
            save_checkpoint(
                checkpoint_path,
                "rerank_ablation",
                split,
                top_k,
                checkpoint_results,
            )

        # 三种策略共享同一组标签。
        rankings = build_rerank_ablation_rankings(raw_result, top_k)
        relevance_by_id = qrels.get(query_id, {})
        relevant_ids = set(gold_qrels.get(query_id, {}))
        for strategy, retrieved_ids in rankings.items():
            query_metrics = calculate_metrics(
                retrieved_ids,
                relevance_by_id,
                relevant_ids,
                top_k,
            )
            for metric_name in metric_names:
                metric_totals[strategy][metric_name] += query_metrics[metric_name]

        if position % 10 == 0 or position == total:
            elapsed = time.perf_counter() - started_at
            print(
                f"进度：{position}/{total}，已用时 {elapsed:.1f} 秒",
                flush=True,
            )

        # 新请求之间保留限流间隔。
        has_unfinished_query = any(
            later_query["id"] not in checkpoint_results
            for later_query in queries[position:]
        )
        if not was_cached and has_unfinished_query:
            print(
                f"等待{RERANK_DELAY_SECONDS}秒后处理下一条……",
                flush=True,
            )
            time.sleep(RERANK_DELAY_SECONDS)

    return {
        strategy: {
            metric_name: total_value / total
            for metric_name, total_value in strategy_totals.items()
        }
        for strategy, strategy_totals in metric_totals.items()
    }


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
    """执行单一检索方式并汇总平均指标。"""
    metric_totals = {
        "precision": 0.0,
        "recall": 0.0,
        "hit_rate": 0.0,
        "ndcg": 0.0,
        "mrr": 0.0,
    }
    started_at = time.perf_counter()
    total = len(queries)
    # 网络型评测支持断点续跑。
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
        relevance_by_id = qrels.get(query_id, {})
        relevant_ids = set(gold_qrels.get(query_id, {}))
        # 无缓存时执行当前检索流程。
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
            query_metrics = {
                **calculate_metrics(
                    retrieved_ids,
                    relevance_by_id,
                    relevant_ids,
                    top_k,
                ),
                "retrieved_ids": retrieved_ids,
            }

            # 保存官方候选池诊断信息。
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

            # 保存融合前后的名次和相关等级。
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

            # 每条查询完成后立即更新断点。
            if checkpoint_path is not None:
                checkpoint_results[query_id] = query_metrics
                save_checkpoint(
                    checkpoint_path,
                    method,
                    split,
                    top_k,
                    checkpoint_results,
                )

        # 从旧断点补算新增指标。
        if "precision" not in query_metrics or "hit_rate" not in query_metrics:
            query_metrics.update(
                calculate_metrics(
                    query_metrics["retrieved_ids"],
                    relevance_by_id,
                    relevant_ids,
                    top_k,
                )
            )

        for metric_name in metric_totals:
            metric_totals[metric_name] += query_metrics[metric_name]

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

        # 仅在新网络请求之间等待。
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
        "rerank_ablation",
        "rerank_pool",
    }:
        checkpoint_path = EVALUATION_ROOT / (
            f"{arguments.method}_{arguments.split}_top{arguments.top_k}.json"
        )
        print(f"断点文件：{checkpoint_path}")

    if arguments.method == "rerank_ablation":
        metrics_by_strategy = evaluate_rerank_ablation(
            queries,
            qrels,
            gold_qrels,
            arguments.top_k,
            arguments.split,
            checkpoint_path,
        )
        print("\n评测结果：")
        print(
            f"| 方案 | Precision@{arguments.top_k} | Recall@{arguments.top_k} | "
            f"HitRate@{arguments.top_k} | NDCG@{arguments.top_k} | "
            f"MRR@{arguments.top_k} |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for strategy, label in RERANK_ABLATION_LABELS.items():
            metrics = metrics_by_strategy[strategy]
            display_label = label.replace("前K名", f"前{arguments.top_k}名")
            print(
                f"| {display_label} | {metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | {metrics['hit_rate']:.4f} | "
                f"{metrics['ndcg']:.4f} | {metrics['mrr']:.4f} |"
            )
        return

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
    print(f"Precision@{arguments.top_k}: {metrics['precision']:.4f}")
    print(f"Recall@{arguments.top_k}: {metrics['recall']:.4f}")
    print(f"HitRate@{arguments.top_k}: {metrics['hit_rate']:.4f}")
    print(f"NDCG@{arguments.top_k}: {metrics['ndcg']:.4f}")
    print(f"MRR@{arguments.top_k}: {metrics['mrr']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"评测失败：{error}", file=sys.stderr)
        raise SystemExit(1) from error
