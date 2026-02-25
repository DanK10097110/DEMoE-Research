"""
Information retrieval evaluation metrics.

Implements NDCG@k, Recall@k, MRR, MAP, and Precision@k inline —
no external beir/pytrec_eval dependency required.

These metrics match the formulations used in:
 - BEIR benchmark (Thakur et al. 2021) — NDCG@10 as primary metric
 - MTEB leaderboard — NDCG@10 for retrieval category
 - Pinecone / Weaviate evaluation guides

Reference:
  NDCG: Järvinen & Kekäläinen (2002)
  MRR:  Voorhees (1999) TREC QA track
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def dcg_at_k(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at rank k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(
        rel / np.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def ndcg_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int,
) -> float:
    """
    Normalized DCG@k for a single query.

    Args:
        retrieved:  Ordered list of retrieved doc IDs (most relevant first).
        qrels:      Ground-truth {doc_id: relevance_score}.
        k:          Cutoff rank.
    """
    rels = [qrels.get(d, 0) for d in retrieved[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    dcg   = dcg_at_k(rels, k)
    idcg  = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int,
    min_relevance: int = 1,
) -> float:
    """Recall@k: fraction of relevant docs that appear in top-k results."""
    relevant = {d for d, r in qrels.items() if r >= min_relevance}
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def precision_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int,
    min_relevance: int = 1,
) -> float:
    """Precision@k: fraction of top-k results that are relevant."""
    relevant = {d for d, r in qrels.items() if r >= min_relevance}
    if not retrieved or not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / min(k, len(retrieved))


def reciprocal_rank(
    retrieved: List[str],
    qrels: Dict[str, int],
    min_relevance: int = 1,
) -> float:
    """Reciprocal Rank for a single query."""
    relevant = {d for d, r in qrels.items() if r >= min_relevance}
    for rank, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int = 1000,
    min_relevance: int = 1,
) -> float:
    """Average Precision@k for a single query."""
    relevant = {d for d, r in qrels.items() if r >= min_relevance}
    if not relevant:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for rank, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            hits += 1
            sum_precisions += hits / rank
    return sum_precisions / len(relevant)


def evaluate_retrieval(
    run: Dict[str, List[str]],                    # query_id → ranked doc list
    qrels: Dict[str, Dict[str, int]],             # query_id → {doc_id: relevance}
    k_values: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """
    Compute all standard retrieval metrics over a full run.

    Returns a flat dict of metric_name → mean_value, e.g.:
        {"ndcg@1": 0.72, "ndcg@10": 0.61, "recall@10": 0.85, "mrr": 0.78, ...}
    """
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    def _add(key: str, val: float):
        totals[key] = totals.get(key, 0.0) + val
        counts[key] = counts.get(key, 0) + 1

    for qid, retrieved in run.items():
        if qid not in qrels:
            continue
        rel = qrels[qid]
        _add("mrr", reciprocal_rank(retrieved, rel))
        _add("map@1000", average_precision(retrieved, rel))
        for k in k_values:
            _add(f"ndcg@{k}",      ndcg_at_k(retrieved, rel, k))
            _add(f"recall@{k}",    recall_at_k(retrieved, rel, k))
            _add(f"precision@{k}", precision_at_k(retrieved, rel, k))

    return {key: totals[key] / counts[key] for key in totals if counts[key] > 0}


def cosine_sim_search(
    query: np.ndarray,
    corpus_ids: List[str],
    corpus_embs: np.ndarray,
    top_k: int = 100,
) -> List[str]:
    """Simple brute-force cosine similarity search. Returns ranked doc IDs."""
    sims = corpus_embs @ query  # (N,)
    idx  = np.argsort(-sims)[:top_k]
    return [corpus_ids[i] for i in idx]
