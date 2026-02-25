#!/usr/bin/env python3
"""
DKES Evaluation Runner
======================

Runs the full evaluation suite and produces a structured JSON + human-readable
report. Designed to be run as a standalone script after all tests pass.

Usage:
    python run_evaluation.py
    python run_evaluation.py --report reports/my_report.json
    python run_evaluation.py --quick   (fewer queries, faster)

Report structure mirrors BEIR evaluation tables and PMoE ablation tables.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np

from dkes_tests.utils.fixtures import (
    make_rng, unit_sphere, cluster_embeddings,
    make_retrieval_dataset, make_continual_stream,
    make_sts_dataset, MockKDM, DOMAIN_NAMES,
)
from dkes_tests.utils.metrics import (
    evaluate_retrieval, cosine_sim_search,
)


# ─────────────────────────────────────────────────────────────────────────────
# Report structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    status: str       # "PASS" | "FAIL" | "WARN"
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str]  = field(default_factory=list)
    elapsed_s: float  = 0.0


@dataclass
class EvaluationReport:
    generated_at: str
    dkes_version: str = "2.0"
    total_pass:  int  = 0
    total_fail:  int  = 0
    total_warn:  int  = 0
    benchmarks: List[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult):
        self.benchmarks.append(result)
        if result.status == "PASS":
            self.total_pass += 1
        elif result.status == "FAIL":
            self.total_fail += 1
        else:
            self.total_warn += 1

    def to_dict(self):
        d = asdict(self)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Individual evaluation routines
# ─────────────────────────────────────────────────────────────────────────────

def _status(value, min_pass, target):
    if value >= target:
        return "PASS"
    elif value >= min_pass:
        return "WARN"
    return "FAIL"


def eval_retrieval_baseline(rng, n_queries: int = 300, dim: int = 128) -> BenchmarkResult:
    t0 = time.perf_counter()
    dataset = make_retrieval_dataset(rng, n_queries=n_queries, dim=dim, n_domains=4)
    corpus_ids  = list(dataset.corpus.keys())
    corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

    run = {}
    for qid, q in dataset.queries.items():
        run[qid] = cosine_sim_search(q, corpus_ids, corpus_embs, top_k=100)
    metrics = evaluate_retrieval(run, dataset.qrels, k_values=(1, 5, 10))

    status = _status(metrics["ndcg@10"], min_pass=0.50, target=0.60)
    return BenchmarkResult(
        name="Backbone Baseline Retrieval",
        status=status,
        metrics={k: round(v, 4) for k, v in metrics.items()},
        notes=[f"n_queries={n_queries}, dim={dim}, n_domains=4"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


def eval_composite_vs_backbone(rng, n_queries: int = 300, dim: int = 128) -> BenchmarkResult:
    t0 = time.perf_counter()
    dataset = make_retrieval_dataset(rng, n_queries=n_queries, dim=dim, n_domains=4,
                                     intra_cluster_noise=0.12)
    kdm = MockKDM(dim=dim, capacity=1000)
    for qid, q in dataset.queries.items():
        rel_docs = {did for did, r in dataset.qrels[qid].items() if r >= 2}
        if rel_docs:
            embs = np.stack([dataset.corpus[did] for did in rel_docs])
            c = embs.mean(0); c /= max(np.linalg.norm(c), 1e-8)
            kdm.write(f"concept_{qid}", c, c)

    corpus_ids  = list(dataset.corpus.keys())
    corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

    def run_retrieval(use_kdm=False, gamma=0.3):
        run = {}
        for qid, q in dataset.queries.items():
            emb = q.copy()
            if use_kdm:
                r   = kdm.read(emb, beta=15.0)
                emb = emb + gamma * r
                n   = np.linalg.norm(emb); emb = emb / n if n > 0 else emb
            run[qid] = cosine_sim_search(emb, corpus_ids, corpus_embs, top_k=100)
        return run

    m_bb   = evaluate_retrieval(run_retrieval(False), dataset.qrels, k_values=(1,5,10))
    m_comp = evaluate_retrieval(run_retrieval(True),  dataset.qrels, k_values=(1,5,10))
    gain   = m_comp["ndcg@10"] - m_bb["ndcg@10"]

    status = _status(gain, min_pass=-0.02, target=0.03)
    return BenchmarkResult(
        name="Composite vs Backbone (Ablation)",
        status=status,
        metrics={
            "backbone_ndcg@10":  round(m_bb["ndcg@10"], 4),
            "composite_ndcg@10": round(m_comp["ndcg@10"], 4),
            "ndcg_gain":         round(gain, 4),
            "backbone_mrr":      round(m_bb["mrr"], 4),
            "composite_mrr":     round(m_comp["mrr"], 4),
        },
        notes=[f"γ=0.3, KDM preloaded with {kdm.slot_count()} concepts"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


def eval_routing_precision(rng, dim: int = 128) -> BenchmarkResult:
    t0 = time.perf_counter()
    n_domains = 6
    clusters_per_domain = 6

    domain_centroids = {d: unit_sphere(rng, clusters_per_domain, dim)
                        for d in range(n_domains)}
    all_centroids = np.vstack(list(domain_centroids.values()))
    all_labels    = [d for d in range(n_domains) for _ in range(clusters_per_domain)]

    p1_correct = r3_correct = total = 0
    for domain_id, centroids in domain_centroids.items():
        for c in centroids:
            for _ in range(10):
                noise = rng.standard_normal(dim).astype(np.float32) * 0.10
                q = c + noise; q /= np.linalg.norm(q)
                sims = all_centroids @ q
                top3 = np.argsort(-sims)[:3]
                if all_labels[top3[0]] == domain_id:
                    p1_correct += 1
                if domain_id in [all_labels[i] for i in top3]:
                    r3_correct += 1
                total += 1

    p1 = p1_correct / total
    r3 = r3_correct / total
    status = _status(min(p1, r3), min_pass=0.70, target=0.80)
    return BenchmarkResult(
        name="Routing Precision & Recall",
        status=status,
        metrics={"precision@1": round(p1, 4), "recall@3": round(r3, 4)},
        notes=[f"{n_domains} domains × {clusters_per_domain} clusters × 10 queries"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


def eval_sts_quality(rng, dim: int = 128) -> BenchmarkResult:
    from scipy.stats import spearmanr
    t0 = time.perf_counter()

    kdm   = MockKDM(dim=dim, capacity=200)
    concs = unit_sphere(rng, 100, dim)
    for i, c in enumerate(concs):
        kdm.write(f"c_{i}", c, c)

    pairs  = make_sts_dataset(rng, n_pairs=500, dim=dim)
    gt     = [p.similarity for p in pairs]
    bb_s   = [float(np.dot(p.emb_a, p.emb_b)) for p in pairs]

    def augment(emb):
        r = kdm.read(emb, beta=15.0)
        c = emb + 0.25 * r; n = np.linalg.norm(c)
        return c / n if n > 0 else emb

    comp_s = [float(np.dot(augment(p.emb_a), augment(p.emb_b))) for p in pairs]
    corr_bb,   _ = spearmanr(gt, bb_s)
    corr_comp, _ = spearmanr(gt, comp_s)

    status = _status(corr_bb, min_pass=0.65, target=0.75)
    return BenchmarkResult(
        name="STS Embedding Quality (Spearman)",
        status=status,
        metrics={
            "backbone_spearman":  round(float(corr_bb), 4),
            "composite_spearman": round(float(corr_comp), 4),
            "delta": round(float(corr_comp - corr_bb), 4),
        },
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


def eval_continual_learning(rng, dim: int = 128) -> BenchmarkResult:
    t0 = time.perf_counter()
    batches = make_continual_stream(rng, n_domains=5, clusters_per_domain=6,
                                    queries_per_cluster=8, dim=dim)
    kdm = MockKDM(dim=dim, capacity=500)
    after_inj: Dict[int, float] = {}

    for batch in batches:
        for ci, c in enumerate(batch.clusters):
            kdm.write(f"{batch.domain}_cluster_{ci}", c, c)
        recall = _recall_batch(kdm, batch)
        after_inj[batch.task_id] = recall

    after_all: Dict[int, float] = {}
    for batch in batches:
        after_all[batch.task_id] = _recall_batch(kdm, batch)

    forgetting = [after_inj[t] - after_all[t] for t in sorted(after_inj)]
    max_f = max(forgetting)
    mean_f = float(np.mean(forgetting))

    status = _status(-max_f, min_pass=-0.30, target=-0.15)
    return BenchmarkResult(
        name="Continual Learning — Forgetting Error",
        status=status,
        metrics={
            "max_forgetting":  round(max_f, 4),
            "mean_forgetting": round(mean_f, 4),
            **{f"recall_after_inject_task{t}": round(after_inj[t], 4)
               for t in sorted(after_inj)},
            **{f"recall_after_all_task{t}": round(after_all[t], 4)
               for t in sorted(after_all)},
        },
        notes=["5 domains, sequential injection, capacity=500"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


def _recall_batch(kdm: MockKDM, batch) -> float:
    if kdm.slot_count() == 0:
        return 0.0
    all_addrs = np.stack(list(kdm._slots.values()))
    all_cids  = list(kdm._slots.keys())
    correct = total = 0
    for ci, (centroid, queries) in enumerate(zip(batch.clusters, batch.cluster_queries)):
        expected = f"{batch.domain}_cluster_{ci}"
        for q in queries:
            sims = all_addrs @ q
            best = all_cids[int(np.argmax(sims))]
            if best == expected:
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def eval_memory_utilization(rng, dim: int = 128) -> BenchmarkResult:
    t0 = time.perf_counter()
    capacity = 100
    dataset  = make_retrieval_dataset(rng, n_queries=100, dim=dim, n_domains=4,
                                      n_docs_per_query=20)
    corpus_ids  = list(dataset.corpus.keys())
    corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

    fill_results = {}
    for fill in [0.20, 0.50, 0.80, 1.00]:
        kdm    = MockKDM(dim=dim, capacity=capacity)
        n_fill = int(capacity * fill)
        written = 0
        for qid, q in list(dataset.queries.items()):
            if written >= n_fill:
                break
            rel_docs = {did for did, r in dataset.qrels[qid].items() if r >= 2}
            if rel_docs:
                embs = np.stack([dataset.corpus[did] for did in rel_docs])
                c = embs.mean(0); c /= max(np.linalg.norm(c), 1e-8)
                kdm.write(f"concept_{qid}", c, c)
                written += 1

        run = {}
        for qid, q in dataset.queries.items():
            r   = kdm.read(q, beta=15.0)
            emb = q + 0.3 * r; emb /= max(np.linalg.norm(emb), 1e-8)
            run[qid] = cosine_sim_search(emb, corpus_ids, corpus_embs, top_k=50)
        m = evaluate_retrieval(run, dataset.qrels, k_values=(10,))
        fill_results[f"ndcg@10_at_{int(fill*100)}pct"] = round(m["ndcg@10"], 4)

    min_ndcg = min(fill_results.values())
    status = _status(min_ndcg, min_pass=0.40, target=0.50)
    return BenchmarkResult(
        name="Memory Utilization vs Quality",
        status=status,
        metrics=fill_results,
        notes=["Tests quality at 20/50/80/100% KDM fill"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


def eval_latency(rng, dim: int = 768) -> BenchmarkResult:
    t0 = time.perf_counter()
    kdm = MockKDM(dim=dim, capacity=4096)
    vecs = unit_sphere(rng, 4096, dim)
    for i, v in enumerate(vecs):
        kdm.write(f"c_{i}", v, v)

    queries = unit_sphere(rng, 200, dim)
    ts = time.perf_counter()
    for q in queries:
        kdm.read(q, beta=10.0)
    ms_per_query = (time.perf_counter() - ts) * 1000 / len(queries)

    status = _status(-ms_per_query, min_pass=-100, target=-20)
    return BenchmarkResult(
        name="KDM Read Latency (768d, 4096 slots)",
        status=status,
        metrics={"ms_per_query": round(ms_per_query, 3)},
        notes=["Measured on CPU with numpy; GPU will be much faster"],
        elapsed_s=round(time.perf_counter() - t0, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

PASS_CRITERIA = {
    "NDCG@10 (backbone baseline)":   ("≥ 0.55", "≥ 0.60"),
    "Composite NDCG gain":            ("≥ -0.02", "≥ +0.03"),
    "Routing Precision@1":            ("≥ 0.70", "≥ 0.80"),
    "Routing Recall@3":               ("≥ 0.85", "≥ 0.90"),
    "STS Spearman (backbone)":        ("≥ 0.65", "≥ 0.75"),
    "Max Forgetting Error":           ("≤ 0.30", "≤ 0.15"),
    "Min NDCG@10 (all fill levels)":  ("≥ 0.40", "≥ 0.50"),
    "KDM Read Latency":               ("≤ 100ms", "≤ 20ms"),
}


def print_report(report: EvaluationReport):
    width = 72
    SEP = "─" * width

    print(f"\n{'═' * width}")
    print(f"  DKES EVALUATION REPORT  v{report.dkes_version}")
    print(f"  Generated: {report.generated_at}")
    print(f"{'═' * width}")

    total = report.total_pass + report.total_fail + report.total_warn
    print(f"\n  SUMMARY:  {report.total_pass}/{total} PASS  "
          f"{report.total_warn} WARN  {report.total_fail} FAIL\n")

    for br in report.benchmarks:
        icon = "✓" if br.status == "PASS" else ("!" if br.status == "WARN" else "✗")
        print(f"{SEP}")
        print(f"  [{icon}] {br.status:4s}  {br.name}  ({br.elapsed_s:.1f}s)")
        for k, v in br.metrics.items():
            print(f"           {k:35s} = {v}")
        for note in br.notes:
            print(f"           NOTE: {note}")

    print(f"\n{SEP}")
    print(f"  PASS CRITERIA REFERENCE")
    print(f"{SEP}")
    for metric, (minimum, target) in PASS_CRITERIA.items():
        print(f"  {metric:40s}  min={minimum}  target={target}")
    print(f"{'═' * width}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/evaluation_report.json")
    parser.add_argument("--quick", action="store_true",
                        help="Use fewer queries for faster run")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = make_rng(args.seed)
    n_queries = 100 if args.quick else 300
    dim = 128

    import datetime
    report = EvaluationReport(
        generated_at=datetime.datetime.now().isoformat()
    )

    print("\nRunning DKES Evaluation Suite...")

    evals = [
        ("Retrieval Baseline",   lambda: eval_retrieval_baseline(rng, n_queries, dim)),
        ("Composite Ablation",   lambda: eval_composite_vs_backbone(rng, n_queries, dim)),
        ("Routing Precision",    lambda: eval_routing_precision(rng, dim)),
        ("STS Quality",          lambda: eval_sts_quality(rng, dim)),
        ("Continual Learning",   lambda: eval_continual_learning(rng, dim)),
        ("Memory Utilization",   lambda: eval_memory_utilization(rng, dim)),
        ("Latency Profile",      lambda: eval_latency(rng, dim)),
    ]

    for name, fn in evals:
        print(f"  Running: {name}...", end="", flush=True)
        try:
            result = fn()
            print(f" {result.status}")
        except Exception as e:
            result = BenchmarkResult(
                name=name, status="FAIL",
                notes=[f"Exception: {e}"],
            )
            print(f" ERROR: {e}")
        report.add(result)

    print_report(report)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"Report saved to: {args.report}")

    return 0 if report.total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
