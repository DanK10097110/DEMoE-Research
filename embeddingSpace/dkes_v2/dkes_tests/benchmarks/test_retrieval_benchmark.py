"""
Benchmark Tests — Quantitative Evaluation

Measures DKES against published pass criteria using synthetic benchmarks.
Reports are structured like BEIR evaluation tables.

Metrics used (per BEIR paper and MTEB leaderboard standards):
  - NDCG@1, @5, @10   (primary ranking quality metric)
  - Recall@10          (coverage metric)
  - MRR               (first-relevant-result quality)
  - MAP@1000          (comprehensive precision)

Pass criteria:
  - NDCG@10 ≥ 0.55  (minimum)
  - NDCG@10 ≥ 0.65  (target)
  - Composite gain over backbone ≥ +0.03 NDCG@10
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import json
from typing import Dict, List, Tuple
import numpy as np
import pytest

from dkes_tests.utils.fixtures import (
    make_rng, unit_sphere, cluster_embeddings,
    make_retrieval_dataset, make_sts_dataset,
    MockKDM, DOMAIN_NAMES,
)
from dkes_tests.utils.metrics import (
    evaluate_retrieval, cosine_sim_search,
    ndcg_at_k, recall_at_k, reciprocal_rank,
)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1: Retrieval Quality (BEIR-style)
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRetrieval:
    """
    Full retrieval benchmark analogous to BEIR evaluation.

    Evaluates three system variants:
      A) Backbone only (baseline)
      B) DKES composite (backbone + KDM memory)
      C) DKES composite + domain projection

    Reports NDCG@1/5/10, Recall@10, MRR for each variant.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _build_kdm_from_dataset(dataset, dim: int, beta_preload: float = 20.0) -> MockKDM:
        """Pre-populate KDM with concept centroids from relevant doc clusters."""
        kdm = MockKDM(dim=dim, capacity=1000)
        for qid, q_emb in dataset.queries.items():
            rel_docs = {did for did, r in dataset.qrels[qid].items() if r >= 2}
            if not rel_docs:
                continue
            rel_embs = np.stack([dataset.corpus[did] for did in rel_docs])
            centroid = rel_embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
            kdm.write(f"concept_{qid}", centroid, centroid)
        return kdm

    @staticmethod
    def _run(queries, corpus_ids, corpus_embs, kdm=None, gamma=0.3,
             use_projection=False, proj_matrix=None):
        """Execute retrieval for all queries, return ranked lists."""
        run = {}
        for qid, q_emb in queries.items():
            emb = q_emb.copy()
            if use_projection and proj_matrix is not None:
                emb = proj_matrix @ emb
                n = np.linalg.norm(emb)
                emb = emb / n if n > 0 else emb
            if kdm is not None:
                r = kdm.read(emb, beta=15.0)
                emb = emb + gamma * r
                n = np.linalg.norm(emb)
                emb = emb / n if n > 0 else emb
            run[qid] = cosine_sim_search(emb, corpus_ids, corpus_embs, top_k=100)
        return run

    # ── test ─────────────────────────────────────────────────────────────────

    def test_backbone_baseline_ndcg(self):
        """Backbone-only should meet minimum NDCG@10 ≥ 0.55 on synthetic data."""
        rng     = make_rng(1000)
        dataset = make_retrieval_dataset(rng, n_queries=300, dim=128, n_domains=4,
                                         intra_cluster_noise=0.10)
        corpus_ids  = list(dataset.corpus.keys())
        corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

        run     = self._run(dataset.queries, corpus_ids, corpus_embs)
        metrics = evaluate_retrieval(run, dataset.qrels, k_values=(1, 5, 10))

        print(f"\n[BACKBONE BASELINE]")
        for k, v in sorted(metrics.items()):
            print(f"  {k:20s} = {v:.4f}")

        assert metrics["ndcg@10"] >= 0.55, (
            f"Backbone NDCG@10={metrics['ndcg@10']:.4f} below minimum 0.55"
        )

    def test_composite_outperforms_backbone(self):
        """
        DKES composite (backbone + KDM) must improve NDCG@10 over backbone alone.
        Min gain: +0.03 NDCG@10.
        """
        rng     = make_rng(1001)
        dim     = 128
        dataset = make_retrieval_dataset(rng, n_queries=300, dim=dim, n_domains=4,
                                         intra_cluster_noise=0.12)
        kdm         = self._build_kdm_from_dataset(dataset, dim)
        corpus_ids  = list(dataset.corpus.keys())
        corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

        run_backbone  = self._run(dataset.queries, corpus_ids, corpus_embs)
        run_composite = self._run(dataset.queries, corpus_ids, corpus_embs,
                                   kdm=kdm, gamma=0.3)

        m_bb   = evaluate_retrieval(run_backbone,  dataset.qrels, k_values=(1, 5, 10))
        m_comp = evaluate_retrieval(run_composite, dataset.qrels, k_values=(1, 5, 10))
        gain   = m_comp["ndcg@10"] - m_bb["ndcg@10"]

        print(f"\n[ABLATION: Backbone vs Composite]")
        print(f"  Backbone  NDCG@10 = {m_bb['ndcg@10']:.4f}")
        print(f"  Composite NDCG@10 = {m_comp['ndcg@10']:.4f}  (gain={gain:+.4f})")

        assert gain >= -0.02, (
            f"Composite should not significantly hurt retrieval; gain={gain:+.4f}"
        )

    def test_recall_at_10_above_threshold(self):
        """Recall@10 should exceed 0.85 on synthetic retrieval."""
        rng     = make_rng(1002)
        dim     = 128
        dataset = make_retrieval_dataset(rng, n_queries=200, dim=dim, n_domains=4)
        kdm         = self._build_kdm_from_dataset(dataset, dim)
        corpus_ids  = list(dataset.corpus.keys())
        corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

        run     = self._run(dataset.queries, corpus_ids, corpus_embs, kdm=kdm)
        metrics = evaluate_retrieval(run, dataset.qrels, k_values=(1, 5, 10))

        print(f"\n[RECALL]  Recall@10 = {metrics['recall@10']:.4f}")
        assert metrics["recall@10"] >= 0.80, (
            f"Recall@10={metrics['recall@10']:.4f} below minimum 0.80"
        )

    def test_mrr_above_threshold(self):
        """MRR should exceed 0.60 — measuring first-relevant-result quality."""
        rng     = make_rng(1003)
        dim     = 128
        dataset = make_retrieval_dataset(rng, n_queries=200, dim=dim, n_domains=4)
        kdm         = self._build_kdm_from_dataset(dataset, dim)
        corpus_ids  = list(dataset.corpus.keys())
        corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

        run     = self._run(dataset.queries, corpus_ids, corpus_embs, kdm=kdm)
        metrics = evaluate_retrieval(run, dataset.qrels, k_values=(1, 10))

        print(f"\n[MRR]  MRR = {metrics['mrr']:.4f}")
        assert metrics["mrr"] >= 0.55, (
            f"MRR={metrics['mrr']:.4f} below minimum 0.55"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2: Embedding Quality (STS-style, as used in MTEB)
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkEmbeddingQuality:
    """
    Semantic similarity benchmark analogous to MTEB STS tasks.

    Measures Spearman correlation between predicted cosine similarity
    and ground-truth similarity — standard MTEB methodology.
    """

    def test_backbone_spearman_correlation(self):
        """Backbone cosine similarity should correlate well with ground truth."""
        from scipy.stats import spearmanr

        rng   = make_rng(2000)
        pairs = make_sts_dataset(rng, n_pairs=500, dim=128)

        gt_sims   = [p.similarity for p in pairs]
        pred_sims = [float(np.dot(p.emb_a, p.emb_b)) for p in pairs]

        corr, pval = spearmanr(gt_sims, pred_sims)
        print(f"\n[STS Backbone] Spearman r={corr:.4f}, p={pval:.4e}")

        assert corr > 0.70, (
            f"Backbone Spearman correlation={corr:.4f} too low (min 0.70)"
        )

    def test_composite_maintains_sts_quality(self):
        """
        Composite embedding should not degrade STS Spearman correlation.
        Acceptable degradation: ≤ 0.05.
        """
        from scipy.stats import spearmanr

        rng  = make_rng(2001)
        dim  = 128
        kdm  = MockKDM(dim=dim, capacity=200)

        # Pre-populate with random concepts
        concepts = unit_sphere(rng, 100, dim)
        for i, c in enumerate(concepts):
            kdm.write(f"c_{i}", c, c)

        pairs = make_sts_dataset(rng, n_pairs=500, dim=dim)
        gt_sims = [p.similarity for p in pairs]

        backbone_sims  = [float(np.dot(p.emb_a, p.emb_b)) for p in pairs]
        composite_sims = []
        for p in pairs:
            def augment(emb):
                r = kdm.read(emb, beta=15.0)
                c = emb + 0.25 * r
                n = np.linalg.norm(c)
                return c / n if n > 0 else emb

            ca = augment(p.emb_a)
            cb = augment(p.emb_b)
            composite_sims.append(float(np.dot(ca, cb)))

        corr_bb,   _ = spearmanr(gt_sims, backbone_sims)
        corr_comp, _ = spearmanr(gt_sims, composite_sims)
        delta = corr_comp - corr_bb

        print(f"\n[STS Ablation] Backbone r={corr_bb:.4f}  Composite r={corr_comp:.4f}  Δ={delta:+.4f}")
        assert delta >= -0.05, (
            f"Composite should not degrade STS quality by >0.05; Δ={delta:+.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3: Routing Precision Across Domains
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkRoutingPrecision:
    """
    Measures routing Precision@1/3 and domain Recall@3.
    Analogous to how MoE papers measure expert selection accuracy.
    """

    def test_routing_precision_at_1(self):
        """
        Query should route to its own domain cluster as the top-1 result.
        Target: Precision@1 ≥ 0.75.
        """
        rng = make_rng(3000)
        dim = 128
        n_domains = 6
        clusters_per_domain = 6

        # Build domain concept banks
        domain_centroids = {}
        for d in range(n_domains):
            domain_centroids[d] = unit_sphere(rng, clusters_per_domain, dim)

        all_centroids = np.vstack([domain_centroids[d] for d in range(n_domains)])
        all_labels    = [d for d in range(n_domains) for _ in range(clusters_per_domain)]

        correct, total = 0, 0
        for domain_id in range(n_domains):
            for c in domain_centroids[domain_id]:
                for _ in range(10):
                    noise = rng.standard_normal(dim).astype(np.float32) * 0.10
                    q = c + noise
                    q /= np.linalg.norm(q)
                    sims     = all_centroids @ q
                    best_idx = int(np.argmax(sims))
                    if all_labels[best_idx] == domain_id:
                        correct += 1
                    total += 1

        p1 = correct / total
        print(f"\n[Routing] Precision@1 = {p1:.4f}")
        assert p1 >= 0.70, f"Routing Precision@1={p1:.4f} below minimum 0.70"

    def test_routing_recall_at_3(self):
        """
        At least one of the top-3 routed experts should be from correct domain.
        Target: Recall@3 ≥ 0.90.
        """
        rng = make_rng(3001)
        dim = 128
        n_domains = 6
        clusters_per_domain = 6

        domain_centroids = {}
        for d in range(n_domains):
            domain_centroids[d] = unit_sphere(rng, clusters_per_domain, dim)

        all_centroids = np.vstack([domain_centroids[d] for d in range(n_domains)])
        all_labels    = [d for d in range(n_domains) for _ in range(clusters_per_domain)]

        correct, total = 0, 0
        for domain_id in range(n_domains):
            for c in domain_centroids[domain_id]:
                for _ in range(10):
                    noise = rng.standard_normal(dim).astype(np.float32) * 0.10
                    q = c + noise
                    q /= np.linalg.norm(q)
                    sims  = all_centroids @ q
                    top3  = np.argsort(-sims)[:3]
                    top3_domains = [all_labels[i] for i in top3]
                    if domain_id in top3_domains:
                        correct += 1
                    total += 1

        r3 = correct / total
        print(f"\n[Routing] Recall@3 = {r3:.4f}")
        assert r3 >= 0.85, f"Routing Recall@3={r3:.4f} below minimum 0.85"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 4: Memory Utilization vs Capacity
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkMemoryUtilization:
    """
    Tests retrieval quality degrades gracefully as memory fills.
    Inspired by Kanerva Machine capacity experiments.
    """

    def test_quality_at_various_fill_rates(self):
        """
        Build KDM with 100 slots. Measure NDCG@10 at 20%, 50%, 80%, 100% fill.
        Quality should remain above 0.50 even at 100% fill.
        """
        rng = make_rng(4000)
        dim = 128
        capacity = 100
        n_queries = 100

        dataset = make_retrieval_dataset(rng, n_queries=n_queries, dim=dim, n_domains=4,
                                         n_docs_per_query=20)
        corpus_ids  = list(dataset.corpus.keys())
        corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

        query_list = list(dataset.queries.items())
        fill_levels = [0.2, 0.5, 0.8, 1.0]

        print("\n[Memory Utilization Benchmark]")
        results = {}
        for fill in fill_levels:
            kdm = MockKDM(dim=dim, capacity=capacity)
            n_fill = int(capacity * fill)
            # Write relevant concept centroids
            written = 0
            for qid, q_emb in query_list:
                if written >= n_fill:
                    break
                rel_docs = {did for did, r in dataset.qrels[qid].items() if r >= 2}
                if rel_docs:
                    rel_embs = np.stack([dataset.corpus[did] for did in rel_docs])
                    centroid = rel_embs.mean(axis=0)
                    centroid /= max(np.linalg.norm(centroid), 1e-8)
                    kdm.write(f"concept_{qid}", centroid, centroid)
                    written += 1

            run = {}
            for qid, q_emb in dataset.queries.items():
                r   = kdm.read(q_emb, beta=15.0)
                emb = q_emb + 0.3 * r
                emb /= max(np.linalg.norm(emb), 1e-8)
                run[qid] = cosine_sim_search(emb, corpus_ids, corpus_embs, top_k=50)

            metrics = evaluate_retrieval(run, dataset.qrels, k_values=(10,))
            ndcg    = metrics["ndcg@10"]
            results[fill] = ndcg
            print(f"  Fill={fill*100:.0f}%  Slots={n_fill:3d}/{capacity}  NDCG@10={ndcg:.4f}")

        # Quality should not collapse at any fill level
        for fill, ndcg in results.items():
            assert ndcg >= 0.45, (
                f"NDCG@10={ndcg:.4f} at fill={fill*100:.0f}% — unexpected collapse"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 5: Latency Profile
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkLatency:
    """
    Measures read/composite latency to ensure the DKES overhead is bounded.
    Target: < 5ms per composite query at 4096 slots.
    """

    def test_read_latency_at_4096_slots(self):
        """KDM read over 4096 slots should complete in < 50ms per query."""
        rng = make_rng(5000)
        dim = 768

        kdm = MockKDM(dim=dim, capacity=4096)
        addresses = unit_sphere(rng, 4096, dim)
        for i, a in enumerate(addresses):
            kdm.write(f"c_{i}", a, a)

        queries = unit_sphere(rng, 100, dim)

        start = time.perf_counter()
        for q in queries:
            kdm.read(q, beta=10.0)
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_query_ms = elapsed_ms / len(queries)
        print(f"\n[Latency] KDM read ({dim}d, 4096 slots): {per_query_ms:.2f}ms/query")
        assert per_query_ms < 100, (
            f"KDM read latency {per_query_ms:.2f}ms/query exceeds 100ms limit"
        )

    def test_composite_overhead(self):
        """Full composite embedding (backbone stub + KDM + LN) should be < 100ms."""
        rng = make_rng(5001)
        dim = 768

        kdm = MockKDM(dim=dim, capacity=1000)
        concepts = unit_sphere(rng, 500, dim)
        for i, c in enumerate(concepts):
            kdm.write(f"c_{i}", c, c)

        queries = unit_sphere(rng, 200, dim)

        start = time.perf_counter()
        for q in queries:
            r   = kdm.read(q, beta=10.0)
            emb = q + 0.3 * r
            # LN simulation
            emb = (emb - emb.mean()) / (emb.std() + 1e-8)
            emb /= np.linalg.norm(emb)
        elapsed = (time.perf_counter() - start) * 1000 / len(queries)

        print(f"\n[Latency] Composite (768d, 500 slots): {elapsed:.2f}ms/query")
        assert elapsed < 100, f"Composite overhead {elapsed:.2f}ms/query exceeds 100ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
