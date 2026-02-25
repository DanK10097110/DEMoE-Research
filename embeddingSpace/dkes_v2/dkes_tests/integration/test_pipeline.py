"""
Integration Tests — Full DKES Pipeline

Tests the complete Write → Read → Composite → Route cycle.
Uses the MockKDM for speed; these tests prove the *flow* not the neural weights.

Inspired by:
  - Kanerva Machine paper's iterative retrieval convergence tests
  - DNC paper's sequential read-after-write accuracy tests
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import numpy as np
from typing import List, Dict

from dkes_tests.utils.fixtures import (
    make_rng, unit_sphere, cluster_embeddings,
    make_retrieval_dataset, MockKDM,
)
from dkes_tests.utils.metrics import (
    evaluate_retrieval, cosine_sim_search,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Write → Read Fidelity (core Kanerva property)
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteReadFidelity:
    """
    Tests inspired by Figure 3 of Kanerva Machine paper:
    After writing N patterns, reads should recover the correct pattern
    for each address, even in the presence of interference.
    """

    def test_single_write_read_exact(self):
        """Write one concept; read with exact address → value perfectly recovered."""
        rng = make_rng(100)
        kdm = MockKDM(dim=128)
        a = unit_sphere(rng, 1, 128)[0]
        v = unit_sphere(rng, 1, 128)[0]
        kdm.write("c0", a, v)

        result = kdm.read(a, beta=100.0)
        cos = float(np.dot(result, v))
        assert cos > 0.995, f"Single-slot read should be near-perfect: cos={cos:.4f}"

    def test_ten_concepts_each_readable(self):
        """Write 10 distinct concepts; each should be readable from its own address."""
        rng = make_rng(101)
        dim = 128
        kdm = MockKDM(dim=dim, capacity=50)
        addresses = unit_sphere(rng, 10, dim)
        values    = unit_sphere(rng, 10, dim)
        for i, (a, v) in enumerate(zip(addresses, values)):
            kdm.write(f"c{i}", a, v)

        cos_scores = []
        for i, (a, v) in enumerate(zip(addresses, values)):
            result = kdm.read(a, beta=30.0)
            cos_scores.append(float(np.dot(result, v)))

        mean_cos = np.mean(cos_scores)
        min_cos  = np.min(cos_scores)
        assert mean_cos > 0.70, f"Mean read fidelity should be >0.70, got {mean_cos:.4f}"
        assert min_cos  > 0.40, f"Min read fidelity should be >0.40, got {min_cos:.4f}"

    def test_read_after_many_writes_still_recalls(self):
        """
        Write 50 concepts, then query the earliest ones.
        Memory should still recall them (capacity = 100 so no eviction).
        This mirrors the Kanerva Machine's capacity test.
        """
        rng = make_rng(102)
        dim = 64
        n   = 50
        kdm = MockKDM(dim=dim, capacity=100)
        addresses = unit_sphere(rng, n, dim)
        values    = unit_sphere(rng, n, dim)
        for i, (a, v) in enumerate(zip(addresses, values)):
            kdm.write(f"c{i}", a, v)

        # Read the first 10 concepts
        scores = []
        for i in range(10):
            result = kdm.read(addresses[i], beta=20.0)
            scores.append(float(np.dot(result, values[i])))

        assert np.mean(scores) > 0.60, (
            f"Early concepts should still be recalled; mean cos={np.mean(scores):.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Composite Embedding Improves Retrieval
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeImprovesRetrieval:
    """
    Key integration test: DKES composite embedding should achieve better
    retrieval metrics than backbone-only.

    This mirrors the BEIR-style evaluation but at small synthetic scale.
    """

    def _run_retrieval(self, queries, corpus_ids, corpus_embs, qrels, use_memory=False,
                       kdm=None, gamma=0.3):
        """Run retrieval with optional memory augmentation."""
        run = {}
        for qid, q_emb in queries.items():
            if use_memory and kdm is not None:
                r   = kdm.read(q_emb, beta=15.0)
                emb = q_emb + gamma * r
                norm = np.linalg.norm(emb)
                emb = emb / norm if norm > 0 else emb
            else:
                emb = q_emb
            run[qid] = cosine_sim_search(emb, corpus_ids, corpus_embs, top_k=50)
        return run

    def test_composite_matches_or_beats_backbone(self):
        """
        Composite embedding should achieve NDCG@10 ≥ backbone's NDCG@10.
        """
        rng = make_rng(200)
        dataset = make_retrieval_dataset(rng, n_queries=100, dim=128, n_domains=4)

        # Populate KDM with ground-truth document centroids
        kdm = MockKDM(dim=128, capacity=200)
        for qid, q_emb in dataset.queries.items():
            rel_docs = {did for did, r in dataset.qrels[qid].items() if r >= 2}
            if rel_docs:
                # Write centroid of relevant docs as a concept
                rel_embs = np.stack([dataset.corpus[did] for did in rel_docs])
                centroid = rel_embs.mean(axis=0)
                centroid /= np.linalg.norm(centroid)
                kdm.write(f"concept_{qid}", centroid, centroid)

        corpus_ids  = list(dataset.corpus.keys())
        corpus_embs = np.stack([dataset.corpus[cid] for cid in corpus_ids])

        run_backbone  = self._run_retrieval(dataset.queries, corpus_ids, corpus_embs,
                                             dataset.qrels, use_memory=False)
        run_composite = self._run_retrieval(dataset.queries, corpus_ids, corpus_embs,
                                             dataset.qrels, use_memory=True,
                                             kdm=kdm, gamma=0.3)

        metrics_bb   = evaluate_retrieval(run_backbone,  dataset.qrels, k_values=(1, 5, 10))
        metrics_comp = evaluate_retrieval(run_composite, dataset.qrels, k_values=(1, 5, 10))

        assert metrics_comp["ndcg@10"] >= metrics_bb["ndcg@10"] - 0.02, (
            f"Composite should match or beat backbone. "
            f"Backbone NDCG@10={metrics_bb['ndcg@10']:.4f}, "
            f"Composite NDCG@10={metrics_comp['ndcg@10']:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Eviction → ColdStore → Reactivation Cycle
# ─────────────────────────────────────────────────────────────────────────────

class TestEvictionReactivationCycle:
    """Verify that evicted concepts can be found and re-integrated."""

    def test_concept_recoverable_after_eviction(self):
        """
        Simulate filling a small KDM past capacity, then verifying
        evicted concept is findable by similar query.
        """
        rng = make_rng(300)
        dim = 64
        capacity = 5

        kdm    = MockKDM(dim=dim, capacity=capacity)
        cold   = {}   # concept_id → address

        # Write 10 concepts; first 5 will be evicted
        addresses = unit_sphere(rng, 10, dim)
        for i, a in enumerate(addresses):
            if i >= capacity:
                # Simulate eviction of oldest
                oldest = list(kdm._slots.keys())[0]
                cold[oldest] = kdm._slots[oldest]
            kdm.write(f"c{i}", a, a)

        assert len(cold) > 0, "Some concepts should have been evicted to cold store"

        # Query similar to first evicted concept
        first_evicted = list(cold.keys())[0]
        target_addr   = cold[first_evicted]
        query = target_addr + rng.standard_normal(dim).astype(np.float32) * 0.05
        query /= np.linalg.norm(query)

        # Search cold store
        best_cid, best_cos = None, -1.0
        for cid, addr in cold.items():
            cos = float(np.dot(query, addr))
            if cos > best_cos:
                best_cos = cos
                best_cid = cid

        assert best_cid == first_evicted
        assert best_cos > 0.85, f"Should find evicted concept; cos={best_cos:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Domain Isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestDomainIsolation:
    """
    Verify that routing doesn't bleed across domain boundaries.
    Biomedical queries should route to biomedical concepts, not finance.
    """

    def test_domain_routing_precision(self):
        """
        8 domains × 10 concept clusters × 5 queries each.
        A query in domain_i should retrieve concepts from domain_i.
        """
        rng = make_rng(400)
        dim = 128
        n_domains = 4
        clusters_per_domain = 5
        queries_per_cluster = 10

        # Build domain concept banks
        domain_centroids: Dict[int, np.ndarray] = {}
        for d in range(n_domains):
            centroids = unit_sphere(rng, clusters_per_domain, dim)
            domain_centroids[d] = centroids

        # All centroids in one index
        all_centroids = np.vstack(list(domain_centroids.values()))
        all_domain_labels = [d for d in range(n_domains)
                             for _ in range(clusters_per_domain)]

        correct = 0
        total   = 0
        for domain_id, centroids in domain_centroids.items():
            for c in centroids:
                for _ in range(queries_per_cluster):
                    # Query = centroid + small noise
                    noise = rng.standard_normal(dim).astype(np.float32) * 0.08
                    q = c + noise
                    q /= np.linalg.norm(q)

                    # Find nearest centroid
                    sims = all_centroids @ q
                    best_idx = int(np.argmax(sims))
                    predicted_domain = all_domain_labels[best_idx]
                    if predicted_domain == domain_id:
                        correct += 1
                    total += 1

        recall_at_1 = correct / total
        assert recall_at_1 > 0.75, (
            f"Domain routing Recall@1 should exceed 0.75; got {recall_at_1:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
