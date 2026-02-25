"""
Shared fixtures and synthetic dataset generators for DKES test suite.

Inspired by how the Kanerva Machine paper used Omniglot (structured concept clusters)
and CIFAR (high-dimensional noisy real data). We generate equivalent synthetic
structures that exercise the same properties without requiring large downloads.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reproducible RNG
# ---------------------------------------------------------------------------

def make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Embedding vector generators
# ---------------------------------------------------------------------------

def unit_sphere(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Generate n normalized random vectors in R^d."""
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-8)


def cluster_embeddings(
    rng: np.random.Generator,
    n_clusters: int,
    n_per_cluster: int,
    dim: int = 768,
    intra_noise: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate clustered embeddings mimicking semantic concept clusters.

    Returns:
        embeddings:    (n_clusters * n_per_cluster, dim)
        centroids:     (n_clusters, dim)
        labels:        (n_clusters * n_per_cluster,) int
    """
    centroids = unit_sphere(rng, n_clusters, dim)
    embeddings = []
    labels = []
    for i, c in enumerate(centroids):
        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * intra_noise
        pts = c[None, :] + noise
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-8)
        embeddings.append(pts)
        labels.extend([i] * n_per_cluster)
    return np.vstack(embeddings), centroids, np.array(labels, dtype=np.int32)


def noisy_copy(
    embeddings: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.10,
    dropout_prob: float = 0.05,
) -> np.ndarray:
    """
    Degrade embeddings with Gaussian noise + random dimension dropout.
    Simulates noisy / partial queries (tested in Kanerva Machine paper).
    """
    out = embeddings.copy()
    out += rng.standard_normal(out.shape).astype(np.float32) * noise_std
    mask = rng.random(out.shape) > dropout_prob
    out *= mask.astype(np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic domain dataset (BEIR-style)
# ---------------------------------------------------------------------------

DOMAIN_NAMES = [
    "biomedical", "legal", "finance", "software_engineering",
    "mathematics", "history", "chemistry", "linguistics",
]


@dataclass
class QueryDocPair:
    query_id: str
    query_emb: np.ndarray      # shape (D,)
    doc_id: str
    doc_emb: np.ndarray        # shape (D,)
    domain: str
    relevance: int             # 0 = irrelevant, 1 = relevant, 2 = highly relevant


@dataclass
class RetrievalDataset:
    """
    Minimal BEIR-style retrieval dataset for benchmarking.
    Each query has exactly one 'ground-truth' document cluster.
    """
    name: str
    queries: Dict[str, np.ndarray]        # query_id → embedding
    corpus:  Dict[str, np.ndarray]        # doc_id   → embedding
    qrels:   Dict[str, Dict[str, int]]    # query_id → {doc_id: relevance}
    query_domains: Dict[str, str]         # query_id → domain name


def make_retrieval_dataset(
    rng: np.random.Generator,
    n_queries: int = 200,
    n_docs_per_query: int = 50,   # pool of docs per query (1 relevant + rest distractors)
    dim: int = 768,
    n_domains: int = 4,
    intra_cluster_noise: float = 0.12,
    name: str = "synthetic_retrieval",
) -> RetrievalDataset:
    """
    Build a synthetic retrieval benchmark.

    Structure mirrors BEIR: each query has a relevance-judged corpus subset.
    Relevant docs sit close to the query (same concept cluster).
    Distractors are randomly sampled from other clusters.
    """
    domains = DOMAIN_NAMES[:n_domains]

    # Build concept clusters — one per query
    query_centroids = unit_sphere(rng, n_queries, dim)

    queries: Dict[str, np.ndarray] = {}
    corpus:  Dict[str, np.ndarray] = {}
    qrels:   Dict[str, Dict[str, int]] = {}
    query_domains: Dict[str, str] = {}

    all_distractor_embs = unit_sphere(rng, n_queries * n_docs_per_query, dim)
    distractor_idx = 0

    for qi in range(n_queries):
        qid = f"q_{qi:04d}"
        domain = domains[qi % n_domains]
        # Query embedding = centroid + small noise
        q_noise = rng.standard_normal(dim).astype(np.float32) * intra_cluster_noise
        q_emb = query_centroids[qi] + q_noise
        q_emb /= np.linalg.norm(q_emb)

        queries[qid] = q_emb
        query_domains[qid] = domain
        qrels[qid] = {}

        # 1 highly relevant doc
        for rank in range(2):
            did = f"d_{qi:04d}_r{rank}"
            noise_scale = intra_cluster_noise * (0.5 + rank * 0.5)
            d_noise = rng.standard_normal(dim).astype(np.float32) * noise_scale
            d_emb = query_centroids[qi] + d_noise
            d_emb /= np.linalg.norm(d_emb)
            corpus[did] = d_emb
            qrels[qid][did] = 2 - rank  # rank0=2, rank1=1

        # n_docs_per_query-2 distractors
        for j in range(n_docs_per_query - 2):
            did = f"d_{qi:04d}_dist{j}"
            corpus[did] = all_distractor_embs[distractor_idx % len(all_distractor_embs)]
            distractor_idx += 1
            qrels[qid][did] = 0

    return RetrievalDataset(
        name=name,
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        query_domains=query_domains,
    )


# ---------------------------------------------------------------------------
# Continual learning stream
# ---------------------------------------------------------------------------

@dataclass
class DomainBatch:
    """A batch of concept clusters belonging to one domain, arriving sequentially."""
    domain: str
    clusters: List[np.ndarray]          # list of centroids
    cluster_queries: List[List[np.ndarray]]  # queries per cluster for testing
    task_id: int


def make_continual_stream(
    rng: np.random.Generator,
    n_domains: int = 6,
    clusters_per_domain: int = 8,
    queries_per_cluster: int = 10,
    dim: int = 768,
) -> List[DomainBatch]:
    """
    Generate sequential domain batches for continual learning evaluation.

    Inspired by how PMoE/MoLE tested task sequences — each domain arrives
    in turn, and we measure backward transfer (forgetting of earlier domains).
    """
    domains = DOMAIN_NAMES[:n_domains]
    batches: List[DomainBatch] = []
    for tid, domain in enumerate(domains):
        centroids = unit_sphere(rng, clusters_per_domain, dim)
        cluster_queries = []
        for c in centroids:
            qs = []
            for _ in range(queries_per_cluster):
                noise = rng.standard_normal(dim).astype(np.float32) * 0.10
                q = c + noise
                q /= np.linalg.norm(q)
                qs.append(q)
            cluster_queries.append(qs)
        batches.append(DomainBatch(
            domain=domain,
            clusters=list(centroids),
            cluster_queries=cluster_queries,
            task_id=tid,
        ))
    return batches


# ---------------------------------------------------------------------------
# STS-style similarity dataset (for composite embedding quality)
# ---------------------------------------------------------------------------

@dataclass
class STSPair:
    """Semantic Textual Similarity pair (no text — just embeddings + score)."""
    emb_a: np.ndarray   # shape (D,)
    emb_b: np.ndarray   # shape (D,)
    similarity: float   # 0.0 – 1.0


def make_sts_dataset(
    rng: np.random.Generator,
    n_pairs: int = 500,
    dim: int = 768,
) -> List[STSPair]:
    """
    Generate STS-style pairs where ground-truth similarity is the actual
    cosine similarity between the two embeddings (so Spearman correlation
    between predicted cosine and ground-truth is computable and high).

    Pairs span the full similarity range [0, 1] by varying how much noise
    is added to the centroid — the similarity label IS the cosine.
    """
    pairs: List[STSPair] = []
    n_centroids = n_pairs

    centroids = unit_sphere(rng, n_centroids, dim)
    for c in centroids:
        # Choose a random noise level, producing a wide range of similarities
        noise_scale = float(rng.uniform(0.01, 1.5))
        n1 = rng.standard_normal(dim).astype(np.float32) * noise_scale * 0.3
        n2 = rng.standard_normal(dim).astype(np.float32) * noise_scale * 0.3
        emb_a = c + n1; emb_a /= np.linalg.norm(emb_a)
        emb_b = c + n2; emb_b /= np.linalg.norm(emb_b)
        # Ground-truth label = actual cosine (what a perfect model would predict)
        cos = float(np.dot(emb_a, emb_b))
        similarity = (cos + 1.0) / 2.0  # map to [0, 1]
        pairs.append(STSPair(emb_a, emb_b, similarity=similarity))

    rng.shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Minimal DKES stub — used by tests that need a mockable interface
# ---------------------------------------------------------------------------

class MockKDM:
    """
    Minimal KDM mock for unit tests that don't want to spin up full DKES.
    Stores concept embeddings in a dict and does exact cosine lookup.
    """
    def __init__(self, dim: int = 768, capacity: int = 100):
        self.dim = dim
        self.capacity = capacity
        self._slots: Dict[str, np.ndarray] = {}  # concept_id → address vector
        self._values: Dict[str, np.ndarray] = {}  # concept_id → value vector

    def write(self, concept_id: str, address: np.ndarray, value: np.ndarray):
        if len(self._slots) >= self.capacity:
            oldest = next(iter(self._slots))
            del self._slots[oldest]
            del self._values[oldest]
        self._slots[concept_id] = address.copy()
        self._values[concept_id] = value.copy()

    def read(self, query: np.ndarray, beta: float = 10.0) -> np.ndarray:
        if not self._slots:
            return np.zeros(self.dim, dtype=np.float32)
        addresses = np.stack(list(self._slots.values()))
        values    = np.stack(list(self._values.values()))
        cosines = (addresses @ query).astype(np.float32)
        weights = _softmax(cosines * beta)
        return (weights[:, None] * values).sum(axis=0)

    def slot_count(self) -> int:
        return len(self._slots)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
