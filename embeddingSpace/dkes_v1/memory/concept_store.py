"""
ConceptStore — accumulates query clusters and triggers KDM writes.

Tracks incoming query embeddings, groups them into clusters, and
fires a write trigger when a cluster meets the novelty + coherence
+ count criteria from Section 1.3 of the spec.

This is the component that bridges the routing system (which sees
every query) and the KDM (which receives finalised write requests).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dkes.utils.config import DKESConfig
from dkes.utils.types import ConceptCluster, EmbeddingVector, WriteSource

logger = logging.getLogger(__name__)


class ConceptStore:
    """
    Accumulates query embeddings into clusters and triggers KDM writes.

    A lightweight online clustering algorithm groups new query embeddings
    into provisional clusters. When a cluster meets all three write
    criteria (count, novelty, coherence), the on_write_ready callback
    is fired with a ConceptCluster object.

    Parameters
    ----------
    cfg:             DKESConfig
    on_write_ready:  callable(ConceptCluster, trace_id) triggered when
                     a cluster is eligible for KDM write
    get_max_cos_to_kdm:  callable(centroid) → float, returns the max
                     cosine similarity of a centroid to any existing
                     KDM address (novelty check)
    """

    def __init__(
        self,
        cfg: DKESConfig,
        on_write_ready: Callable[[ConceptCluster, Optional[str]], None],
        get_max_cos_to_kdm: Callable[[EmbeddingVector], float],
    ):
        self._cfg = cfg
        self._kdm_cfg = cfg.kdm
        self._on_write_ready = on_write_ready
        self._get_max_cos = get_max_cos_to_kdm
        self._lock = threading.Lock()

        # cluster_id → list of (embedding, u_blob, expert_id, timestamp)
        self._clusters: Dict[str, list] = defaultdict(list)
        # cluster_id → current centroid (running mean)
        self._centroids: Dict[str, np.ndarray] = {}
        # cluster_id → creation time
        self._cluster_created: Dict[str, float] = {}

        # Intra-cluster assignment threshold
        self._assignment_threshold: float = 1 - cfg.kdm.write_coherence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        embedding: EmbeddingVector,
        u_blob: float,
        expert_id: str,
        trace_id: Optional[str] = None,
    ):
        """
        Register a query observation.

        The embedding is assigned to the nearest existing provisional
        cluster (if within assignment threshold) or starts a new cluster.
        If the assigned cluster now satisfies all write criteria, the
        on_write_ready callback is fired asynchronously.
        """
        with self._lock:
            cluster_id = self._assign_or_create(embedding)
            self._clusters[cluster_id].append({
                "embedding": embedding.copy(),
                "u_blob": u_blob,
                "expert_id": expert_id,
                "timestamp": time.time(),
            })
            # Update running centroid
            n = len(self._clusters[cluster_id])
            old_c = self._centroids.get(cluster_id, embedding.copy())
            new_c = old_c + (embedding - old_c) / n
            self._centroids[cluster_id] = new_c

            # Check write eligibility
            cluster = self._build_cluster(cluster_id)

        if cluster is not None:
            self._check_and_fire(cluster, trace_id)

    def get_cluster_stats(self) -> List[dict]:
        """Return a snapshot of all provisional clusters for inspection."""
        with self._lock:
            stats = []
            for cid, points in self._clusters.items():
                stats.append({
                    "cluster_id": cid,
                    "n_queries": len(points),
                    "centroid_norm": float(np.linalg.norm(
                        self._centroids.get(cid, np.zeros(1))
                    )),
                    "created_at": self._cluster_created.get(cid, 0.0),
                    "age_hours": (time.time() - self._cluster_created.get(cid, time.time())) / 3600,
                })
            return stats

    def prune_old_clusters(self, max_age_days: float = 30.0):
        """Remove clusters that have been inactive for max_age_days."""
        cutoff = time.time() - max_age_days * 86400
        with self._lock:
            stale = [
                cid for cid, t in self._cluster_created.items()
                if t < cutoff and len(self._clusters[cid]) < self._kdm_cfg.write_min_queries
            ]
            for cid in stale:
                del self._clusters[cid]
                self._centroids.pop(cid, None)
                self._cluster_created.pop(cid, None)
        if stale:
            logger.info("ConceptStore: pruned %d stale clusters", len(stale))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assign_or_create(self, embedding: EmbeddingVector) -> str:
        """
        Assign embedding to nearest cluster within threshold, or create new.
        Must be called under self._lock.
        """
        if not self._centroids:
            return self._new_cluster()

        e_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        best_cid = None
        best_dist = float("inf")

        for cid, centroid in self._centroids.items():
            c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            cos_dist = 1.0 - float(e_norm @ c_norm)
            if cos_dist < best_dist:
                best_dist = cos_dist
                best_cid = cid

        if best_dist <= self._assignment_threshold:
            return best_cid
        return self._new_cluster()

    def _new_cluster(self) -> str:
        cid = str(uuid.uuid4())[:12]
        self._cluster_created[cid] = time.time()
        return cid

    def _build_cluster(self, cluster_id: str) -> Optional[ConceptCluster]:
        """
        Compute cluster statistics. Returns None if count < write_min_queries.
        Must be called under self._lock.
        """
        points = self._clusters[cluster_id]
        n = len(points)
        if n < self._kdm_cfg.write_min_queries:
            return None

        centroid = self._centroids[cluster_id]
        embeddings = np.stack([p["embedding"] for p in points])

        # Coherence: mean pairwise cosine similarity (sampled for speed)
        sample_size = min(n, 100)
        sample_idx = np.random.choice(n, sample_size, replace=False)
        sample = embeddings[sample_idx]
        norms = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-8)
        cos_matrix = norms @ norms.T
        coherence = float(cos_matrix[np.triu_indices(sample_size, k=1)].mean())

        # Mean BLoB uncertainty
        mean_u = float(np.mean([p["u_blob"] for p in points]))

        # Expert routing distribution
        expert_dist: Dict[str, float] = defaultdict(float)
        for p in points:
            expert_dist[p["expert_id"]] += 1.0 / n

        return ConceptCluster(
            cluster_id=cluster_id,
            centroid=centroid.astype(np.float32),
            n_queries=n,
            coherence=coherence,
            max_novelty=0.0,  # filled in _check_and_fire
            mean_u_blobs=mean_u,
            expert_dist=dict(expert_dist),
        )

    def _check_and_fire(
        self,
        cluster: ConceptCluster,
        trace_id: Optional[str],
    ):
        """Evaluate all three write criteria and fire callback if met."""
        cfg = self._kdm_cfg

        # 1. Coherence check
        if cluster.coherence < cfg.write_coherence_threshold:
            return

        # 2. Novelty check (live query to KDM)
        max_cos = self._get_max_cos(cluster.centroid)
        novelty = 1.0 - max_cos
        if max_cos >= cfg.write_novelty_threshold:
            return

        # Rebuild cluster with filled novelty field
        cluster = ConceptCluster(
            cluster_id=cluster.cluster_id,
            centroid=cluster.centroid,
            n_queries=cluster.n_queries,
            coherence=cluster.coherence,
            max_novelty=novelty,
            mean_u_blobs=cluster.mean_u_blobs,
            expert_dist=cluster.expert_dist,
        )

        logger.info(
            "ConceptStore: write trigger for cluster=%s n=%d coherence=%.3f novelty=%.3f",
            cluster.cluster_id, cluster.n_queries, cluster.coherence, novelty,
        )

        # Remove cluster from accumulation (it's being written)
        with self._lock:
            self._clusters.pop(cluster.cluster_id, None)
            self._centroids.pop(cluster.cluster_id, None)
            self._cluster_created.pop(cluster.cluster_id, None)

        # Fire in the calling thread (KDM write is fast; async if needed)
        self._on_write_ready(cluster, trace_id)