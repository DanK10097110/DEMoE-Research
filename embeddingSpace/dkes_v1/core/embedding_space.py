"""
DKESEmbeddingSpace — top-level orchestrator for the DKES.

This is the primary interface for the Dynamic Kanerva-Enhanced Embedding
Space used in the DEMoE architecture. It wires together:

  1. MRLBackbone          — frozen MRL encoder (no gradient updates)
  2. KanervaMemory        — associative concept memory (read/write/evict)
  3. CompositeEmbeddingLayer — q̃(q) = LayerNorm(e(q) + γ·r(q))
  4. FAISSIndexManager    — coarse+fine MRL routing funnel + KDM address index
  5. MetaConceptLayer     — M1-M4 meta-concept sublayer
  6. MetricsAggregator    — rolling statistics for health monitoring
  7. AuditLog             — structured audit record stream
  8. HealthMonitor        — background daemon health checks

Public interface
----------------
  route(query)              → RoutingResult
  record_query(query, ...)  → accumulate into concept clusters
  flush_pending_writes()    → commit mature clusters to KDM
  register_expert(...)      → add expert to the routing index
  metrics()                 → live MetricsAggregator snapshot
  save_checkpoint(path)     → serialise full DKES state
  load_checkpoint(path)     → restore from checkpoint

All public methods are thread-safe and designed for concurrent
inference traffic without blocking on memory writes.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dkes.audit.audit_log import AuditLog
from dkes.audit.metrics import MetricsAggregator
from dkes.audit.health_monitor import HealthMonitor
from dkes.core.backbone import MRLBackbone
from dkes.core.composite import CompositeEmbeddingLayer
from dkes.memory.kdm import KanervaMemory
from dkes.memory.meta_concepts import MetaConceptLayer
from dkes.routing.faiss_index import FAISSIndexManager
from dkes.utils.config import DKESConfig
from dkes.utils.types import (
    AuditEventKind,
    CompositeEmbedding,
    ConceptCluster,
    EmbeddingVector,
    RoutingResult,
    WriteSource,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ConceptAccumulator — query buffer that feeds into KDM writes
# ---------------------------------------------------------------------------

class _QueryBuffer:
    """
    Accumulates per-cluster query embeddings until write threshold.

    Each incoming query is soft-assigned to its nearest cluster centroid.
    A new cluster is created when a query is sufficiently novel (cosine
    distance > write_novelty_threshold to all existing centroids).

    This class is internal to DKESEmbeddingSpace and not part of the
    public interface.
    """

    def __init__(self, cfg: DKESConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        # cluster_id → {"centroid", "embeddings", "u_blobs", "expert_dists", "created_at"}
        self._clusters: Dict[str, Dict[str, Any]] = {}

    def observe(
        self,
        embedding: EmbeddingVector,
        u_base: float,
        expert_ids: List[str],
        expert_scores: List[float],
    ) -> Optional[ConceptCluster]:
        """
        Record a single query observation.

        Returns a mature ConceptCluster ready for KDM write if the
        write_min_queries threshold has been crossed, else None.
        """
        cluster_id = self._assign_or_create(embedding)
        with self._lock:
            buf = self._clusters[cluster_id]
            buf["embeddings"].append(embedding.copy())
            buf["u_blobs"].append(u_base)
            # Accumulate routing distribution
            for eid, score in zip(expert_ids, expert_scores):
                buf["expert_dist"][eid] = buf["expert_dist"].get(eid, 0.0) + score

            n = len(buf["embeddings"])
            if n < self._cfg.kdm.write_min_queries:
                return None

            # Compute cluster statistics
            embs = np.stack(buf["embeddings"])
            centroid = embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroid = (centroid / norm).astype(np.float32) if norm > 0 else centroid

            # Intra-cluster coherence: mean pairwise cosine similarity
            normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
            gram = normed @ normed.T
            # Exclude diagonal (self-similarity = 1.0)
            mask = ~np.eye(len(embs), dtype=bool)
            coherence = float(gram[mask].mean()) if mask.any() else 1.0

            if coherence < self._cfg.kdm.write_coherence_threshold:
                # Cluster too incoherent — reset and wait for a tighter cluster
                logger.debug(
                    "QueryBuffer: cluster %s coherence=%.3f below threshold %.3f — reset",
                    cluster_id, coherence, self._cfg.kdm.write_coherence_threshold,
                )
                buf["embeddings"] = []
                buf["u_blobs"] = []
                buf["expert_dist"] = {}
                return None

            # Normalise expert distribution
            total = sum(buf["expert_dist"].values()) or 1.0
            expert_dist = {k: v / total for k, v in buf["expert_dist"].items()}

            # Max novelty: 1 - max cosine similarity of centroid to all other cluster centroids
            other_centroids = [
                v["centroid"] for cid, v in self._clusters.items() if cid != cluster_id
            ]
            if other_centroids:
                others = np.stack(other_centroids)
                o_norms = others / (np.linalg.norm(others, axis=1, keepdims=True) + 1e-8)
                c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                max_novelty = float(1.0 - (o_norms @ c_norm).max())
            else:
                max_novelty = 1.0

            mature_cluster = ConceptCluster(
                cluster_id=cluster_id,
                centroid=centroid,
                n_queries=n,
                coherence=coherence,
                max_novelty=max_novelty,
                mean_u_blobs=float(np.mean(buf["u_blobs"])),
                expert_dist=expert_dist,
            )

            # Update stored centroid with refined estimate
            buf["centroid"] = centroid.copy()
            # Reset for next epoch
            buf["embeddings"] = []
            buf["u_blobs"] = []
            buf["expert_dist"] = {}

        return mature_cluster

    def _assign_or_create(self, embedding: EmbeddingVector) -> str:
        """Assign embedding to nearest cluster or spawn a new one."""
        with self._lock:
            if not self._clusters:
                return self._new_cluster(embedding)

            e_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            best_id, best_sim = None, -1.0
            for cid, buf in self._clusters.items():
                c = buf["centroid"]
                c_norm = c / (np.linalg.norm(c) + 1e-8)
                sim = float(e_norm @ c_norm)
                if sim > best_sim:
                    best_sim = sim
                    best_id = cid

            # cos distance > novelty threshold → new cluster
            if (1.0 - best_sim) > (1.0 - self._cfg.kdm.write_novelty_threshold):
                return self._new_cluster(embedding)
            return best_id

    def _new_cluster(self, embedding: EmbeddingVector) -> str:
        cid = str(uuid.uuid4())[:12]
        self._clusters[cid] = {
            "centroid": embedding.copy(),
            "embeddings": [],
            "u_blobs": [],
            "expert_dist": {},
            "created_at": time.time(),
        }
        return cid

    def pending_count(self) -> int:
        with self._lock:
            return sum(len(v["embeddings"]) for v in self._clusters.values())

    def n_clusters(self) -> int:
        with self._lock:
            return len(self._clusters)


# ---------------------------------------------------------------------------
# Fast-path routing cache
# ---------------------------------------------------------------------------

class _RoutingCache:
    """
    LRU-style cache mapping composite embedding → RoutingResult.

    Uses cosine-distance < cache_epsilon to determine a cache hit.
    Thread-safe. Bounded to cfg.routing.cache_size entries.
    """

    def __init__(self, cfg: DKESConfig):
        self._epsilon = cfg.routing.cache_epsilon
        self._max_size = cfg.routing.cache_size
        self._lock = threading.Lock()
        # Ordered list of (embedding, RoutingResult) — most recent last
        self._entries: List[Tuple[np.ndarray, RoutingResult]] = []
        self._hits = 0
        self._misses = 0

    def lookup(self, composite: EmbeddingVector) -> Optional[RoutingResult]:
        q = composite / (np.linalg.norm(composite) + 1e-8)
        with self._lock:
            for emb, result in reversed(self._entries):
                cos_dist = 1.0 - float(q @ (emb / (np.linalg.norm(emb) + 1e-8)))
                if cos_dist < self._epsilon:
                    self._hits += 1
                    return result
            self._misses += 1
        return None

    def insert(self, composite: EmbeddingVector, result: RoutingResult):
        with self._lock:
            if len(self._entries) >= self._max_size:
                self._entries.pop(0)
            self._entries.append((composite.copy(), result))

    def invalidate(self):
        """Flush cache when expert index is rebuilt."""
        with self._lock:
            self._entries.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)


# ---------------------------------------------------------------------------
# DKESEmbeddingSpace — the public orchestrator
# ---------------------------------------------------------------------------

class DKESEmbeddingSpace:
    """
    Dynamic Kanerva-Enhanced Embedding Space for the DEMoE architecture.

    This class is the single entry point for all embedding space operations.
    It wires together the backbone encoder, associative memory, composite
    embedding layer, FAISS routing indices, meta-concept sublayer, audit
    logging, and health monitoring into one coherent, thread-safe interface.

    Parameters
    ----------
    cfg : DKESConfig
        Master configuration (all defaults are architecture-spec compliant).
    audit : AuditLog, optional
        External audit log instance. A new one is created if not provided.

    Thread Safety
    -------------
    All public methods acquire only the locks they need and are safe to call
    concurrently from multiple inference threads. Write operations are
    asynchronous by default (query accumulation) and only briefly block
    during the actual slot write.

    Lifecycle
    ---------
    1. Construct:   space = DKESEmbeddingSpace(cfg)
    2. Register:    space.register_expert(expert_id, coarse_emb, fine_emb)
    3. Route:       result = space.route(query_text)
    4. Feedback:    space.record_routing_feedback(result, quality_signal)
    5. Checkpoint:  space.save_checkpoint(path)
    """

    def __init__(
        self,
        cfg: DKESConfig,
        audit: Optional[AuditLog] = None,
    ):
        self._cfg = cfg
        self._init_time = time.time()

        # --- Audit infrastructure (must be first) ---
        self._audit = audit or AuditLog(cfg.audit)
        self._metrics = MetricsAggregator(
            window_hours=cfg.audit.metrics_window_hours,
            kdm_total_slots=cfg.kdm.total_slots,
        )

        # --- Core components ---
        self._backbone = MRLBackbone(cfg)
        self._kdm = KanervaMemory(cfg, audit=self._audit)
        self._composite = CompositeEmbeddingLayer(cfg, audit=self._audit)
        self._faiss = FAISSIndexManager(cfg, audit=self._audit)
        self._meta = MetaConceptLayer(cfg, self._kdm, audit=self._audit)

        # --- Routing cache ---
        self._cache = _RoutingCache(cfg)

        # --- Query accumulation buffer ---
        self._query_buffer = _QueryBuffer(cfg)

        # --- Write queue (concept clusters awaiting KDM commit) ---
        self._write_queue: List[ConceptCluster] = []
        self._write_lock = threading.Lock()

        # --- Health monitor ---
        self._health = HealthMonitor(
            audit=self._audit,
            metrics=self._metrics,
            cfg=cfg.audit,
            get_kdm_meta_free_slots=self._meta.n_free_slots,
            get_gamma=lambda: self._composite.gamma,
            get_gamma_ceiling_active=lambda: self._composite.gamma_ceiling_active(
                self._kdm.n_populated
            ),
        )
        self._health.start()

        # --- Background write flusher ---
        self._flush_stop = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._background_flush_loop,
            daemon=True,
            name="dkes-write-flush",
        )
        self._flush_thread.start()

        # --- Operational counters ---
        self._total_queries: int = 0
        self._total_cache_hits: int = 0
        self._total_meta_shortcuts: int = 0
        self._query_counter_lock = threading.Lock()

        self._audit.emit(
            AuditEventKind.CHECKPOINT_LOAD,
            "embedding_space",
            {
                "event": "init",
                "backbone": cfg.backbone.model_name,
                "total_slots": cfg.kdm.total_slots,
                "meta_slots": cfg.kdm.meta_slots,
                "gamma_init": cfg.kdm.gamma_init,
                "coarse_dim": cfg.backbone.coarse_dim,
                "fine_dim": cfg.backbone.fine_dim,
            },
        )
        logger.info(
            "DKESEmbeddingSpace initialised: backbone=%s slots=%d gamma=%.3f",
            cfg.backbone.model_name, cfg.kdm.total_slots, cfg.kdm.gamma_init,
        )

    # ------------------------------------------------------------------
    # Expert registration
    # ------------------------------------------------------------------

    def register_expert(
        self,
        expert_id: str,
        description: str,
        coarse_embedding: Optional[EmbeddingVector] = None,
        fine_embedding: Optional[EmbeddingVector] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Register a new expert in the FAISS routing index.

        If embeddings are not provided, the description text is encoded
        by the backbone to produce both coarse and fine centroids.

        Parameters
        ----------
        expert_id:        unique identifier for this expert
        description:      human-readable text describing the expert's domain
        coarse_embedding: optional pre-computed 64-dim centroid
        fine_embedding:   optional pre-computed 768-dim centroid
        trace_id:         optional trace context for audit records
        """
        tid = trace_id or str(uuid.uuid4())[:8]

        if coarse_embedding is None:
            coarse_embedding = self._backbone.encode_coarse(description)[0]
        if fine_embedding is None:
            fine_embedding = self._backbone.encode_fine(description)[0]

        self._faiss.register_expert(expert_id, coarse_embedding, fine_embedding)

        # Invalidate routing cache — expert index has changed
        self._cache.invalidate()

        self._audit.emit(
            AuditEventKind.INDEX_REBUILD,
            "embedding_space",
            {
                "event": "expert_registered",
                "expert_id": expert_id,
                "n_experts_now": self._faiss.n_experts,
                "trace_id": tid,
            },
        )
        logger.info(
            "Expert registered: %s (total experts=%d)",
            expert_id, self._faiss.n_experts,
        )

    # ------------------------------------------------------------------
    # Primary routing interface
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        trace_id: Optional[str] = None,
        force_full_pipeline: bool = False,
    ) -> RoutingResult:
        """
        Route a query text through the full DKES pipeline.

        Pipeline stages:
          1.  Backbone encode (coarse + fine)
          2.  KDM read → composite embedding q̃(q)
          3.  M2 template check (fast-path shortcut)
          4.  Routing cache lookup
          5.  FAISS Stage 1: coarse k-NN
          6.  FAISS Stage 2: fine re-rank
          7.  M3 uncertainty zone check
          8.  Expert budget capping to M_active
          9.  Metrics + audit emission

        Parameters
        ----------
        query:              raw query string
        trace_id:           optional trace context
        force_full_pipeline: bypass cache even on a cache hit

        Returns
        -------
        RoutingResult
        """
        t_start = time.perf_counter()
        tid = trace_id or str(uuid.uuid4())[:8]

        # ---- Stage 1: Backbone encoding ----
        coarse_emb = self._backbone.encode_coarse(query)[0]
        fine_emb   = self._backbone.encode_fine(query)[0]

        # ---- Stage 2: KDM memory read ----
        memory_read, top_slot_ids, top_slot_weights = self._kdm.read(fine_emb)
        n_populated = self._kdm.n_populated

        # ---- Stage 3: Composite embedding ----
        effective_gamma = self._composite.get_effective_gamma(n_populated)
        comp_result: CompositeEmbedding = self._composite.compute(
            backbone_emb=fine_emb,
            memory_read=memory_read,
            top_slot_indices=top_slot_ids,
            top_slot_weights=top_slot_weights,
            query_id=tid,
            trace_id=tid,
        )

        # Update KDM metrics
        top_weight = top_slot_weights[0] if top_slot_weights else 0.0
        self._metrics.record_kdm_read(top_weight=top_weight, gamma=effective_gamma)
        self._metrics.set_populated_slots(n_populated)

        # ---- Stage 4: M2 template (fast-path) ----
        used_meta_shortcut = False
        m2_result = self._meta.match_query_pattern(comp_result.composite)
        if m2_result and not force_full_pipeline:
            used_meta_shortcut = True
            result = self._build_result_from_m2(
                m2_result=m2_result,
                comp_result=comp_result,
                coarse_emb=coarse_emb,
                t_start=t_start,
                tid=tid,
            )
            self._finalize_route(result, t_start, tid)
            with self._query_counter_lock:
                self._total_queries += 1
                self._total_meta_shortcuts += 1
            return result

        # ---- Stage 5: Routing cache lookup ----
        composite_vec = comp_result.composite
        if not force_full_pipeline:
            cached = self._cache.lookup(composite_vec)
            if cached is not None:
                with self._query_counter_lock:
                    self._total_queries += 1
                    self._total_cache_hits += 1
                self._audit.emit(
                    AuditEventKind.ROUTING_CACHE_HIT,
                    "embedding_space",
                    {
                        "trace_id": tid,
                        "expert_ids": cached.expert_ids,
                        "cache_size": self._cache.size,
                        "cache_hit_rate": round(self._cache.hit_rate, 4),
                    },
                )
                return cached

        # ---- Stage 6: FAISS coarse search ----
        # Expand k_coarse if query is in M3 uncertainty zone
        m3_markers = self._meta.get_uncertainty_markers(comp_result.composite)
        in_uncertainty_region = len(m3_markers) > 0
        k_coarse = (
            self._cfg.routing.k_coarse_expanded
            if in_uncertainty_region
            else self._cfg.routing.k_coarse
        )
        coarse_candidates = self._faiss.coarse_search(coarse_emb, k=k_coarse)

        if not coarse_candidates:
            logger.warning("DKES routing: no coarse candidates for query (trace=%s)", tid)
            result = self._empty_routing_result(comp_result, in_uncertainty_region)
            self._finalize_route(result, t_start, tid)
            return result

        # ---- Stage 7: FAISS fine re-rank ----
        fine_ranked = self._faiss.fine_rerank(comp_result.composite, coarse_candidates)

        # Take top-k_fine per... (full list, not per domain — domain labelling
        # happens inside the DEMoE adapter layer above this module)
        k_fine = self._cfg.routing.k_fine
        fine_candidates = [eid for eid, _ in fine_ranked[:k_fine * 2]]

        # ---- Stage 8: Score-based filtering & M_active budget cap ----
        threshold = self._cfg.routing.default_routing_threshold
        selected: List[Tuple[str, float]] = []
        for eid, score in fine_ranked:
            if score >= threshold:
                selected.append((eid, score))
            if len(selected) >= self._cfg.routing.max_active_experts:
                break

        # Deadlock guard: if nothing clears threshold, take best M_active
        if not selected:
            margin = self._cfg.routing.routing_deadlock_margin
            if fine_ranked and fine_ranked[0][1] >= margin:
                selected = fine_ranked[: self._cfg.routing.max_active_experts]
            else:
                logger.warning(
                    "DKES routing deadlock: best score=%.3f below margin=%.3f (trace=%s)",
                    fine_ranked[0][1] if fine_ranked else 0.0, margin, tid,
                )

        expert_ids    = [eid for eid, _ in selected]
        u_base_scores = [score for _, score in selected]
        # adapter_ids resolved by caller (domain projection layer above DKES)
        adapter_ids   = [None] * len(expert_ids)

        below_threshold = bool(selected and u_base_scores[0] >= threshold)

        result = RoutingResult(
            expert_ids=expert_ids,
            adapter_ids=adapter_ids,
            u_base_scores=u_base_scores,
            coarse_candidates=coarse_candidates,
            fine_candidates=fine_candidates,
            used_meta_shortcut=False,
            uncertainty_region=in_uncertainty_region,
            composite_emb=comp_result,
        )

        # ---- Cache and finalise ----
        self._cache.insert(composite_vec, result)
        self._finalize_route(result, t_start, tid, below_threshold=below_threshold)

        with self._query_counter_lock:
            self._total_queries += 1

        return result

    def _build_result_from_m2(
        self,
        m2_result: dict,
        comp_result: CompositeEmbedding,
        coarse_emb: EmbeddingVector,
        t_start: float,
        tid: str,
    ) -> RoutingResult:
        """Construct a RoutingResult from a cached M2 template match."""
        expert_ids = m2_result.get("expert_ids", [])
        u_scores   = m2_result.get("u_base_scores", [1.0] * len(expert_ids))

        self._audit.emit(
            AuditEventKind.ROUTING_QUERY,
            "embedding_space",
            {
                "trace_id": tid,
                "shortcut": "M2",
                "pattern_key": m2_result.get("pattern_key"),
                "confidence": m2_result.get("confidence"),
                "expert_ids": expert_ids,
                "slot_id": m2_result.get("slot_id"),
            },
        )
        return RoutingResult(
            expert_ids=expert_ids,
            adapter_ids=[None] * len(expert_ids),
            u_base_scores=u_scores,
            coarse_candidates=[],
            fine_candidates=[],
            used_meta_shortcut=True,
            uncertainty_region=False,
            composite_emb=comp_result,
        )

    def _empty_routing_result(
        self,
        comp_result: CompositeEmbedding,
        uncertainty_region: bool,
    ) -> RoutingResult:
        return RoutingResult(
            expert_ids=[],
            adapter_ids=[],
            u_base_scores=[],
            coarse_candidates=[],
            fine_candidates=[],
            used_meta_shortcut=False,
            uncertainty_region=uncertainty_region,
            composite_emb=comp_result,
        )

    def _finalize_route(
        self,
        result: RoutingResult,
        t_start: float,
        tid: str,
        below_threshold: bool = True,
    ):
        """Emit metrics and audit record after a routing decision."""
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        self._metrics.record_routing(
            latency_ms=latency_ms,
            u_base_scores=result.u_base_scores or [0.0],
            gamma=result.composite_emb.gamma,
            used_meta_shortcut=result.used_meta_shortcut,
            uncertainty_region=result.uncertainty_region,
            below_threshold=below_threshold,
        )

        self._audit.emit(
            AuditEventKind.ROUTING_QUERY,
            "embedding_space",
            {
                "trace_id": tid,
                "expert_ids": result.expert_ids,
                "u_base_scores": [round(u, 4) for u in result.u_base_scores],
                "latency_ms": round(latency_ms, 2),
                "coarse_candidates": len(result.coarse_candidates),
                "fine_candidates": len(result.fine_candidates),
                "used_meta_shortcut": result.used_meta_shortcut,
                "uncertainty_region": result.uncertainty_region,
                "gamma": round(result.composite_emb.gamma, 5),
                "top_kdm_weight": round(
                    result.composite_emb.top_slot_weights[0], 5
                ) if result.composite_emb.top_slot_weights else 0.0,
                "cache_hit_rate": round(self._cache.hit_rate, 4),
                "kdm_populated_slots": self._kdm.n_populated,
            },
        )

    # ------------------------------------------------------------------
    # Query accumulation and KDM write pipeline
    # ------------------------------------------------------------------

    def record_query(
        self,
        query: str,
        routing_result: RoutingResult,
        u_base_override: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Record a routed query into the concept accumulation buffer.

        Called after route() to feed the unsupervised concept-formation
        loop. When a cluster matures (≥ write_min_queries coherent
        observations), it is queued for KDM write.

        Parameters
        ----------
        query:            raw query string (re-encoded to get fine embedding)
        routing_result:   the RoutingResult returned by route()
        u_base_override:  optional uncertainty score override (uses mean
                          u_base_scores from routing if not supplied)
        trace_id:         optional trace context
        """
        tid = trace_id or str(uuid.uuid4())[:8]

        fine_emb = self._backbone.encode_fine(query)[0]
        u_base = u_base_override or (
            float(np.mean(routing_result.u_base_scores))
            if routing_result.u_base_scores else 0.5
        )

        mature_cluster = self._query_buffer.observe(
            embedding=fine_emb,
            u_base=u_base,
            expert_ids=routing_result.expert_ids,
            expert_scores=routing_result.u_base_scores,
        )

        if mature_cluster is not None:
            with self._write_lock:
                self._write_queue.append(mature_cluster)
            self._audit.emit(
                AuditEventKind.KDM_WRITE,
                "embedding_space",
                {
                    "event": "cluster_queued",
                    "cluster_id": mature_cluster.cluster_id,
                    "n_queries": mature_cluster.n_queries,
                    "coherence": round(mature_cluster.coherence, 4),
                    "max_novelty": round(mature_cluster.max_novelty, 4),
                    "mean_u_blobs": round(mature_cluster.mean_u_blobs, 4),
                    "pending_clusters_in_queue": len(self._write_queue),
                    "trace_id": tid,
                },
            )

        # Also offer the query to M2 pattern tracking
        # Pattern key: sorted tuple of top expert IDs (domain fingerprint)
        if routing_result.expert_ids:
            pattern_key = "|".join(sorted(routing_result.expert_ids[:2]))
            template = {
                "expert_ids": routing_result.expert_ids,
                "u_base_scores": routing_result.u_base_scores,
            }
            self._meta.observe_query_pattern(
                pattern_key=pattern_key,
                pattern_embedding=fine_emb,
                decomposition_template=template,
                trace_id=tid,
            )

    def flush_pending_writes(self, trace_id: Optional[str] = None) -> int:
        """
        Commit all queued concept clusters to the KDM.

        Called automatically by the background flush thread every
        checkpoint_interval_minutes, but can be triggered manually for
        testing or operator-initiated flushes.

        Returns
        -------
        int: number of clusters successfully written to KDM
        """
        tid = trace_id or str(uuid.uuid4())[:8]
        written = 0

        with self._write_lock:
            queue_snapshot = list(self._write_queue)
            self._write_queue.clear()

        for cluster in queue_snapshot:
            slot_idx = self._kdm.write_concept(
                cluster, source=WriteSource.CLUSTER_THRESHOLD, trace_id=tid
            )
            if slot_idx is not None:
                written += 1
                self._metrics.record_kdm_write()
                self._health.notify_kdm_write()

                # Rebuild KDM address sub-index after each write
                populated = self._kdm.populated_concept_slots()
                if populated:
                    addresses = self._kdm.addresses[populated]
                    self._faiss.rebuild_kdm_index(addresses, populated)
            else:
                self._metrics.record_eviction()

        if written > 0:
            self._metrics.set_populated_slots(self._kdm.n_populated)
            self._audit.emit(
                AuditEventKind.KDM_WRITE,
                "embedding_space",
                {
                    "event": "flush_complete",
                    "clusters_attempted": len(queue_snapshot),
                    "clusters_written": written,
                    "kdm_populated_slots": self._kdm.n_populated,
                    "trace_id": tid,
                },
            )
            logger.info(
                "KDM flush: %d/%d clusters written (total populated=%d)",
                written, len(queue_snapshot), self._kdm.n_populated,
            )

        return written

    def _background_flush_loop(self):
        """Daemon thread: periodically flush pending KDM writes."""
        interval = self._cfg.checkpoint_interval_minutes * 60
        while not self._flush_stop.wait(interval):
            try:
                written = self.flush_pending_writes(trace_id="bg_flush")
                if written > 0:
                    self.save_checkpoint(
                        self._cfg.checkpoint_dir / f"auto_{int(time.time())}.json"
                    )
            except Exception:
                logger.exception("Background flush/checkpoint failed")

    # ------------------------------------------------------------------
    # Routing quality feedback — gamma updates
    # ------------------------------------------------------------------

    def record_routing_feedback(
        self,
        routing_result: RoutingResult,
        quality_signal: float,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Provide a routing quality signal to update gamma.

        This closes the DKES learning loop: downstream task performance
        signals flow back to adjust the KDM interpolation weight.

        Parameters
        ----------
        routing_result: the RoutingResult to provide feedback on
        quality_signal: float in [-1, +1]
          +1 = KDM contribution improved routing
          -1 = KDM contribution hurt routing
          0  = neutral
        trace_id:       optional trace context
        """
        tid = trace_id or str(uuid.uuid4())[:8]
        self._composite.update_gamma(
            routing_quality_signal=quality_signal,
            n_populated_slots=self._kdm.n_populated,
            trace_id=tid,
        )

        self._audit.emit(
            AuditEventKind.GAMMA_UPDATE,
            "embedding_space",
            {
                "event": "feedback",
                "quality_signal": round(quality_signal, 4),
                "new_gamma": round(self._composite.gamma, 5),
                "expert_ids": routing_result.expert_ids,
                "trace_id": tid,
            },
        )

    # ------------------------------------------------------------------
    # Meta-concept write convenience methods
    # ------------------------------------------------------------------

    def inject_domain_proximity(
        self,
        domain_a: str,
        domain_b: str,
        overlap_score: float,
        synthesis_quality: float,
        trace_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Write an M1 domain proximity meta-concept.

        Computes the domain embeddings from their names and stores the
        overlap relationship in the reserved M1 meta slots.
        """
        emb_a = self._backbone.encode_fine(domain_a)[0]
        emb_b = self._backbone.encode_fine(domain_b)[0]
        return self._meta.write_domain_proximity(
            domain_a, domain_b, emb_a, emb_b,
            overlap_score, synthesis_quality, trace_id=trace_id,
        )

    def inject_analogy_bridge(
        self,
        expert_a: str,
        expert_b: str,
        shared_concept: str,
        equivalence_score: float,
        trace_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Write an M4 cross-domain analogy bridge meta-concept.
        """
        shared_emb = self._backbone.encode_fine(shared_concept)[0]
        return self._meta.write_analogy_bridge(
            expert_a, expert_b, shared_emb, equivalence_score, trace_id=trace_id,
        )

    def inject_uncertainty_marker(
        self,
        region_description: str,
        mean_uncertainty: float,
        n_uncertain_queries: int,
        trace_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Write an M3 uncertainty landscape marker meta-concept.
        """
        region_emb = self._backbone.encode_fine(region_description)[0]
        return self._meta.write_uncertainty_marker(
            region_centroid=region_emb,
            mean_u_base=mean_uncertainty,
            n_uncertain_queries=n_uncertain_queries,
            trace_id=trace_id,
        )

    # ------------------------------------------------------------------
    # Manual KDM concept injection
    # ------------------------------------------------------------------

    def inject_concept(
        self,
        concept_text: str,
        expert_ids: List[str],
        u_base: float = 0.5,
        trace_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Manually inject a concept into the KDM from operator-supplied text.

        Bypasses the cluster accumulation threshold — useful for seeding
        the memory with known-important concepts before traffic ramps up.

        Parameters
        ----------
        concept_text: text describing the concept
        expert_ids:   list of expert IDs this concept routes to
        u_base:       uncertainty score for this concept
        trace_id:     optional trace context
        """
        tid = trace_id or str(uuid.uuid4())[:8]
        embedding = self._backbone.encode_fine(concept_text)[0]
        expert_dist = {eid: 1.0 / len(expert_ids) for eid in expert_ids}

        cluster = ConceptCluster(
            cluster_id=f"manual:{uuid.uuid4().hex[:8]}",
            centroid=embedding,
            n_queries=1,
            coherence=1.0,
            max_novelty=1.0,
            mean_u_blobs=u_base,
            expert_dist=expert_dist,
        )

        slot_idx = self._kdm.write_concept(
            cluster, source=WriteSource.MANUAL_INJECTION, trace_id=tid
        )

        if slot_idx is not None:
            self._metrics.record_kdm_write()
            self._health.notify_kdm_write()
            self._metrics.set_populated_slots(self._kdm.n_populated)

            populated = self._kdm.populated_concept_slots()
            if populated:
                self._faiss.rebuild_kdm_index(
                    self._kdm.addresses[populated], populated
                )

            self._audit.emit(
                AuditEventKind.KDM_WRITE,
                "embedding_space",
                {
                    "event": "manual_inject",
                    "concept_text": concept_text[:80],
                    "slot_idx": slot_idx,
                    "expert_ids": expert_ids,
                    "trace_id": tid,
                },
            )

        return slot_idx

    # ------------------------------------------------------------------
    # Invariant checks
    # ------------------------------------------------------------------

    def assert_invariants(self) -> bool:
        """
        Run all DKES structural invariant checks.

        Checks:
          - Backbone parameters remain frozen (no grad updates)
          - Meta slot indices are within the reserved range
          - Gamma is within [0.01, 1.0]
          - FAISS index sizes are consistent with slot counts

        Returns True if all invariants hold; raises on violation.
        """
        self._backbone.assert_frozen()

        gamma = self._composite.gamma
        assert 0.01 <= gamma <= 1.0, f"Gamma {gamma} out of bounds [0.01, 1.0]"

        n_meta_free = self._meta.n_free_slots()
        assert 0 <= n_meta_free <= self._cfg.kdm.meta_slots, (
            f"Meta free slots {n_meta_free} inconsistent"
        )

        logger.debug("DKESEmbeddingSpace invariants: all OK")
        return True

    # ------------------------------------------------------------------
    # Metrics and observability
    # ------------------------------------------------------------------

    def metrics(self) -> Dict[str, Any]:
        """
        Return a live metrics snapshot.

        Includes routing latency, KDM hit rate, gamma trend, slot
        utilization, meta slot usage, query buffer state, and cache stats.
        """
        snap = self._metrics.snapshot()
        with self._query_counter_lock:
            total_q = self._total_queries
            cache_hits = self._total_cache_hits
            meta_shortcuts = self._total_meta_shortcuts

        snap["embedding_space"] = {
            "total_queries": total_q,
            "total_cache_hits": cache_hits,
            "total_meta_shortcuts": meta_shortcuts,
            "cache_hit_rate_lifetime": cache_hits / total_q if total_q > 0 else 0.0,
            "meta_shortcut_rate_lifetime": meta_shortcuts / total_q if total_q > 0 else 0.0,
            "routing_cache_size": self._cache.size,
            "routing_cache_epsilon": self._cfg.routing.cache_epsilon,
            "pending_write_queue": len(self._write_queue),
            "query_buffer_pending": self._query_buffer.pending_count(),
            "query_buffer_clusters": self._query_buffer.n_clusters(),
            "kdm_n_populated": self._kdm.n_populated,
            "kdm_n_free": self._kdm.n_free_concept_slots,
            "meta_free_slots": self._meta.n_free_slots(),
            "n_experts": self._faiss.n_experts,
            "gamma": self._composite.gamma,
            "gamma_ceiling_active": self._composite.gamma_ceiling_active(
                self._kdm.n_populated
            ),
            "uptime_hours": (time.time() - self._init_time) / 3600,
        }
        return snap

    def status(self) -> str:
        """Return a human-readable one-line status string."""
        m = self.metrics()
        es = m["embedding_space"]
        kdm = m["kdm"]
        rt = m["routing"]
        return (
            f"DKES | experts={es['n_experts']} | "
            f"slots={es['kdm_n_populated']}/{self._cfg.kdm.total_slots} "
            f"({kdm['slot_utilization']:.1%}) | "
            f"gamma={es['gamma']:.4f}"
            f"{'[ceil]' if es['gamma_ceiling_active'] else ''} | "
            f"p50_lat={rt['latency_ms'].get('mean', 0):.1f}ms | "
            f"cache_hit={es['cache_hit_rate_lifetime']:.1%} | "
            f"uptime={es['uptime_hours']:.1f}h"
        )

    # ------------------------------------------------------------------
    # Checkpoint serialisation
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Serialise the full mutable DKES state to a JSON checkpoint.

        Immutable parts (backbone weights) are not included — the same
        backbone model name in config is sufficient to reload them.

        Checkpoint structure:
          {
            "version": "1.0.0",
            "timestamp": <unix>,
            "config": <DKESConfig dict>,
            "kdm": <KanervaMemory state_dict>,
            "composite": <CompositeEmbeddingLayer state_dict>,
            "metrics": <MetricsAggregator snapshot>,
          }
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "version": "1.0.0",
            "timestamp": time.time(),
            "config": self._cfg.to_dict(),
            "kdm": self._kdm.state_dict(),
            "composite": self._composite.state_dict(),
            "metrics_snapshot": self._metrics.snapshot(),
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=_json_default)

        self._audit.emit(
            AuditEventKind.CHECKPOINT_SAVE,
            "embedding_space",
            {
                "path": str(path),
                "kdm_populated_slots": self._kdm.n_populated,
                "gamma": self._composite.gamma,
                "timestamp": checkpoint["timestamp"],
            },
        )
        logger.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Restore mutable DKES state from a JSON checkpoint.

        The config embedded in the checkpoint is used only for validation;
        the live config (passed at construction) takes precedence.
        Backbone weights are NOT loaded from the checkpoint.
        """
        path = Path(path)
        with open(path) as f:
            checkpoint = json.load(f)

        version = checkpoint.get("version", "unknown")
        logger.info("Loading checkpoint v%s from %s", version, path)

        self._kdm.load_state_dict(checkpoint["kdm"])
        self._composite.load_state_dict(checkpoint["composite"])

        # Rebuild KDM address index after restore
        populated = self._kdm.populated_concept_slots()
        if populated:
            self._faiss.rebuild_kdm_index(
                self._kdm.addresses[populated], populated
            )
        self._metrics.set_populated_slots(self._kdm.n_populated)

        # Invalidate routing cache (state has changed)
        self._cache.invalidate()

        self._audit.emit(
            AuditEventKind.CHECKPOINT_LOAD,
            "embedding_space",
            {
                "path": str(path),
                "checkpoint_version": version,
                "checkpoint_timestamp": checkpoint.get("timestamp"),
                "kdm_populated_slots": self._kdm.n_populated,
                "gamma_restored": self._composite.gamma,
            },
        )
        logger.info(
            "Checkpoint loaded: kdm_slots=%d gamma=%.4f",
            self._kdm.n_populated, self._composite.gamma,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, flush: bool = True) -> None:
        """
        Graceful shutdown: stop background threads and optionally flush.

        Parameters
        ----------
        flush: if True, commit pending write queue before stopping
        """
        logger.info("DKESEmbeddingSpace: initiating shutdown (flush=%s)", flush)
        self._flush_stop.set()
        self._health.stop()
        if flush:
            n = self.flush_pending_writes(trace_id="shutdown_flush")
            logger.info("Shutdown flush: %d clusters written", n)
        self._audit.emit(
            AuditEventKind.HEALTH_CHECK,
            "embedding_space",
            {
                "event": "shutdown",
                "flush": flush,
                "kdm_populated_slots": self._kdm.n_populated,
                "total_queries_served": self._total_queries,
                "uptime_hours": (time.time() - self._init_time) / 3600,
            },
        )
        logger.info("DKESEmbeddingSpace shutdown complete.")

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialisation fallback for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
