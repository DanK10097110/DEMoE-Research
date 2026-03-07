"""
DEMoE Section 1.3 - Domain Projection Adapter Manager

Handles the full lifecycle of domain projection adapters:
  - Creation with identity-init verification
  - 50-adapter cap enforcement with LRU-based subsumption and human escalation
  - O(log N) adapter selection at inference via FAISS centroid lookup
  - Cache invalidation on any adapter create/delete event
  - InfoNCE contrastive training interface (wiring point for actual trainer)

Expert drift addressed here:
  - Routing coherence is monitored over a rolling 3-day window.  When a
    domain's coherence drops below 0.70, a projection adapter is created to
    correct the local embedding geometry without global recalibration.
  - This avoids the need to retrain the global encoder when a sub-domain's
    vocabulary drifts from the backbone's training distribution.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .types import (
    AdapterCapExhaustedError,
    AdapterConstants,
    CapExhaustionEvent,
    DomainProjectionAdapter,
    EncoderEpoch,
    MRLDimensions,
    cosine_distance,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingCoherenceRecord:
    """
    Tracks daily routing coherence for a domain.  Section 1.3.

    Routing coherence = fraction of near-identical queries that route to the
    same expert over a day.  When coherence < 0.70 for > 3 consecutive days,
    a domain projection adapter is created.
    """
    domain_label: str
    daily_scores: deque = field(default_factory=lambda: deque(maxlen=7))  # 7-day window

    def record_daily_coherence(self, score: float) -> None:
        self.daily_scores.append(score)

    def adapter_creation_triggered(self) -> bool:
        """
        Return True when coherence has been below threshold for more than
        3 consecutive days.  Section 1.3.
        """
        if len(self.daily_scores) < AdapterConstants.ROUTING_COHERENCE_WINDOW_DAYS:
            return False
        recent = list(self.daily_scores)[-AdapterConstants.ROUTING_COHERENCE_WINDOW_DAYS:]
        return all(s < AdapterConstants.ROUTING_COHERENCE_THRESHOLD for s in recent)


class DomainProjectionAdapterManager:
    """
    Manages the pool of up to 50 domain projection adapters.

    Responsibilities:
      - Adapter creation, registration, and deletion
      - Cap enforcement with subsumption logic and human escalation
      - O(log N) adapter selection at inference
      - Cache-invalidation event emission on any pool change
      - Routing coherence monitoring

    The adapter pool is kept small by design: at 50 adapters the FAISS
    centroid lookup is O(log 50) ≈ O(1), making inference overhead negligible.
    """

    def __init__(
        self,
        faiss_adapter_index,           # FAISS index over 768-dim adapter centroids
        cache_invalidation_callback: Callable[[FrozenSet[str]], None],
        human_escalation_callback:   Callable[[CapExhaustionEvent], None],
    ) -> None:
        """
        Parameters
        ----------
        faiss_adapter_index
            FAISS index over 768-dim domain centroids of active adapters.
            Used for O(log N) adapter selection at query time.
        cache_invalidation_callback
            Called with the new frozenset of active adapter IDs whenever
            the pool changes.  Triggers routing cache invalidation.  Section 1.3.
        human_escalation_callback
            Called with a CapExhaustionEvent when the cap is reached and
            subsumption is not possible.  Section 1.3.
        """
        self._adapters: Dict[str, DomainProjectionAdapter] = {}
        self._faiss_index = faiss_adapter_index
        self._on_cache_invalidate = cache_invalidation_callback
        self._on_escalate = human_escalation_callback

        # Routing coherence monitoring (Section 1.3 / 9.5)
        self._coherence_records: Dict[str, RoutingCoherenceRecord] = {}
        # 30-day rolling query counts per adapter (for LRU eviction)
        self._query_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=30))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def active_adapter_ids(self) -> FrozenSet[str]:
        return frozenset(
            aid for aid, a in self._adapters.items() if a.is_active
        )

    def get_adapter_for_query(
        self,
        query_embedding: np.ndarray,
    ) -> Optional[DomainProjectionAdapter]:
        """
        Select the most relevant adapter for a query via a single FAISS lookup
        of the query against adapter centroid embeddings.  O(log N_adapters).
        Section 1.3.

        Returns None if no adapters are registered.
        """
        if not self._adapters:
            return None
        active_adapters = [a for a in self._adapters.values() if a.is_active]
        if not active_adapters:
            return None

        # FAISS search: 1-NN over adapter centroids
        q = query_embedding.reshape(1, -1).astype(np.float32)
        _, indices = self._faiss_index.search(q, 1)
        best_idx = int(indices[0][0])
        if best_idx < 0 or best_idx >= len(active_adapters):
            return None

        adapter = active_adapters[best_idx]
        self._record_query(adapter.adapter_id)
        return adapter

    def record_routing_coherence(self, domain_label: str, coherence_score: float) -> None:
        """
        Record a daily routing coherence score for a domain.
        If the 3-day consecutive threshold is breached, trigger adapter
        creation.  Section 1.3.
        """
        if domain_label not in self._coherence_records:
            self._coherence_records[domain_label] = RoutingCoherenceRecord(domain_label)
        record = self._coherence_records[domain_label]
        record.record_daily_coherence(coherence_score)

        if record.adapter_creation_triggered():
            logger.warning(
                "Routing coherence for domain '%s' has been below %.2f "
                "for %d consecutive days.  Triggering adapter creation. "
                "Section 1.3.",
                domain_label,
                AdapterConstants.ROUTING_COHERENCE_THRESHOLD,
                AdapterConstants.ROUTING_COHERENCE_WINDOW_DAYS,
            )
            return True   # Caller responsible for initiating creation pipeline
        return False

    def create_adapter(
        self,
        domain_label: str,
        domain_centroid: np.ndarray,
        projection_matrix: Optional[np.ndarray] = None,
        A: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
    ) -> DomainProjectionAdapter:
        """
        Create and register a new domain projection adapter.

        Enforces the 50-adapter cap via subsumption and LRU eviction.
        Verifies identity initialization.
        Emits cache invalidation event.
        Section 1.3.

        Parameters
        ----------
        domain_label : str
            Natural language domain name.
        domain_centroid : np.ndarray
            768-dim centroid of the domain's query/document distribution.
        projection_matrix : np.ndarray, optional
            Full (768, 768) projection P.  Mutually exclusive with A/B.
        A, B : np.ndarray, optional
            Low-rank factors for P = A @ B.T.
        """
        import uuid as _uuid

        if projection_matrix is not None and (A is not None or B is not None):
            raise ValueError("Provide either projection_matrix or (A, B), not both.")

        # --- Cap enforcement ---
        n_active = sum(1 for a in self._adapters.values() if a.is_active)
        if n_active >= AdapterConstants.MAX_ADAPTERS:
            resolved = self._handle_cap_exhaustion(domain_label, domain_centroid)
            if not resolved:
                raise AdapterCapExhaustedError(
                    f"Adapter cap ({AdapterConstants.MAX_ADAPTERS}) reached and "
                    "subsumption was not possible.  Human escalation queued. "
                    "Section 1.3."
                )

        # --- Build adapter ---
        adapter_id = str(_uuid.uuid4())
        d = MRLDimensions.FULL_DIM

        if projection_matrix is not None:
            if projection_matrix.shape != (d, d):
                raise ValueError(f"projection_matrix must be ({d}, {d}); got {projection_matrix.shape}")
            A_stored = projection_matrix
            B_stored = None
        elif A is not None and B is not None:
            A_stored, B_stored = A, B
        else:
            # Default: identity (full-rank square)
            A_stored = np.eye(d, dtype=np.float32)
            B_stored = None

        adapter = DomainProjectionAdapter(
            adapter_id=adapter_id,
            domain_label=domain_label,
            A=A_stored,
            B=B_stored,
            domain_centroid=domain_centroid.astype(np.float32),
        )

        # --- Identity-init verification ---
        if not adapter.validate_identity_init():
            raise RuntimeError(
                f"Adapter '{adapter_id}' failed identity-init verification. "
                f"A freshly-initialized adapter must change no embedding by more "
                f"than {AdapterConstants.INIT_COSINE_TOLERANCE} in cosine distance. "
                "Section 1.3."
            )

        self._adapters[adapter_id] = adapter

        # --- Update FAISS index ---
        self._faiss_index.add(domain_centroid.reshape(1, -1).astype(np.float32))

        # --- Emit cache invalidation ---
        self._emit_cache_invalidation()
        logger.info("Created domain projection adapter '%s' for domain '%s'.", adapter_id[:8], domain_label)
        return adapter

    def delete_adapter(self, adapter_id: str) -> None:
        """
        Deactivate an adapter and emit cache invalidation.  Section 1.3.
        Note: FAISS does not support deletion; the adapter is marked inactive
        and filtered at query time.  For full removal, rebuild the FAISS index.
        """
        if adapter_id not in self._adapters:
            raise KeyError(f"Adapter '{adapter_id}' not found.")
        self._adapters[adapter_id].is_active = False
        self._emit_cache_invalidation()
        logger.info("Deactivated adapter '%s'.", adapter_id[:8])

    # ------------------------------------------------------------------
    # Cap exhaustion logic (Section 1.3)
    # ------------------------------------------------------------------

    def _handle_cap_exhaustion(
        self,
        candidate_domain: str,
        candidate_centroid: np.ndarray,
    ) -> bool:
        """
        When cap is reached:
          1. Find lowest-utilisation adapter (30-day rolling query count).
          2. If candidate centroid subsumes lowest-util adapter (cosine
             distance < 0.15), replace it.
          3. Otherwise, emit CapExhaustionEvent and escalate to human.
             Return False to signal that creation cannot proceed.
        Section 1.3.
        """
        active_adapters = [a for a in self._adapters.values() if a.is_active]
        if not active_adapters:
            return True  # Should never happen; handle gracefully

        # Lowest-utilisation adapter by 30-day rolling query count
        def util_count(a: DomainProjectionAdapter) -> int:
            return sum(self._query_counts.get(a.adapter_id, []))

        lowest_util = min(active_adapters, key=util_count)

        # Subsumption check: cosine centroid distance < 0.15
        dist = cosine_distance(candidate_centroid, lowest_util.domain_centroid)
        subsumption_possible = dist < AdapterConstants.SUBSUMPTION_CENTROID_THRESHOLD

        event = CapExhaustionEvent(
            timestamp=time.time(),
            candidate_domain=candidate_domain,
            candidate_centroid=candidate_centroid,
            lowest_util_adapter_id=lowest_util.adapter_id,
            subsumption_possible=subsumption_possible,
        )

        if subsumption_possible:
            logger.info(
                "Cap exhaustion: replacing lowest-util adapter '%s' (domain='%s', "
                "util=%d) with new adapter for domain '%s' (centroid distance=%.4f). "
                "Section 1.3.",
                lowest_util.adapter_id[:8],
                lowest_util.domain_label,
                util_count(lowest_util),
                candidate_domain,
                dist,
            )
            self.delete_adapter(lowest_util.adapter_id)
            return True  # Proceed with creation

        # Cannot subsume; escalate to human within 24 hours
        event.escalated_to_human = True
        logger.error(
            "Cap exhaustion: cannot subsume.  Candidate domain '%s' is too far "
            "(distance=%.4f > %.4f) from lowest-util adapter '%s'.  "
            "Deferring creation and escalating to human operator.  "
            "SLA: within %d hours.  Section 1.3.",
            candidate_domain,
            dist,
            AdapterConstants.SUBSUMPTION_CENTROID_THRESHOLD,
            lowest_util.adapter_id[:8],
            AdapterConstants.ESCALATION_WINDOW_HOURS,
        )
        self._on_escalate(event)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_query(self, adapter_id: str) -> None:
        """Record a query hit for rolling utilisation tracking."""
        today = int(time.time() // 86_400)
        self._query_counts[adapter_id].append(today)

    def _emit_cache_invalidation(self) -> None:
        """
        Notify the routing cache that the adapter set has changed.
        All cache entries tagged with the old adapter frozenset are invalid.
        Section 1.3.
        """
        self._on_cache_invalidate(self.active_adapter_ids)
