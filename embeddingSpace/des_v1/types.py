"""
DEMoE Embedding Space - Types, Constants, and Data Structures
Covers: Section 1 of the DEMoE Architecture Specification
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants derived directly from the specification
# ---------------------------------------------------------------------------

class MRLDimensions:
    """Nested MRL embedding sizes. Section 1.2."""
    NESTED_SIZES: Tuple[int, ...] = (32, 64, 128, 256, 512, 768)
    COARSE_DIM: int = 64    # Stage 1 coarse shortlisting
    FULL_DIM:   int = 768   # Stage 2 fine re-ranking
    # Minimum rank-order preservation (Spearman ρ) between 64-dim prefix
    # and full 768-dim embeddings, required before live traffic.
    MIN_SPEARMAN_RHO: float = 0.90


class RoutingConstants:
    """MRL two-stage funnel parameters. Section 1.2."""
    K_COARSE: int = 20           # Candidates retrieved in Stage 1
    K_FINE:   int = 5            # Candidates retained after Stage 2 re-ranking
    # Recall guarantee: shortlist must contain the exact 768-dim NN for ≥ 95%
    # of queries.  Verified on 2,000 held-out queries before deployment.
    RECALL_GUARANTEE: float = 0.95
    RECALL_EVAL_N:    int = 2_000


class AdapterConstants:
    """Domain projection adapter constraints. Section 1.3."""
    MAX_ADAPTERS: int = 50
    # Create adapter when routing coherence < 0.70 for > 3 consecutive days
    ROUTING_COHERENCE_THRESHOLD: float = 0.70
    ROUTING_COHERENCE_WINDOW_DAYS: int = 3
    # Adapter centroid distance for subsumption during cap exhaustion
    SUBSUMPTION_CENTROID_THRESHOLD: float = 0.15
    # Human escalation SLA when cap is exhausted and new event detected
    ESCALATION_WINDOW_HOURS: int = 24
    # Identity initialization tolerance: freshly-initialized adapter must
    # change no embedding by more than 1e-6 in cosine distance.
    INIT_COSINE_TOLERANCE: float = 1e-6
    # 30-day rolling window for utilisation counts during cap exhaustion
    UTILISATION_WINDOW_DAYS: int = 30


class StreamingCentroidConstants:
    """EMA centroid update parameters. Section 1.4."""
    ALPHA: float = 0.01
    # After streaming 10,000 docs the streaming centroid must lie within
    # cosine distance 0.01 of the exact batch centroid.
    CONVERGENCE_DOCS: int = 10_000
    CONVERGENCE_TOLERANCE: float = 0.01


class EncoderHealthConstants:
    """Encoder replica / health monitoring. Section 1.5."""
    MIN_HEALTHY_REPLICAS: int = 2
    P99_LATENCY_ALERT_MS: float = 500.0
    ROUTING_SUCCESS_ALERT_RATE: float = 0.95
    ROUTING_SUCCESS_WINDOW_MINUTES: int = 10
    CACHE_SIZE: int = 10_000   # Most-frequent queries for partial fallback


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class DEMoEError(Exception):
    """Base error class."""


class EncoderFrozenViolationError(DEMoEError):
    """Raised when code attempts to modify the frozen backbone encoder."""


class AdapterCapExhaustedError(DEMoEError):
    """Raised when the 50-adapter cap is reached and no auto-resolution is
    possible, triggering the human escalation path."""


class StaleRoutingCacheError(DEMoEError):
    """Raised when a cache entry has a stale epoch or adapter-set tag."""


class GapEventError(DEMoEError):
    """Raised when a routing deadlock occurs (all K_fine candidates exceed
    threshold), triggers a GAP_EVENT in the system log."""


class HardProhibitionViolationError(DEMoEError):
    """Raised when code attempts to use a DEMoE expert as a synthetic data
    source. Section 8.4 / Fatal T3."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderEpoch:
    """
    Immutable identifier for an encoder snapshot.  Every routing cache entry
    and FAISS index version is tagged with this epoch so stale entries can be
    invalidated after outages or adapter changes.  Section 1.3.
    """
    epoch_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def create(cls) -> "EncoderEpoch":
        return cls(epoch_id=str(uuid.uuid4()))


@dataclass
class RoutingCacheEntry:
    """
    A single entry in the routing cache.  Tagged with the encoder epoch and
    the frozenset of active adapter IDs at cache time.  Section 1.3 / 7.5.
    """
    query_embedding_64:  np.ndarray    # 64-dim prefix for cache lookup
    selected_expert_ids: List[str]
    encoder_epoch:       EncoderEpoch
    active_adapter_ids:  FrozenSet[str]
    timestamp:           float         # UNIX time for LRU eviction

    def is_stale(
        self,
        current_epoch: EncoderEpoch,
        current_adapter_ids: FrozenSet[str],
    ) -> bool:
        """Return True if this entry is no longer valid. Section 1.3."""
        if self.encoder_epoch.epoch_id != current_epoch.epoch_id:
            return True
        if self.active_adapter_ids != current_adapter_ids:
            return True
        return False


@dataclass
class ExpertCentroid:
    """
    Embedding-space representation of a base expert model.
    Stored as both a full 768-dim centroid and a 64-dim prefix centroid.
    For broad experts, multiple centroids may be maintained.  Section 1.4.
    """
    expert_id: str
    # Single or multi-centroid.  Multi-centroid for broad experts (Section 1.4).
    centroids_full:   List[np.ndarray]   # Each shape (768,)
    centroids_prefix: List[np.ndarray]   # Each shape (64,),  Section 1.2 Stage 1
    # Streaming EMA state (Section 1.4 / 9.2)
    ema_centroid:     np.ndarray         # shape (768,)
    ema_centroid_64:  np.ndarray         # shape (64,)
    # Diagonal covariance for Mahalanobis OOD (Section 6.2).
    # Stored as inverse-variance vector for O(d) inference cost.
    diag_inv_cov:     np.ndarray         # shape (768,)

    def update_ema(self, batch_centroid: np.ndarray, alpha: float = StreamingCentroidConstants.ALPHA) -> None:
        """
        Online streaming centroid update. Section 1.4 / 9.2.
        centroid_new = (1-α)·centroid_old + α·batch_centroid
        """
        if batch_centroid.shape != (MRLDimensions.FULL_DIM,):
            raise ValueError(
                f"batch_centroid must have shape ({MRLDimensions.FULL_DIM},), "
                f"got {batch_centroid.shape}"
            )
        self.ema_centroid    = (1 - alpha) * self.ema_centroid    + alpha * batch_centroid
        self.ema_centroid_64 = (1 - alpha) * self.ema_centroid_64 + alpha * batch_centroid[:MRLDimensions.COARSE_DIM]


@dataclass
class DomainProjectionAdapter:
    """
    Lightweight linear map P ∈ ℝ^(d×d) or low-rank factorisation P = AB^T
    that shifts query/document embeddings into a domain-adjusted space.
    Section 1.3.

    Design properties:
      - Initialised to identity.
      - Trained with InfoNCE contrastive loss on domain-specific triplets.
      - Registered per domain alongside the expert registry.
      - Applied to query embeddings only at inference time.
    """
    adapter_id:    str
    domain_label:  str
    # Low-rank factors (None → full-rank square matrix).
    # Full-rank: P shape (d, d).  Low-rank: A shape (d, rank), B shape (d, rank).
    A: Optional[np.ndarray]    # Low-rank factor or full P when B is None
    B: Optional[np.ndarray]    # Low-rank factor;  P = A @ B.T
    # Domain centroid used for adapter selection at inference (Section 1.3).
    domain_centroid: np.ndarray   # shape (768,)
    # Rolling 30-day query count for LRU eviction at cap exhaustion.
    rolling_query_count: int = 0
    is_active: bool = True

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Apply the projection to a query embedding. Section 1.3."""
        _validate_embedding(embedding, MRLDimensions.FULL_DIM)
        if self.B is not None:
            # Low-rank: P·x = A·(B^T·x)
            return self.A @ (self.B.T @ embedding)
        else:
            return self.A @ embedding

    def validate_identity_init(self) -> bool:
        """
        Verify that a freshly-initialised adapter changes no embedding by more
        than 1e-6 in cosine distance (identity check). Section 1.3.
        """
        rng = np.random.default_rng(seed=42)
        test_vecs = rng.standard_normal((10, MRLDimensions.FULL_DIM)).astype(np.float32)
        test_vecs /= np.linalg.norm(test_vecs, axis=1, keepdims=True)
        for v in test_vecs:
            projected = self.project(v)
            projected_norm = projected / (np.linalg.norm(projected) + 1e-12)
            cos_diff = 1.0 - float(np.dot(v, projected_norm))
            if abs(cos_diff) > AdapterConstants.INIT_COSINE_TOLERANCE:
                return False
        return True


@dataclass
class CapExhaustionEvent:
    """
    Emitted when the adapter cap is reached and a new routing degradation
    event is detected.  Triggers human escalation if subsumption is not
    possible.  Section 1.3.
    """
    timestamp:              float
    candidate_domain:       str
    candidate_centroid:     np.ndarray
    lowest_util_adapter_id: str
    subsumption_possible:   bool
    escalated_to_human:     bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_embedding(embedding: np.ndarray, expected_dim: int) -> None:
    if embedding.ndim != 1 or embedding.shape[0] != expected_dim:
        raise ValueError(
            f"Expected 1D embedding of dim {expected_dim}, "
            f"got shape {embedding.shape}"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Numerically stable cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)
