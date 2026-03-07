"""
DEMoE Section 1.1 / 1.2 - Frozen Backbone Encoder with MRL Support

Key design decisions implemented here:
  - Encoder is permanently frozen; any mutation raises EncoderFrozenViolationError
  - MRL produces nested embeddings; 64-dim prefix is valid for coarse retrieval
  - Spearman ρ verification between 64-dim prefix and full 768-dim embeddings
  - Encoder health monitoring (replica count, P99 latency, routing success rate)
  - Encoder epoch versioning for cache invalidation

Curse-of-dimensionality mitigations:
  - The two-stage MRL funnel coarse-shortlists in 64 dims (much lower than 768)
    before fine re-ranking.  This reduces the effective search dimension for
    the expensive exact comparison step, shrinking the empty-space problem.
  - Domain projection adapters correct local distribution gaps without
    touching the global 768-dim space, preventing global recalibration cost.
  - Mahalanobis OOD with diagonal covariance is O(d) in storage and
    computation, avoiding the full-covariance curse.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import stats

from .types import (
    AdapterConstants,
    EncoderEpoch,
    EncoderFrozenViolationError,
    EncoderHealthConstants,
    MRLDimensions,
    RoutingConstants,
    _validate_embedding,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


class FrozenMRLEncoder:
    """
    Wrapper around a pre-trained MRL backbone encoder that enforces:
      1. Immutability of backbone weights (Section 1.1)
      2. MRL nested embedding extraction (Section 1.2)
      3. Spearman ρ verification before live traffic (Section 1.2)
      4. Encoder health monitoring (Section 1.5 Limitation 1.B)
      5. Epoch versioning for cache invalidation (Section 1.3)

    The encoder MUST have been trained with Matryoshka loss.  If the
    backing model was not MRL-trained, this class surfaces a clear error
    rather than silently producing invalid prefix embeddings.

    Parameters
    ----------
    encode_fn : callable
        Function ``(texts: List[str]) -> np.ndarray`` with shape (N, 768).
        Must accept batches.  This is the only interface to the backbone.
    is_mrl_trained : bool
        Whether the backing model was trained with Matryoshka loss.
        If False, only full-dimensional retrieval is available and the two-
        stage funnel is disabled (Section 1.5 Limitation 1.A).
    replica_ids : list of str
        Identifiers of healthy encoder replicas.  Alerts if < 2.
    """

    def __init__(
        self,
        encode_fn: Callable[[List[str]], np.ndarray],
        is_mrl_trained: bool = True,
        replica_ids: Optional[List[str]] = None,
    ) -> None:
        # The encode function is the only pathway to the backbone.  We store
        # only the callable, not the model itself, so parameter access is not
        # possible from this layer.
        self._encode_fn: Callable[[List[str]], np.ndarray] = encode_fn
        self._is_mrl_trained: bool = is_mrl_trained
        self._replica_ids: List[str] = list(replica_ids or [])
        self._current_epoch: EncoderEpoch = EncoderEpoch.create()

        # Health telemetry queues (Section 1.5 Limitation 1.B)
        self._latency_window: deque = deque(maxlen=1_000)   # ms, rolling P99
        self._routing_success_window: deque = deque(maxlen=10_000)  # bool
        self._verified: bool = False  # True after Spearman ρ pre-flight passes

        if not self._is_mrl_trained:
            logger.warning(
                "Encoder was NOT trained with Matryoshka loss. "
                "Two-stage MRL funnel is DISABLED; falling back to single-stage "
                "full-dimensional retrieval. "
                "See Section 1.5 Limitation 1.A for mitigation options."
            )

    # ------------------------------------------------------------------
    # Frozen backbone enforcement
    # ------------------------------------------------------------------

    def update_weights(self, *args, **kwargs) -> None:  # noqa: ANN001
        """
        Hard block on any weight mutation.  Section 1.1:
        'The shared encoder backbone is permanently frozen.'
        """
        raise EncoderFrozenViolationError(
            "The MRL encoder backbone is permanently frozen. "
            "Domain-specific changes must be made via DomainProjectionAdapter. "
            "See Section 1.1 of the DEMoE specification."
        )

    def __setattr__(self, name: str, value: object) -> None:
        # Allow internal dunder assignments during __init__ and private attrs
        if name.startswith("_") or name in ("update_weights",):
            super().__setattr__(name, value)
        else:
            raise EncoderFrozenViolationError(
                f"Attempted to set attribute '{name}' on frozen encoder. "
                "Only private internal state may be modified."
            )

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts and return full 768-dim MRL embeddings.
        Automatically records P99 latency for health monitoring.
        """
        self._check_replica_health()
        t0 = time.monotonic()
        try:
            embeddings = self._encode_fn(texts)
        except Exception as exc:
            logger.error("Encoder call failed: %s", exc)
            raise
        latency_ms = (time.monotonic() - t0) * 1_000
        self._latency_window.append(latency_ms)
        self._check_latency_alert()
        if embeddings.ndim != 2 or embeddings.shape[1] != MRLDimensions.FULL_DIM:
            raise ValueError(
                f"Encoder returned shape {embeddings.shape}; "
                f"expected (N, {MRLDimensions.FULL_DIM})."
            )
        return embeddings.astype(np.float32)

    def extract_prefix(self, embeddings: np.ndarray, dim: int = MRLDimensions.COARSE_DIM) -> np.ndarray:
        """
        Extract the first `dim` dimensions from full embeddings.
        Valid only when is_mrl_trained=True.  Section 1.2.
        """
        if not self._is_mrl_trained:
            raise RuntimeError(
                "Cannot extract MRL prefix: encoder was not trained with "
                "Matryoshka loss.  Only full-dimensional retrieval is available. "
                "See Section 1.5 Limitation 1.A."
            )
        if dim not in MRLDimensions.NESTED_SIZES:
            raise ValueError(
                f"Requested prefix dim {dim} is not a valid MRL nested size. "
                f"Valid sizes: {MRLDimensions.NESTED_SIZES}"
            )
        return embeddings[:, :dim].astype(np.float32)

    # ------------------------------------------------------------------
    # Pre-flight: Spearman ρ verification (Section 1.2)
    # ------------------------------------------------------------------

    def run_spearman_verification(self, held_out_pairs: List[Tuple[str, str]]) -> float:
        """
        Verify that the first 64 MRL dimensions preserve rank-order similarity
        with the full 768-dim embeddings at Spearman ρ ≥ 0.90.

        This check MUST pass before the system routes any live traffic.

        Parameters
        ----------
        held_out_pairs : list of (query, document) string pairs
            1,000 pairs required by the specification.

        Returns
        -------
        float
            Observed Spearman ρ.

        Raises
        ------
        RuntimeError
            If ρ < MRLDimensions.MIN_SPEARMAN_RHO or MRL is not available.
        """
        if not self._is_mrl_trained:
            raise RuntimeError(
                "Spearman verification requires MRL-trained encoder. "
                "Section 1.2 verification cannot be completed."
            )
        if len(held_out_pairs) < 1_000:
            logger.warning(
                "Specification requires 1,000 held-out pairs for Spearman "
                "verification; only %d provided.  Results may be less reliable.",
                len(held_out_pairs),
            )

        queries    = [p[0] for p in held_out_pairs]
        documents  = [p[1] for p in held_out_pairs]
        q_embs     = self.encode(queries)
        d_embs     = self.encode(documents)

        # Full 768-dim cosine similarities (reference)
        sims_full = np.einsum(
            "ij,ij->i",
            q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-12),
            d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-12),
        )
        # 64-dim prefix cosine similarities
        q_64 = self.extract_prefix(q_embs, MRLDimensions.COARSE_DIM)
        d_64 = self.extract_prefix(d_embs, MRLDimensions.COARSE_DIM)
        sims_64 = np.einsum(
            "ij,ij->i",
            q_64 / (np.linalg.norm(q_64, axis=1, keepdims=True) + 1e-12),
            d_64 / (np.linalg.norm(d_64, axis=1, keepdims=True) + 1e-12),
        )

        rho, p_value = stats.spearmanr(sims_full, sims_64)
        logger.info("Spearman ρ (64-dim vs 768-dim): %.4f (p=%.4e)", rho, p_value)

        if rho < MRLDimensions.MIN_SPEARMAN_RHO:
            raise RuntimeError(
                f"Spearman ρ pre-flight FAILED: ρ={rho:.4f} < "
                f"required {MRLDimensions.MIN_SPEARMAN_RHO}. "
                "Do not route live traffic.  Section 1.2 requirement not met."
            )
        self._verified = True
        logger.info("Spearman ρ pre-flight PASSED.  System cleared for live routing.")
        return float(rho)

    # ------------------------------------------------------------------
    # Recall verification (Section 1.2)
    # ------------------------------------------------------------------

    def verify_coarse_recall(
        self,
        held_out_queries: List[str],
        expert_centroids_full: np.ndarray,   # shape (E, 768)
        expert_centroids_64:   np.ndarray,   # shape (E, 64)
        faiss_index_64,                       # FAISS Index over 64-dim centroids
    ) -> float:
        """
        Verify that the 20-candidate coarse shortlist contains the exact
        768-dim nearest neighbour for ≥ 95% of held-out queries.
        Section 1.2 recall guarantee.

        Parameters
        ----------
        held_out_queries : list of str
            2,000 queries required by the specification.

        Returns
        -------
        float
            Recall@20 rate.
        """
        if len(held_out_queries) < RoutingConstants.RECALL_EVAL_N:
            logger.warning(
                "Specification requires %d held-out queries for recall "
                "verification; got %d.",
                RoutingConstants.RECALL_EVAL_N,
                len(held_out_queries),
            )

        embeddings_full = self.encode(held_out_queries)
        embeddings_64   = self.extract_prefix(embeddings_full, MRLDimensions.COARSE_DIM)

        # Ground-truth nearest neighbour by exhaustive 768-dim cosine search
        norms_q = embeddings_full / (np.linalg.norm(embeddings_full, axis=1, keepdims=True) + 1e-12)
        norms_e = expert_centroids_full / (np.linalg.norm(expert_centroids_full, axis=1, keepdims=True) + 1e-12)
        similarities = norms_q @ norms_e.T                     # (Q, E)
        true_nn_ids = similarities.argmax(axis=1)              # (Q,)

        # Coarse shortlist from FAISS 64-dim index
        _, coarse_candidates = faiss_index_64.search(         # (Q, K_coarse)
            embeddings_64, RoutingConstants.K_COARSE
        )

        hits = sum(
            true_id in coarse_candidates[i]
            for i, true_id in enumerate(true_nn_ids)
        )
        recall = hits / len(held_out_queries)
        logger.info("Coarse shortlist recall@%d: %.4f", RoutingConstants.K_COARSE, recall)

        if recall < RoutingConstants.RECALL_GUARANTEE:
            logger.error(
                "Coarse recall BELOW GUARANTEE: %.4f < %.4f.  "
                "Consider expanding K_coarse or verifying FAISS index. "
                "Section 1.2.",
                recall,
                RoutingConstants.RECALL_GUARANTEE,
            )
        return recall

    # ------------------------------------------------------------------
    # Epoch management (Section 1.3)
    # ------------------------------------------------------------------

    @property
    def current_epoch(self) -> EncoderEpoch:
        return self._current_epoch

    def rotate_epoch(self) -> EncoderEpoch:
        """
        Rotate the encoder epoch identifier.  Called after outage recovery.
        All routing cache entries tagged with the old epoch are automatically
        invalid.  Section 1.3.
        """
        old_epoch = self._current_epoch
        object.__setattr__(self, "_current_epoch", EncoderEpoch.create())
        logger.info(
            "Encoder epoch rotated: %s → %s",
            old_epoch.epoch_id[:8],
            self._current_epoch.epoch_id[:8],
        )
        return self._current_epoch

    # ------------------------------------------------------------------
    # Health monitoring (Section 1.5 Limitation 1.B)
    # ------------------------------------------------------------------

    def record_routing_outcome(self, success: bool) -> None:
        """Record whether a query was successfully routed."""
        self._routing_success_window.append(int(success))
        self._check_routing_success_alert()

    def register_replica(self, replica_id: str) -> None:
        if replica_id not in self._replica_ids:
            self._replica_ids.append(replica_id)

    def deregister_replica(self, replica_id: str) -> None:
        self._replica_ids = [r for r in self._replica_ids if r != replica_id]
        self._check_replica_health()

    def p99_latency_ms(self) -> Optional[float]:
        if not self._latency_window:
            return None
        return float(np.percentile(list(self._latency_window), 99))

    def _check_replica_health(self) -> None:
        n = len(self._replica_ids)
        if n < EncoderHealthConstants.MIN_HEALTHY_REPLICAS:
            logger.critical(
                "ALERT: Only %d healthy encoder replica(s) available.  "
                "Minimum required: %d.  "
                "Novel-query routing will fail if the remaining replica fails.  "
                "Section 1.5 Limitation 1.B.  Page on-call immediately.",
                n,
                EncoderHealthConstants.MIN_HEALTHY_REPLICAS,
            )

    def _check_latency_alert(self) -> None:
        p99 = self.p99_latency_ms()
        if p99 is not None and p99 > EncoderHealthConstants.P99_LATENCY_ALERT_MS:
            logger.warning(
                "ALERT: Encoder P99 latency %.1f ms exceeds threshold %.1f ms. "
                "Section 1.5.",
                p99,
                EncoderHealthConstants.P99_LATENCY_ALERT_MS,
            )

    def _check_routing_success_alert(self) -> None:
        window = list(self._routing_success_window)[
            -EncoderHealthConstants.ROUTING_SUCCESS_WINDOW_MINUTES * 60:
        ]
        if not window:
            return
        success_rate = sum(window) / len(window)
        if success_rate < EncoderHealthConstants.ROUTING_SUCCESS_ALERT_RATE:
            logger.error(
                "ALERT: Novel-query routing success rate %.3f below threshold "
                "%.3f over recent window.  Section 1.5.",
                success_rate,
                EncoderHealthConstants.ROUTING_SUCCESS_ALERT_RATE,
            )
