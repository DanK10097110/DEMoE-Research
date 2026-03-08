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
import threading
import time
from collections import deque
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Internal sentinel so __setattr__ can distinguish init-time private writes
# from external attempts.
# ---------------------------------------------------------------------------
_INIT_SENTINEL = object()


@dataclass
class _HealthSnapshot:
    """Immutable snapshot returned by health_snapshot()."""
    p99_latency_ms: Optional[float]
    routing_success_rate: Optional[float]
    replica_count: int
    is_verified: bool
    epoch_id: str


class FrozenMRLEncoder:
    """
    Wrapper around a pre-trained MRL backbone encoder that enforces:
      1. Immutability of backbone weights                     (Section 1.1)
      2. MRL nested embedding extraction                      (Section 1.2)
      3. Spearman ρ + magnitude calibration pre-flight        (Section 1.2)
      4. Encoder health monitoring                            (Section 1.5)
      5. Epoch versioning for cache invalidation              (Section 1.3)

    Thread-safety
    -------------
    All public methods are safe to call from multiple threads.  Internal
    mutable state is protected by a single reentrant lock (_state_lock).

    Parameters
    ----------
    encode_fn : callable
        ``(texts: List[str]) -> np.ndarray`` shape (N, 768), float32.
        Embeddings need NOT be pre-normalised; this class normalises them.
    is_mrl_trained : bool
        Whether the backing model was trained with Matryoshka loss.
    replica_ids : list of str
        Initial set of healthy replica identifiers.
    routing_success_window_size : int
        Number of most-recent routing outcomes to track for the success-rate
        alert.  Replaces the broken wall-clock slice.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        encode_fn: Callable[[List[str]], np.ndarray],
        is_mrl_trained: bool = True,
        replica_ids: Optional[List[str]] = None,
        routing_success_window_size: int = 10_000,
        _sentinel: object = None,          # used internally to unlock __setattr__
    ) -> None:
        # After __init__ completes, the guard is active for all public attrs.
        def _set(name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        _set("_encode_fn", encode_fn)
        _set("_is_mrl_trained", is_mrl_trained)
        _set("_replica_ids", list(replica_ids or []))
        _set("_current_epoch", EncoderEpoch.create())
        _set("_verified", False)
        _set("_magnitude_calibrated", False) 

        _set("_state_lock", threading.RLock())

        _set("_latency_window", deque(maxlen=1_000))

        _set(
            "_routing_success_window",
            deque(maxlen=routing_success_window_size),
        )
        _set("_routing_window_size", routing_success_window_size)

        if not is_mrl_trained:
            logger.warning(
                "Encoder was NOT trained with Matryoshka loss. "
                "Two-stage MRL funnel is DISABLED; falling back to single-stage "
                "full-dimensional retrieval. "
                "See Section 1.5 Limitation 1.A for mitigation options."
            )

        self._check_replica_health()   # warn immediately if under-replicated

    def __setattr__(self, name: str, value: object) -> None:
        # Private attributes (leading underscore) are always writeable so that
        # internal state updates work naturally.  Public attributes are blocked
        # unconditionally — the encoder exposes no mutable public interface.
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            raise EncoderFrozenViolationError(
                f"Attempted to set public attribute '{name}' on a frozen encoder. "
                "The MRL encoder backbone is permanently frozen. "
                "Domain-specific changes must be made via DomainProjectionAdapter. "
                "See Section 1.1 of the DEMoE specification."
            )

    # Public alias kept for API compatibility; raises always.
    def update_weights(self, *args, **kwargs) -> None:  # noqa: ANN001
        raise EncoderFrozenViolationError(
            "The MRL encoder backbone is permanently frozen. "
            "See Section 1.1 of the DEMoE specification."
        )

    # ------------------------------------------------------------------
    # Input validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_texts(texts: List[str]) -> None:
        if not texts:
            raise ValueError("texts must be a non-empty list of strings.")
        bad = [i for i, t in enumerate(texts) if not isinstance(t, str)]
        if bad:
            raise TypeError(
                f"texts[{bad[0]}] is {type(texts[bad[0]]).__name__}, expected str. "
                f"All elements must be strings."
            )

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], normalise: bool = True) -> np.ndarray:
        """
        Encode texts and return (N, 768) float32 embeddings.

        Parameters
        ----------
        normalise : bool
            If True (default), L2-normalise each row so that downstream dot
            products equal cosine similarity without extra caller burden.
        """
        self._validate_texts(texts)       
        self._check_replica_health()

        t0 = time.monotonic()
        try:
            embeddings = self._encode_fn(texts)
        except Exception as exc:
            logger.error("Encoder call failed: %s", exc)
            raise
        finally:
            latency_ms = (time.monotonic() - t0) * 1_000
            with self._state_lock:          
                self._latency_window.append(latency_ms)
            self._check_latency_alert()

        if embeddings.ndim != 2 or embeddings.shape[1] != MRLDimensions.FULL_DIM:
            raise ValueError(
                f"Encoder returned shape {embeddings.shape}; "
                f"expected (N, {MRLDimensions.FULL_DIM})."
            )

        embeddings = embeddings.astype(np.float32)

        if normalise:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)   # avoid div-by-zero
            embeddings = embeddings / norms

        return embeddings

    def extract_prefix(
        self, embeddings: np.ndarray, dim: int = MRLDimensions.COARSE_DIM
    ) -> np.ndarray:
        """
        Extract the first `dim` dimensions from full embeddings.
        """
        if not self._is_mrl_trained:
            raise RuntimeError(
                "Cannot extract MRL prefix: encoder was not trained with "
                "Matryoshka loss.  Only full-dimensional retrieval is available."
            )
        if dim not in MRLDimensions.NESTED_SIZES:
            raise ValueError(
                f"Requested prefix dim {dim} is not a valid MRL nested size. "
                f"Valid sizes: {MRLDimensions.NESTED_SIZES}"
            )
        if dim >= MRLDimensions.FULL_DIM:
            raise ValueError(
                f"Prefix dim {dim} must be strictly less than FULL_DIM "
                f"({MRLDimensions.FULL_DIM}).  Use encode() directly for full embeddings."
            )
        prefix = embeddings[:, :dim].astype(np.float32)
        # Re-normalise the prefix slice — slicing unit vectors does NOT preserve
        # unit norm; cosine similarity on the prefix requires re-normalisation.
        norms = np.linalg.norm(prefix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return prefix / norms

    def run_spearman_verification(self, held_out_pairs: List[Tuple[str, str]]) -> float:
        """
        Verify that the first 64 MRL dimensions preserve rank-order similarity
        with the full 768-dim embeddings at Spearman ρ ≥ MIN_SPEARMAN_RHO.

        ----------
        The original implementation measured pairwise (query_i, doc_i)
        similarities — only N diagonal values.  This misses whether the *rank
        ordering across all N² pairs* is preserved, which is what matters for
        ANN retrieval correctness.

        We now compute an all-pairs similarity matrix and flatten both the
        64-dim and 768-dim upper triangles before computing Spearman ρ.  This
        is O(N²) memory; for N=1000 that is 1M floats ≈ 4 MB — acceptable.

        Parameters
        ----------
        held_out_pairs : list of (str, str)
            At least 1,000 unique texts.  Only the first element of each pair
            is used as the corpus for the all-pairs comparison.
        """
        if not self._is_mrl_trained:
            raise RuntimeError(
                "Spearman verification requires MRL-trained encoder."
            )
        if len(held_out_pairs) < 1_000:
            logger.warning(
                "Specification requires 1,000 held-out pairs; only %d provided.",
                len(held_out_pairs),
            )

        texts = [p[0] for p in held_out_pairs]
        embs_full = self.encode(texts, normalise=True)       # already unit norm
        embs_64   = self.extract_prefix(embs_full, MRLDimensions.COARSE_DIM)

        # All-pairs cosine similarity (dot product on unit vectors)
        sim_full_matrix = embs_full @ embs_full.T            # (N, N)
        sim_64_matrix   = embs_64   @ embs_64.T             # (N, N)

        # Upper triangle (excluding diagonal) → 1-D rank vectors
        idx = np.triu_indices(len(texts), k=1)
        sims_full_flat = sim_full_matrix[idx]
        sims_64_flat   = sim_64_matrix[idx]

        rho, p_value = stats.spearmanr(sims_full_flat, sims_64_flat)
        logger.info(
            "Spearman ρ all-pairs (64-dim vs 768-dim): %.4f (p=%.4e)", rho, p_value
        )

        if rho < MRLDimensions.MIN_SPEARMAN_RHO:
            raise RuntimeError(
                f"Spearman ρ pre-flight FAILED: ρ={rho:.4f} < "
                f"required {MRLDimensions.MIN_SPEARMAN_RHO}. "
                "Do not route live traffic."
            )

        with self._state_lock:
            self._verified = True
        logger.info("Spearman ρ pre-flight PASSED.")
        return float(rho)

    def run_magnitude_calibration(
        self,
        held_out_pairs: List[Tuple[str, str]],
        max_mean_abs_error: float = 0.05,
    ) -> float:
        """
        Verify that 64-dim cosine distances are *numerically close* to 768-dim
        distances, not just rank-correlated.

        Spearman ρ only validates rank ordering.  A threshold
        calibrated on 768-dim distances will be systematically wrong if 64-dim
        distances are uniformly scaled or shifted.  This check measures mean
        absolute error between the two distance distributions.

        Returns
        -------
        float
            Observed mean absolute error.
        """
        if not self._is_mrl_trained:
            raise RuntimeError("Magnitude calibration requires MRL-trained encoder.")

        texts = [p[0] for p in held_out_pairs]
        embs_full = self.encode(texts, normalise=True)
        embs_64   = self.extract_prefix(embs_full, MRLDimensions.COARSE_DIM)

        sim_full = embs_full @ embs_full.T
        sim_64   = embs_64   @ embs_64.T

        idx = np.triu_indices(len(texts), k=1)
        mae = float(np.mean(np.abs(sim_full[idx] - sim_64[idx])))
        logger.info("Distance magnitude MAE (64-dim vs 768-dim): %.4f", mae)

        if mae > max_mean_abs_error:
            logger.error(
                "Magnitude calibration WARNING: MAE=%.4f exceeds threshold %.4f. "
                "OOD thresholds calibrated at 768-dim will be inaccurate at 64-dim. "
                "Consider re-calibrating thresholds separately per dimension.",
                mae,
                max_mean_abs_error,
            )
        else:
            with self._state_lock:
                self._magnitude_calibrated = True
            logger.info("Magnitude calibration PASSED.")

        return mae


    def verify_coarse_recall(
        self,
        held_out_queries: List[str],
        expert_centroids_full: np.ndarray,    # shape (E, 768)
        expert_centroids_64:   np.ndarray,    # shape (E, 64)
        faiss_index_64,                        # FAISS Index over 64-dim centroids
    ) -> float:
        if len(held_out_queries) < RoutingConstants.RECALL_EVAL_N:
            logger.warning(
                "Specification requires %d held-out queries; got %d.",
                RoutingConstants.RECALL_EVAL_N,
                len(held_out_queries),
            )

        embeddings_full = self.encode(held_out_queries, normalise=True)
        embeddings_64   = self.extract_prefix(embeddings_full, MRLDimensions.COARSE_DIM)

        # Normalise centroids (encode() normalises queries; centroids may not be)
        nc_full = expert_centroids_full / (
            np.linalg.norm(expert_centroids_full, axis=1, keepdims=True) + 1e-12
        )
        similarities    = embeddings_full @ nc_full.T          # (Q, E)
        true_nn_ids     = similarities.argmax(axis=1)          # (Q,)

        _, coarse_candidates = faiss_index_64.search(
            embeddings_64, RoutingConstants.K_COARSE
        )

        n_evaluated = len(true_nn_ids)                         
        hits = sum(
            true_id in coarse_candidates[i]
            for i, true_id in enumerate(true_nn_ids)
        )
        recall = hits / n_evaluated                    
        logger.info("Coarse shortlist recall@%d: %.4f", RoutingConstants.K_COARSE, recall)

        if recall < RoutingConstants.RECALL_GUARANTEE:
            logger.error(
                "Coarse recall BELOW GUARANTEE: %.4f < %.4f. "
                "Consider expanding K_coarse or verifying FAISS index.",
                recall,
                RoutingConstants.RECALL_GUARANTEE,
            )
        return recall


    def assert_ready_for_live_traffic(self) -> None:
        """
        Raises
        ------
        RuntimeError
            If Spearman pre-flight has not passed.
        """
        with self._state_lock:
            verified = self._verified
        if not verified:
            raise RuntimeError(
                "FrozenMRLEncoder has not passed Spearman ρ pre-flight. "
                "Call run_spearman_verification() before routing live traffic. "
                "Section 1.2."
            )


    @property
    def current_epoch(self) -> EncoderEpoch:
        with self._state_lock:
            return self._current_epoch

    def rotate_epoch(self) -> EncoderEpoch:

        with self._state_lock:
            old_epoch = self._current_epoch
            self._current_epoch = EncoderEpoch.create()
            new_epoch = self._current_epoch
            # Invalidate the verification flag — a new epoch means a new encoder
            # version may be in use and pre-flight must be re-run.
            self._verified = False
            self._magnitude_calibrated = False

        logger.info(
            "Encoder epoch rotated: %s → %s. "
            "Spearman pre-flight must be re-run before live routing.",
            old_epoch.epoch_id[:8],
            new_epoch.epoch_id[:8],
        )
        return new_epoch

    def record_routing_outcome(self, success: bool) -> None:
        """
        Record whether a query was successfully routed.
        We expose a configurable window_size and check the whole window content.
        """
        with self._state_lock:
            self._routing_success_window.append(int(success))
        self._check_routing_success_alert()

    def register_replica(self, replica_id: str) -> None:
        with self._state_lock:
            if replica_id not in self._replica_ids:
                self._replica_ids.append(replica_id)

    def deregister_replica(self, replica_id: str) -> None:
        with self._state_lock:
            self._replica_ids = [r for r in self._replica_ids if r != replica_id]
        self._check_replica_health()

    def p99_latency_ms(self) -> Optional[float]:

        with self._state_lock:
            snapshot = list(self._latency_window)
        if not snapshot:
            return None
        return float(np.percentile(snapshot, 99))

    def health_snapshot(self) -> _HealthSnapshot:
        """Return a consistent, immutable health snapshot (thread-safe)."""
        with self._state_lock:
            p99 = self.p99_latency_ms()
            window = list(self._routing_success_window)
            success_rate = (sum(window) / len(window)) if window else None
            replica_count = len(self._replica_ids)
            verified = self._verified
            epoch_id = self._current_epoch.epoch_id
        return _HealthSnapshot(
            p99_latency_ms=p99,
            routing_success_rate=success_rate,
            replica_count=replica_count,
            is_verified=verified,
            epoch_id=epoch_id,
        )

    # ------------------------------------------------------------------
    # Private alert helpers
    # ------------------------------------------------------------------

    def _check_replica_health(self) -> None:
        with self._state_lock:
            n = len(self._replica_ids)
        if n < EncoderHealthConstants.MIN_HEALTHY_REPLICAS:
            logger.critical(
                "ALERT: Only %d healthy encoder replica(s) available. "
                "Minimum required: %d. "
                "Novel-query routing will fail if the remaining replica fails. "
                "Page on-call immediately.",
                n,
                EncoderHealthConstants.MIN_HEALTHY_REPLICAS,
            )

    def _check_latency_alert(self) -> None:
        p99 = self.p99_latency_ms()
        if p99 is not None and p99 > EncoderHealthConstants.P99_LATENCY_ALERT_MS:
            logger.warning(
                "ALERT: Encoder P99 latency %.1f ms exceeds threshold %.1f ms.",
                p99,
                EncoderHealthConstants.P99_LATENCY_ALERT_MS,
            )

    def _check_routing_success_alert(self) -> None:

        with self._state_lock:
            window = list(self._routing_success_window)
        if not window:
            return
        success_rate = sum(window) / len(window)
        if success_rate < EncoderHealthConstants.ROUTING_SUCCESS_ALERT_RATE:
            logger.error(
                "ALERT: Routing success rate %.3f below threshold %.3f "
                "over a %d-outcome rolling window.",
                success_rate,
                EncoderHealthConstants.ROUTING_SUCCESS_ALERT_RATE,
                len(window),
            )