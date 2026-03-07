"""
DEMoE Section 1.4 / 9.2 - Expert Centroid Registry

What gets embedded and how it's managed:
  - Expert centroids: 768-dim + 64-dim prefix, updated via streaming EMA
  - Multi-centroid support for broad Tier-3 experts
  - Streaming centroid convergence verification (Section 9.2)
  - Expert drift sentinel metrics (Section 3.5 Limitation 3.B)

Expert drift problem addressed here:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ Long-term identity drift under continual EWC updates.                  │
  │                                                                         │
  │ EWC prevents catastrophic forgetting of specific weights, but cannot   │
  │ prevent gradual semantic drift where the expert slowly shifts domain   │
  │ without any single update being large enough to trigger Fisher         │
  │ protection.                                                             │
  │                                                                         │
  │ Mitigation: centroid drift monitoring.                                  │
  │   - Registration centroid is stored permanently at expert creation.    │
  │   - On each EMA update, cosine distance from the registration          │
  │     centroid is tracked.  Alert if distance > 0.15.                    │
  │   - Rolling 90-day benchmark performance trend is monitored.           │
  │     Alert if slope < -0.002/week.                                      │
  │   - Expert utilisation rate is monitored.  Alert if declining          │
  │     > 20% over 90 days without new competing experts.                  │
  │                                                                         │
  │ These are SENTINEL METRICS for human operators.  There is currently    │
  │ no automated correction.  Confirmed drift > 5pp on domain benchmarks  │
  │ requires a managed retraining event with full K-FAC Fisher reset.      │
  └─────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import (
    ExpertCentroid,
    MRLDimensions,
    StreamingCentroidConstants,
    _validate_embedding,
    cosine_distance,
    cosine_similarity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drift sentinel thresholds  (Section 3.5 Limitation 3.B)
# ---------------------------------------------------------------------------

CENTROID_DRIFT_ALERT_THRESHOLD    = 0.15   # cosine distance from registration centroid
BENCHMARK_SLOPE_ALERT_PER_WEEK   = -0.002  # minimum acceptable slope
UTILISATION_DECLINE_ALERT_PCT    = 0.20    # 20% decline over 90 days
DRIFT_RETRAINING_THRESHOLD_PP    = 5.0    # percentage points degradation


@dataclass
class ExpertDriftRecord:
    """Drift sentinel state for one expert.  Section 3.5 Limitation 3.B."""
    expert_id: str
    registration_centroid: np.ndarray        # Immutable snapshot at registration
    # Rolling 90-day benchmark scores (one entry per week)
    benchmark_scores: deque = field(default_factory=lambda: deque(maxlen=13))   # 13 weeks ≈ 90 days
    # Daily utilisation counts
    daily_utilisation: deque = field(default_factory=lambda: deque(maxlen=90))
    registration_benchmark: Optional[float] = None   # Benchmark at registration time

    def record_benchmark(self, score: float) -> None:
        self.benchmark_scores.append((time.time(), score))

    def record_daily_utilisation(self, count: int) -> None:
        self.daily_utilisation.append(count)

    def check_centroid_drift(self, current_centroid: np.ndarray) -> bool:
        """Return True (and log alert) if centroid has drifted beyond threshold."""
        dist = cosine_distance(current_centroid, self.registration_centroid)
        if dist > CENTROID_DRIFT_ALERT_THRESHOLD:
            logger.error(
                "DRIFT ALERT [expert=%s]: Centroid cosine distance from "
                "registration centroid = %.4f > threshold %.4f.  "
                "Human operator review required.  Section 3.5 Limitation 3.B.",
                self.expert_id[:8],
                dist,
                CENTROID_DRIFT_ALERT_THRESHOLD,
            )
            return True
        return False

    def check_benchmark_slope(self) -> bool:
        """Return True (and log alert) if benchmark slope < -0.002/week."""
        scores = list(self.benchmark_scores)
        if len(scores) < 4:
            return False  # Insufficient history
        # Linear regression over (week_index, score)
        weeks = np.arange(len(scores), dtype=float)
        values = np.array([s[1] for s in scores], dtype=float)
        coeffs = np.polyfit(weeks, values, 1)
        slope = float(coeffs[0])
        if slope < BENCHMARK_SLOPE_ALERT_PER_WEEK:
            logger.error(
                "DRIFT ALERT [expert=%s]: Rolling 90-day benchmark slope "
                "= %.4f/week < threshold %.4f/week.  "
                "Human operator review required.  Section 3.5.",
                self.expert_id[:8],
                slope,
                BENCHMARK_SLOPE_ALERT_PER_WEEK,
            )
            return True
        return False

    def check_utilisation_decline(self, competing_expert_added: bool = False) -> bool:
        """Return True (and log alert) if utilisation declined > 20% over 90 days
        without a new competing expert being added."""
        if competing_expert_added:
            return False  # Decline expected; not a drift signal
        util = list(self.daily_utilisation)
        if len(util) < 30:
            return False
        first_30 = sum(util[:30])
        last_30  = sum(util[-30:])
        if first_30 == 0:
            return False
        decline = (first_30 - last_30) / first_30
        if decline > UTILISATION_DECLINE_ALERT_PCT:
            logger.warning(
                "DRIFT ALERT [expert=%s]: Expert utilisation declined %.1f%% "
                "over 90 days without new competing experts.  "
                "Consider routing diagnosis.  Section 3.5.",
                self.expert_id[:8],
                decline * 100,
            )
            return True
        return False

    def needs_managed_retraining(self) -> bool:
        """Return True if confirmed drift exceeds 5pp on domain benchmarks."""
        if not self.benchmark_scores or self.registration_benchmark is None:
            return False
        latest_score = self.benchmark_scores[-1][1]
        drop_pp = (self.registration_benchmark - latest_score) * 100
        if drop_pp > DRIFT_RETRAINING_THRESHOLD_PP:
            logger.critical(
                "MANAGED RETRAINING REQUIRED [expert=%s]: Benchmark degradation "
                "= %.2f pp > threshold %.2f pp.  "
                "Full K-FAC Fisher reset and experience replay required.  "
                "Section 3.5 Limitation 3.B.",
                self.expert_id[:8],
                drop_pp,
                DRIFT_RETRAINING_THRESHOLD_PP,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Expert Centroid Registry
# ---------------------------------------------------------------------------

class ExpertCentroidRegistry:
    """
    Central registry of all expert centroids with:
      - Registration and lookup
      - Streaming EMA centroid updates (Section 1.4 / 9.2)
      - Multi-centroid support for broad experts (Section 1.4)
      - Streaming convergence verification (Section 9.2)
      - Expert drift sentinel monitoring (Section 3.5 Limitation 3.B)
    """

    def __init__(self) -> None:
        self._experts: Dict[str, ExpertCentroid] = {}
        self._drift_records: Dict[str, ExpertDriftRecord] = {}
        # Track how many documents have been streamed per expert
        # (for convergence verification gate)
        self._streamed_doc_counts: Dict[str, int] = defaultdict(int)
        # Snapshot at convergence checkpoint for verification
        self._batch_centroid_snapshots: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_expert(
        self,
        expert_id: str,
        training_corpus_embeddings: np.ndarray,  # (N, 768)
        registration_benchmark_score: Optional[float] = None,
        is_broad_expert: bool = False,
        n_centroids: int = 1,
    ) -> ExpertCentroid:
        """
        Register a new expert by computing its centroid(s) from training
        corpus embeddings.  Section 1.4 / Section 3 Step 5.

        For broad experts (Tier-3), multiple centroids are computed via
        K-means (k = n_centroids).

        Parameters
        ----------
        training_corpus_embeddings : np.ndarray
            Shape (N, 768).  Used to compute centroid and initial covariance.
        is_broad_expert : bool
            If True, compute multiple centroids.
        n_centroids : int
            Number of centroids for broad experts (default 1 for narrow).
        """
        if training_corpus_embeddings.ndim != 2 or \
                training_corpus_embeddings.shape[1] != MRLDimensions.FULL_DIM:
            raise ValueError(
                f"training_corpus_embeddings must be (N, {MRLDimensions.FULL_DIM}); "
                f"got {training_corpus_embeddings.shape}"
            )

        embs = training_corpus_embeddings.astype(np.float32)

        if is_broad_expert and n_centroids > 1:
            centroids_full, centroids_prefix = self._kmeans_centroids(embs, n_centroids)
        else:
            c = embs.mean(axis=0)
            centroids_full   = [c]
            centroids_prefix = [c[:MRLDimensions.COARSE_DIM]]

        # Initial EMA centroid (full-dimensional mean)
        ema_centroid    = embs.mean(axis=0)
        ema_centroid_64 = ema_centroid[:MRLDimensions.COARSE_DIM]

        # Diagonal inverse covariance for Mahalanobis OOD (Section 6.2)
        # Diagonal variance: var per dimension
        var = embs.var(axis=0)
        # Floor variance to prevent division by near-zero (numerical stability)
        var = np.maximum(var, 1e-6)
        diag_inv_cov = (1.0 / var).astype(np.float32)

        expert = ExpertCentroid(
            expert_id=expert_id,
            centroids_full=centroids_full,
            centroids_prefix=centroids_prefix,
            ema_centroid=ema_centroid,
            ema_centroid_64=ema_centroid_64,
            diag_inv_cov=diag_inv_cov,
        )
        self._experts[expert_id] = expert

        # Drift record with immutable registration snapshot
        self._drift_records[expert_id] = ExpertDriftRecord(
            expert_id=expert_id,
            registration_centroid=ema_centroid.copy(),
            registration_benchmark=registration_benchmark_score,
        )
        logger.info(
            "Registered expert '%s' with %d centroid(s), %d training docs.",
            expert_id[:8],
            len(centroids_full),
            len(embs),
        )
        return expert

    # ------------------------------------------------------------------
    # Streaming centroid update (Section 1.4 / 9.2)
    # ------------------------------------------------------------------

    def update_centroid_streaming(
        self,
        expert_id: str,
        new_document_embedding: np.ndarray,   # shape (768,)
        alpha: float = StreamingCentroidConstants.ALPHA,
    ) -> None:
        """
        Online EMA centroid update.  Called per document at fast timescale.

        centroid_new = (1 - α) · centroid_old + α · new_embedding

        Section 1.4 / 9.2.
        """
        if expert_id not in self._experts:
            raise KeyError(f"Expert '{expert_id}' not registered.")
        _validate_embedding(new_document_embedding, MRLDimensions.FULL_DIM)
        expert = self._experts[expert_id]
        expert.update_ema(new_document_embedding.astype(np.float32), alpha)
        self._streamed_doc_counts[expert_id] += 1

        # Check convergence at the spec-required 10,000-doc threshold
        if self._streamed_doc_counts[expert_id] == StreamingCentroidConstants.CONVERGENCE_DOCS:
            self._snapshot_for_convergence_check(expert_id)

        # Drift monitoring
        self._drift_records[expert_id].check_centroid_drift(expert.ema_centroid)

    def _snapshot_for_convergence_check(self, expert_id: str) -> None:
        """
        At exactly CONVERGENCE_DOCS documents, snapshot the streaming centroid
        for later comparison against batch-computed centroid.  Section 9.2.
        """
        self._batch_centroid_snapshots[expert_id] = \
            self._experts[expert_id].ema_centroid.copy()
        logger.debug(
            "Streaming convergence snapshot taken for expert '%s' at %d docs.",
            expert_id[:8],
            StreamingCentroidConstants.CONVERGENCE_DOCS,
        )

    def verify_streaming_convergence(
        self,
        expert_id: str,
        exact_batch_centroid: np.ndarray,
    ) -> bool:
        """
        Verify streaming centroid convergence.  Section 9.2.

        After streaming 10,000 document embeddings with α = 0.01, the
        streaming centroid must lie within cosine distance 0.01 of the
        exact batch-computed centroid.
        """
        if expert_id not in self._batch_centroid_snapshots:
            raise RuntimeError(
                f"No convergence snapshot found for expert '{expert_id}'.  "
                f"Stream at least {StreamingCentroidConstants.CONVERGENCE_DOCS} documents first."
            )
        streaming_centroid = self._batch_centroid_snapshots[expert_id]
        dist = cosine_distance(streaming_centroid, exact_batch_centroid)
        passed = dist <= StreamingCentroidConstants.CONVERGENCE_TOLERANCE
        if passed:
            logger.info(
                "Streaming centroid convergence VERIFIED for expert '%s': "
                "cosine distance = %.5f ≤ %.5f.",
                expert_id[:8],
                dist,
                StreamingCentroidConstants.CONVERGENCE_TOLERANCE,
            )
        else:
            logger.error(
                "Streaming centroid convergence FAILED for expert '%s': "
                "cosine distance = %.5f > %.5f.  "
                "Check EMA alpha and document ordering.  Section 9.2.",
                expert_id[:8],
                dist,
                StreamingCentroidConstants.CONVERGENCE_TOLERANCE,
            )
        return passed

    # ------------------------------------------------------------------
    # Drift monitoring (Section 3.5 Limitation 3.B)
    # ------------------------------------------------------------------

    def record_benchmark(self, expert_id: str, score: float) -> None:
        self._drift_records[expert_id].record_benchmark(score)
        self._drift_records[expert_id].check_benchmark_slope()
        self._drift_records[expert_id].needs_managed_retraining()

    def record_utilisation(self, expert_id: str, daily_count: int,
                           competing_expert_added: bool = False) -> None:
        self._drift_records[expert_id].record_daily_utilisation(daily_count)
        self._drift_records[expert_id].check_utilisation_decline(competing_expert_added)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, expert_id: str) -> ExpertCentroid:
        if expert_id not in self._experts:
            raise KeyError(f"Expert '{expert_id}' not found in registry.")
        return self._experts[expert_id]

    def all_experts(self) -> Dict[str, ExpertCentroid]:
        return dict(self._experts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kmeans_centroids(
        self,
        embeddings: np.ndarray,
        k: int,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Compute k centroids via K-means for broad experts.  Section 1.4.
        Uses a simple k-means implementation; production should use
        faiss.Kmeans for scale.
        """
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3)
        km.fit(embeddings)
        centroids_full   = [km.cluster_centers_[i].astype(np.float32) for i in range(k)]
        centroids_prefix = [c[:MRLDimensions.COARSE_DIM] for c in centroids_full]
        return centroids_full, centroids_prefix
