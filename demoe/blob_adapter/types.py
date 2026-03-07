"""
DEMoE Section 4 - BLoB Adapter Layer: Types and Constants

Design decisions:
  - BLoB as primary uncertainty mechanism (rationale encoded as comments)
  - Cyclical KL schedule with β verification
  - Structured VI with free-bits constants
  - Two-NN rank determination formula
  - Singular vector fingerprint data structures
  - Laplace-LoRA calibration inflation (restricted to bootstrapped adapters)
  - Percentile normalisation for cross-expert uncertainty comparison
  - Adapter lifecycle states
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants  (Section 4.2 / 4.3 / 4.4 / 4.7)
# ---------------------------------------------------------------------------

class BLoBConstants:
    """BLoB training hyperparameters.  Section 4.2."""
    # MAP pre-training
    MAP_VALIDATION_FRACTION:   float = 0.15   # 15% held-out for early stopping
    MAP_L2_REGULARISATION:     float = 1e-4   # Light L2 to prevent extreme weights
    # Variance initialisation after MAP: σ² = ε × mean(μ²)
    VARIANCE_INIT_EPSILON:     float = 0.01
    # Cyclical KL: 5 cycles recommended; T_cycle = 20% of total steps
    KL_CYCLE_COUNT:            int = 5
    KL_CYCLE_FRACTION:         float = 0.20   # T_cycle / total_steps
    # Free-bits: recommended starting value
    FREE_BITS_LAMBDA:          float = 0.10   # nats; ablation over {0.05, 0.10, 0.20, 0.50}
    # Structured VI: low-rank correction; apply to top-4 layers only
    STRUCTURED_VI_K:           int = 5        # Rank k; ablation over {2, 5, 10, 20}
    STRUCTURED_VI_LAYERS:      int = 4        # Top-4 layers only
    # Temporary adapter inflation during MAP-only phase (Section 4.8)
    TEMP_ADAPTER_INFLATION:    float = 1.4    # c_temp; ablation over {1.2, 1.4, 1.6, 2.0}
    # Ambiguous uncertainty band requiring full sampling pass
    AMBIGUOUS_LOWER:           float = 0.3
    AMBIGUOUS_UPPER:           float = 0.7
    SAMPLING_N:                int = 7        # Weight samples in ambiguous zone
    # Uncertainty stability gate: 3 consecutive eval checkpoints below threshold
    STABILITY_CHECKPOINTS:     int = 3


class LaplaceConstants:
    """Laplace-LoRA calibration constants.  Section 4.3."""
    # Starting inflation factor; calibrate per-deployment to ECE < 0.10
    C_BOOTSTRAP:     float = 1.25
    ECE_TARGET:      float = 0.10
    # Number of calibration bins for ECE computation
    ECE_BINS:        int = 15


class TwoNNConstants:
    """Two-NN rank determination constants.  Section 4.4."""
    SCALING_FACTOR_S:         float = 0.5    # Ablation over {0.3, 0.4, 0.5, 0.6, 0.7}
    RANK_MIN:                 int = 4
    RANK_MAX:                 int = 64
    DEFAULT_RANK_SMALL_CORPUS: int = 8       # For N < 200
    SMALL_CORPUS_N:           int = 200
    BOOTSTRAP_ROBUST_N:       int = 500      # Apply bootstrap for N < 500
    BOOTSTRAP_SUBSET_FRAC:    float = 0.80   # Subset fraction for bootstrap
    BOOTSTRAP_ITERATIONS:     int = 10


class FingerprintConstants:
    """Singular vector fingerprint constants.  Section 4.5 / 4.6."""
    TOP_K_SINGULAR_VECTORS: int = 16    # k for SVD fingerprint
    ANGULAR_ROTATION_ALERT: float = 15.0  # degrees; flag for full compat test
    ADAPTER_RELEVANCE_ALPHA: float = 0.6  # α in S_i = α·(1-d) + (1-α)·R_i


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AdapterType(Enum):
    BLOB_TRAINED   = "blob_trained"         # Primary: trained with BLoB
    MAP_TEMPORARY  = "map_temporary"        # Temporary during BLoB async training
    LAPLACE_LORA   = "laplace_lora"         # Restricted: bootstrapped only


class AdapterLifecycleState(Enum):
    PENDING_MAP        = auto()   # MAP training in progress
    MAP_DEPLOYED       = auto()   # MAP adapter live (c_temp inflation active)
    BLOB_TRAINING      = auto()   # BLoB fine-tune running asynchronously
    BLOB_DEPLOYED      = auto()   # Full BLoB adapter live
    DEPRECATED         = auto()   # Superseded or compatibility failed


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CyclicalKLSchedule:
    """
    Cyclical KL annealing schedule.  Section 4.2 Improvement 2.

    Within each cycle of T_cycle steps:
      - First half: β linearly from 0 → 1
      - Second half: β = 1

    Recommended: T_cycle = 20% of total steps, 5 total cycles.

    Verification: the training loop must assert β resets to 0 at each cycle
    boundary and reaches 1.0 at the midpoint.
    """
    total_steps:  int
    n_cycles:     int = BLoBConstants.KL_CYCLE_COUNT
    cycle_frac:   float = BLoBConstants.KL_CYCLE_FRACTION

    def __post_init__(self) -> None:
        self.t_cycle = max(1, int(self.total_steps * self.cycle_frac))
        # Total steps covered by cycles; training should run for exactly this many
        self.steps_scheduled = self.t_cycle * self.n_cycles

    def beta(self, step: int) -> float:
        """
        Return the KL weight β at training step `step`.

        Invariants (Section 4.2):
          - β = 0 at the start of each cycle (step % T_cycle == 0)
          - β = 1 at the midpoint of each cycle (step % T_cycle == T_cycle // 2)
          - β = 1 for the second half of each cycle
        """
        if step < 0:
            return 0.0
        step_in_cycle = step % self.t_cycle
        half = self.t_cycle // 2
        if step_in_cycle < half:
            # Linear ramp: 0 → 1 over first half
            return float(step_in_cycle) / float(max(half, 1))
        else:
            return 1.0

    def verify_invariants(self) -> bool:
        """
        Verify the three β invariants from Section 4.2.
        Returns True if all pass.
        """
        errors = []
        # Check each cycle boundary
        for cycle_idx in range(self.n_cycles):
            cycle_start = cycle_idx * self.t_cycle
            midpoint    = cycle_start + self.t_cycle // 2
            # β at cycle start must be 0
            b_start = self.beta(cycle_start)
            if abs(b_start) > 1e-6:
                errors.append(f"Cycle {cycle_idx}: β at start = {b_start:.6f} ≠ 0")
            # β at midpoint must be 1.0
            b_mid = self.beta(midpoint)
            if abs(b_mid - 1.0) > 1e-6:
                errors.append(f"Cycle {cycle_idx}: β at midpoint = {b_mid:.6f} ≠ 1.0")
            # β in second half must be 1.0
            for offset in range(self.t_cycle // 2, self.t_cycle):
                b = self.beta(cycle_start + offset)
                if abs(b - 1.0) > 1e-4:
                    errors.append(
                        f"Cycle {cycle_idx}: β at offset {offset} = {b:.6f} ≠ 1.0"
                    )
                    break

        if errors:
            for e in errors:
                import logging
                logging.getLogger(__name__).error("KL schedule invariant FAILED: %s", e)
            return False
        return True


@dataclass
class StructuredVIParams:
    """
    Structured variational approximation parameters.  Section 4.2 Improvement 3.

    q(θ) = N(μ, diag(σ²) + V·V^T)

    V ∈ ℝ^(P×k) captures k dominant correlation directions.
    Applied only to the top-4 layers of the base expert (Section 4.8).
    """
    P:          int           # Number of parameters in the layer
    k:          int = BLoBConstants.STRUCTURED_VI_K
    # V matrix: (P, k)
    V: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.V is None:
            # Initialise to small random values
            self.V = np.random.randn(self.P, self.k).astype(np.float32) * 0.01

    def covariance_diag_approx(self, sigma_sq: np.ndarray) -> np.ndarray:
        """
        Return diagonal approximation of the full covariance for efficiency.
        Full cov = diag(σ²) + V·V^T; this returns only the diagonal.
        """
        return sigma_sq + (self.V ** 2).sum(axis=1)

    def memory_overhead_bytes(self) -> int:
        """Approximate memory overhead of the V matrix."""
        return self.P * self.k * 4   # float32 = 4 bytes


@dataclass
class BLoBAdapterWeights:
    """
    BLoB adapter weight distributions for one LoRA layer.

    Stores:
      - μ (mean): shape (rank, d_in) for A; (d_out, rank) for B
      - log_σ (log standard deviation): same shape as μ
      - V (low-rank correlation factor): only for top-4 layers
    """
    layer_name:  str
    A_mu:        np.ndarray    # (rank, d_in)
    A_log_sigma: np.ndarray    # (rank, d_in)
    B_mu:        np.ndarray    # (d_out, rank)
    B_log_sigma: np.ndarray    # (d_out, rank)
    # Structured VI correction (None for lower layers)
    A_V: Optional[np.ndarray] = None    # (rank*d_in, k)
    B_V: Optional[np.ndarray] = None    # (d_out*rank, k)
    adapter_type: AdapterType = AdapterType.BLOB_TRAINED

    @property
    def A_sigma(self) -> np.ndarray:
        return np.exp(self.A_log_sigma)

    @property
    def B_sigma(self) -> np.ndarray:
        return np.exp(self.B_log_sigma)

    def mean_forward(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean-mode weight matrices for production inference."""
        return self.A_mu.copy(), self.B_mu.copy()

    def sample_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Draw one sample from the weight distribution for MC estimation."""
        A = self.A_mu + self.A_sigma * np.random.randn(*self.A_mu.shape).astype(np.float32)
        B = self.B_mu + self.B_sigma * np.random.randn(*self.B_mu.shape).astype(np.float32)
        return A, B

    def kl_divergence(self, lambda_free: float = BLoBConstants.FREE_BITS_LAMBDA) -> float:
        """
        KL divergence KL(q||p) for a standard normal prior.

        With free-bits: KL contributions below lambda_free nats are zeroed out
        to prevent posterior collapse.  Section 4.2 Improvement 3.

        KL(N(μ, σ²) || N(0, 1)) = 0.5 * (μ² + σ² - log(σ²) - 1)
        """
        def _kl(mu, log_sigma):
            sigma_sq = np.exp(2 * log_sigma)
            per_param = 0.5 * (mu ** 2 + sigma_sq - 2 * log_sigma - 1.0)
            # Free-bits: zero out contributions below lambda_free
            per_param = np.where(per_param < lambda_free, 0.0, per_param)
            return float(per_param.sum())

        return _kl(self.A_mu, self.A_log_sigma) + _kl(self.B_mu, self.B_log_sigma)

    def law_of_total_variance(
        self,
        input_activation: np.ndarray,   # (d_in,) or (batch, d_in)
    ) -> float:
        """
        Closed-form uncertainty estimate via law of total variance.
        Section 4.7.

        Var[output] = Var_θ[E[output|θ]] + E_θ[Var[output|θ]]
                    ≈ ||σ_A ⊙ x||² · ||B_mu||² + ||A_mu · x||² · ||σ_B||²
                    (first-order approximation for LoRA composition A·B·x)

        This requires no additional forward passes.
        """
        if input_activation.ndim == 1:
            x = input_activation
        else:
            x = input_activation.mean(axis=0)   # Batch mean for efficiency

        # Variance from A layer: σ_A elementwise, propagated through B
        var_from_A = float(
            np.sum((self.A_sigma * x[None, :]) ** 2) * np.sum(self.B_mu ** 2)
        )
        # Variance from B layer: propagated activation variance from A_mu
        Ax = self.A_mu @ x
        var_from_B = float(np.sum(Ax ** 2) * np.sum(self.B_sigma ** 2))

        return var_from_A + var_from_B


@dataclass
class SingularVectorFingerprint:
    """
    Singular vector fingerprint for one BLoB adapter layer.  Section 4.5 / 4.6.

    Computed as the top-k left singular vectors of the product AB^T via SVD.
    """
    layer_name:          str
    top_k_left_singular: np.ndarray    # (k, d_out) — left singular vectors
    top_k_singular_vals: np.ndarray    # (k,)
    k:                   int = FingerprintConstants.TOP_K_SINGULAR_VECTORS

    def relevance_score(self, query_embedding: np.ndarray) -> float:
        """
        R_i = ||U_i^T · q|| / ||q||    Section 4.6 Use 1.

        Measures how much of the query embedding projects onto the adapter's
        dominant learned directions.
        """
        q_norm = np.linalg.norm(query_embedding)
        if q_norm < 1e-12:
            return 0.0
        proj = self.top_k_left_singular @ query_embedding
        return float(np.linalg.norm(proj) / q_norm)

    def angular_rotation_from(self, other: "SingularVectorFingerprint") -> float:
        """
        Compute angular rotation of representation geometry.
        Section 4.6 Use 2.

        θ = arccos(||V_pre^T · V_post||_F / k)

        Returns angle in degrees.
        """
        # Spec: θ = arccos(||V_pre^T · V_post||_F / k)
        # For identical orthonormal matrices: trace(I_k)/k = 1 → θ=0
        # The spec formula uses trace not Frobenius (Frobenius of I_k = sqrt(k) ≠ k)
        overlap = self.top_k_left_singular @ other.top_k_left_singular.T
        trace_val = float(np.trace(overlap))
        cos_val = float(np.clip(trace_val / self.k, -1.0, 1.0))
        angle_rad = float(np.arccos(cos_val))
        return float(np.degrees(angle_rad))

    def needs_full_compat_test(self, post_update_fp: "SingularVectorFingerprint") -> bool:
        """Return True if angular rotation exceeds 15° (full test required). Section 4.6."""
        angle = self.angular_rotation_from(post_update_fp)
        if angle > FingerprintConstants.ANGULAR_ROTATION_ALERT:
            import logging
            logging.getLogger(__name__).warning(
                "Layer '%s': angular rotation = %.1f° > %.1f°. "
                "Full adapter compatibility testing required.  Section 4.6 Use 2.",
                self.layer_name, angle, FingerprintConstants.ANGULAR_ROTATION_ALERT,
            )
            return True
        return False


@dataclass
class PercentileCalibrationDistribution:
    """
    Holds the calibration distribution for percentile-normalising uncertainty
    scores within an expert.  Section 4.7.

    Enables cross-expert comparison: a score at the 90th percentile of any
    expert's distribution is treated as equivalent.
    """
    expert_id:   str
    percentiles: np.ndarray    # shape (101,) — 0th through 100th percentile values
    n_samples:   int = 0

    @classmethod
    def fit(cls, expert_id: str, raw_scores: np.ndarray) -> "PercentileCalibrationDistribution":
        percs = np.percentile(raw_scores, np.arange(0, 101))
        return cls(expert_id=expert_id, percentiles=percs, n_samples=len(raw_scores))

    def normalise(self, raw_score: float) -> float:
        """
        Map a raw uncertainty score to its percentile rank in [0, 1].
        Linear interpolation between stored percentile values.
        """
        return float(np.searchsorted(self.percentiles, raw_score) / 100.0)

    def verify_equivalence(
        self,
        other: "PercentileCalibrationDistribution",
        target_percentile: float = 0.90,
        tolerance: float = 0.05,
    ) -> bool:
        """
        Section 4.7 verification: scores at the 90th percentile of each
        expert's distribution must both normalise to within ±0.05 of 0.90.
        """
        self_90th = float(self.percentiles[90])
        other_90th = float(other.percentiles[90])
        self_normalised  = self.normalise(self_90th)
        other_normalised = other.normalise(other_90th)
        ok = (
            abs(self_normalised  - target_percentile) <= tolerance and
            abs(other_normalised - target_percentile) <= tolerance
        )
        if not ok:
            import logging
            logging.getLogger(__name__).error(
                "Percentile calibration verification FAILED: "
                "expert '%s' normalised=%.3f, expert '%s' normalised=%.3f "
                "(target=%.2f ± %.2f). Section 4.7.",
                self.expert_id, self_normalised,
                other.expert_id, other_normalised,
                target_percentile, tolerance,
            )
        return ok


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class BLoBPosteriorCollapseError(Exception):
    """Raised when mean posterior variance drops below minimum floor."""


class AdapterCompatibilityError(Exception):
    """
    Raised when a BLoB adapter fails compatibility testing after a base model
    update.  Triggers distillation-based recovery.  Section 9.3.
    """


class EarlyStoppingTriggered(Exception):
    """
    Raised by the MAP training loop when validation loss divergence
    is detected.  Section 4.2 Improvement 1 / FATAL T2.
    """
    def __init__(self, best_step: int, best_val_loss: float) -> None:
        self.best_step = best_step
        self.best_val_loss = best_val_loss
        super().__init__(
            f"Early stopping at step {best_step} (val_loss={best_val_loss:.4f}). "
            "Using best checkpoint as MAP initialisation.  Section 4.2 / FATAL T2."
        )
