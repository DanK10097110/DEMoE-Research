"""
DEMoE Section 3 - Base Expert Model Types and Constants

Design decisions:
  - Tier framework with parameter bounds and hardware requirements
  - Latency budget enforcement with alerting
  - TIES merging interface (preferred over SLERP for multi-model merging)
  - Corpus assembly with hard synthetic data cap (token counter, not guideline)
  - EWC + K-FAC Fisher types
  - Experience replay buffer for catastrophic forgetting mitigation
  - Progressive layer unfreezing state machine
  - Catastrophic forgetting acceptance criterion (< 5pp degradation)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Tier constants  (Section 3.2 / 3.3)
# ---------------------------------------------------------------------------

class ExpertTier(Enum):
    TIER1_NARROW       = 1   # 1B–3B parameters, single narrow sub-field
    TIER2_DOMAIN       = 2   # 7B–13B parameters, broad professional domain
    TIER3_CROSSDOMAIN  = 3   # 30B–70B parameters, multi-domain cluster


@dataclass(frozen=True)
class TierSpec:
    """Hardware and budget specifications for one tier. Section 3.3 / 3.4."""
    tier:                  ExpertTier
    min_params_b:          float    # Billions
    max_params_b:          float
    min_training_gpus:     int
    min_vram_per_gpu_gb:   int
    requires_nvlink:       bool
    requires_infiniband:   bool
    phase1_budget_hours:   float    # Section 3.4 wall-clock budget
    # Phase 2 adds ≤ 20% to Phase-1 duration
    phase2_budget_hours:   float
    # 4 hours from training completion to routing availability (Step 5)
    registration_budget_hours: float = 4.0


TIER_SPECS: Dict[ExpertTier, TierSpec] = {
    ExpertTier.TIER1_NARROW: TierSpec(
        tier=ExpertTier.TIER1_NARROW,
        min_params_b=1.0, max_params_b=3.0,
        min_training_gpus=2,
        min_vram_per_gpu_gb=40,
        requires_nvlink=False,    # NVLink preferred but PCIe 4.0 acceptable
        requires_infiniband=False,
        phase1_budget_hours=48.0,
        phase2_budget_hours=48.0 * 1.20,
    ),
    ExpertTier.TIER2_DOMAIN: TierSpec(
        tier=ExpertTier.TIER2_DOMAIN,
        min_params_b=7.0, max_params_b=13.0,
        min_training_gpus=4,
        min_vram_per_gpu_gb=80,
        requires_nvlink=True,    # Required for Phase-1
        requires_infiniband=False,
        phase1_budget_hours=5 * 24.0,  # 5 days
        phase2_budget_hours=5 * 24.0 * 1.20,
    ),
    ExpertTier.TIER3_CROSSDOMAIN: TierSpec(
        tier=ExpertTier.TIER3_CROSSDOMAIN,
        min_params_b=30.0, max_params_b=70.0,
        min_training_gpus=8,
        min_vram_per_gpu_gb=80,
        requires_nvlink=True,
        requires_infiniband=True,   # HDR for multi-node
        phase1_budget_hours=14 * 24.0,  # 14 days
        phase2_budget_hours=14 * 24.0 * 1.20,
    ),
}

# Registration pipeline budget (Step 5)
REGISTRATION_BUDGET_HOURS: float = 4.0

# Catastrophic forgetting acceptance criterion. Section 9.3.
FORGETTING_ACCEPTANCE_MAX_PP: float = 5.0   # percentage points

# Corpus coherence floor for adaptive radius expansion. Section 3.4 Step 3.
CORPUS_COHERENCE_FLOOR: float = 0.65

# Synthetic data hard caps. Section 3.4 Step 3.
# "This cap is enforced by a per-run token counter that halts synthetic data
#  ingestion, not a guideline."
SYNTHETIC_DATA_CAP_BASE_MODEL: float = 0.30    # 30% of tokens
SYNTHETIC_DATA_CAP_ADAPTER:    float = 0.40    # 40% of tokens (Section 8.4)


# ---------------------------------------------------------------------------
# Initialisation strategy constants  (Section 3.4 Step 2)
# ---------------------------------------------------------------------------

class InitialisationStrategy(Enum):
    DOMAIN_ADJACENT_TRANSFER = "domain_adjacent_transfer"   # distance ≤ 0.25
    TIES_MERGE               = "ties_merge"                  # distance ≤ 0.40
    GENERAL_CHECKPOINT       = "general_checkpoint"          # beyond 0.40

TRANSFER_DISTANCE_THRESHOLD: float = 0.25
TIES_DISTANCE_THRESHOLD:     float = 0.40


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CorpusAssemblyResult:
    """
    Result of corpus assembly for a base expert.  Section 3.4 Step 3.

    The synthetic data cap is enforced by token counter during assembly.
    This object records the final token breakdown for audit.
    """
    real_token_count:      int
    curated_token_count:   int
    synthetic_token_count: int
    total_token_count:     int
    semantic_coherence:    float    # Must be ≥ 0.65 for corpus to be accepted
    sources:               List[str] = field(default_factory=list)
    cap_hit:               bool = False    # True if synthetic cap was triggered

    @property
    def synthetic_fraction(self) -> float:
        if self.total_token_count == 0:
            return 0.0
        return self.synthetic_token_count / self.total_token_count

    def validate(self, cap: float = SYNTHETIC_DATA_CAP_BASE_MODEL) -> Tuple[bool, str]:
        """Return (valid, reason)."""
        if self.semantic_coherence < CORPUS_COHERENCE_FLOOR:
            return False, (
                f"Semantic coherence {self.semantic_coherence:.3f} < "
                f"floor {CORPUS_COHERENCE_FLOOR}"
            )
        if self.synthetic_fraction > cap + 1e-6:
            # Should never occur if token counter is working correctly
            return False, (
                f"Synthetic fraction {self.synthetic_fraction:.3f} > cap {cap:.3f}. "
                "Token counter enforcement has failed — CRITICAL BUG."
            )
        return True, "OK"


@dataclass
class HardwareConfig:
    """
    Describes the available training hardware.  Used to validate that
    the minimum configuration for a tier is met before training begins.
    Section 3.3.
    """
    gpu_count:           int
    vram_per_gpu_gb:     int
    has_nvlink:          bool
    has_infiniband:      bool
    interconnect_label:  str = ""

    def meets_tier_requirements(self, tier: ExpertTier) -> Tuple[bool, str]:
        """Return (meets, reason)."""
        spec = TIER_SPECS[tier]
        if self.gpu_count < spec.min_training_gpus:
            return False, (
                f"Need ≥ {spec.min_training_gpus} GPUs; "
                f"have {self.gpu_count}."
            )
        if self.vram_per_gpu_gb < spec.min_vram_per_gpu_gb:
            return False, (
                f"Need ≥ {spec.min_vram_per_gpu_gb} GB VRAM/GPU; "
                f"have {self.vram_per_gpu_gb} GB."
            )
        if spec.requires_nvlink and not self.has_nvlink:
            return False, (
                f"Tier {tier.name} requires NVLink. "
                "K-FAC on Tier-3 models without NVLink will exceed the weekly "
                "update budget and must not be attempted."
            )
        if spec.requires_infiniband and not self.has_infiniband:
            return False, (
                f"Tier {tier.name} requires InfiniBand HDR for multi-node training."
            )
        return True, "OK"


@dataclass
class KFACFactors:
    """
    Per-layer K-FAC Fisher approximation: F_layer ≈ A_layer ⊗ G_layer.
    Section 9.3.

    A_layer (d_in × d_in): input activation covariance.
    G_layer (d_out × d_out): gradient covariance.

    These are updated as exponential moving averages during training.
    They MUST be reset before each base model update to ensure Fisher accuracy.
    Carrying accumulated factors across updates degrades approximation quality.
    """
    layer_name:  str
    A:           np.ndarray    # (d_in, d_in)
    G:           np.ndarray    # (d_out, d_out)
    update_count: int = 0

    def update_ema(
        self,
        new_A: np.ndarray,
        new_G: np.ndarray,
        decay: float = 0.95,
    ) -> None:
        """Online EMA update. Section 9.3."""
        self.A = decay * self.A + (1 - decay) * new_A
        self.G = decay * self.G + (1 - decay) * new_G
        self.update_count += 1

    def ewc_penalty(
        self,
        current_layer_weights: np.ndarray,
        reference_weights:     np.ndarray,
    ) -> float:
        """
        Compute per-layer EWC penalty: (A ⊗ G) ⊙ (θ - θ*)².
        Section 9.3.

        Uses the Kronecker product property: for a weight matrix W,
        vec(W)^T (A ⊗ G) vec(W) = tr(A W^T G W).

        For efficient computation with large weight matrices, we compute
        the trace form rather than materialising the full Kronecker product.
        """
        diff = current_layer_weights - reference_weights
        # tr(A · diff^T · G · diff)
        return float(np.trace(self.A @ diff.T @ self.G @ diff))


@dataclass
class ExperienceReplayBuffer:
    """
    Experience replay buffer for catastrophic forgetting mitigation.
    Section 3.5 Limitation 3.A.

    Stores a representative sample of training examples from prior updates.
    Used during medium-timescale base model updates to prevent forgetting.
    """
    max_size:     int
    # Stores (input_token_ids, target_token_ids, weight) tuples
    buffer:       List[Tuple] = field(default_factory=list)
    total_added:  int = 0

    def add(self, example: Tuple, weight: float = 1.0) -> None:
        """Add an example, evicting oldest if at capacity."""
        self.buffer.append((example, weight))
        self.total_added += 1
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # FIFO; reservoir sampling preferred for large buffers

    def sample(self, n: int) -> List[Tuple]:
        """Sample n examples from the buffer."""
        if not self.buffer:
            return []
        indices = np.random.choice(len(self.buffer), size=min(n, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class TrainingPhaseResult:
    """Records outcome of one training phase for latency budget monitoring."""
    phase:              str            # "phase1", "phase2", "map_stage", "blob_stage"
    tier:               ExpertTier
    duration_hours:     float
    budget_hours:       float
    final_val_loss:     float
    uncertainty_passed: bool           # Did uncertainty drop below threshold?
    forgetting_pp:      Optional[float] = None  # Post-update forgetting %

    @property
    def over_budget(self) -> bool:
        return self.duration_hours > self.budget_hours

    @property
    def acceptable_forgetting(self) -> bool:
        if self.forgetting_pp is None:
            return True
        return self.forgetting_pp <= FORGETTING_ACCEPTANCE_MAX_PP


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class HardwareInsufficientError(Exception):
    """Raised when hardware config does not meet the tier's minimum requirements."""


class CorpusAssemblyError(Exception):
    """Raised when corpus assembly fails quality/coherence checks."""


class ForgettingThresholdExceededError(Exception):
    """
    Raised when sub-domain accuracy degrades > 5pp after a base expert update.
    Triggers rollback and λ increase.  Section 9.3.
    """


class SyntheticDataCapViolationError(Exception):
    """
    Raised when the synthetic data token counter would be exceeded.
    This should never reach production — it indicates a bug in the
    per-run token counter.  Section 3.4 Step 3 / Section 8.4.
    """
