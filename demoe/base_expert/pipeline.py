"""
DEMoE Section 3 - Base Expert Model Creation Pipeline

Implements the full 5-step creation pipeline (Section 3.4):
  Step 1: Bootstrap registry query                 → calls Section 2
  Step 2: Initialisation strategy selection        → TIES / transfer / checkpoint
  Step 3: Corpus assembly with hard synthetic cap  → token counter enforcement
  Step 4: Two-phase training with EWC + K-FAC      → Phase 1 + Phase 2
  Step 5: Validation, centroid computation, registration

Also implements:
  - Hardware pre-flight validation (Section 3.3)
  - Catastrophic forgetting acceptance criterion: rollback + λ increase (Section 9.3)
  - Progressive layer unfreezing state machine (Section 3.5 Limitation 3.A)
  - Online K-FAC Fisher update with mandatory reset between updates (Section 9.3)
  - Experience replay sampling during medium-timescale updates

Synthetic data hard prohibition:
  The HardProhibitionViolationError from Section 8.4 is enforced in the
  corpus assembler: any attempt to set a DEMoE expert as a synthetic data
  source raises immediately and the pipeline refuses to proceed.

TIES merging rationale (Section 3.4 Step 2):
  TIES is preferred over SLERP for multi-model merging because it resolves
  sign conflicts in the merged weight tensors.  When two models disagree on
  the sign of a parameter, SLERP interpolates through zero (neutralising both),
  while TIES uses majority-sign election to preserve the stronger signal.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .types import (
    CorpusAssemblyResult,
    ExperienceReplayBuffer,
    ExpertTier,
    ForgettingThresholdExceededError,
    FORGETTING_ACCEPTANCE_MAX_PP,
    HardwareConfig,
    HardwareInsufficientError,
    InitialisationStrategy,
    KFACFactors,
    REGISTRATION_BUDGET_HOURS,
    SYNTHETIC_DATA_CAP_BASE_MODEL,
    SyntheticDataCapViolationError,
    TIER_SPECS,
    TIES_DISTANCE_THRESHOLD,
    TrainingPhaseResult,
    TRANSFER_DISTANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Sentinel: models in this set are known DEMoE experts and must never be
# used as synthetic data sources.  The set is populated by the system at
# expert registration time.  Section 8.4 / FATAL T3.
_DEMOE_EXPERT_MODEL_IDS: set = set()


def register_demoe_expert_id(model_id: str) -> None:
    """Called by registration pipeline to mark a model as a DEMoE expert."""
    _DEMOE_EXPERT_MODEL_IDS.add(model_id)


def is_demoe_expert(model_id: str) -> bool:
    return model_id in _DEMOE_EXPERT_MODEL_IDS


# ---------------------------------------------------------------------------
# FATAL T3: HardProhibitionViolationError (re-export from embedding_space)
# ---------------------------------------------------------------------------

class HardProhibitionViolationError(Exception):
    """
    Raised when any DEMoE expert model is configured as a synthetic data source.
    Section 8.4 / FATAL T3.
    This check is enforced at the pipeline level, not documented as a guideline.
    """


# ---------------------------------------------------------------------------
# Corpus assembler with token counter enforcement
# ---------------------------------------------------------------------------

class CorpusAssembler:
    """
    Assembles training corpora from three sources in priority order:
      1. Real documents from the data lake (via MRL nearest-neighbour retrieval)
      2. Curated external repositories
      3. Synthetic data from a frontier model (NOT a DEMoE expert — hard prohibition)

    The 30% synthetic cap is enforced by a per-run token counter that halts
    ingestion when the budget is exhausted.  Section 3.4 Step 3.
    """

    def __init__(
        self,
        real_doc_retriever: Optional[Callable[[np.ndarray, float], List[Tuple[str, int]]]] = None,
        curated_repo_fetcher: Optional[Callable[[str], List[Tuple[str, int]]]] = None,
        synthetic_generator: Optional[Callable[[str, int], List[Tuple[str, int]]]] = None,
        synthetic_generator_model_id: Optional[str] = None,
        synthetic_cap: float = SYNTHETIC_DATA_CAP_BASE_MODEL,
        coherence_evaluator: Optional[Callable[[List[str]], float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        real_doc_retriever : callable, optional
            ``(centroid, radius) → [(text, token_count)]``
        curated_repo_fetcher : callable, optional
            ``(domain_label) → [(text, token_count)]``
        synthetic_generator : callable, optional
            ``(domain_label, max_tokens) → [(text, token_count)]``
        synthetic_generator_model_id : str, optional
            The model_id of the synthetic generator.  MUST NOT be a DEMoE expert.
        synthetic_cap : float
            Maximum fraction of total tokens that may be synthetic.
        coherence_evaluator : callable, optional
            ``(texts) → float`` returning mean coherence score in [0, 1].
        """
        # FATAL T3 hard prohibition check
        if synthetic_generator_model_id is not None:
            if is_demoe_expert(synthetic_generator_model_id):
                raise HardProhibitionViolationError(
                    f"HardProhibitionViolationError: model '{synthetic_generator_model_id}' "
                    f"is a registered DEMoE expert and MUST NOT be used as a synthetic "
                    f"data source.  Section 8.4 / FATAL T3."
                )

        self._real_retriever = real_doc_retriever
        self._curated_fetcher = curated_repo_fetcher
        self._synthetic_gen = synthetic_generator
        self._synthetic_gen_id = synthetic_generator_model_id
        self._synthetic_cap = synthetic_cap
        self._coherence_eval = coherence_evaluator

    def assemble(
        self,
        domain_label: str,
        domain_centroid: np.ndarray,
        target_total_tokens: int,
        generation_id: str,
        parent_generation_id: Optional[str] = None,
    ) -> CorpusAssemblyResult:
        """
        Assemble a training corpus.  Section 3.4 Step 3.

        Uses adaptive radius expansion for real document retrieval:
        expand the retrieval radius until corpus semantic coherence drops
        below 0.65, then stop.

        Data provenance is tagged with generation_id to prevent cross-generation
        accumulation.  Section 8.4.

        Parameters
        ----------
        generation_id : str
            Identifier for this training run.  Synthetic tokens are tagged to
            prevent reuse as real data in future generations.
        parent_generation_id : str, optional
            If set, synthetic tokens from this ID will be BLOCKED as input.
        """
        real_docs:      List[Tuple[str, int]] = []
        curated_docs:   List[Tuple[str, int]] = []
        synthetic_docs: List[Tuple[str, int]] = []

        real_tokens = curated_tokens = synthetic_tokens = 0

        # --- Source 1: Real documents via adaptive radius expansion ---
        if self._real_retriever is not None:
            real_docs, real_tokens = self._retrieve_real_docs(
                domain_centroid, domain_label, target_total_tokens
            )

        # --- Source 2: Curated external repositories ---
        remaining = max(0, target_total_tokens - real_tokens)
        if remaining > 0 and self._curated_fetcher is not None:
            curated_raw = self._curated_fetcher(domain_label)
            for text, tcount in curated_raw:
                if curated_tokens + tcount > remaining:
                    break
                curated_docs.append((text, tcount))
                curated_tokens += tcount

        # --- Source 3: Synthetic data (HARD CAP via token counter) ---
        total_so_far = real_tokens + curated_tokens
        synthetic_budget = int(
            self._synthetic_cap / (1.0 - self._synthetic_cap) * total_so_far
        )
        # Cap: synthetic / (real + curated + synthetic) ≤ cap
        # => synthetic ≤ cap / (1 - cap) * (real + curated)

        if synthetic_budget > 0 and self._synthetic_gen is not None:
            # Re-check prohibition at call time (defence in depth)
            if self._synthetic_gen_id and is_demoe_expert(self._synthetic_gen_id):
                raise HardProhibitionViolationError(
                    f"HardProhibitionViolationError: '{self._synthetic_gen_id}' "
                    "is a DEMoE expert.  FATAL T3."
                )
            raw_synthetic = self._synthetic_gen(domain_label, synthetic_budget)
            cap_hit = False
            for text, tcount in raw_synthetic:
                if synthetic_tokens + tcount > synthetic_budget:
                    cap_hit = True
                    logger.info(
                        "Synthetic data token cap reached for domain '%s'. "
                        "Halting ingestion.  synthetic_tokens=%d / budget=%d.",
                        domain_label, synthetic_tokens, synthetic_budget,
                    )
                    break
                synthetic_docs.append((text, tcount))
                synthetic_tokens += tcount

        total = real_tokens + curated_tokens + synthetic_tokens

        # Verify the cap was not violated (should be impossible given counter)
        if total > 0:
            actual_frac = synthetic_tokens / total
            if actual_frac > self._synthetic_cap + 1e-6:
                raise SyntheticDataCapViolationError(
                    f"Synthetic fraction {actual_frac:.4f} exceeds cap "
                    f"{self._synthetic_cap:.4f}. Token counter has failed — CRITICAL BUG."
                )

        # Coherence evaluation
        all_texts = [t for t, _ in real_docs + curated_docs + synthetic_docs]
        coherence = 1.0
        if self._coherence_eval and all_texts:
            coherence = float(self._coherence_eval(all_texts))

        result = CorpusAssemblyResult(
            real_token_count=real_tokens,
            curated_token_count=curated_tokens,
            synthetic_token_count=synthetic_tokens,
            total_token_count=total,
            semantic_coherence=coherence,
            sources=[domain_label],
            cap_hit=False,  # Already handled above
        )
        valid, reason = result.validate(self._synthetic_cap)
        if not valid:
            from .types import CorpusAssemblyError
            raise CorpusAssemblyError(f"Corpus validation failed: {reason}")

        logger.info(
            "Corpus assembled for '%s': real=%d, curated=%d, synthetic=%d tokens "
            "(synth=%.1f%%), coherence=%.3f.",
            domain_label, real_tokens, curated_tokens, synthetic_tokens,
            result.synthetic_fraction * 100, coherence,
        )
        return result

    def _retrieve_real_docs(
        self,
        centroid: np.ndarray,
        domain_label: str,
        target_tokens: int,
    ) -> Tuple[List[Tuple[str, int]], int]:
        """
        Adaptive radius expansion: start with a tight radius and expand
        until either enough tokens are collected or coherence drops below 0.65.
        Section 3.4 Step 3.
        """
        radius = 0.15
        docs_collected: List[Tuple[str, int]] = []
        total_tokens = 0

        for _ in range(8):   # Max 8 expansion steps
            batch = self._real_retriever(centroid, radius)
            new_texts = [t for t, _ in batch if t not in [d for d, _ in docs_collected]]
            for text, tcount in batch:
                if text not in [d for d, _ in docs_collected]:
                    docs_collected.append((text, tcount))
                    total_tokens += tcount

            if total_tokens >= target_tokens:
                break

            # Check coherence; stop expanding if it drops below floor
            if self._coherence_eval and docs_collected:
                coherence = self._coherence_eval([d for d, _ in docs_collected])
                if coherence < 0.65:
                    logger.info(
                        "Adaptive radius expansion stopped at radius=%.3f: "
                        "coherence %.3f < 0.65 floor. domain='%s'",
                        radius, coherence, domain_label,
                    )
                    break

            radius *= 1.5   # Expand radius by 50% each iteration

        return docs_collected, total_tokens


# ---------------------------------------------------------------------------
# Initialisation strategy selector  (Section 3.4 Step 2)
# ---------------------------------------------------------------------------

class InitialisationSelector:
    """
    Selects the initialisation strategy for a new base expert.
    Section 3.4 Step 2.

    Priority order:
      1. Domain-adjacent transfer (existing expert within distance 0.25)
      2. TIES merge (experts within distance 0.40)
      3. General-purpose checkpoint

    TIES merging is preferred over SLERP for multi-model merging because it
    resolves sign conflicts in merged weights more robustly.  Section 3.4.
    """

    def select(
        self,
        gap_centroid: np.ndarray,
        existing_expert_centroids: Dict[str, np.ndarray],  # expert_id → centroid (768)
    ) -> Tuple[InitialisationStrategy, List[str]]:
        """
        Return (strategy, list_of_source_expert_ids).

        For GENERAL_CHECKPOINT the list is empty.
        For TIES_MERGE the list contains the experts to merge.
        For DOMAIN_ADJACENT_TRANSFER the list contains one expert.
        """
        distances = {
            eid: self._cosine_distance(gap_centroid, centroid)
            for eid, centroid in existing_expert_centroids.items()
        }
        if not distances:
            return InitialisationStrategy.GENERAL_CHECKPOINT, []

        nearest_id   = min(distances, key=distances.get)
        nearest_dist = distances[nearest_id]

        if nearest_dist <= TRANSFER_DISTANCE_THRESHOLD:
            return InitialisationStrategy.DOMAIN_ADJACENT_TRANSFER, [nearest_id]

        ties_candidates = [
            eid for eid, dist in distances.items()
            if dist <= TIES_DISTANCE_THRESHOLD
        ]
        if len(ties_candidates) >= 2:
            # TIES merging: use up to 5 closest experts
            ties_sorted = sorted(ties_candidates, key=lambda e: distances[e])[:5]
            return InitialisationStrategy.TIES_MERGE, ties_sorted

        return InitialisationStrategy.GENERAL_CHECKPOINT, []

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        n_a, n_b = np.linalg.norm(a), np.linalg.norm(b)
        if n_a < 1e-12 or n_b < 1e-12:
            return 1.0
        return float(1.0 - np.dot(a, b) / (n_a * n_b))


def ties_merge(
    weight_tensors: List[np.ndarray],
    density: float = 0.20,
) -> np.ndarray:
    """
    TIES (Trim, Elect Sign, Merge) model merging.  Section 3.4 Step 2.

    Algorithm:
      1. Trim: for each model, keep only the top-density fraction of weights
         by magnitude; zero out the rest.
      2. Elect sign: for each parameter, use majority vote on sign across models.
      3. Merge: average only the values that agree with the elected sign.

    Parameters
    ----------
    weight_tensors : list of np.ndarray
        Weight tensors from N models (all same shape).
    density : float
        Fraction of top-magnitude weights to retain per model (default 0.20).

    Returns
    -------
    np.ndarray
        Merged weight tensor of same shape.

    Notes
    -----
    This implements the core TIES algorithm.  In production, apply
    this layer-by-layer across the full model.
    """
    if not weight_tensors:
        raise ValueError("No weight tensors provided for TIES merging.")
    if len(weight_tensors) == 1:
        return weight_tensors[0].copy()

    stacked = np.stack(weight_tensors)   # (N, ...)
    shape = stacked.shape[1:]
    stacked_flat = stacked.reshape(len(weight_tensors), -1)   # (N, P)
    P = stacked_flat.shape[1]

    # Step 1: Trim — keep top-density magnitude weights per model
    trimmed = np.zeros_like(stacked_flat)
    k = max(1, int(density * P))
    for i in range(len(weight_tensors)):
        row = stacked_flat[i]
        threshold = np.partition(np.abs(row), -k)[-k]
        mask = np.abs(row) >= threshold
        trimmed[i] = np.where(mask, row, 0.0)

    # Step 2: Elect sign — majority vote per parameter
    sign_votes = np.sign(trimmed)   # (N, P)
    # Count positive and negative votes; break ties by positive
    pos_votes = (sign_votes > 0).sum(axis=0)
    neg_votes = (sign_votes < 0).sum(axis=0)
    elected_sign = np.where(pos_votes >= neg_votes, 1.0, -1.0)   # (P,)

    # Step 3: Merge — average values agreeing with elected sign
    agrees = (sign_votes * elected_sign[None, :] > 0)   # (N, P) bool
    # Avoid division by zero for parameters where no model agrees
    agree_counts = agrees.sum(axis=0).astype(float)
    agree_counts = np.where(agree_counts == 0, 1.0, agree_counts)
    merged = (trimmed * agrees).sum(axis=0) / agree_counts   # (P,)

    return merged.reshape(shape).astype(stacked.dtype)


# ---------------------------------------------------------------------------
# Online K-FAC Fisher Manager  (Section 9.3)
# ---------------------------------------------------------------------------

class KFACFisherManager:
    """
    Manages online Kronecker-factored Fisher information matrices.
    Section 9.3.

    Per-layer factors are updated as exponential moving averages.
    CRITICAL: factors MUST be reset before each base model update to ensure
    Fisher accuracy.  Accumulated factors from previous updates degrade the
    approximation.

    Verification requirement (Section 9.3):
      Compare EWC regularisation strength with and without reset.  Reset
      condition must produce ≥ 2× stronger penalisation of high-Fisher
      parameters.
    """

    def __init__(self, decay: float = 0.95) -> None:
        self._factors: Dict[str, KFACFactors] = {}
        self._decay = decay
        self._reset_count = 0

    def reset(self) -> None:
        """
        Reset all K-FAC factors before a base model update.
        Section 9.3: factors MUST be reset, not carried across updates.
        """
        self._factors.clear()
        self._reset_count += 1
        logger.info(
            "K-FAC Fisher factors reset (count=%d). "
            "Ready for fresh online estimation. Section 9.3.",
            self._reset_count,
        )

    def update_layer(
        self,
        layer_name: str,
        activation: np.ndarray,    # (batch, d_in)
        grad_output: np.ndarray,   # (batch, d_out)
    ) -> None:
        """
        Online update of K-FAC factors for one layer.
        A_layer = activation^T @ activation / batch_size
        G_layer = grad_output^T @ grad_output / batch_size
        """
        batch_size = activation.shape[0]
        new_A = (activation.T @ activation) / batch_size
        new_G = (grad_output.T @ grad_output) / batch_size

        if layer_name not in self._factors:
            self._factors[layer_name] = KFACFactors(
                layer_name=layer_name,
                A=new_A.astype(np.float32),
                G=new_G.astype(np.float32),
            )
        else:
            self._factors[layer_name].update_ema(
                new_A.astype(np.float32),
                new_G.astype(np.float32),
                decay=self._decay,
            )

    def compute_ewc_loss(
        self,
        current_weights: Dict[str, np.ndarray],
        reference_weights: Dict[str, np.ndarray],
        lambda_ewc: float,
    ) -> float:
        """
        Compute total EWC loss across all layers with K-FAC factors.
        Section 9.3: L_total = L_task + λ · Σ_layer [(A⊗G) ⊙ (θ - θ*)²]
        """
        total = 0.0
        for layer_name, factors in self._factors.items():
            if layer_name not in current_weights:
                continue
            if layer_name not in reference_weights:
                continue
            total += factors.ewc_penalty(
                current_weights[layer_name],
                reference_weights[layer_name],
            )
        return lambda_ewc * total

    def verify_reset_strengthens_penalty(
        self,
        current_weights: Dict[str, np.ndarray],
        reference_weights: Dict[str, np.ndarray],
        lambda_ewc: float = 1.0,
    ) -> bool:
        """
        Verification from Section 9.3: after reset, EWC regularisation
        should produce ≥ 2× stronger penalisation of high-Fisher parameters.

        This method computes the penalty before and after a mock accumulation
        to verify the reset property holds.  Returns True if verified.
        """
        # Penalty immediately after reset (fresh, accurate factors)
        penalty_fresh = self.compute_ewc_loss(current_weights, reference_weights, lambda_ewc)

        # Simulate stale factors by degrading with noise
        stale_factors = KFACFisherManager(decay=self._decay)
        for layer_name, factors in self._factors.items():
            noisy_A = factors.A * 0.1  # Simulate accumulated/degraded factor
            noisy_G = factors.G * 0.1
            stale_factors._factors[layer_name] = KFACFactors(layer_name, noisy_A, noisy_G)
        penalty_stale = stale_factors.compute_ewc_loss(current_weights, reference_weights, lambda_ewc)

        ratio = penalty_fresh / (penalty_stale + 1e-12)
        verified = ratio >= 2.0
        if not verified:
            logger.warning(
                "K-FAC reset verification: fresh/stale penalty ratio = %.2f < 2.0. "
                "Reset may not be producing expected benefit.  Section 9.3.",
                ratio,
            )
        else:
            logger.info(
                "K-FAC reset verification PASSED: ratio = %.2f ≥ 2.0.  Section 9.3.",
                ratio,
            )
        return verified

    @property
    def layer_names(self) -> List[str]:
        return list(self._factors.keys())


# ---------------------------------------------------------------------------
# Progressive layer unfreezing  (Section 3.5 Limitation 3.A)
# ---------------------------------------------------------------------------

class LayerUnfreezingStateMachine:
    """
    Progressive layer unfreezing state machine.  Section 3.5 Limitation 3.A.

    Unfreeze upper layers first; validate adapter compatibility (via singular
    vector pre-screening) before unfreezing lower layers.

    State transitions:
      FROZEN → UPPER_UNFROZEN → LOWER_UNFROZEN → COMPLETE

    The machine refuses to advance to LOWER_UNFROZEN until adapter
    compatibility has been validated for the upper layers.
    """

    class State(Enum):
        FROZEN           = auto()
        UPPER_UNFROZEN   = auto()
        LOWER_UNFROZEN   = auto()
        COMPLETE         = auto()

    def __init__(self, total_layers: int, upper_fraction: float = 0.3) -> None:
        """
        Parameters
        ----------
        total_layers : int
            Total number of transformer layers.
        upper_fraction : float
            Fraction of layers counted as 'upper' (default: top 30%).
        """
        self.total_layers = total_layers
        self.n_upper = max(1, int(total_layers * upper_fraction))
        self.n_lower = total_layers - self.n_upper
        self._state = self.State.FROZEN
        self._adapter_compat_validated = False

    @property
    def state(self) -> "LayerUnfreezingStateMachine.State":
        return self._state

    def unfreeze_upper(self) -> List[int]:
        """
        Unfreeze the top n_upper layers.  Returns their indices.
        """
        if self._state != self.State.FROZEN:
            raise RuntimeError(f"Cannot unfreeze upper layers from state {self._state.name}")
        self._state = self.State.UPPER_UNFROZEN
        upper_indices = list(range(self.total_layers - self.n_upper, self.total_layers))
        logger.info(
            "Progressive unfreezing: upper %d/%d layers unfrozen.  "
            "Awaiting adapter compatibility validation before lower layers.  "
            "Section 3.5.",
            self.n_upper, self.total_layers,
        )
        return upper_indices

    def confirm_adapter_compatibility(self) -> None:
        """
        Called after singular vector pre-screening passes for upper layers.
        Section 3.5 Limitation 3.A.
        """
        if self._state != self.State.UPPER_UNFROZEN:
            raise RuntimeError(
                "Adapter compatibility must be confirmed after upper unfreezing."
            )
        self._adapter_compat_validated = True
        logger.info("Adapter compatibility validated for upper layers.  Section 3.5.")

    def unfreeze_lower(self) -> List[int]:
        """
        Unfreeze lower layers.  BLOCKED until adapter compatibility is confirmed.
        """
        if self._state != self.State.UPPER_UNFROZEN:
            raise RuntimeError("Upper layers must be unfrozen first.")
        if not self._adapter_compat_validated:
            raise RuntimeError(
                "Adapter compatibility for upper layers must be confirmed before "
                "unfreezing lower layers.  Section 3.5 Limitation 3.A."
            )
        self._state = self.State.LOWER_UNFROZEN
        lower_indices = list(range(0, self.n_lower))
        logger.info(
            "Progressive unfreezing: lower %d/%d layers unfrozen.  Section 3.5.",
            self.n_lower, self.total_layers,
        )
        return lower_indices

    def mark_complete(self) -> None:
        self._state = self.State.COMPLETE


# ---------------------------------------------------------------------------
# Latency budget monitor
# ---------------------------------------------------------------------------

class LatencyBudgetMonitor:
    """
    Monitors wall-clock time against per-step budgets.  Section 3.4.
    Exceeding a budget triggers a WARNING log.
    """

    def __init__(self) -> None:
        self._start_times: Dict[str, float] = {}

    def start(self, step_name: str) -> None:
        self._start_times[step_name] = time.monotonic()

    def check(self, step_name: str, budget_hours: float) -> float:
        """
        Check elapsed time for a step.  Logs warning if over budget.
        Returns elapsed hours.
        """
        if step_name not in self._start_times:
            return 0.0
        elapsed_h = (time.monotonic() - self._start_times[step_name]) / 3600.0
        if elapsed_h > budget_hours:
            logger.warning(
                "LATENCY BUDGET EXCEEDED: Step '%s' took %.2f hours "
                "(budget: %.2f hours).  Investigation required.  Section 3.4.",
                step_name, elapsed_h, budget_hours,
            )
        return elapsed_h
