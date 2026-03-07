"""
DEMoE Section 2 - Bootstrap Registry: Types, Constants, and Data Structures

Design decisions captured here:
  - Licence tiers with hard enforcement at registration time
  - Quality gate constants per domain benchmark
  - Bootstrap decision thresholds (distance 0.20 / 0.30 / 0.50)
  - Safety evaluation rejection machinery with mandatory logging
  - Adapter bootstrap → BLoB conversion pathway
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Decision thresholds (Section 2.2)
# ---------------------------------------------------------------------------

class BootstrapDistanceThresholds:
    """Cosine-distance cutoffs that drive the bootstrap vs. train decision."""
    DIRECT_REGISTER:      float = 0.20  # Register directly, skip all training
    PHASE2_ONLY:          float = 0.30  # Phase-2 fine-tuning only
    TIES_INIT:            float = 0.50  # Use as TIES init; beyond → general checkpoint
    COVERAGE_RELEVANT_MAX: float = 0.20  # Used alongside DIRECT_REGISTER


class QualityThresholds:
    """
    Minimum acceptable benchmark scores per domain before a model may enter
    the registry. Section 2.4 Limitation 2.B.
    Values should be calibrated to each benchmark's score distribution;
    these are reasonable starting points.
    """
    BIOMEDICAL_MEDQA:     float = 0.65   # MedQA 4-option accuracy
    LEGAL_LEGALBENCH:     float = 0.60   # LegalBench overall accuracy
    CODE_HUMANEVAL:       float = 0.55   # HumanEval pass@1
    FINANCE_FINBENCH:     float = 0.60
    MATH_MATH_BENCH:      float = 0.55
    SCIENCE_MMLU:         float = 0.65   # MMLU science subtasks
    GENERAL_MMLU:         float = 0.60
    DEFAULT_MINIMUM:      float = 0.55   # Used when no domain-specific threshold exists


class ValidationConstants:
    """Post-bootstrap validation parameters. Section 2.4."""
    # Uncertainty must drop below creation threshold on triggering queries
    UNCERTAINTY_DROP_REQUIRED: bool = True
    # Safety evaluation failure → hard rejection + log
    SAFETY_EVAL_FAILURE_REJECTS: bool = True
    # Maximum 30-min budget for the full bootstrap decision incl. safety eval
    BOOTSTRAP_DECISION_BUDGET_MINUTES: float = 30.0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LicenceTier(Enum):
    """
    Licence permissiveness tiers. Section 2.4 Limitation 2.A.
    Used to filter candidates by deployment context.
    """
    PERMISSIVE   = "permissive"    # Apache 2.0, MIT, BSD — usable everywhere
    RESEARCH     = "research"      # CC BY, non-commercial only
    RESTRICTED   = "restricted"    # Model-specific EULA with redistribution limits
    PROPRIETARY  = "proprietary"   # Cannot be used without explicit agreement
    UNKNOWN      = "unknown"       # Not yet reviewed — blocked by default


class DeploymentContext(Enum):
    """Describes the sensitivity of the deployment environment."""
    COMMERCIAL_GENERAL = "commercial_general"
    COMMERCIAL_REGULATED = "commercial_regulated"   # Finance, healthcare, legal
    RESEARCH_ONLY = "research_only"
    INTERNAL_ONLY = "internal_only"


class BootstrapOutcome(Enum):
    DIRECT_REGISTER = "direct_register"
    PHASE2_FINETUNE = "phase2_finetune"
    FULL_TRAINING   = "full_training"
    REJECTED_QUALITY = "rejected_quality"
    REJECTED_LICENCE = "rejected_licence"
    REJECTED_SAFETY  = "rejected_safety"
    REJECTED_DISTANCE = "rejected_distance"


class AdapterBootstrapOutcome(Enum):
    BLOB_WARMSTART   = "blob_warmstart"    # LoRA → BLoB fine-tune (preferred)
    LAPLACE_FALLBACK = "laplace_fallback"  # No training data; post-hoc Laplace
    REJECTED         = "rejected"


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A single benchmark evaluation result for a model."""
    benchmark_name: str
    score: float
    evaluated_at_iso: str   # ISO-8601 timestamp
    dataset_version: str = "unknown"


@dataclass
class RegistryEntry:
    """
    One record in the bootstrap registry catalogue. Section 2.1.

    Covers all metadata required for the bootstrap decision tree (Section 2.2),
    licence filtering (Section 2.4 Limitation 2.A), and quality gating
    (Section 2.4 Limitation 2.B).
    """
    model_id:           str                         # e.g. "Meditron-70B"
    hf_repo:            Optional[str]               # HuggingFace repo path
    domain:             str                         # Broad domain label
    sub_domains:        List[str]                   # Finer-grained coverage
    architecture:       str                         # e.g. "LLaMA-2"
    parameter_count_b:  float                       # Billions of parameters
    training_corpus:    str                         # Short description
    licence_tier:       LicenceTier
    licence_name:       str
    benchmark_results:  List[BenchmarkResult] = field(default_factory=list)
    # Embedding-space centroid of this model's domain (set when indexed)
    domain_centroid_768: Optional[object] = None    # np.ndarray set at index time
    coverage_notes:     str = ""
    compliance_vetted:  bool = False                # True = manually reviewed
    is_fallback_model:  bool = False                # General-purpose fallback
    registry_id:        str = field(default_factory=lambda: str(uuid.uuid4()))

    def best_benchmark_score(self, benchmark_prefix: str = "") -> Optional[float]:
        """Return the best score among benchmarks matching the given prefix."""
        matching = [
            b.score for b in self.benchmark_results
            if b.benchmark_name.lower().startswith(benchmark_prefix.lower())
        ]
        return max(matching) if matching else None

    def passes_quality_gate(self, domain_label: str) -> Tuple[bool, str]:
        """
        Return (passes, reason).  The quality gate requires at least one
        benchmark above the domain-specific minimum threshold.
        Section 2.4 Limitation 2.B.
        """
        thresholds = {
            "biomedical": ("medqa", QualityThresholds.BIOMEDICAL_MEDQA),
            "clinical":   ("medqa", QualityThresholds.BIOMEDICAL_MEDQA),
            "legal":      ("legalbench", QualityThresholds.LEGAL_LEGALBENCH),
            "code":       ("humaneval", QualityThresholds.CODE_HUMANEVAL),
            "software":   ("humaneval", QualityThresholds.CODE_HUMANEVAL),
            "finance":    ("finbench", QualityThresholds.FINANCE_FINBENCH),
            "math":       ("math", QualityThresholds.MATH_MATH_BENCH),
            "science":    ("mmlu", QualityThresholds.SCIENCE_MMLU),
        }
        d = domain_label.lower()
        for key, (bench_prefix, minimum) in thresholds.items():
            if key in d:
                score = self.best_benchmark_score(bench_prefix)
                if score is None:
                    return False, f"No {bench_prefix} benchmark found for domain '{domain_label}'"
                if score < minimum:
                    return False, (
                        f"Score {score:.3f} on {bench_prefix} below minimum "
                        f"{minimum:.3f} for domain '{domain_label}'"
                    )
                return True, f"Score {score:.3f} ≥ {minimum:.3f} on {bench_prefix}"
        # Default minimum
        any_score = max((b.score for b in self.benchmark_results), default=None)
        if any_score is None:
            return False, "No benchmark results available"
        if any_score < QualityThresholds.DEFAULT_MINIMUM:
            return False, f"Best score {any_score:.3f} below default minimum {QualityThresholds.DEFAULT_MINIMUM:.3f}"
        return True, f"Best score {any_score:.3f} ≥ {QualityThresholds.DEFAULT_MINIMUM:.3f}"


@dataclass
class SafetyEvaluationResult:
    """
    Result of the mandatory post-bootstrap safety evaluation suite.
    Section 2.4 Limitation 2.B.

    A SAFETY_EVALUATION_REJECTION event must be logged on failure.
    Human review is recommended for high-stakes domain bootstraps.
    """
    passed:              bool
    probes_run:          int
    probes_failed:       int
    failure_categories:  List[str] = field(default_factory=list)
    high_stakes_domain:  bool = False
    human_review_flagged: bool = False
    evaluation_id:       str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def failure_rate(self) -> float:
        if self.probes_run == 0:
            return 0.0
        return self.probes_failed / self.probes_run


@dataclass
class BootstrapDecision:
    """
    The output of the bootstrap decision tree for a single gap event.
    Section 2.2.
    """
    outcome:              BootstrapOutcome
    entry:                Optional[RegistryEntry]
    cosine_distance:      Optional[float]
    quality_passed:       bool
    safety_result:        Optional[SafetyEvaluationResult]
    rationale:            str
    decision_duration_s:  float = 0.0    # Wall-clock time for budget monitoring


@dataclass
class AdapterBootstrapCandidate:
    """
    A LoRA adapter from the open-source ecosystem, catalogued as a
    BLoB adapter bootstrap candidate. Section 2.3.
    """
    adapter_id:           str
    hf_repo:              str
    parent_model_id:      str
    domain:               str
    sub_domain:           str
    lora_rank:            int
    training_data_available: bool    # If False → Laplace-LoRA fallback
    licence_tier:         LicenceTier
    benchmark_results:    List[BenchmarkResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class SafetyEvaluationRejectionError(Exception):
    """
    Raised (and logged as SAFETY_EVALUATION_REJECTION) when a candidate
    fails the safety evaluation suite. Section 2.4 / FATAL T1.
    """


class QualityGateRejectionError(Exception):
    """Raised when a candidate model falls below minimum quality threshold."""


class LicenceRejectionError(Exception):
    """Raised when a model's licence is incompatible with the deployment context."""


# ---------------------------------------------------------------------------
# Licence compatibility matrix  (Section 2.4 Limitation 2.A)
# ---------------------------------------------------------------------------

# Maps (deployment_context, licence_tier) → allowed
LICENCE_COMPATIBILITY: Dict[Tuple[DeploymentContext, LicenceTier], bool] = {
    (DeploymentContext.COMMERCIAL_GENERAL,    LicenceTier.PERMISSIVE):  True,
    (DeploymentContext.COMMERCIAL_GENERAL,    LicenceTier.RESEARCH):    False,
    (DeploymentContext.COMMERCIAL_GENERAL,    LicenceTier.RESTRICTED):  False,
    (DeploymentContext.COMMERCIAL_GENERAL,    LicenceTier.PROPRIETARY): False,
    (DeploymentContext.COMMERCIAL_GENERAL,    LicenceTier.UNKNOWN):     False,
    (DeploymentContext.COMMERCIAL_REGULATED,  LicenceTier.PERMISSIVE):  True,
    (DeploymentContext.COMMERCIAL_REGULATED,  LicenceTier.RESEARCH):    False,
    (DeploymentContext.COMMERCIAL_REGULATED,  LicenceTier.RESTRICTED):  False,
    (DeploymentContext.COMMERCIAL_REGULATED,  LicenceTier.PROPRIETARY): False,
    (DeploymentContext.COMMERCIAL_REGULATED,  LicenceTier.UNKNOWN):     False,
    (DeploymentContext.RESEARCH_ONLY,         LicenceTier.PERMISSIVE):  True,
    (DeploymentContext.RESEARCH_ONLY,         LicenceTier.RESEARCH):    True,
    (DeploymentContext.RESEARCH_ONLY,         LicenceTier.RESTRICTED):  False,
    (DeploymentContext.RESEARCH_ONLY,         LicenceTier.PROPRIETARY): False,
    (DeploymentContext.RESEARCH_ONLY,         LicenceTier.UNKNOWN):     False,
    (DeploymentContext.INTERNAL_ONLY,         LicenceTier.PERMISSIVE):  True,
    (DeploymentContext.INTERNAL_ONLY,         LicenceTier.RESEARCH):    True,
    (DeploymentContext.INTERNAL_ONLY,         LicenceTier.RESTRICTED):  True,
    (DeploymentContext.INTERNAL_ONLY,         LicenceTier.PROPRIETARY): False,
    (DeploymentContext.INTERNAL_ONLY,         LicenceTier.UNKNOWN):     False,
}


def is_licence_compatible(context: DeploymentContext, tier: LicenceTier) -> bool:
    return LICENCE_COMPATIBILITY.get((context, tier), False)


# ---------------------------------------------------------------------------
# Curated default registry  (Table 2.1)
# ---------------------------------------------------------------------------

HIGH_STAKES_DOMAINS = frozenset(["biomedical", "clinical", "legal", "finance"])
