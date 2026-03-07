"""
DEMoE Section 2 - Bootstrap Registry

Manages pre-trained expert bootstrapping with quality gating, licence validation,
and safety evaluation.
"""

from .registry import BootstrapRegistry
from .types import (
    AdapterBootstrapCandidate,
    AdapterBootstrapOutcome,
    BenchmarkResult,
    BootstrapDecision,
    BootstrapDistanceThresholds,
    BootstrapOutcome,
    DeploymentContext,
    LicenceTier,
    RegistryEntry,
    SafetyEvaluationResult,
    SafetyEvaluationRejectionError,
    is_licence_compatible,
)

__all__ = [
    "BootstrapRegistry",
    "AdapterBootstrapCandidate",
    "AdapterBootstrapOutcome",
    "BenchmarkResult",
    "BootstrapDecision",
    "BootstrapDistanceThresholds",
    "BootstrapOutcome",
    "DeploymentContext",
    "LicenceTier",
    "RegistryEntry",
    "SafetyEvaluationResult",
    "SafetyEvaluationRejectionError",
    "is_licence_compatible",
]
