"""
DEMoE Section 3 - Base Expert Model Creation Pipeline

Handles the 5-step creation pipeline including TIES merging, corpus assembly,
two-phase training with EWC, K-FAC Fisher updates, and progressive unfreezing.
"""

from .pipeline import (
    CorpusAssembler,
    HardProhibitionViolationError,
    InitialisationSelector,
    KFACFisherManager,
    LayerUnfreezingStateMachine,
    register_demoe_expert_id,
)
from .types import (
    CorpusAssemblyResult,
    ExpertTier,
    ExperienceReplayBuffer,
    HardwareConfig,
    HardwareInsufficientError,
    InitialisationStrategy,
    TrainingPhaseResult,
)

__all__ = [
    "CorpusAssembler",
    "HardProhibitionViolationError",
    "InitialisationSelector",
    "KFACFisherManager",
    "LayerUnfreezingStateMachine",
    "register_demoe_expert_id",
    "CorpusAssemblyResult",
    "ExpertTier",
    "ExperienceReplayBuffer",
    "HardwareConfig",
    "HardwareInsufficientError",
    "InitialisationStrategy",
    "TrainingPhaseResult",
]
