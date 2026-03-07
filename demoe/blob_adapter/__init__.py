"""
DEMoE Section 4 - BLoB (Bayesian LoRA Blocks) Adapter Layer

Implements Bayesian two-parameter blocks for uncertainty quantification,
including Two-NN estimator, MAP pre-training, cyclical KL scheduling,
and singular vector fingerprinting for compatibility assessment.
"""

from .core import (
    BLoBAdapterPipeline,
    BLoBInferenceEngine,
    LaplaceCalibrationManager,
    MAPTrainer,
    TwoNNRankEstimator,
)
from .types import (
    AdapterLifecycleState,
    AdapterType,
    BLoBAdapterWeights,
    BLoBConstants,
    CyclicalKLSchedule,
    SingularVectorFingerprint,
)

__all__ = [
    "BLoBAdapterPipeline",
    "BLoBInferenceEngine",
    "LaplaceCalibrationManager",
    "MAPTrainer",
    "TwoNNRankEstimator",
    "AdapterLifecycleState",
    "AdapterType",
    "BLoBAdapterWeights",
    "BLoBConstants",
    "CyclicalKLSchedule",
    "SingularVectorFingerprint",
]
