"""
DKESConfig — single source of truth for all DKES hyperparameters.

Every magic constant in the architecture spec is parameterised here so
that experiments can swap values without touching implementation code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import json


@dataclass
class KDMConfig:
    """Hyperparameters for the Kanerva Dynamic Memory layer."""

    # Memory capacity
    total_slots: int = 4096
    meta_slots: int = 256          # top-N slots reserved for meta-concepts
    write_mlp_hidden: int = 256    # hidden dim of f_write MLP

    # Addressing
    sharpness_beta: float = 10.0   # softmax sharpness for attention weights
    top_k_read: int = 16           # number of address matches evaluated in full
    gamma_init: float = 0.1        # initial KDM interpolation weight
    gamma_ceiling_concepts: int = 500    # N_populated before γ ceiling lifted
    gamma_ceiling_max: float = 0.05      # hard ceiling during ramp-up

    # Write triggers
    write_min_queries: int = 20    # N_write threshold
    write_novelty_threshold: float = 0.75   # max cos-sim to existing slot
    write_coherence_threshold: float = 0.65  # min intra-cluster cos-sim

    # Eviction
    cold_store_days: int = 90      # days inactive before demotion
    lfu_correction_alpha: float = 0.1  # recency correction for LFU

    # Meta-concept specific
    meta_m2_template_threshold: int = 15   # query pattern seen N times → write
    meta_m2_match_distance: float = 0.12   # cosine distance for template match
    meta_m2_match_confidence: float = 0.85  # min confidence to use cached template


@dataclass
class BackboneConfig:
    """Configuration for the frozen MRL encoder backbone."""
    model_name: str = "BAAI/bge-m3"
    embedding_dim: int = 768
    mrl_dims: Tuple[int, ...] = (32, 64, 128, 256, 512, 768)
    coarse_dim: int = 64           # Stage 1 FAISS uses this prefix
    fine_dim: int = 768            # Stage 2 re-ranking dimension
    device: str = "cpu"            # "cuda" | "cpu" | "mps"
    batch_size: int = 64
    normalize_embeddings: bool = True


@dataclass
class RoutingConfig:
    """Configuration for the MRL two-stage routing funnel."""
    k_coarse: int = 20             # Stage 1 candidates
    k_coarse_expanded: int = 30    # Stage 1 when query is in M3 zone
    k_fine: int = 5                # Stage 2 candidates per domain label
    max_active_experts: int = 3    # M_active budget cap
    concept_overlap_threshold: float = 0.80  # merge domain labels above this

    # Routing selection thresholds (domain-conditioned MLP overrides these)
    default_routing_threshold: float = 0.60
    routing_deadlock_margin: float = 0.30

    # Fast-path routing cache
    cache_size: int = 10_000
    cache_epsilon: float = 0.02    # cosine distance for cache hit


@dataclass
class DomainProjectionConfig:
    """Configuration for per-domain linear projection adapters."""
    max_adapters: int = 50
    coherence_threshold: float = 0.70   # routing coherence below → create adapter
    coherence_window_days: int = 3
    low_rank_dim: Optional[int] = None  # None = full d×d, else low-rank AB^T
    contrastive_loss: str = "infonce"


@dataclass
class AuditConfig:
    """Configuration for audit record emission and retention."""
    log_dir: Path = Path("./dkes_audit_logs")
    log_to_file: bool = True
    log_to_stdout: bool = False
    emit_routing_records: bool = True
    emit_kdm_read_records: bool = True    # can be noisy at scale; toggle off
    emit_kdm_write_records: bool = True
    emit_health_records: bool = True
    health_check_interval_seconds: int = 300
    metrics_window_hours: int = 24        # rolling window for metric aggregation
    max_log_size_mb: int = 512
    rotate_logs: bool = True

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)


@dataclass
class DKESConfig:
    """
    Master configuration for the full DKES embedding space.

    Usage
    -----
    cfg = DKESConfig()               # all defaults
    cfg = DKESConfig.from_json("cfg.json")
    cfg.to_json("cfg.json")
    """
    backbone:    BackboneConfig         = field(default_factory=BackboneConfig)
    kdm:         KDMConfig              = field(default_factory=KDMConfig)
    routing:     RoutingConfig          = field(default_factory=RoutingConfig)
    projection:  DomainProjectionConfig = field(default_factory=DomainProjectionConfig)
    audit:       AuditConfig            = field(default_factory=AuditConfig)

    # Checkpoint / persistence
    checkpoint_dir: Path = Path("./dkes_checkpoints")
    checkpoint_interval_minutes: int = 60

    # Reproducibility
    random_seed: int = 42

    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        return _convert(self)

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "DKESConfig":
        with open(path) as f:
            data = json.load(f)
        backbone = BackboneConfig(**{k: tuple(v) if isinstance(v, list) and k == "mrl_dims" else v
                                     for k, v in data.pop("backbone", {}).items()})
        kdm         = KDMConfig(**data.pop("kdm", {}))
        routing     = RoutingConfig(**data.pop("routing", {}))
        projection  = DomainProjectionConfig(**data.pop("projection", {}))
        audit_data  = data.pop("audit", {})
        if "log_dir" in audit_data:
            audit_data["log_dir"] = Path(audit_data["log_dir"])
        audit = AuditConfig(**audit_data)
        return cls(backbone=backbone, kdm=kdm, routing=routing,
                   projection=projection, audit=audit, **data)