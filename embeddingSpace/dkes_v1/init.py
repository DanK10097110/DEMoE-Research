"""
DKES — Dynamic Kanerva-Enhanced Embedding Space
================================================
A hybrid embedding space combining a frozen MRL backbone with a
write-capable Kanerva Dynamic Memory (KDM) layer for the DEMoE
architecture.

Package layout
--------------
dkes/
  core/
    embedding_space.py   — Top-level DKES orchestrator
    backbone.py          — Frozen MRL encoder wrapper
    composite.py         — q̃(q) = LayerNorm(e(q) + γ·r(q)) logic
  memory/
    kdm.py               — Kanerva Dynamic Memory (read/write/evict)
    concept_store.py     — Concept cluster accumulation & write triggers
    meta_concepts.py     — Meta-concept sublayer (M1–M4 types)
    cold_store.py        — Evicted-slot archive & reactivation
  routing/
    faiss_index.py       — Main + KDM-address FAISS sub-index wrapper
    domain_projection.py — Per-domain linear projection adapters
    mrl_funnel.py        — Two-stage coarse→fine MRL routing
  audit/
    audit_log.py         — Structured audit record emission
    metrics.py           — Routing quality, KDM hit-rate, γ drift
    health_monitor.py    — Continuous health checks & alerts
  utils/
    config.py            — DKESConfig dataclass
    serialization.py     — Save/load checkpoints
    types.py             — Shared type aliases
"""

from dkes.core.embedding_space import DKESEmbeddingSpace
from dkes.utils.config import DKESConfig
from dkes.audit.audit_log import AuditLog

__all__ = ["DKESEmbeddingSpace", "DKESConfig", "AuditLog"]
__version__ = "1.0.0"