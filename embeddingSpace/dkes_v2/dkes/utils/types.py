"""
Shared type aliases and enumerations used across all DKES modules.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Basic array aliases
# ---------------------------------------------------------------------------
EmbeddingVector = np.ndarray          # shape (D,)
EmbeddingMatrix = np.ndarray          # shape (N, D)
IndexArray      = np.ndarray          # shape (N,) dtype int
ScoreArray      = np.ndarray          # shape (N,) dtype float32

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MetaConceptType(Enum):
    """The four meta-concept slot types stored in the KDM sublayer."""
    DOMAIN_PROXIMITY   = "M1"   # pairwise domain relationship summaries
    QUERY_PATTERN      = "M2"   # recurring decomposition templates
    UNCERTAINTY_MARKER = "M3"   # persistent high-uncertainty regions
    ANALOGY_BRIDGE     = "M4"   # cross-domain paraphrase equivalences


class EvictionReason(Enum):
    """Why a KDM slot was evicted to cold store."""
    LFU_CAPACITY    = auto()   # least-frequently-used, memory full
    AGE_DEMOTION    = auto()   # not accessed in 90+ days
    MANUAL_PRUNE    = auto()   # operator-initiated pruning


class WriteSource(Enum):
    """What triggered a KDM concept write."""
    CLUSTER_THRESHOLD    = auto()   # N_write queries accumulated
    MANUAL_INJECTION     = auto()   # operator-supplied concept
    COLD_REACTIVATION    = auto()   # reactivated from cold store


class AuditEventKind(Enum):
    """Top-level categories for structured audit records."""
    KDM_WRITE          = "kdm.write"
    KDM_READ           = "kdm.read"
    KDM_EVICTION       = "kdm.eviction"
    KDM_COLD_RESTORE   = "kdm.cold_restore"
    META_WRITE         = "meta.write"
    META_UPDATE        = "meta.update"
    ROUTING_QUERY      = "routing.query"
    ROUTING_CACHE_HIT  = "routing.cache_hit"
    GAMMA_UPDATE       = "gamma.update"
    HEALTH_CHECK       = "health.check"
    HEALTH_ALERT       = "health.alert"
    PROJECTION_CREATE  = "projection.create"
    INDEX_REBUILD      = "index.rebuild"
    CHECKPOINT_SAVE    = "checkpoint.save"
    CHECKPOINT_LOAD    = "checkpoint.load"


# ---------------------------------------------------------------------------
# Named return types
# ---------------------------------------------------------------------------

class CompositeEmbedding(NamedTuple):
    """Result of computing q̃(q) = LayerNorm(e(q) + γ·r(q))."""
    composite:        EmbeddingVector   # q̃(q)
    backbone:         EmbeddingVector   # e(q)
    memory_read:      EmbeddingVector   # r(q)
    gamma:            float             # current interpolation weight
    top_slot_indices: List[int]         # KDM slots with highest attention
    top_slot_weights: List[float]       # corresponding attention weights


class RoutingResult(NamedTuple):
    """Output of the DKES routing pipeline for a single query."""
    expert_ids:        List[str]         # ordered selected expert IDs
    adapter_ids:       List[Optional[str]]
    u_base_scores:     List[float]
    coarse_candidates: List[str]         # K_coarse IDs from Stage 1
    fine_candidates:   List[str]         # K_fine IDs from Stage 2
    used_meta_shortcut: bool             # was M2 pattern template used?
    uncertainty_region: bool             # does query fall in M3 zone?
    composite_emb:     CompositeEmbedding


class ConceptCluster(NamedTuple):
    """Accumulated query cluster awaiting potential KDM write."""
    cluster_id:   str
    centroid:     EmbeddingVector
    n_queries:    int
    coherence:    float             # mean intra-cluster cosine similarity
    max_novelty:  float             # 1 - max_cos_to_any_existing_slot
    mean_u_blobs: float             # mean BLoB uncertainty across cluster
    expert_dist:  Dict[str, float]  # routing distribution over cluster


class KDMSlot(NamedTuple):
    """Snapshot of a single KDM memory slot."""
    slot_id:       int
    address:       EmbeddingVector   # a_i  (backbone-grounded centroid)
    value:         EmbeddingVector   # v_i  (f_write MLP output)
    concept_id:    str
    write_source:  WriteSource
    created_at:    float             # unix timestamp
    last_accessed: float
    access_count:  int
    meta_type:     Optional[MetaConceptType]  # None for regular slots