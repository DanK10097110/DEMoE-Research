"""
KanervaMemory — the core dynamic memory layer of the DKES.

Implements the M-slot associative memory described in Section 1.2-1.3 of
the architectural specification:

  Read:   r(q) = sum_i w_i(q) * v_i
          where w_i(q) = softmax(beta * cos(q, a_i))

  Write:  a_new = cluster_centroid (backbone-grounded)
          v_new = f_write(concat(centroid, mean_u_blob, expert_dist_vector))

  Evict:  LFU with recency correction; evicted slots archived to ColdStore

The top-256 slots are reserved for meta-concepts (MetaConceptLayer manages
those separately). This class manages the remaining concept slots.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from dkes.utils.config import DKESConfig
from dkes.utils.types import (
    ConceptCluster, EmbeddingVector, EvictionReason,
    KDMSlot, MetaConceptType, WriteSource,
)

logger = logging.getLogger(__name__)


class _WriteMLPStub:
    """
    Stub implementation of f_write MLP for use when torch is unavailable.
    Computes a weighted average of the input components as a stand-in.
    """
    def __call__(self, centroid: np.ndarray, mean_u: float,
                 expert_dist_vec: np.ndarray) -> np.ndarray:
        D = len(centroid)
        # Simple weighted blend — not trainable, only for integration testing
        uncertainty_signal = np.full(D, mean_u, dtype=np.float32)
        expert_signal = expert_dist_vec[:D] if len(expert_dist_vec) >= D else \
            np.pad(expert_dist_vec, (0, D - len(expert_dist_vec)))
        value = 0.7 * centroid + 0.2 * uncertainty_signal + 0.1 * expert_signal
        norm = np.linalg.norm(value)
        return (value / norm).astype(np.float32) if norm > 0 else value


class KanervaMemory:
    """
    Write-capable associative memory with M concept slots.

    Thread-safe: all slot mutations acquire self._lock.
    The top `meta_slots` indices are reserved; this class only manages
    indices [meta_slots, total_slots).

    Parameters
    ----------
    cfg:    DKESConfig
    audit:  AuditLog (optional; pass None to disable audit emission here)
    """

    def __init__(self, cfg: DKESConfig, audit=None):
        self._cfg = cfg
        self._kdm_cfg = cfg.kdm
        self._audit = audit
        self._lock = threading.RLock()

        D = cfg.backbone.embedding_dim
        M = cfg.kdm.total_slots

        # Memory arrays — addresses (a) and values (v)
        self.addresses: np.ndarray = np.zeros((M, D), dtype=np.float32)
        self.values:    np.ndarray = np.zeros((M, D), dtype=np.float32)

        # Slot metadata (concept_id, write_source, timestamps, access counts)
        self._slots: Dict[int, KDMSlot] = {}
        self._populated: set = set()         # slot indices with data

        # Concept-id → slot-id reverse index
        self._concept_to_slot: Dict[str, int] = {}

        # Beta (sharpness) — may be learned; start from config
        self._beta: float = cfg.kdm.sharpness_beta

        # f_write MLP stub (replaced with torch MLP when available)
        self._f_write = _WriteMLPStub()
        self._try_load_torch_mlp(D)

        logger.info(
            "KanervaMemory initialised: M=%d concept_slots=%d meta_reserved=%d",
            M, self._concept_capacity, cfg.kdm.meta_slots,
        )

    def _try_load_torch_mlp(self, D: int):
        try:
            import torch
            import torch.nn as nn

            H = self._kdm_cfg.write_mlp_hidden
            # Input: centroid (D) + scalar u (1) + expert_dist (D padded) = 2D+1
            in_dim = D + 1 + D
            self._f_write_nn = nn.Sequential(
                nn.Linear(in_dim, H), nn.GELU(),
                nn.Linear(H, H // 2), nn.GELU(),
                nn.Linear(H // 2, D),
                nn.LayerNorm(D),
            )
            self._f_write_nn.eval()

            def _torch_write_fn(centroid, mean_u, expert_dist_vec):
                import torch
                D = len(centroid)
                edv = expert_dist_vec[:D] if len(expert_dist_vec) >= D else \
                    np.pad(expert_dist_vec, (0, D - len(expert_dist_vec)))
                x = np.concatenate([centroid, [mean_u], edv]).astype(np.float32)
                with torch.no_grad():
                    out = self._f_write_nn(torch.from_numpy(x)).numpy()
                norm = np.linalg.norm(out)
                return (out / norm).astype(np.float32) if norm > 0 else out

            self._f_write = _torch_write_fn
            logger.debug("KanervaMemory: using torch f_write MLP")
        except ImportError:
            logger.debug("KanervaMemory: torch not available, using stub f_write")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _concept_capacity(self) -> int:
        """Number of non-meta slots available for concept storage."""
        return self._kdm_cfg.total_slots - self._kdm_cfg.meta_slots

    @property
    def _concept_start_idx(self) -> int:
        """First slot index available for concept storage."""
        return self._kdm_cfg.meta_slots

    @property
    def n_populated(self) -> int:
        """Number of concept slots with data (excluding meta slots)."""
        with self._lock:
            return len([i for i in self._populated if i >= self._concept_start_idx])

    @property
    def n_free_concept_slots(self) -> int:
        return self._concept_capacity - self.n_populated

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def read(
        self,
        query: EmbeddingVector,
        top_k: Optional[int] = None,
    ) -> Tuple[EmbeddingVector, List[int], List[float]]:
        """
        Soft read from the KDM given query embedding.

        Returns
        -------
        memory_read:      r(q) — weighted sum of values, shape (D,)
        top_slot_indices: indices of the top-k slots by attention weight
        top_slot_weights: their attention weights (sums to < 1 if < M populated)
        """
        top_k = top_k or self._kdm_cfg.top_k_read

        with self._lock:
            populated_indices = sorted(self._populated)
            if not populated_indices:
                D = self._cfg.backbone.embedding_dim
                return (
                    np.zeros(D, dtype=np.float32),
                    [],
                    [],
                )

            # Compute cosine similarities to populated addresses
            pop_addresses = self.addresses[populated_indices]  # (P, D)
            q_norm = query / (np.linalg.norm(query) + 1e-8)
            a_norms = pop_addresses / (
                np.linalg.norm(pop_addresses, axis=1, keepdims=True) + 1e-8
            )
            cos_sims = a_norms @ q_norm  # (P,)

            # Softmax with sharpness beta (over populated slots only)
            logits = self._beta * cos_sims
            logits -= logits.max()  # numerical stability
            weights = np.exp(logits)
            weights /= weights.sum()

            # Select top-k for audit reporting
            if len(weights) <= top_k:
                top_local = np.argsort(weights)[::-1]
            else:
                top_local = np.argpartition(weights, -top_k)[-top_k:]
                top_local = top_local[np.argsort(weights[top_local])[::-1]]

            top_global = [populated_indices[i] for i in top_local]
            top_weights = [float(weights[i]) for i in top_local]

            # Soft-read: weighted sum of all populated values
            pop_values = self.values[populated_indices]  # (P, D)
            memory_read = (weights[:, None] * pop_values).sum(axis=0)

            # Update access tracking for top slots
            now = time.time()
            for slot_idx in top_global:
                if slot_idx in self._slots:
                    slot = self._slots[slot_idx]
                    self._slots[slot_idx] = KDMSlot(
                        slot_id=slot.slot_id,
                        address=slot.address,
                        value=slot.value,
                        concept_id=slot.concept_id,
                        write_source=slot.write_source,
                        created_at=slot.created_at,
                        last_accessed=now,
                        access_count=slot.access_count + 1,
                        meta_type=slot.meta_type,
                    )

        return (
            memory_read.astype(np.float32),
            top_global,
            top_weights,
        )

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    def write_concept(
        self,
        cluster: ConceptCluster,
        source: WriteSource = WriteSource.CLUSTER_THRESHOLD,
        trace_id: Optional[str] = None,
    ) -> Optional[int]:
        """
        Write a concept cluster to a new KDM slot.

        Returns the allocated slot index, or None if write was rejected
        (e.g., the concept is no longer novel at write time).
        """
        # Final novelty check at write time (cluster may have been queued)
        with self._lock:
            max_cos = self._max_cos_to_existing(cluster.centroid)

        if max_cos >= self._kdm_cfg.write_novelty_threshold:
            logger.debug(
                "KDM write rejected for %s: max_cos=%.3f >= threshold %.3f",
                cluster.cluster_id, max_cos, self._kdm_cfg.write_novelty_threshold,
            )
            return None

        # Compute value via f_write MLP
        expert_dist_vec = self._expert_dist_to_vector(cluster.expert_dist)
        value = self._f_write(cluster.centroid, cluster.mean_u_blobs, expert_dist_vec)

        with self._lock:
            slot_idx = self._allocate_slot(trace_id=trace_id)
            self.addresses[slot_idx] = cluster.centroid.astype(np.float32)
            self.values[slot_idx] = value

            now = time.time()
            concept_id = cluster.cluster_id
            new_slot = KDMSlot(
                slot_id=slot_idx,
                address=self.addresses[slot_idx].copy(),
                value=self.values[slot_idx].copy(),
                concept_id=concept_id,
                write_source=source,
                created_at=now,
                last_accessed=now,
                access_count=0,
                meta_type=None,
            )
            self._slots[slot_idx] = new_slot
            self._populated.add(slot_idx)
            self._concept_to_slot[concept_id] = slot_idx

        logger.info(
            "KDM concept write: slot=%d concept=%s source=%s",
            slot_idx, concept_id, source.name,
        )

        if self._audit:
            self._audit.kdm_write(
                concept_id=concept_id,
                slot_id=slot_idx,
                address_norm=float(np.linalg.norm(self.addresses[slot_idx])),
                value_norm=float(np.linalg.norm(self.values[slot_idx])),
                write_source=source.name,
                n_queries_triggered=cluster.n_queries,
                coherence=cluster.coherence,
                novelty=cluster.max_novelty,
                trace_id=trace_id,
            )

        return slot_idx

    def write_meta_slot(
        self,
        slot_idx: int,
        address: EmbeddingVector,
        value: EmbeddingVector,
        meta_type: MetaConceptType,
        concept_id: str,
        source: WriteSource = WriteSource.MANUAL_INJECTION,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Write directly to a reserved meta-concept slot.
        Only MetaConceptLayer should call this method.
        """
        if slot_idx >= self._kdm_cfg.meta_slots:
            raise ValueError(
                f"Slot {slot_idx} is not in the meta-reserved range "
                f"[0, {self._kdm_cfg.meta_slots})"
            )

        with self._lock:
            self.addresses[slot_idx] = address.astype(np.float32)
            self.values[slot_idx] = value.astype(np.float32)
            now = time.time()
            existing = self._slots.get(slot_idx)
            new_slot = KDMSlot(
                slot_id=slot_idx,
                address=self.addresses[slot_idx].copy(),
                value=self.values[slot_idx].copy(),
                concept_id=concept_id,
                write_source=source,
                created_at=existing.created_at if existing else now,
                last_accessed=now,
                access_count=(existing.access_count if existing else 0),
                meta_type=meta_type,
            )
            self._slots[slot_idx] = new_slot
            self._populated.add(slot_idx)
            self._concept_to_slot[concept_id] = slot_idx

    # ------------------------------------------------------------------
    # EVICT
    # ------------------------------------------------------------------

    def _allocate_slot(self, trace_id: Optional[str] = None) -> int:
        """
        Find a free concept slot, or evict the least-valuable one.
        Must be called under self._lock.
        """
        # Prefer empty slots first
        for idx in range(self._concept_start_idx, self._kdm_cfg.total_slots):
            if idx not in self._populated:
                return idx

        # All slots full — evict LFU with recency correction
        return self._evict_lfu(trace_id=trace_id)

    def _evict_lfu(self, trace_id: Optional[str] = None) -> int:
        """
        Evict the concept slot with the lowest LFU-recency score.
        Score = access_count * recency_factor
        recency_factor = exp(-alpha * days_since_access)
        """
        now = time.time()
        best_idx = None
        best_score = float("inf")

        for idx in range(self._concept_start_idx, self._kdm_cfg.total_slots):
            slot = self._slots.get(idx)
            if slot is None:
                return idx  # should not happen, but handle gracefully
            days = (now - slot.last_accessed) / 86400
            recency_factor = np.exp(-self._kdm_cfg.lfu_correction_alpha * days)
            score = slot.access_count * recency_factor
            if score < best_score:
                best_score = score
                best_idx = idx

        slot = self._slots[best_idx]
        days_since = (now - slot.last_accessed) / 86400
        reason = (
            EvictionReason.AGE_DEMOTION
            if days_since > self._kdm_cfg.cold_store_days
            else EvictionReason.LFU_CAPACITY
        )

        logger.info(
            "KDM eviction: slot=%d concept=%s reason=%s accesses=%d days_idle=%.1f",
            best_idx, slot.concept_id, reason.name, slot.access_count, days_since,
        )

        if self._audit:
            self._audit.kdm_eviction(
                slot_id=best_idx,
                concept_id=slot.concept_id,
                reason=reason.name,
                access_count=slot.access_count,
                days_since_access=days_since,
                trace_id=trace_id,
            )

        # Clear slot and remove from indices
        self._populated.discard(best_idx)
        self._concept_to_slot.pop(slot.concept_id, None)
        self._slots.pop(best_idx, None)
        self.addresses[best_idx] = 0.0
        self.values[best_idx] = 0.0

        return best_idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _max_cos_to_existing(self, centroid: EmbeddingVector) -> float:
        """Compute max cosine similarity between centroid and all populated addresses."""
        if not self._populated:
            return 0.0
        populated_indices = sorted(self._populated)
        pop_addresses = self.addresses[populated_indices]
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
        a_norms = pop_addresses / (
            np.linalg.norm(pop_addresses, axis=1, keepdims=True) + 1e-8
        )
        return float((a_norms @ c_norm).max())

    def _expert_dist_to_vector(self, expert_dist: Dict[str, float]) -> np.ndarray:
        """
        Convert expert routing distribution to a fixed-size vector.
        Uses a consistent hash-based mapping from expert_id → dimension.
        """
        D = self._cfg.backbone.embedding_dim
        vec = np.zeros(D, dtype=np.float32)
        for expert_id, prob in expert_dist.items():
            dim = abs(hash(expert_id)) % D
            vec[dim] += prob
        # Normalise
        s = vec.sum()
        return (vec / s).astype(np.float32) if s > 0 else vec

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_slot(self, slot_id: int) -> Optional[KDMSlot]:
        with self._lock:
            return self._slots.get(slot_id)

    def get_slot_by_concept(self, concept_id: str) -> Optional[KDMSlot]:
        with self._lock:
            idx = self._concept_to_slot.get(concept_id)
            if idx is None:
                return None
            return self._slots.get(idx)

    def all_slots(self) -> List[KDMSlot]:
        with self._lock:
            return list(self._slots.values())

    def populated_concept_slots(self) -> List[int]:
        with self._lock:
            return sorted(
                i for i in self._populated if i >= self._concept_start_idx
            )

    def state_dict(self) -> dict:
        """Serialise all memory state for checkpointing."""
        with self._lock:
            return {
                "addresses": self.addresses.tolist(),
                "values": self.values.tolist(),
                "slots": {
                    str(k): {
                        "slot_id": v.slot_id,
                        "concept_id": v.concept_id,
                        "write_source": v.write_source.name,
                        "created_at": v.created_at,
                        "last_accessed": v.last_accessed,
                        "access_count": v.access_count,
                        "meta_type": v.meta_type.value if v.meta_type else None,
                    }
                    for k, v in self._slots.items()
                },
                "beta": self._beta,
            }

    def load_state_dict(self, state: dict) -> None:
        """Restore memory state from a checkpoint."""
        with self._lock:
            self.addresses = np.array(state["addresses"], dtype=np.float32)
            self.values = np.array(state["values"], dtype=np.float32)
            self._beta = state.get("beta", self._kdm_cfg.sharpness_beta)
            self._slots = {}
            self._populated = set()
            self._concept_to_slot = {}
            for k, v in state["slots"].items():
                idx = int(k)
                meta = MetaConceptType(v["meta_type"]) if v["meta_type"] else None
                slot = KDMSlot(
                    slot_id=idx,
                    address=self.addresses[idx].copy(),
                    value=self.values[idx].copy(),
                    concept_id=v["concept_id"],
                    write_source=WriteSource[v["write_source"]],
                    created_at=v["created_at"],
                    last_accessed=v["last_accessed"],
                    access_count=v["access_count"],
                    meta_type=meta,
                )
                self._slots[idx] = slot
                self._populated.add(idx)
                self._concept_to_slot[v["concept_id"]] = idx