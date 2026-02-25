"""
ColdStore — archive for evicted KDM concept slots with reactivation support.

When a KDM slot is evicted (via LFU or age demotion), its address, value,
and metadata are moved here rather than discarded. The cold store supports:

  - Persistence across restarts (JSON serialisation)
  - Reactivation on access: if a cold concept is queried again, it can
    be written back to the live KDM
  - Pruning of concepts that remain unaccessed for cold_ttl_days

Architecture Spec reference: Section 1.3 (Slot allocation / cold store),
Section 1.A (Limitation and mitigation)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dkes.utils.config import DKESConfig
from dkes.utils.types import (
    AuditEventKind,
    ConceptCluster,
    EmbeddingVector,
    EvictionReason,
    KDMSlot,
    WriteSource,
)

logger = logging.getLogger(__name__)


class ColdStoreEntry:
    """
    An archived KDM slot, preserved for potential reactivation.

    Fields beyond the original KDMSlot:
      evicted_at:    unix timestamp of eviction
      eviction_reason: why it was evicted
      reactivation_count: how many times it has been evicted→reactivated
    """

    __slots__ = (
        "slot",
        "evicted_at",
        "eviction_reason",
        "reactivation_count",
        "address",
        "value",
    )

    def __init__(
        self,
        slot: KDMSlot,
        address: EmbeddingVector,
        value: EmbeddingVector,
        eviction_reason: EvictionReason,
    ):
        self.slot = slot
        self.address = address.copy().astype(np.float32)
        self.value = value.copy().astype(np.float32)
        self.evicted_at = time.time()
        self.eviction_reason = eviction_reason
        self.reactivation_count = 0

    def to_dict(self) -> dict:
        return {
            "concept_id": self.slot.concept_id,
            "write_source": self.slot.write_source.name,
            "created_at": self.slot.created_at,
            "last_accessed": self.slot.last_accessed,
            "access_count": self.slot.access_count,
            "meta_type": self.slot.meta_type.value if self.slot.meta_type else None,
            "address": self.address.tolist(),
            "value": self.value.tolist(),
            "evicted_at": self.evicted_at,
            "eviction_reason": self.eviction_reason.name,
            "reactivation_count": self.reactivation_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ColdStoreEntry":
        from dkes.utils.types import MetaConceptType
        slot = KDMSlot(
            slot_id=-1,  # no longer in live memory
            address=np.array(d["address"], dtype=np.float32),
            value=np.array(d["value"], dtype=np.float32),
            concept_id=d["concept_id"],
            write_source=WriteSource[d["write_source"]],
            created_at=d["created_at"],
            last_accessed=d["last_accessed"],
            access_count=d["access_count"],
            meta_type=MetaConceptType(d["meta_type"]) if d["meta_type"] else None,
        )
        entry = cls.__new__(cls)
        entry.slot = slot
        entry.address = np.array(d["address"], dtype=np.float32)
        entry.value = np.array(d["value"], dtype=np.float32)
        entry.evicted_at = d["evicted_at"]
        entry.eviction_reason = EvictionReason[d["eviction_reason"]]
        entry.reactivation_count = d.get("reactivation_count", 0)
        return entry


class ColdStore:
    """
    Archive of evicted KDM concept slots, with similarity-based lookup
    for reactivation and periodic pruning of permanently stale entries.

    Parameters
    ----------
    cfg:    DKESConfig
    audit:  AuditLog (optional)

    Thread Safety
    -------------
    All public methods are thread-safe.

    Reactivation Protocol
    ---------------------
    When the calling code detects a query that semantically matches a
    cold concept (cosine distance < reactivation_threshold), it can call
    pop_for_reactivation() to retrieve the entry and pass it back to
    KanervaMemory.write_concept() via a ConceptCluster. The reactivation
    count is incremented; entries reactivated repeatedly may warrant
    permanent concept-slot allocation.
    """

    # Cosine distance threshold to trigger a reactivation recommendation
    REACTIVATION_DISTANCE_THRESHOLD = 0.15

    def __init__(self, cfg: DKESConfig, audit=None):
        self._cfg = cfg
        self._audit = audit
        self._lock = threading.RLock()

        # concept_id → ColdStoreEntry
        self._entries: Dict[str, ColdStoreEntry] = {}

        # Cold TTL: entries older than this (days) are permanently pruned
        self._cold_ttl_days: float = cfg.kdm.cold_store_days * 3

        logger.info("ColdStore initialised (cold_ttl_days=%.0f)", self._cold_ttl_days)

    # ------------------------------------------------------------------
    # Archival
    # ------------------------------------------------------------------

    def archive(
        self,
        slot: KDMSlot,
        address: EmbeddingVector,
        value: EmbeddingVector,
        reason: EvictionReason,
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Archive an evicted KDM slot.

        Called by KanervaMemory._evict_lfu() immediately before clearing
        the slot. The slot is stored indexed by concept_id for O(1) lookup.

        Audit record emitted: AuditEventKind.KDM_COLD_RESTORE (archived event).
        """
        entry = ColdStoreEntry(
            slot=slot,
            address=address,
            value=value,
            eviction_reason=reason,
        )

        with self._lock:
            existing = self._entries.get(slot.concept_id)
            if existing is not None:
                # Preserve the higher reactivation count on re-eviction
                entry.reactivation_count = existing.reactivation_count
            self._entries[slot.concept_id] = entry

        logger.info(
            "ColdStore archived: concept=%s reason=%s (total_cold=%d)",
            slot.concept_id, reason.name, self.size,
        )

        if self._audit:
            self._audit.emit(
                AuditEventKind.KDM_EVICTION,
                "cold_store",
                {
                    "event": "archived",
                    "concept_id": slot.concept_id,
                    "reason": reason.name,
                    "access_count_at_eviction": slot.access_count,
                    "days_in_hot_store": (slot.last_accessed - slot.created_at) / 86400,
                    "cold_store_size": self.size,
                },
                trace_id=trace_id,
            )

    # ------------------------------------------------------------------
    # Lookup and Reactivation
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: EmbeddingVector,
        top_k: int = 5,
        max_distance: float = REACTIVATION_DISTANCE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Find cold entries whose addresses are within max_distance of query.

        Returns a list of matches sorted by ascending cosine distance.
        Each match dict contains: concept_id, distance, reactivation_count,
        access_count, days_cold.

        Used by the embedding space to detect reactivation opportunities.
        """
        if not self._entries:
            return []

        q_norm = query / (np.linalg.norm(query) + 1e-8)
        now = time.time()
        matches = []

        with self._lock:
            for cid, entry in self._entries.items():
                a_norm = entry.address / (np.linalg.norm(entry.address) + 1e-8)
                cos_dist = 1.0 - float(q_norm @ a_norm)
                if cos_dist <= max_distance:
                    matches.append({
                        "concept_id": cid,
                        "distance": cos_dist,
                        "reactivation_count": entry.reactivation_count,
                        "access_count": entry.slot.access_count,
                        "days_cold": (now - entry.evicted_at) / 86400,
                        "eviction_reason": entry.eviction_reason.name,
                    })

        matches.sort(key=lambda x: x["distance"])
        return matches[:top_k]

    def pop_for_reactivation(
        self,
        concept_id: str,
        trace_id: Optional[str] = None,
    ) -> Optional[ConceptCluster]:
        """
        Remove a concept from cold store and return it as a ConceptCluster
        suitable for re-writing to the live KDM.

        The ConceptCluster centroid is the archived address, and the expert
        distribution is reconstructed from the archived value vector.

        Returns None if the concept_id is not in cold store.
        """
        with self._lock:
            entry = self._entries.pop(concept_id, None)

        if entry is None:
            return None

        entry.reactivation_count += 1

        if self._audit:
            self._audit.emit(
                AuditEventKind.KDM_COLD_RESTORE,
                "cold_store",
                {
                    "event": "reactivated",
                    "concept_id": concept_id,
                    "reactivation_count": entry.reactivation_count,
                    "days_cold": (time.time() - entry.evicted_at) / 86400,
                    "original_access_count": entry.slot.access_count,
                    "cold_store_size_after": self.size,
                },
                trace_id=trace_id,
            )

        logger.info(
            "ColdStore reactivation: concept=%s (reactivation #%d)",
            concept_id, entry.reactivation_count,
        )

        # Reconstruct ConceptCluster from archived state.
        # expert_dist is approximated as uniform over a placeholder expert;
        # the caller should supply real expert_ids if available.
        cluster = ConceptCluster(
            cluster_id=concept_id,
            centroid=entry.address,
            n_queries=entry.slot.access_count,
            coherence=1.0,   # treat as coherent (was previously accepted)
            max_novelty=1.0,  # novelty re-evaluated by KDM at write time
            mean_u_blobs=0.5,
            expert_dist={"reactivated": 1.0},
        )
        return cluster

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_expired(self, trace_id: Optional[str] = None) -> int:
        """
        Remove entries older than cold_ttl_days from the cold store.

        Called by the background flush loop on the medium timescale.
        Returns the number of entries pruned.
        """
        cutoff = time.time() - (self._cold_ttl_days * 86400)
        pruned = []

        with self._lock:
            for cid, entry in list(self._entries.items()):
                if entry.evicted_at < cutoff:
                    pruned.append(cid)
                    del self._entries[cid]

        if pruned:
            logger.info("ColdStore pruned %d expired entries", len(pruned))
            if self._audit:
                self._audit.emit(
                    AuditEventKind.KDM_EVICTION,
                    "cold_store",
                    {
                        "event": "pruned_expired",
                        "count": len(pruned),
                        "concept_ids_sample": pruned[:5],
                        "cold_store_size_after": self.size,
                        "cold_ttl_days": self._cold_ttl_days,
                    },
                    trace_id=trace_id,
                )

        return len(pruned)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of concepts currently archived in cold store."""
        with self._lock:
            return len(self._entries)

    def get_entry(self, concept_id: str) -> Optional[ColdStoreEntry]:
        with self._lock:
            return self._entries.get(concept_id)

    def all_concept_ids(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict for metrics and status reporting."""
        now = time.time()
        with self._lock:
            entries = list(self._entries.values())

        if not entries:
            return {
                "size": 0,
                "mean_days_cold": 0.0,
                "max_days_cold": 0.0,
                "total_reactivation_count": 0,
                "by_eviction_reason": {},
            }

        days_cold = [(now - e.evicted_at) / 86400 for e in entries]
        by_reason: Dict[str, int] = {}
        for e in entries:
            by_reason[e.eviction_reason.name] = by_reason.get(e.eviction_reason.name, 0) + 1

        return {
            "size": len(entries),
            "mean_days_cold": float(np.mean(days_cold)),
            "max_days_cold": float(np.max(days_cold)),
            "total_reactivation_count": sum(e.reactivation_count for e in entries),
            "by_eviction_reason": by_reason,
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialise cold store for checkpoint inclusion."""
        with self._lock:
            return {
                "cold_ttl_days": self._cold_ttl_days,
                "entries": {cid: e.to_dict() for cid, e in self._entries.items()},
            }

    def load_state_dict(self, state: dict) -> None:
        """Restore cold store from a checkpoint."""
        with self._lock:
            self._cold_ttl_days = state.get("cold_ttl_days", self._cold_ttl_days)
            self._entries = {
                cid: ColdStoreEntry.from_dict(d)
                for cid, d in state.get("entries", {}).items()
            }
        logger.info("ColdStore restored: %d entries", self.size)

    def save_to_file(self, path: str | Path) -> None:
        """Save cold store independently (for operational inspection)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state_dict(), f, indent=2, default=str)
        logger.info("ColdStore saved to %s (%d entries)", path, self.size)

    def load_from_file(self, path: str | Path) -> None:
        """Load cold store from an independent snapshot."""
        with open(path) as f:
            state = json.load(f)
        self.load_state_dict(state)
