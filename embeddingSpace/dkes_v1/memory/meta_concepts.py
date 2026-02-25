"""
MetaConceptLayer - manages the top-256 reserved meta-concept slots.

M1 - Domain Proximity Summaries (weekly)
M2 - Query Pattern Templates (on-demand, after N uses)
M3 - Uncertainty Landscape Markers (continuous fast timescale)
M4 - Cross-Domain Analogy Bridges (on-demand from EASM)
"""
from __future__ import annotations
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
from dkes.utils.config import DKESConfig
from dkes.utils.types import EmbeddingVector, MetaConceptType, WriteSource

logger = logging.getLogger(__name__)

_META_TYPE_QUOTAS = {
    MetaConceptType.DOMAIN_PROXIMITY:   64,
    MetaConceptType.QUERY_PATTERN:      96,
    MetaConceptType.UNCERTAINTY_MARKER: 64,
    MetaConceptType.ANALOGY_BRIDGE:     32,
}
assert sum(_META_TYPE_QUOTAS.values()) == 256


class MetaConceptLayer:
    def __init__(self, cfg: DKESConfig, kdm, audit=None):
        self._cfg = cfg
        self._kdm = kdm
        self._audit = audit
        self._lock = threading.Lock()
        self._type_ranges: Dict[MetaConceptType, range] = {}
        offset = 0
        for mtype, quota in _META_TYPE_QUOTAS.items():
            self._type_ranges[mtype] = range(offset, offset + quota)
            offset += quota
        self._slot_to_id: Dict[int, str] = {}
        self._id_to_slot: Dict[str, int] = {}
        self._free: Dict[MetaConceptType, List[int]] = {
            mtype: list(r) for mtype, r in self._type_ranges.items()
        }
        self._m2_templates: Dict[int, dict] = {}
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        logger.info("MetaConceptLayer initialised: %d total meta slots", sum(_META_TYPE_QUOTAS.values()))

    def n_free_slots(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._free.values())

    def n_free_by_type(self, mtype: MetaConceptType) -> int:
        with self._lock:
            return len(self._free[mtype])

    def match_query_pattern(self, query_embedding: EmbeddingVector, threshold: float = None):
        threshold = threshold or self._cfg.kdm.meta_m2_match_distance
        if not self._m2_templates:
            return None
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        best_slot, best_dist = None, float("inf")
        with self._lock:
            for slot_idx in self._m2_templates:
                addr = self._kdm.addresses[slot_idx]
                a_norm = addr / (np.linalg.norm(addr) + 1e-8)
                cos_dist = 1.0 - float(q_norm @ a_norm)
                if cos_dist < best_dist:
                    best_dist = cos_dist
                    best_slot = slot_idx
        if best_slot is not None and best_dist <= threshold:
            confidence = 1.0 - best_dist
            if confidence >= self._cfg.kdm.meta_m2_match_confidence:
                return {**self._m2_templates[best_slot], "confidence": confidence, "slot_id": best_slot}
        return None

    def get_uncertainty_markers(self, query_embedding: EmbeddingVector, radius: float = 0.20):
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        markers = []
        with self._lock:
            for slot_idx in self._type_ranges[MetaConceptType.UNCERTAINTY_MARKER]:
                slot = self._kdm.get_slot(slot_idx)
                if slot is None:
                    continue
                addr = self._kdm.addresses[slot_idx]
                a_norm = addr / (np.linalg.norm(addr) + 1e-8)
                cos_dist = 1.0 - float(q_norm @ a_norm)
                if cos_dist <= radius:
                    markers.append({"slot_id": slot_idx, "concept_id": slot.concept_id, "distance": cos_dist})
        return sorted(markers, key=lambda x: x["distance"])

    def get_analogy_bridge(self, expert_a: str, expert_b: str):
        bridge_id = self._bridge_id(expert_a, expert_b)
        with self._lock:
            slot_idx = self._id_to_slot.get(bridge_id)
            if slot_idx is None:
                return None
            slot = self._kdm.get_slot(slot_idx)
            if slot is None:
                return None
            return {"bridge_id": bridge_id, "slot_id": slot_idx,
                    "value": self._kdm.values[slot_idx].copy(), "access_count": slot.access_count}

    def write_domain_proximity(self, domain_a, domain_b, embedding_a, embedding_b,
                                overlap_score, synthesis_quality, trace_id=None):
        concept_id = f"M1:{domain_a}:{domain_b}"
        addr = (embedding_a + embedding_b) / 2.0
        addr = (addr / (np.linalg.norm(addr) + 1e-8)).astype(np.float32)
        value = addr.copy()
        value[0] = overlap_score
        value[1] = synthesis_quality
        value = (value / (np.linalg.norm(value) + 1e-8)).astype(np.float32)
        slot_idx = self._get_or_allocate(concept_id, MetaConceptType.DOMAIN_PROXIMITY)
        if slot_idx is None:
            logger.warning("MetaConceptLayer: no M1 slots available")
            return None
        self._kdm.write_meta_slot(slot_idx, addr, value, MetaConceptType.DOMAIN_PROXIMITY,
                                   concept_id, WriteSource.MANUAL_INJECTION)
        if self._audit:
            self._audit.meta_write(MetaConceptType.DOMAIN_PROXIMITY.value, slot_idx,
                                   [domain_a, domain_b], f"overlap={overlap_score:.3f}", trace_id)
        return slot_idx

    def observe_query_pattern(self, pattern_key, pattern_embedding, decomposition_template, trace_id=None):
        with self._lock:
            self._pattern_counts[pattern_key] += 1
            count = self._pattern_counts[pattern_key]
        if count >= self._cfg.kdm.meta_m2_template_threshold:
            return self._write_query_pattern(pattern_key, pattern_embedding, decomposition_template, trace_id)
        return False

    def write_uncertainty_marker(self, region_centroid, mean_u_base, n_uncertain_queries,
                                  concept_id=None, trace_id=None):
        cid = concept_id or f"M3:unc:{time.time():.0f}"
        addr = (region_centroid / (np.linalg.norm(region_centroid) + 1e-8)).astype(np.float32)
        value = addr.copy()
        value[0] = mean_u_base
        value[1] = min(1.0, n_uncertain_queries / 100)
        value = (value / (np.linalg.norm(value) + 1e-8)).astype(np.float32)
        slot_idx = self._get_or_allocate(cid, MetaConceptType.UNCERTAINTY_MARKER)
        if slot_idx is None:
            return None
        self._kdm.write_meta_slot(slot_idx, addr, value, MetaConceptType.UNCERTAINTY_MARKER,
                                   cid, WriteSource.CLUSTER_THRESHOLD)
        if self._audit:
            self._audit.meta_write(MetaConceptType.UNCERTAINTY_MARKER.value, slot_idx,
                                   [cid], f"mean_u={mean_u_base:.3f} n={n_uncertain_queries}", trace_id)
        return slot_idx

    def write_analogy_bridge(self, expert_a, expert_b, shared_concept_embedding,
                              equivalence_score, trace_id=None):
        bridge_id = self._bridge_id(expert_a, expert_b)
        addr = (shared_concept_embedding / (np.linalg.norm(shared_concept_embedding) + 1e-8)).astype(np.float32)
        value = addr.copy()
        value[0] = equivalence_score
        value = (value / (np.linalg.norm(value) + 1e-8)).astype(np.float32)
        slot_idx = self._get_or_allocate(bridge_id, MetaConceptType.ANALOGY_BRIDGE)
        if slot_idx is None:
            logger.warning("MetaConceptLayer: no M4 slots for bridge %s", bridge_id)
            return None
        self._kdm.write_meta_slot(slot_idx, addr, value, MetaConceptType.ANALOGY_BRIDGE,
                                   bridge_id, WriteSource.MANUAL_INJECTION)
        if self._audit:
            self._audit.meta_write(MetaConceptType.ANALOGY_BRIDGE.value, slot_idx,
                                   [expert_a, expert_b], f"equiv={equivalence_score:.3f}", trace_id)
        return slot_idx

    def _get_or_allocate(self, concept_id, mtype):
        with self._lock:
            if concept_id in self._id_to_slot:
                return self._id_to_slot[concept_id]
            if not self._free[mtype]:
                return None
            slot_idx = self._free[mtype].pop(0)
            self._slot_to_id[slot_idx] = concept_id
            self._id_to_slot[concept_id] = slot_idx
            return slot_idx

    def _write_query_pattern(self, pattern_key, pattern_embedding, template, trace_id):
        cid = f"M2:{pattern_key}"
        if cid in self._id_to_slot:
            return False
        addr = (pattern_embedding / (np.linalg.norm(pattern_embedding) + 1e-8)).astype(np.float32)
        slot_idx = self._get_or_allocate(cid, MetaConceptType.QUERY_PATTERN)
        if slot_idx is None:
            logger.warning("MetaConceptLayer: M2 quota exhausted for %s", pattern_key)
            return False
        self._kdm.write_meta_slot(slot_idx, addr, addr.copy(), MetaConceptType.QUERY_PATTERN,
                                   cid, WriteSource.CLUSTER_THRESHOLD)
        with self._lock:
            self._m2_templates[slot_idx] = {**template, "pattern_key": pattern_key}
        if self._audit:
            self._audit.meta_write(MetaConceptType.QUERY_PATTERN.value, slot_idx, [pattern_key],
                                   f"written after {self._pattern_counts[pattern_key]} uses", trace_id)
        logger.info("MetaConceptLayer: M2 pattern written for %s", pattern_key)
        return True

    @staticmethod
    def _bridge_id(a, b):
        return "M4:" + ":".join(sorted([a, b]))