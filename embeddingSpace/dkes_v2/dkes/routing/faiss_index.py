"""
FAISSIndexManager - manages main expert indices and KDM address sub-index.

Uses double-buffering (active/staging) for atomic swaps so routing is
never blocked during index updates.
"""
from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np
from dkes.utils.config import DKESConfig
from dkes.utils.types import EmbeddingVector

logger = logging.getLogger(__name__)


class _NumpyIndex:
    """Brute-force cosine index (fallback when faiss unavailable)."""
    def __init__(self, dim: int):
        self.d = dim
        self._vecs: Optional[np.ndarray] = None
        self._ids: List[int] = []
        self.ntotal = 0

    def add_with_ids(self, vecs: np.ndarray, ids):
        self._vecs = vecs.copy() if self._vecs is None else np.vstack([self._vecs, vecs])
        self._ids.extend(list(ids))
        self.ntotal = len(self._ids)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._vecs is None or self.ntotal == 0:
            return np.array([[-1.0]*k]), np.array([[-1]*k])
        k = min(k, self.ntotal)
        q = query[0] / (np.linalg.norm(query[0]) + 1e-8)
        n = self._vecs / (np.linalg.norm(self._vecs, axis=1, keepdims=True) + 1e-8)
        sims = n @ q
        top = np.argsort(sims)[::-1][:k]
        return np.array([[sims[i] for i in top]]), np.array([[self._ids[i] for i in top]])

    def reset(self):
        self._vecs = None; self._ids = []; self.ntotal = 0


def _make_index(dim: int):
    try:
        import faiss
        return faiss.IndexFlatIP(dim)
    except ImportError:
        return _NumpyIndex(dim)


class _DoubleBufferIndex:
    """Atomic-swap double-buffered index for safe concurrent reads during updates."""
    def __init__(self, dim: int):
        self._lock = threading.Lock()
        self._active = _make_index(dim)
        self.d = dim

    def search(self, query: np.ndarray, k: int):
        with self._lock:
            idx = self._active
        return idx.search(query, k)

    def rebuild(self, vecs: np.ndarray, ids: np.ndarray):
        staging = _make_index(self.d)
        if hasattr(staging, "add_with_ids"):
            try:
                import faiss
                staging.add_with_ids(vecs.astype(np.float32), ids.astype(np.int64))
            except ImportError:
                staging.add_with_ids(vecs.astype(np.float32), ids)
        with self._lock:
            self._active = staging

    @property
    def ntotal(self):
        with self._lock:
            return self._active.ntotal


class FAISSIndexManager:
    """
    Manages coarse/fine expert indices and KDM address sub-index.

    Coarse index: 64-dim for Stage 1 candidate shortlisting.
    Fine index:   768-dim for Stage 2 re-ranking.
    KDM index:    768-dim for memory address lookup.
    """

    def __init__(self, cfg: DKESConfig, audit=None):
        self._cfg = cfg
        self._audit = audit
        D_c = cfg.backbone.coarse_dim
        D_f = cfg.backbone.fine_dim

        self._coarse = _DoubleBufferIndex(D_c)
        self._fine   = _DoubleBufferIndex(D_f)
        self._kdm    = _NumpyIndex(D_f)
        self._kdm_lock = threading.Lock()

        self._expert_to_int: Dict[str, int] = {}
        self._int_to_expert: Dict[int, str] = {}
        self._next_int: int = 0

        self._coarse_centroids: Dict[str, np.ndarray] = {}
        self._fine_centroids: Dict[str, np.ndarray] = {}

    # -- Expert index --

    def register_expert(self, expert_id: str, coarse: np.ndarray, fine: np.ndarray):
        """Add a single expert and rebuild both indices atomically."""
        eid = self._get_or_register_int(expert_id)
        self._coarse_centroids[expert_id] = coarse.copy()
        self._fine_centroids[expert_id] = fine.copy()
        self._rebuild_expert_indices()
        logger.info("FAISSIndexManager: registered expert %s (int=%d)", expert_id, eid)

    def _rebuild_expert_indices(self):
        ids = sorted(self._expert_to_int.values())
        id_order = sorted(self._expert_to_int.keys(), key=lambda e: self._expert_to_int[e])
        if not id_order:
            return
        coarse_m = np.stack([self._coarse_centroids[e] for e in id_order]).astype(np.float32)
        fine_m   = np.stack([self._fine_centroids[e]   for e in id_order]).astype(np.float32)
        coarse_m /= np.linalg.norm(coarse_m, axis=1, keepdims=True) + 1e-8
        fine_m   /= np.linalg.norm(fine_m,   axis=1, keepdims=True) + 1e-8
        int_ids = np.array([self._expert_to_int[e] for e in id_order])
        self._coarse.rebuild(coarse_m, int_ids)
        self._fine.rebuild(fine_m, int_ids)
        if self._audit:
            from dkes.utils.types import AuditEventKind
            self._audit.emit(AuditEventKind.INDEX_REBUILD, "faiss_index",
                             {"n_experts": len(id_order), "action": "rebuild"})

    def coarse_search(self, query_coarse: np.ndarray, k: int) -> List[str]:
        """Return top-k expert IDs by 64-dim cosine similarity."""
        q = (query_coarse / (np.linalg.norm(query_coarse) + 1e-8)).astype(np.float32)
        _, ids = self._coarse.search(q[None, :], k)
        return [self._int_to_expert[i] for i in ids[0].tolist() if i in self._int_to_expert]

    def fine_rerank(self, query_fine: np.ndarray, candidates: List[str]) -> List[Tuple[str, float]]:
        """Re-rank a candidate list by 768-dim cosine similarity."""
        if not candidates:
            return []
        q = (query_fine / (np.linalg.norm(query_fine) + 1e-8)).astype(np.float32)
        results = []
        for eid in candidates:
            fine_c = self._fine_centroids.get(eid)
            if fine_c is None:
                continue
            fine_c_n = fine_c / (np.linalg.norm(fine_c) + 1e-8)
            sim = float(q @ fine_c_n)
            results.append((eid, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    # -- KDM address index --

    def rebuild_kdm_index(self, addresses: np.ndarray, slot_ids: List[int]):
        """Rebuild KDM address sub-index after batch updates."""
        with self._kdm_lock:
            self._kdm = _NumpyIndex(self._cfg.backbone.fine_dim)
            if len(slot_ids) > 0:
                self._kdm.add_with_ids(addresses.astype(np.float32), np.array(slot_ids))
        logger.debug("FAISSIndexManager: KDM address index rebuilt with %d slots", len(slot_ids))

    def kdm_address_search(self, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        """Find top-k KDM slot addresses nearest the query."""
        q = (query / (np.linalg.norm(query) + 1e-8)).astype(np.float32)
        with self._kdm_lock:
            scores, ids = self._kdm.search(q[None, :], k)
        return ids[0].tolist(), scores[0].tolist()

    # -- Helpers --

    def _get_or_register_int(self, expert_id: str) -> int:
        if expert_id not in self._expert_to_int:
            i = self._next_int
            self._expert_to_int[expert_id] = i
            self._int_to_expert[i] = expert_id
            self._next_int += 1
        return self._expert_to_int[expert_id]

    @property
    def n_experts(self) -> int:
        return len(self._expert_to_int)