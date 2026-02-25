"""
CompositeEmbedding — computes q~(q) = LayerNorm(e(q) + gamma * r(q)).

This module is the mathematical core of the DKES architecture: it
combines the frozen backbone embedding with the KDM memory read to
produce the composite representation used in routing.

Responsibilities
----------------
- Maintain and update the gamma interpolation scalar
- Apply domain projection adapters to backbone embeddings before KDM read
- Normalise the composite via LayerNorm
- Enforce the gamma ceiling schedule during ramp-up
- Emit gamma update audit records on meaningful changes
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from dkes.utils.config import DKESConfig
from dkes.utils.types import CompositeEmbedding, EmbeddingVector

logger = logging.getLogger(__name__)


class CompositeEmbeddingLayer:
    """
    Manages the composite embedding q~(q) = LayerNorm(e(q) + gamma * r(q)).

    Parameters
    ----------
    cfg:   DKESConfig
    audit: AuditLog (optional)
    """

    # Minimum gamma change to trigger an audit record (avoid log spam)
    _AUDIT_GAMMA_DELTA_THRESHOLD = 0.001

    def __init__(self, cfg: DKESConfig, audit=None):
        self._cfg = cfg
        self._kdm_cfg = cfg.kdm
        self._audit = audit
        self._lock = threading.Lock()

        self._gamma: float = cfg.kdm.gamma_init
        self._last_audited_gamma: float = cfg.kdm.gamma_init

        # LayerNorm running statistics (simplified: tracked per-dimension mean/var)
        D = cfg.backbone.embedding_dim
        self._ln_mean: np.ndarray = np.zeros(D, dtype=np.float64)
        self._ln_var:  np.ndarray = np.ones(D, dtype=np.float64)
        self._ln_n:    int = 0
        self._ln_eps:  float = 1e-6

        # Gradient accumulator for gamma (soft SGD updates from routing quality)
        self._gamma_grad_accum: float = 0.0
        self._gamma_lr: float = 1e-4

        logger.info(
            "CompositeEmbeddingLayer init: gamma=%.4f ceiling_threshold=%d",
            self._gamma, cfg.kdm.gamma_ceiling_concepts,
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        backbone_emb: EmbeddingVector,
        memory_read: EmbeddingVector,
        top_slot_indices: list,
        top_slot_weights: list,
        query_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> CompositeEmbedding:
        """
        Compute the composite embedding.

        Parameters
        ----------
        backbone_emb:       e(q) from the frozen MRL encoder
        memory_read:        r(q) from KanervaMemory.read()
        top_slot_indices:   indices of top KDM slots (for audit)
        top_slot_weights:   their weights (for audit)
        query_id:           optional identifier for the query
        trace_id:           optional trace context

        Returns
        -------
        CompositeEmbedding named tuple
        """
        with self._lock:
            gamma = self._gamma

        # Raw combination: e(q) + gamma * r(q)
        raw = backbone_emb + gamma * memory_read

        # LayerNorm normalisation
        composite = self._layer_norm(raw)

        # Update running LayerNorm statistics (Welford online algorithm)
        self._update_ln_stats(raw)

        # Audit if meaningful
        if self._audit:
            self._audit.kdm_read(
                query_id=query_id or "",
                top_slot_ids=top_slot_indices,
                top_weights=top_slot_weights,
                gamma=gamma,
                memory_read_norm=float(np.linalg.norm(memory_read)),
                composite_norm=float(np.linalg.norm(composite)),
                trace_id=trace_id,
            )

        return CompositeEmbedding(
            composite=composite.astype(np.float32),
            backbone=backbone_emb.astype(np.float32),
            memory_read=memory_read.astype(np.float32),
            gamma=gamma,
            top_slot_indices=top_slot_indices,
            top_slot_weights=top_slot_weights,
        )

    # ------------------------------------------------------------------
    # Gamma management
    # ------------------------------------------------------------------

    @property
    def gamma(self) -> float:
        with self._lock:
            return self._gamma

    def gamma_ceiling_active(self, n_populated_slots: int) -> bool:
        """Returns True if the gamma ceiling constraint is currently active."""
        return n_populated_slots < self._kdm_cfg.gamma_ceiling_concepts

    def get_effective_gamma(self, n_populated_slots: int) -> float:
        """
        Return the effective gamma, respecting the ceiling schedule.
        The ceiling is: gamma_max(t) = gamma_ceiling_max * min(1, N / N_ceiling)
        """
        with self._lock:
            raw_gamma = self._gamma

        if n_populated_slots >= self._kdm_cfg.gamma_ceiling_concepts:
            return raw_gamma  # ceiling lifted

        ceiling = (
            self._kdm_cfg.gamma_ceiling_max
            * (n_populated_slots / self._kdm_cfg.gamma_ceiling_concepts)
        )
        return min(raw_gamma, ceiling)

    def update_gamma(
        self,
        routing_quality_signal: float,
        n_populated_slots: int,
        trace_id: Optional[str] = None,
    ):
        """
        Update gamma via a soft gradient step based on routing quality signal.

        routing_quality_signal: in [-1, +1]
          +1 means the KDM read helped routing (should increase gamma)
          -1 means the KDM read hurt routing (should decrease gamma)
        """
        with self._lock:
            old_gamma = self._gamma
            # Accumulate gradient
            self._gamma_grad_accum += routing_quality_signal
            # Apply update
            grad = np.tanh(self._gamma_grad_accum * 0.1)
            new_gamma = old_gamma + self._gamma_lr * grad
            # Clamp to [0.01, 1.0]
            new_gamma = float(np.clip(new_gamma, 0.01, 1.0))
            self._gamma = new_gamma
            # Decay accumulator
            self._gamma_grad_accum *= 0.9

        delta = abs(new_gamma - old_gamma)
        ceiling_active = self.gamma_ceiling_active(n_populated_slots)

        if self._audit and delta > self._AUDIT_GAMMA_DELTA_THRESHOLD:
            self._audit.gamma_update(
                old_gamma=old_gamma,
                new_gamma=new_gamma,
                n_populated_slots=n_populated_slots,
                ceiling_active=ceiling_active,
                trace_id=trace_id,
            )
            self._last_audited_gamma = new_gamma

        if delta > 0.01:
            logger.debug(
                "Gamma updated: %.5f -> %.5f (ceiling_active=%s populated=%d)",
                old_gamma, new_gamma, ceiling_active, n_populated_slots,
            )

    # ------------------------------------------------------------------
    # LayerNorm
    # ------------------------------------------------------------------

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """
        Online LayerNorm using running mean and variance.
        Falls back to instance-level norm (standard LN) for the first
        1000 observations before the running stats are reliable.
        """
        if self._ln_n < 1000:
            # Instance LayerNorm: normalise the current vector itself
            mean = x.mean()
            var = x.var()
            return ((x - mean) / np.sqrt(var + self._ln_eps)).astype(np.float32)

        mean = self._ln_mean.astype(np.float32)
        var = self._ln_var.astype(np.float32)
        return ((x - mean) / np.sqrt(var + self._ln_eps)).astype(np.float32)

    def _update_ln_stats(self, x: np.ndarray):
        """Welford online update of running mean and variance."""
        with self._lock:
            self._ln_n += 1
            n = self._ln_n
            delta = x - self._ln_mean
            self._ln_mean += delta / n
            delta2 = x - self._ln_mean
            # Running M2 approximation (simplified to moving average of squared diff)
            self._ln_var = (self._ln_var * (n - 1) + delta * delta2) / n

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        with self._lock:
            return {
                "gamma": self._gamma,
                "gamma_grad_accum": self._gamma_grad_accum,
                "ln_mean": self._ln_mean.tolist(),
                "ln_var": self._ln_var.tolist(),
                "ln_n": self._ln_n,
            }

    def load_state_dict(self, state: dict):
        with self._lock:
            self._gamma = state["gamma"]
            self._gamma_grad_accum = state.get("gamma_grad_accum", 0.0)
            self._ln_mean = np.array(state["ln_mean"], dtype=np.float64)
            self._ln_var = np.array(state["ln_var"], dtype=np.float64)
            self._ln_n = state["ln_n"]