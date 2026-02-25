"""
MRLFunnel — two-stage coarse→fine MRL routing funnel.

Implements the two-stage FAISS routing described in Section 7.2 of the
architecture spec, with full integration of:
  - Domain projection adapters (applied before KDM read and Stage 1)
  - M2 template shortcut check (pre-Stage 1)
  - M3 uncertainty zone expansion (k_coarse boost)
  - Deadlock guard (routing deadlock protocol)
  - Budget-constrained multi-expert activation (M_active cap)
  - Per-domain learned routing thresholds (MLP or config fallback)

This class does NOT own any state — it is a stateless routing coordinator
that wraps FAISSIndexManager, DomainProjectionManager, and MetaConceptLayer.
DKESEmbeddingSpace is the owner of all state; MRLFunnel is a helper.

Architecture Spec reference:
  Section 7   — Geometric Router
  Section 7.2 — Routing Algorithm (Steps 1–5)
  Section 7.3 — Budget-Constrained Multi-Expert Activation
  Section 7.5 — Fast-Path Routing Cache
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dkes.utils.config import DKESConfig
from dkes.utils.types import (
    AuditEventKind,
    CompositeEmbedding,
    EmbeddingVector,
    RoutingResult,
)

logger = logging.getLogger(__name__)


class RoutingDeadlockError(Exception):
    """Raised when no expert candidate meets minimum routing margin."""


class MRLFunnel:
    """
    Stateless two-stage coarse→fine MRL routing coordinator.

    Accepts the current composite embedding and the live FAISS index,
    and executes the full routing algorithm from Stage 1 through expert
    selection and budget capping.

    Parameters
    ----------
    cfg:        DKESConfig
    faiss:      FAISSIndexManager instance
    meta:       MetaConceptLayer instance (for M2 shortcuts and M3 zones)
    projection: DomainProjectionManager instance
    audit:      AuditLog (optional)
    """

    def __init__(self, cfg: DKESConfig, faiss, meta, projection, audit=None):
        self._cfg = cfg
        self._rcfg = cfg.routing
        self._faiss = faiss
        self._meta = meta
        self._projection = projection
        self._audit = audit

        # Per-domain learned threshold MLP (domain_id → callable returning float).
        # Populated externally by the threshold learning subsystem.
        # Falls back to cfg.routing.default_routing_threshold if not set.
        self._threshold_mlp: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Primary routing entry point
    # ------------------------------------------------------------------

    def route(
        self,
        composite_result: CompositeEmbedding,
        coarse_embedding: EmbeddingVector,
        domain_id: Optional[str] = None,
        force_full_pipeline: bool = False,
        trace_id: Optional[str] = None,
    ) -> RoutingResult:
        """
        Execute the full MRL routing funnel.

        Stages
        ------
        1. M2 template shortcut check (pre-FAISS fast path)
        2. FAISS Stage 1: coarse 64-dim k-NN with M3 zone expansion
        3. FAISS Stage 2: fine 768-dim re-ranking
        4. Expert selection via U_base threshold
        5. Budget cap enforcement (M_active)
        6. Concept overlap merging
        7. Deadlock resolution

        Parameters
        ----------
        composite_result: output of CompositeEmbeddingLayer.compute()
        coarse_embedding: 64-dim backbone embedding for Stage 1
        domain_id:        optional domain label for threshold MLP lookup
        force_full_pipeline: skip M2 shortcut and cache checks
        trace_id:         optional trace context

        Returns
        -------
        RoutingResult
        """
        t_start = time.perf_counter()
        composite_vec = composite_result.composite

        # ---- M2 template shortcut ----
        if not force_full_pipeline:
            m2 = self._meta.match_query_pattern(composite_vec)
            if m2:
                return self._result_from_m2(m2, composite_result, t_start, trace_id)

        # ---- M3 zone check: expand k_coarse if in uncertainty region ----
        m3_markers = self._meta.get_uncertainty_markers(composite_vec)
        in_unc_region = len(m3_markers) > 0
        k_coarse = (
            self._rcfg.k_coarse_expanded if in_unc_region else self._rcfg.k_coarse
        )

        if self._audit and in_unc_region:
            self._audit.emit(
                AuditEventKind.ROUTING_QUERY,
                "mrl_funnel",
                {
                    "event": "m3_zone_expansion",
                    "k_coarse_expanded": k_coarse,
                    "n_markers": len(m3_markers),
                    "nearest_marker_distance": m3_markers[0]["distance"] if m3_markers else None,
                    "trace_id": trace_id,
                },
                trace_id=trace_id,
            )

        # ---- Stage 1: coarse k-NN ----
        t_stage1 = time.perf_counter()

        # Apply domain projection to coarse query if adapter exists
        projected_coarse, proj_applied = self._apply_projection(
            domain_id, coarse_embedding, trace_id
        )
        coarse_candidates = self._faiss.coarse_search(projected_coarse, k=k_coarse)

        stage1_latency_ms = (time.perf_counter() - t_stage1) * 1000

        if not coarse_candidates:
            logger.warning(
                "MRLFunnel Stage 1 returned no candidates (domain=%s trace=%s)",
                domain_id, trace_id,
            )
            return self._empty_result(composite_result, in_unc_region)

        # ---- Stage 2: fine 768-dim re-ranking ----
        t_stage2 = time.perf_counter()

        # Apply projection to fine composite embedding
        projected_fine, _ = self._apply_projection(domain_id, composite_vec, trace_id)
        fine_ranked: List[Tuple[str, float]] = self._faiss.fine_rerank(
            projected_fine, coarse_candidates
        )

        stage2_latency_ms = (time.perf_counter() - t_stage2) * 1000

        # Retain top k_fine × 2 for uncertainty evaluation
        k_fine = self._rcfg.k_fine
        top_candidates = fine_ranked[:k_fine * 2]
        fine_candidate_ids = [eid for eid, _ in top_candidates]

        # ---- Expert selection: threshold filtering ----
        threshold = self._get_threshold(domain_id)
        selected = self._apply_threshold(top_candidates, threshold)

        # ---- Deadlock guard ----
        deadlock_triggered = False
        if not selected:
            deadlock_triggered = True
            margin = self._rcfg.routing_deadlock_margin
            if fine_ranked and fine_ranked[0][1] >= margin:
                # Permit best expert below threshold, flag as deadlock
                selected = fine_ranked[:1]
                logger.warning(
                    "MRLFunnel routing deadlock resolved: best_score=%.3f "
                    "margin=%.3f (trace=%s)",
                    fine_ranked[0][1], margin, trace_id,
                )
            else:
                logger.warning(
                    "MRLFunnel routing deadlock: best_score=%.3f below margin=%.3f "
                    "(trace=%s) — returning empty result",
                    fine_ranked[0][1] if fine_ranked else 0.0, margin, trace_id,
                )
                if self._audit:
                    self._audit.emit(
                        AuditEventKind.ROUTING_QUERY,
                        "mrl_funnel",
                        {
                            "event": "routing_deadlock",
                            "best_score": fine_ranked[0][1] if fine_ranked else 0.0,
                            "margin": margin,
                            "n_coarse_candidates": len(coarse_candidates),
                            "trace_id": trace_id,
                        },
                        trace_id=trace_id,
                    )
                return self._empty_result(composite_result, in_unc_region)

        # ---- Concept overlap merging (Section 7.3) ----
        selected = self._merge_overlapping(selected)

        # ---- Budget cap: max M_active experts ----
        if len(selected) > self._rcfg.max_active_experts:
            selected = selected[:self._rcfg.max_active_experts]

        expert_ids    = [eid for eid, _ in selected]
        u_base_scores = [s for _, s in selected]
        adapter_ids   = [None] * len(expert_ids)

        # ---- Emit detailed routing audit record ----
        total_latency_ms = (time.perf_counter() - t_start) * 1000
        if self._audit:
            self._audit.emit(
                AuditEventKind.ROUTING_QUERY,
                "mrl_funnel",
                {
                    "event": "route_complete",
                    "trace_id": trace_id,
                    "domain_id": domain_id,
                    "expert_ids": expert_ids,
                    "u_base_scores": [round(u, 4) for u in u_base_scores],
                    "stage1_latency_ms": round(stage1_latency_ms, 2),
                    "stage2_latency_ms": round(stage2_latency_ms, 2),
                    "total_latency_ms": round(total_latency_ms, 2),
                    "k_coarse_used": k_coarse,
                    "n_coarse_candidates": len(coarse_candidates),
                    "n_fine_candidates": len(fine_ranked),
                    "n_selected": len(expert_ids),
                    "threshold_used": round(threshold, 4),
                    "projection_applied": proj_applied,
                    "in_uncertainty_region": in_unc_region,
                    "deadlock_triggered": deadlock_triggered,
                    "gamma": round(composite_result.gamma, 5),
                },
                trace_id=trace_id,
            )

        return RoutingResult(
            expert_ids=expert_ids,
            adapter_ids=adapter_ids,
            u_base_scores=u_base_scores,
            coarse_candidates=coarse_candidates,
            fine_candidates=fine_candidate_ids,
            used_meta_shortcut=False,
            uncertainty_region=in_unc_region,
            composite_emb=composite_result,
        )

    # ------------------------------------------------------------------
    # Multi-expert activation planning (Section 7.3)
    # ------------------------------------------------------------------

    def plan_multi_expert_activation(
        self,
        domain_labels: List[str],
        integration_structure: str,
        composite_vec: EmbeddingVector,
        coarse_embedding: EmbeddingVector,
        composite_result: CompositeEmbedding,
        trace_id: Optional[str] = None,
    ) -> RoutingResult:
        """
        Route a multi-expert query with EASM-provided domain labels.

        Executes the full routing funnel for each domain label and applies
        budget-constrained activation based on integration structure.

        Parameters
        ----------
        domain_labels:        list of natural-language domain strings from EASM
        integration_structure: "parallel" | "sequential" | "hierarchical"
        composite_vec:        composite embedding for the full query
        coarse_embedding:     coarse embedding for Stage 1
        composite_result:     full CompositeEmbedding object
        trace_id:             optional trace context

        Returns
        -------
        RoutingResult — merged result with up to M_active experts
        """
        all_candidates: Dict[str, List[Tuple[str, float]]] = {}

        for label in domain_labels:
            # Route per domain label; use label as domain_id for projection lookup
            m3_markers = self._meta.get_uncertainty_markers(composite_vec)
            in_unc = len(m3_markers) > 0
            k = self._rcfg.k_coarse_expanded if in_unc else self._rcfg.k_coarse

            projected_coarse, _ = self._apply_projection(label, coarse_embedding, trace_id)
            coarse_cands = self._faiss.coarse_search(projected_coarse, k=k)

            if not coarse_cands:
                continue

            projected_fine, _ = self._apply_projection(label, composite_vec, trace_id)
            ranked = self._faiss.fine_rerank(projected_fine, coarse_cands)
            all_candidates[label] = ranked

        # Merge candidates across labels, deduplicate by expert_id
        merged: Dict[str, float] = {}
        for label, candidates in all_candidates.items():
            threshold = self._get_threshold(label)
            for eid, score in candidates:
                if score >= threshold:
                    # Take max score if same expert appears for multiple labels
                    merged[eid] = max(merged.get(eid, 0.0), score)

        # Apply integration structure priority
        ordered = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        ordered = self._merge_overlapping(ordered)

        # Apply budget cap
        capped = ordered[:self._rcfg.max_active_experts]
        expert_ids    = [eid for eid, _ in capped]
        u_base_scores = [s for _, s in capped]

        if self._audit and capped:
            self._audit.emit(
                AuditEventKind.ROUTING_QUERY,
                "mrl_funnel",
                {
                    "event": "multi_expert_activation",
                    "domain_labels": domain_labels,
                    "integration_structure": integration_structure,
                    "n_labels": len(domain_labels),
                    "n_unique_candidates": len(merged),
                    "n_selected": len(expert_ids),
                    "expert_ids": expert_ids,
                    "trace_id": trace_id,
                },
                trace_id=trace_id,
            )

        return RoutingResult(
            expert_ids=expert_ids,
            adapter_ids=[None] * len(expert_ids),
            u_base_scores=u_base_scores,
            coarse_candidates=list(merged.keys()),
            fine_candidates=[eid for eid, _ in ordered],
            used_meta_shortcut=False,
            uncertainty_region=False,
            composite_emb=composite_result,
        )

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def register_threshold_mlp(self, domain_id: str, mlp_callable) -> None:
        """
        Register a learned threshold callable for a domain.

        The callable should accept no arguments and return a float in [0, 1].
        Used by the threshold learning subsystem to override config defaults.
        """
        self._threshold_mlp[domain_id] = mlp_callable
        logger.info("MRLFunnel: registered threshold MLP for domain=%s", domain_id)

    def _get_threshold(self, domain_id: Optional[str]) -> float:
        """Lookup learned threshold or fall back to config default."""
        if domain_id and domain_id in self._threshold_mlp:
            try:
                return float(self._threshold_mlp[domain_id]())
            except Exception as exc:
                logger.warning("Threshold MLP failed for %s: %s", domain_id, exc)
        return self._rcfg.default_routing_threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_projection(
        self,
        domain_id: Optional[str],
        embedding: EmbeddingVector,
        trace_id: Optional[str],
    ) -> Tuple[EmbeddingVector, bool]:
        """Apply domain projection if available; return (embedding, applied)."""
        if domain_id is None or self._projection is None:
            return embedding, False
        return self._projection.apply(domain_id, embedding, trace_id=trace_id)

    def _apply_threshold(
        self,
        ranked: List[Tuple[str, float]],
        threshold: float,
    ) -> List[Tuple[str, float]]:
        """Return all candidates with score >= threshold."""
        return [(eid, s) for eid, s in ranked if s >= threshold]

    def _merge_overlapping(
        self,
        candidates: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Merge domain-label candidates with cosine similarity > concept_overlap_threshold.

        Uses centroid similarity of expert embeddings (from FAISS fine centroids).
        """
        if len(candidates) <= 1:
            return candidates

        threshold = self._rcfg.concept_overlap_threshold
        kept = []
        seen_centroids: List[np.ndarray] = []

        for eid, score in candidates:
            fine_centroid = self._faiss._fine_centroids.get(eid)
            if fine_centroid is None:
                kept.append((eid, score))
                continue

            fc_norm = fine_centroid / (np.linalg.norm(fine_centroid) + 1e-8)
            is_duplicate = False

            for prev_c in seen_centroids:
                prev_norm = prev_c / (np.linalg.norm(prev_c) + 1e-8)
                sim = float(fc_norm @ prev_norm)
                if sim >= threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append((eid, score))
                seen_centroids.append(fine_centroid.copy())

        return kept

    def _result_from_m2(
        self,
        m2_result: dict,
        composite_result: CompositeEmbedding,
        t_start: float,
        trace_id: Optional[str],
    ) -> RoutingResult:
        """Build a RoutingResult from an M2 template cache hit."""
        expert_ids = m2_result.get("expert_ids", [])
        u_scores   = m2_result.get("u_base_scores", [1.0] * len(expert_ids))
        latency_ms = (time.perf_counter() - t_start) * 1000

        if self._audit:
            self._audit.emit(
                AuditEventKind.ROUTING_QUERY,
                "mrl_funnel",
                {
                    "event": "m2_shortcut",
                    "pattern_key": m2_result.get("pattern_key"),
                    "confidence": m2_result.get("confidence"),
                    "expert_ids": expert_ids,
                    "slot_id": m2_result.get("slot_id"),
                    "latency_ms": round(latency_ms, 2),
                    "trace_id": trace_id,
                },
                trace_id=trace_id,
            )

        return RoutingResult(
            expert_ids=expert_ids,
            adapter_ids=[None] * len(expert_ids),
            u_base_scores=u_scores,
            coarse_candidates=[],
            fine_candidates=[],
            used_meta_shortcut=True,
            uncertainty_region=False,
            composite_emb=composite_result,
        )

    def _empty_result(
        self,
        composite_result: CompositeEmbedding,
        uncertainty_region: bool,
    ) -> RoutingResult:
        return RoutingResult(
            expert_ids=[],
            adapter_ids=[],
            u_base_scores=[],
            coarse_candidates=[],
            fine_candidates=[],
            used_meta_shortcut=False,
            uncertainty_region=uncertainty_region,
            composite_emb=composite_result,
        )
