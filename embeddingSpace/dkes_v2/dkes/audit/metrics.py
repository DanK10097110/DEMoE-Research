"""
MetricsAggregator — rolling statistics computed over the audit stream.

Consumed by HealthMonitor and exposed via DKESEmbeddingSpace.metrics().
All statistics operate on a configurable rolling time window.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from dkes.utils.types import AuditEventKind


class _RollingStats:
    """Lightweight rolling-window statistics for a numeric series."""

    def __init__(self, window_seconds: float):
        self._window = window_seconds
        self._values: Deque[Tuple[float, float]] = deque()  # (timestamp, value)
        self._lock = threading.Lock()

    def record(self, value: float):
        now = time.time()
        with self._lock:
            self._values.append((now, value))
            self._evict(now)

    def _evict(self, now: float):
        cutoff = now - self._window
        while self._values and self._values[0][0] < cutoff:
            self._values.popleft()

    def snapshot(self) -> Dict[str, float]:
        now = time.time()
        with self._lock:
            self._evict(now)
            vals = [v for _, v in self._values]
        if not vals:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "last": 0.0}
        import statistics
        return {
            "count":  len(vals),
            "mean":   statistics.mean(vals),
            "min":    min(vals),
            "max":    max(vals),
            "last":   vals[-1],
            "stdev":  statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }


class MetricsAggregator:
    """
    Aggregates audit records into rolling statistical summaries.

    Key metrics tracked
    -------------------
    routing_latency_ms          — per-query routing latency
    kdm_hit_rate                — fraction of reads with top weight > 0.5
    gamma_value                 — current interpolation weight trend
    kdm_slot_utilization        — fraction of M slots populated
    meta_shortcut_rate          — fraction of queries using M2 template
    uncertainty_region_rate     — fraction of queries landing in M3 zones
    eviction_rate               — KDM evictions per hour
    routing_coverage            — fraction of queries routed below threshold
    u_base_mean                 — mean U_base across selected experts
    concept_write_rate          — new concept writes per hour
    """

    def __init__(self, window_hours: float = 24.0, kdm_total_slots: int = 4096):
        self._window = window_hours * 3600
        self._total_slots = kdm_total_slots

        self._routing_latency  = _RollingStats(self._window)
        self._u_base           = _RollingStats(self._window)
        self._gamma            = _RollingStats(self._window)
        self._kdm_top_weight   = _RollingStats(self._window)

        # Counters (events per window)
        self._event_times: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=10_000)
        )
        self._lock = threading.Lock()

        # Populated slot count (set externally by KDM)
        self._populated_slots: int = 0

    # ------------------------------------------------------------------
    # Feed methods — called by DKES components after each operation
    # ------------------------------------------------------------------

    def record_routing(
        self,
        latency_ms: float,
        u_base_scores: List[float],
        gamma: float,
        used_meta_shortcut: bool,
        uncertainty_region: bool,
        below_threshold: bool,
    ):
        now = time.time()
        self._routing_latency.record(latency_ms)
        for u in u_base_scores:
            self._u_base.record(u)
        self._gamma.record(gamma)

        with self._lock:
            self._event_times["routing"].append(now)
            if used_meta_shortcut:
                self._event_times["meta_shortcut"].append(now)
            if uncertainty_region:
                self._event_times["uncertainty_region"].append(now)
            if below_threshold:
                self._event_times["below_threshold"].append(now)

    def record_kdm_read(self, top_weight: float, gamma: float):
        self._kdm_top_weight.record(top_weight)
        self._gamma.record(gamma)

    def record_kdm_write(self):
        with self._lock:
            self._event_times["concept_write"].append(time.time())

    def record_eviction(self):
        with self._lock:
            self._event_times["eviction"].append(time.time())

    def set_populated_slots(self, n: int):
        self._populated_slots = n

    # ------------------------------------------------------------------
    # Snapshot — returns a dict suitable for health checks and dashboards
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        cutoff = now - self._window

        def _rate_per_hour(key: str) -> float:
            with self._lock:
                buf = self._event_times[key]
                count = sum(1 for t in buf if t >= cutoff)
            return count / max(1.0, self._window / 3600)

        def _fraction(numerator_key: str, denominator_key: str) -> float:
            with self._lock:
                num = sum(1 for t in self._event_times[numerator_key] if t >= cutoff)
                den = sum(1 for t in self._event_times[denominator_key] if t >= cutoff)
            return num / den if den > 0 else 0.0

        routing   = self._routing_latency.snapshot()
        u_base    = self._u_base.snapshot()
        gamma     = self._gamma.snapshot()
        top_w     = self._kdm_top_weight.snapshot()

        return {
            "routing": {
                "latency_ms":          routing,
                "u_base":              u_base,
                "meta_shortcut_rate":  _fraction("meta_shortcut", "routing"),
                "uncertainty_region_rate": _fraction("uncertainty_region", "routing"),
                "coverage_rate":       _fraction("below_threshold", "routing"),
                "queries_per_hour":    _rate_per_hour("routing"),
            },
            "kdm": {
                "slot_utilization":    self._populated_slots / self._total_slots,
                "populated_slots":     self._populated_slots,
                "total_slots":         self._total_slots,
                "hit_rate":            top_w.get("mean", 0.0),  # mean top attention weight
                "top_weight_stats":    top_w,
                "concept_write_rate_per_hour": _rate_per_hour("concept_write"),
                "eviction_rate_per_hour": _rate_per_hour("eviction"),
            },
            "gamma": {
                "current":   gamma.get("last", 0.0),
                "trend":     gamma,
            },
            "window_hours": self._window / 3600,
            "snapshot_time": now,
        }