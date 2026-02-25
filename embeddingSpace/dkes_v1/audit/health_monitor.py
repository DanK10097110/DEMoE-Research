"""
HealthMonitor — continuous background health checks for the DKES.

Runs as a daemon thread that periodically evaluates system health
indicators and emits structured alerts via the AuditLog when thresholds
are breached. All checks are designed to be non-blocking and read-only
relative to the embedding space state.

Health Checks
-------------
HC-1  KDM slot utilization — warn at 80%, critical at 95%
HC-2  Mean top attention weight falling (KDM read quality)
HC-3  gamma ceiling still active past expected threshold
HC-4  Routing coverage degradation
HC-5  Concept write drought despite traffic
HC-6  Eviction storm (evictions >> writes)
HC-7  Routing latency spike
HC-8  Meta-concept slot exhaustion
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from dkes.audit.audit_log import AuditLog
from dkes.audit.metrics import MetricsAggregator
from dkes.utils.config import AuditConfig

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Runs background health checks and emits alerts via AuditLog.

    Usage
    -----
    monitor = HealthMonitor(audit, metrics, cfg, ...)
    monitor.start()
    # ... system runs ...
    monitor.stop()
    """

    def __init__(
        self,
        audit: AuditLog,
        metrics: MetricsAggregator,
        cfg: AuditConfig,
        get_kdm_meta_free_slots: Callable[[], int],
        get_gamma: Callable[[], float],
        get_gamma_ceiling_active: Callable[[], bool],
        baseline_latency_ms: float = 50.0,
    ):
        self._audit = audit
        self._metrics = metrics
        self._cfg = cfg
        self._get_meta_free = get_kdm_meta_free_slots
        self._get_gamma = get_gamma
        self._get_gamma_ceiling_active = get_gamma_ceiling_active
        self._baseline_latency = baseline_latency_ms

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_write_time: float = time.time()

    def start(self):
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="dkes-health-monitor"
        )
        self._thread.start()
        logger.info("HealthMonitor started (interval=%ds)",
                    self._cfg.health_check_interval_seconds)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def notify_kdm_write(self):
        self._last_write_time = time.time()

    def _run_loop(self):
        while not self._stop_event.wait(self._cfg.health_check_interval_seconds):
            try:
                self._run_all_checks()
            except Exception as exc:
                logger.exception("HealthMonitor check failed: %s", exc)

    def _run_all_checks(self):
        snap = self._metrics.snapshot()
        kdm = snap["kdm"]
        rt = snap["routing"]
        gamma = snap["gamma"]

        checks_passed = []
        checks_warned = []
        checks_critical = []

        # HC-1: Slot utilization
        util = kdm["slot_utilization"]
        if util >= 0.95:
            checks_critical.append(f"HC-1: KDM slot utilization critical ({util:.1%})")
        elif util >= 0.80:
            checks_warned.append(f"HC-1: KDM slot utilization high ({util:.1%})")
        else:
            checks_passed.append(f"HC-1: ok ({util:.1%})")

        # HC-2: Mean top attention weight
        mean_top_w = kdm["hit_rate"]
        if mean_top_w < 0.05 and kdm["populated_slots"] > 200:
            checks_warned.append(
                f"HC-2: Low mean top attention weight {mean_top_w:.3f}")
        else:
            checks_passed.append(f"HC-2: ok ({mean_top_w:.3f})")

        # HC-3: gamma ceiling
        if self._get_gamma_ceiling_active():
            if kdm["populated_slots"] > 600:
                checks_warned.append(
                    f"HC-3: gamma ceiling still active at {kdm['populated_slots']} slots")
            else:
                checks_passed.append("HC-3: gamma ceiling active (expected)")
        else:
            checks_passed.append(f"HC-3: gamma ceiling lifted, gamma={gamma['current']:.4f}")

        # HC-4: Routing coverage
        coverage = rt.get("coverage_rate", 1.0)
        if coverage < 0.60:
            checks_critical.append(f"HC-4: Routing coverage critical ({coverage:.1%})")
        elif coverage < 0.75:
            checks_warned.append(f"HC-4: Routing coverage low ({coverage:.1%})")
        else:
            checks_passed.append(f"HC-4: ok ({coverage:.1%})")

        # HC-5: Write drought
        hours_since_write = (time.time() - self._last_write_time) / 3600
        queries_per_hour = rt.get("queries_per_hour", 0.0)
        if hours_since_write > 48 and queries_per_hour > 10:
            checks_warned.append(
                f"HC-5: No KDM writes in {hours_since_write:.0f}h "
                f"despite {queries_per_hour:.0f} queries/hr")
        else:
            checks_passed.append(f"HC-5: ok ({hours_since_write:.1f}h since last write)")

        # HC-6: Eviction storm
        write_rate = kdm["concept_write_rate_per_hour"]
        evict_rate = kdm["eviction_rate_per_hour"]
        if evict_rate > write_rate * 2 and evict_rate > 5:
            checks_warned.append(
                f"HC-6: Eviction storm (evict={evict_rate:.1f}/hr, write={write_rate:.1f}/hr)")
        else:
            checks_passed.append(f"HC-6: ok (evict={evict_rate:.1f}, write={write_rate:.1f}/hr)")

        # HC-7: Latency spike
        p95_latency = rt["latency_ms"].get("max", 0.0)
        if p95_latency > self._baseline_latency * 3:
            checks_warned.append(
                f"HC-7: Routing latency spike (max={p95_latency:.0f}ms)")
        else:
            checks_passed.append(f"HC-7: ok (max={p95_latency:.0f}ms)")

        # HC-8: Meta slot exhaustion
        free_meta = self._get_meta_free()
        if free_meta < 5:
            checks_critical.append(f"HC-8: Meta slots nearly full ({free_meta} remaining)")
        elif free_meta < 10:
            checks_warned.append(f"HC-8: Meta slots low ({free_meta} remaining)")
        else:
            checks_passed.append(f"HC-8: ok ({free_meta} meta slots free)")

        if checks_critical:
            status = "critical"
            msg = "; ".join(checks_critical + checks_warned)
        elif checks_warned:
            status = "warning"
            msg = "; ".join(checks_warned)
        else:
            status = "ok"
            msg = None

        self._audit.health_check(
            component="health_monitor",
            status=status,
            metrics={
                "passed": checks_passed,
                "warnings": checks_warned,
                "critical": checks_critical,
                "kdm_utilization": util,
                "coverage_rate": coverage,
                "gamma": gamma["current"],
                "latency_ms_max": p95_latency,
                "write_rate_per_hour": write_rate,
                "evict_rate_per_hour": evict_rate,
                "meta_free_slots": free_meta,
            },
            alert_message=msg,
        )

        if status != "ok":
            logger.warning("DKES health %s: %s", status.upper(), msg)