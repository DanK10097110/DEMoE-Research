"""
AuditLog — structured audit record emission for the DKES.

Emits structured JSON records to file and/or stdout for every significant
event in the embedding space lifecycle. All methods are thread-safe.

Record schema (every record has these fields):
  {
    "ts":        <unix float>,
    "iso":       <ISO-8601 string>,
    "kind":      <AuditEventKind.value>,
    "component": <str>,
    "trace_id":  <str | null>,
    ... <event-specific fields>
  }
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dkes.utils.config import AuditConfig
from dkes.utils.types import AuditEventKind

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLog:
    """
    Structured audit log for the full DKES.

    Accepts typed emit() calls from all subsystems and writes
    newline-delimited JSON records to a rotating log file and/or stdout.

    Parameters
    ----------
    cfg : AuditConfig
    """

    def __init__(self, cfg: AuditConfig):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._file = None

        if cfg.log_to_file:
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = cfg.log_dir / f"dkes_audit_{int(time.time())}.jsonl"
            self._file = open(log_path, "a", buffering=1)  # line-buffered
            logger.info("AuditLog: writing to %s", log_path)

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    def emit(
        self,
        kind: AuditEventKind,
        component: str,
        payload: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> None:
        """
        Emit a raw audit record.

        Parameters
        ----------
        kind:       AuditEventKind enum value
        component:  subsystem name (e.g. "kdm", "composite", "health_monitor")
        payload:    event-specific data dict
        trace_id:   optional trace context for request-level correlation
        """
        now = time.time()
        record = {
            "ts": now,
            "iso": _now_iso(),
            "kind": kind.value,
            "component": component,
            "trace_id": trace_id or payload.pop("trace_id", None),
            **payload,
        }
        self._write(record)

    def _write(self, record: Dict[str, Any]):
        line = json.dumps(record, default=str)
        with self._lock:
            if self._file:
                self._file.write(line + "\n")
            if self._cfg.log_to_stdout:
                print(line)

    # ------------------------------------------------------------------
    # Convenience typed emitters
    # ------------------------------------------------------------------

    def kdm_write(
        self,
        concept_id: str,
        slot_id: int,
        address_norm: float,
        value_norm: float,
        write_source: str,
        n_queries_triggered: int,
        coherence: float,
        novelty: float,
        trace_id: Optional[str] = None,
    ):
        if not self._cfg.emit_kdm_write_records:
            return
        self.emit(AuditEventKind.KDM_WRITE, "kdm", {
            "concept_id": concept_id,
            "slot_id": slot_id,
            "address_norm": round(address_norm, 5),
            "value_norm": round(value_norm, 5),
            "write_source": write_source,
            "n_queries_triggered": n_queries_triggered,
            "coherence": round(coherence, 4),
            "novelty": round(novelty, 4),
        }, trace_id=trace_id)

    def kdm_read(
        self,
        query_id: str,
        top_slot_ids: List[int],
        top_weights: List[float],
        gamma: float,
        memory_read_norm: float,
        composite_norm: float,
        trace_id: Optional[str] = None,
    ):
        if not self._cfg.emit_kdm_read_records:
            return
        self.emit(AuditEventKind.KDM_READ, "kdm", {
            "query_id": query_id,
            "top_slot_ids": top_slot_ids[:5],
            "top_weights": [round(w, 5) for w in top_weights[:5]],
            "gamma": round(gamma, 5),
            "memory_read_norm": round(memory_read_norm, 5),
            "composite_norm": round(composite_norm, 5),
        }, trace_id=trace_id)

    def kdm_eviction(
        self,
        slot_id: int,
        concept_id: str,
        reason: str,
        access_count: int,
        days_since_access: float,
        trace_id: Optional[str] = None,
    ):
        self.emit(AuditEventKind.KDM_EVICTION, "kdm", {
            "slot_id": slot_id,
            "concept_id": concept_id,
            "reason": reason,
            "access_count": access_count,
            "days_since_access": round(days_since_access, 2),
        }, trace_id=trace_id)

    def gamma_update(
        self,
        old_gamma: float,
        new_gamma: float,
        n_populated_slots: int,
        ceiling_active: bool,
        trace_id: Optional[str] = None,
    ):
        self.emit(AuditEventKind.GAMMA_UPDATE, "composite", {
            "old_gamma": round(old_gamma, 5),
            "new_gamma": round(new_gamma, 5),
            "delta": round(new_gamma - old_gamma, 5),
            "n_populated_slots": n_populated_slots,
            "ceiling_active": ceiling_active,
        }, trace_id=trace_id)

    def meta_write(
        self,
        meta_type: str,
        slot_id: int,
        concept_ids: List[str],
        detail: str,
        trace_id: Optional[str] = None,
    ):
        self.emit(AuditEventKind.META_WRITE, "meta_concepts", {
            "meta_type": meta_type,
            "slot_id": slot_id,
            "concept_ids": concept_ids,
            "detail": detail,
        }, trace_id=trace_id)

    def health_check(
        self,
        component: str,
        status: str,
        metrics: Dict[str, Any],
        alert_message: Optional[str] = None,
    ):
        if not self._cfg.emit_health_records:
            return
        kind = AuditEventKind.HEALTH_ALERT if alert_message else AuditEventKind.HEALTH_CHECK
        self.emit(kind, component, {
            "status": status,
            "alert_message": alert_message,
            **{k: v for k, v in metrics.items() if not isinstance(v, list) or len(v) < 20},
        })

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        with self._lock:
            if self._file:
                self._file.flush()
                self._file.close()
                self._file = None

    def __del__(self):
        self.close()
