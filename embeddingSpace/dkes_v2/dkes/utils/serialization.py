"""
serialization.py — checkpoint save/load utilities for the full DKES state.

Provides a unified serialisation layer used by DKESEmbeddingSpace and
individual subsystems. Supports:

  - Full checkpoint: all mutable state in one JSON file
  - Partial checkpoints: individual component state_dicts
  - Version migration: handles version string checks with clear errors
  - Numpy array round-trip via base64 encoding (preserves dtype/shape)
  - Atomic write (write to .tmp then rename) to prevent partial writes

Architecture Spec reference:
  Section 1.7 — Checkpoint serialisation / restoration
  Section 11.2 — System Health Monitoring (checkpoint as health artifact)
"""
from __future__ import annotations

import base64
import json
import logging
import os
import struct
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bump this when checkpoint schema changes in a backwards-incompatible way
CHECKPOINT_VERSION = "1.1.0"


# ---------------------------------------------------------------------------
# Numpy serialisation helpers
# ---------------------------------------------------------------------------

def ndarray_to_b64(arr: np.ndarray) -> Dict[str, Any]:
    """
    Serialise a numpy array to a base64-encoded dict.

    Preserves shape and dtype. More space-efficient than .tolist() for
    float32 arrays and avoids floating-point precision loss.
    """
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def ndarray_from_b64(d: Dict[str, Any]) -> np.ndarray:
    """Restore a numpy array from a base64-encoded dict."""
    dtype = np.dtype(d["dtype"])
    shape = tuple(d["shape"])
    data = base64.b64decode(d["data"])
    return np.frombuffer(data, dtype=dtype).reshape(shape)


class _NumpyAwareEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and common non-serialisable types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return ndarray_to_b64(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _numpy_aware_decoder(dct: dict) -> Any:
    """JSON object_hook that restores numpy arrays from base64 dicts."""
    if dct.get("__ndarray__"):
        return ndarray_from_b64(dct)
    return dct


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, suffix=".dkes_tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Full checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    config_dict: Dict[str, Any],
    kdm_state: Dict[str, Any],
    composite_state: Dict[str, Any],
    metrics_snapshot: Dict[str, Any],
    cold_store_state: Optional[Dict[str, Any]] = None,
    projection_state: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a full DKES checkpoint.

    Parameters
    ----------
    path:               destination .json file path
    config_dict:        DKESConfig.to_dict()
    kdm_state:          KanervaMemory.state_dict()
    composite_state:    CompositeEmbeddingLayer.state_dict()
    metrics_snapshot:   MetricsAggregator.snapshot()
    cold_store_state:   ColdStore.state_dict() (optional)
    projection_state:   DomainProjectionManager.state_dict() (optional)
    extra:              any additional metadata to include

    Notes
    -----
    Backbone weights are NOT saved — the model_name in config is sufficient
    to reload from the original checkpoint.
    """
    path = Path(path)
    checkpoint: Dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "saved_at": time.time(),
        "saved_at_iso": _utc_iso(),
        "config": config_dict,
        "kdm": kdm_state,
        "composite": composite_state,
        "metrics_snapshot": metrics_snapshot,
    }
    if cold_store_state is not None:
        checkpoint["cold_store"] = cold_store_state
    if projection_state is not None:
        checkpoint["domain_projection"] = projection_state
    if extra:
        checkpoint["extra"] = extra

    content = json.dumps(checkpoint, cls=_NumpyAwareEncoder, indent=2)
    _atomic_write(path, content)

    size_mb = len(content) / (1024 * 1024)
    logger.info(
        "Checkpoint saved: %s (%.1f MB, version=%s)",
        path, size_mb, CHECKPOINT_VERSION,
    )


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    """
    Load a DKES checkpoint from disk.

    Returns the full checkpoint dict with numpy arrays deserialised.

    Raises
    ------
    FileNotFoundError:  if path does not exist
    ValueError:         if checkpoint version is incompatible
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path) as f:
        checkpoint = json.load(f, object_hook=_numpy_aware_decoder)

    version = checkpoint.get("version", "unknown")
    _check_version(version)

    logger.info(
        "Checkpoint loaded: %s (version=%s, saved_at=%s)",
        path, version, checkpoint.get("saved_at_iso", "unknown"),
    )
    return checkpoint


def _check_version(version: str) -> None:
    """Validate checkpoint version compatibility."""
    if version == "unknown":
        logger.warning("Checkpoint has no version field — loading anyway")
        return

    major_cur = int(CHECKPOINT_VERSION.split(".")[0])
    try:
        major_ckpt = int(version.split(".")[0])
    except (ValueError, IndexError):
        logger.warning("Cannot parse checkpoint version %r", version)
        return

    if major_ckpt != major_cur:
        raise ValueError(
            f"Checkpoint version {version!r} is incompatible with "
            f"current version {CHECKPOINT_VERSION!r}. "
            f"Major version must match."
        )


# ---------------------------------------------------------------------------
# Incremental / component-level save
# ---------------------------------------------------------------------------

def save_kdm_snapshot(
    path: str | Path,
    kdm_state: Dict[str, Any],
    note: Optional[str] = None,
) -> None:
    """
    Save only the KDM memory state (lightweight incremental snapshot).

    Used for frequent incremental saves without the overhead of a full
    checkpoint. The cold_store and projection states are not included.
    """
    path = Path(path)
    payload = {
        "version": CHECKPOINT_VERSION,
        "saved_at": time.time(),
        "kind": "kdm_snapshot",
        "note": note or "",
        "kdm": kdm_state,
    }
    content = json.dumps(payload, cls=_NumpyAwareEncoder, indent=2)
    _atomic_write(path, content)
    logger.debug("KDM snapshot saved: %s", path)


def load_kdm_snapshot(path: str | Path) -> Dict[str, Any]:
    """Load a KDM-only snapshot."""
    with open(path) as f:
        d = json.load(f, object_hook=_numpy_aware_decoder)
    return d["kdm"]


# ---------------------------------------------------------------------------
# Checkpoint listing and rotation
# ---------------------------------------------------------------------------

def list_checkpoints(directory: str | Path) -> list:
    """
    List all .json checkpoint files in a directory, sorted by saved_at.

    Returns a list of dicts with keys: path, saved_at, version, size_mb.
    """
    directory = Path(directory)
    if not directory.exists():
        return []

    results = []
    for f in directory.glob("*.json"):
        try:
            with open(f) as fh:
                header = json.load(fh)
            results.append({
                "path": f,
                "saved_at": header.get("saved_at", 0.0),
                "version": header.get("version", "unknown"),
                "size_mb": f.stat().st_size / (1024 * 1024),
            })
        except Exception:
            continue  # skip malformed files

    results.sort(key=lambda x: x["saved_at"])
    return results


def rotate_checkpoints(
    directory: str | Path,
    keep_last: int = 5,
    keep_pattern: Optional[str] = None,
) -> int:
    """
    Delete old checkpoints, keeping the most recent `keep_last`.

    Files matching keep_pattern (substring) are never deleted.
    Returns the number of files deleted.
    """
    all_ckpts = list_checkpoints(directory)
    if len(all_ckpts) <= keep_last:
        return 0

    to_delete = all_ckpts[:-keep_last]
    deleted = 0
    for ckpt in to_delete:
        p = ckpt["path"]
        if keep_pattern and keep_pattern in p.name:
            continue
        try:
            p.unlink()
            deleted += 1
            logger.info("Checkpoint rotated (deleted): %s", p)
        except OSError as exc:
            logger.warning("Failed to delete checkpoint %s: %s", p, exc)

    return deleted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
