"""
MRLBackbone — frozen encoder backbone wrapper.

Wraps a sentence-transformer / HuggingFace encoder and enforces:
  1. Permanent weight freezing (no gradient updates ever)
  2. Matryoshka nested embedding slicing
  3. Batched encoding with consistent normalisation
  4. Coarse (64-dim) and fine (768-dim) embedding extraction

This module has NO side-effects on model weights. It is the only
component in DKES that touches the underlying neural encoder.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

from dkes.utils.config import BackboneConfig
from dkes.utils.types import EmbeddingMatrix, EmbeddingVector

logger = logging.getLogger(__name__)


class MRLBackbone:
    """
    Frozen MRL encoder backbone.

    The encoder is loaded once and all parameters are immediately frozen.
    Any attempt to call a gradient-requiring operation will raise.

    Parameters
    ----------
    cfg : BackboneConfig

    Notes
    -----
    Requires either `sentence-transformers` or `transformers` + `torch`.
    Falls back to a random-projection stub for unit testing when neither
    is available (set cfg.model_name = "stub").
    """

    def __init__(self, cfg):
        # Accept either DKESConfig or BackboneConfig directly
        if hasattr(cfg, "backbone"):
            cfg = cfg.backbone
        self._cfg = cfg
        self._model = None
        self._stub_mode = False
        self._load_and_freeze()
        logger.info(
            "MRLBackbone loaded: model=%s dim=%d device=%s",
            cfg.model_name, cfg.embedding_dim, cfg.device,
        )

    # ------------------------------------------------------------------
    # Loading and freezing
    # ------------------------------------------------------------------

    def _load_and_freeze(self):
        if self._cfg.model_name == "stub":
            self._stub_mode = True
            logger.warning("MRLBackbone: using random-projection stub (testing only)")
            return

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self._cfg.model_name)
            # Move to device
            if self._cfg.device != "cpu":
                model = model.to(self._cfg.device)
            # Freeze all parameters — the ONLY place this is enforced
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            self._model = model
        except ImportError:
            logger.warning(
                "sentence-transformers not available; falling back to stub mode"
            )
            self._stub_mode = True

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        texts: Union[str, List[str]],
        dim: Optional[int] = None,
        show_progress: bool = False,
    ) -> EmbeddingMatrix:
        """
        Encode one or more texts.

        Parameters
        ----------
        texts:        string or list of strings
        dim:          if provided, return only the first `dim` MRL dimensions;
                      must be one of cfg.mrl_dims. None → full cfg.embedding_dim.
        show_progress: show tqdm bar for large batches

        Returns
        -------
        np.ndarray of shape (N, dim) — L2-normalised if cfg.normalize_embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        target_dim = dim or self._cfg.embedding_dim
        if target_dim not in self._cfg.mrl_dims:
            raise ValueError(
                f"Requested dim {target_dim} not in MRL dims {self._cfg.mrl_dims}"
            )

        embeddings = self._raw_encode(texts, show_progress)

        # Slice to requested MRL granularity
        embeddings = embeddings[:, :target_dim]

        if self._cfg.normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings.astype(np.float32)

    def encode_coarse(self, texts: Union[str, List[str]]) -> EmbeddingMatrix:
        """Encode at coarse (64-dim) MRL granularity for Stage 1 FAISS."""
        return self.encode(texts, dim=self._cfg.coarse_dim)

    def encode_fine(self, texts: Union[str, List[str]]) -> EmbeddingMatrix:
        """Encode at full (768-dim) MRL granularity for Stage 2 re-ranking."""
        return self.encode(texts, dim=self._cfg.fine_dim)

    def encode_single(self, text: str, dim: Optional[int] = None) -> EmbeddingVector:
        """Encode a single text and return a 1-D array."""
        return self.encode([text], dim=dim)[0]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _raw_encode(self, texts: List[str], show_progress: bool) -> np.ndarray:
        if self._stub_mode:
            return self._stub_encode(texts)

        import torch
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                batch_size=self._cfg.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we normalise ourselves after slicing
            )
        return np.array(embeddings, dtype=np.float32)

    def _stub_encode(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic random-projection stub for unit testing.
        Maps text→int seed via hash, then uses np.random with that seed.
        Preserves referential equality: same text → same embedding.
        """
        out = np.zeros((len(texts), self._cfg.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = abs(hash(text)) % (2 ** 31)
            rng = np.random.RandomState(seed)
            out[i] = rng.randn(self._cfg.embedding_dim).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Invariant assertions
    # ------------------------------------------------------------------

    def assert_frozen(self):
        """
        Verify that no backbone parameters have requires_grad=True.
        Call periodically in health checks to guard against accidental unfreezing.
        """
        if self._stub_mode:
            return
        try:
            for name, param in self._model.named_parameters():
                assert not param.requires_grad, (
                    f"Backbone parameter {name!r} has requires_grad=True! "
                    "This violates the frozen backbone invariant."
                )
        except Exception as exc:
            raise RuntimeError(f"Backbone frozen-invariant violation: {exc}") from exc

    @property
    def embedding_dim(self) -> int:
        return self._cfg.embedding_dim

    @property
    def coarse_dim(self) -> int:
        return self._cfg.coarse_dim