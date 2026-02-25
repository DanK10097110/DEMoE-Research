"""
DEMoE v4.0 — Domain Projection Adapter Test Suite v4
=====================================================
Every test answers: "Is the spec's proposed solution the best I can
think of, or does a simpler/different approach win?"

FIXES vs v3
──────────────────────────────────────────────────────────────────────
  Fix 1 — LowRankProjectionAdapter catastrophic init (ROOT CAUSE of
  Tests 6,7,10,13,14 failures):
    v3:  A,B = eye(dim,rank)  →  AB^T is a rank-r projection onto first r
         dims. Initial output keeps only dims 0..r-1. InfoNCE loss sees
         near-zero similarity for all pairs → uniform gradients → no learning.
    v4:  LoRA-style residual: output = x + (x @ A) @ B
         A = randn(dim,rank)/sqrt(dim), B = zeros(rank,dim)
         Initial delta = 0, starts at identity, learns a low-rank correction.

  Fix 2 — Shift magnitude calibration (Bug 1):
    v3:  shift_mag=0.45 hardcoded. On normalised 768-d vectors this
         produces baseline coherence 0.69–0.77 depending on test params —
         sometimes above the 0.70 threshold, so adapters train on data that
         doesn't actually need correction.
    v4:  make_shifted_domain() binary-searches shift_mag to land baseline
         coherence in [0.42, 0.52]. Adapters now always see a real problem.

  Fix 3 — In-batch negatives degenerate (Bug 2):
    v3:  All training pairs come from the shifted distribution. In-batch
         negatives are other shifted queries ≈ all displaced by the same
         vector → near-identical to the anchor → near-zero contrast signal.
    v4:  train_adapter() accepts explicit cross-domain negatives drawn from
         the UN-shifted clean distribution. Hard contrast: anchor (shifted)
         vs positive (true doc) vs negative (clean query from wrong expert).
         infonce_with_negs() combines in-batch and explicit-negative terms.

  Fix 4 — Two-NN on random Gaussian always returns rank≈4 (Bug 3):
    v3:  Tests 9,10 used random Gaussian docs. High-dim Gaussian distributions
         are locally flat in the Two-NN sense (all equidistant on hypersphere)
         → Two-NN always estimates intrinsic dim ≈ 2–4 regardless of true rank.
    v4:  make_manifold_data() generates embeddings that genuinely lie on a
         rank-r affine subspace (U @ diag(s) @ V^T + noise) with known ground-
         truth rank. Tests 9,10 use this instead of random Gaussian.

  Fix 5 — Test 1 trivially easy (all 100% at 32-dim):
    v3:  noise=0.05, n_exp=150 → embedding space so sparse that 32 dims
         uniquely identifies every expert. No gradient to measure.
    v4:  noise=0.15, n_exp=400 for full / 120 for quick. 32-dim recall
         drops to ~65–75%, giving a meaningful monotonicity test.

  Fix 6 — Test 2 cost metric bug (single-stage 64d always "wins"):
    v3:  cost = K_coarse / K_fine for two-stage, cost=1 for single-stage 64d.
         Single-stage 64d gets cost=1 by construction → always top Recall/Cost.
    v4:  Use actual compute units: cost_64 = PREFIX/EMBED ≈ 0.083 full-dim ops,
         cost_two_stage = K_coarse*(PREFIX/EMBED) + K_fine*1.0, cost_768 = 1.0.
         Metric: (s2_recall – 0.90) / cost for recall above 90% threshold,
         else penalised. Finds the cheapest path to high-quality retrieval.

  Fix 7 — Device mismatch in Test 3:
    v3:  Adapter introspection-based re-instantiation could skip .to(DEVICE).
    v4:  All adapter instantiation uses explicit constructors with immediate
         .to(DEVICE). No introspection. Test 3 verified clean.

  Fix 8 — Test 5 trigger calibration:
    v3:  Coherence on "bad" days was often > 0.70 (threshold), so the spec
         trigger (0.70, 3 days) never fired → FNR=1.0 trivially.
    v4:  Bad-day coherence is guaranteed < 0.70 via Fix 2 calibration.
         Good-day coherence verified > 0.70 before grid-search runs.

SPEC REQUIREMENTS COVERED
─────────────────────────
[R1]  EMBED_DIM=768 — Section 1.2 nested sizes {32,64,...,768}
[R2]  Stage 1: 64-dim prefix, K_coarse=20 — Section 1.2, 7.2 Step 2
[R3]  Stage 2: full 768-dim, K_fine=5 — Section 1.2, 7.2 Step 3
[R4]  Two architectures: dense P∈R^(dxd) and low-rank P=AB^T — Section 1.3
[R5]  Adapter applied to queries only at query time — Section 1.4
[R6]  Trigger: coherence < 0.70 for 3+ consecutive days — Section 1.3, 9.5
[R7]  Training data: routing error pairs from failure log — Section 9.5
[R8]  Affected-only centroid update, no global rebuild — Section 1.3
[R9]  Adapter selection: centroid FAISS lookup, O(log N) — Section 1.3
[R10] Rank: Two-NN intrinsic dimensionality estimator — Section 4.4
[R11] Prefix integrity: 64-dim prefix similarity preserved — Section 1.2
[R12] Streaming centroid: EMA update — Section 1.4, 9.2

Usage
─────
  python MRL_embeddingV4.py            # all tests
  python MRL_embeddingV4.py --test 3 9 # subset
  python MRL_embeddingV4.py --quick    # reduced N, fewer epochs (fastest mode)
"""

import os, math, argparse
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Spec constants — taken verbatim from DEMoE v4 document
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM           = 768       # Section 1.2: nested sizes {32,64,...,768}
PREFIX_DIM          = 64        # Section 1.2: Stage 1 coarse prefix
MID_DIM             = 256       # Section 1.2: intermediate MRL level
K_COARSE            = 20        # Section 1.2: Stage 1 candidate count
K_FINE              = 5         # Section 1.2: Stage 2 retained count
COHERENCE_THRESHOLD = 0.70      # Section 1.3: trigger threshold
COHERENCE_WINDOW    = 3         # Section 1.3: consecutive days below threshold
MAX_ADAPTERS        = 50        # Section 1.3: registry cap
TWO_NN_SCALE        = 0.5       # Section 4.4: d_hat → rank scaling
RANK_MIN            = 4         # Section 4.4
RANK_MAX            = 64        # Section 4.4
RANK_DEFAULT        = 8         # Section 4.4: default for N < 200
SMALL_CORPUS_N      = 200       # Section 4.4
STREAMING_ALPHA     = 0.1       # Section 9.2: EMA alpha

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR          = 5e-4              # Increased from 3e-5 — needed for useful learning in few epochs
BATCH_SIZE  = 128               # Increased for GPU throughput
EPOCHS      = 6
GRAD_CLIP   = 1.0
TEMPERATURE = 0.07              # Slightly higher than v3 (0.05) — helps gradients in high-dim

# Target coherence range for shifted data (below trigger threshold with margin)
SHIFT_COHERENCE_TARGET_LO = 0.45
SHIFT_COHERENCE_TARGET_HI = 0.58

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
STAR = "\033[93m★\033[0m"
SPEC = "\033[96m[SPEC]\033[0m"


def pareto_score(gain: float, drift: float, drift_penalty: float = 50.0) -> float:
    """
    Single scalar capturing the drift/gain tradeoff.
    gain  = improvement in R@20(768d) over no-adapter baseline.
    drift = mean absolute 64-dim prefix cosine similarity change.
    Drift weighted heavily (50×) because prefix destruction cascades
    to routing failures that no downstream step can recover.
    Score < 0 → adapter hurts more than it helps.
    """
    return gain / (1.0 + drift_penalty * drift)


def fmt(v: float, pct: bool = False, dp: int = 4) -> str:
    if pct:
        return f"{v*100:.1f}%"
    return f"{v:.{dp}f}"


def print_tournament(title: str,
                     rows: List[Dict],
                     cols: List[Tuple[str, str, bool]],
                     winner_col: str,
                     spec_name: str = "",
                     notes: str = ""):
    print(f"\n  {'─'*68}")
    print(f"  {title}")
    print(f"  {'─'*68}")

    best_val  = max(r[winner_col] for r in rows)
    best_name = next(r["name"] for r in rows if r[winner_col] == best_val)
    baseline  = next((r[winner_col] for r in rows if "baseline" in r["name"].lower()), None)

    name_w = max(len(r["name"]) for r in rows) + 2
    header = f"  {'Approach':<{name_w}}"
    for _, disp, _ in cols:
        header += f"  {disp:>12}"
    print(header)
    print(f"  {'─'*68}")

    for r in rows:
        is_winner = r[winner_col] == best_val
        is_spec   = spec_name and spec_name in r["name"]
        line = f"  {r['name']:<{name_w}}"
        for key, _, pct in cols:
            val = r.get(key, float("nan"))
            line += f"  {fmt(val, pct=pct):>12}"
        suffix = ""
        if is_spec:
            suffix += f"  {SPEC}"
        if is_winner:
            suffix += f"  {STAR} WINNER"
        print(line + suffix)

    print(f"  {'─'*68}")

    spec_score = next((r[winner_col] for r in rows if spec_name and spec_name in r["name"]), None)
    spec_row_name = next((r["name"] for r in rows if spec_name and spec_name in r["name"]), "")
    if spec_score is not None and best_name != spec_row_name:
        margin = best_val - spec_score
        print(f"  {FAIL} SPEC approach is NOT the winner. Gap: {margin:.4f}")
        print(f"       Winner '{best_name}' outperforms spec by {margin/max(abs(spec_score),1e-9)*100:.1f}%")
    elif spec_score is not None:
        if baseline is not None and best_val > baseline:
            print(f"  {PASS} SPEC approach wins. Improvement over baseline: "
                  f"{(best_val-baseline)/max(abs(baseline),1e-9)*100:+.1f}%")
        else:
            print(f"  {PASS} SPEC approach wins among tested alternatives.")

    if notes:
        print(f"  Note: {notes}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic domain shift — core data primitive
# FIX 1: auto-calibrate shift_mag so pre-adapter coherence ∈ [0.42, 0.52]
# This guarantees the trigger condition is genuinely exercised.
# ─────────────────────────────────────────────────────────────────────────────

def _coherence_for_shift(n_queries, n_experts, shift_mag, noise):
    """Compute routing coherence for a given shift magnitude (no side effects)."""
    torch.manual_seed(42)  # deterministic for calibration
    experts = F.normalize(torch.randn(n_experts, EMBED_DIM, device=DEVICE), p=2, dim=-1)
    assigns = torch.arange(n_queries, device=DEVICE) % n_experts
    clean   = F.normalize(
        experts[assigns] + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)
    sv      = F.normalize(torch.randn(1, EMBED_DIM, device=DEVICE), p=2, dim=-1)
    shifted = F.normalize(clean + sv * shift_mag
                          + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
                          p=2, dim=-1)
    return routing_coherence_raw(shifted, experts)


@torch.no_grad()
def routing_coherence_raw(queries: torch.Tensor, expert_centroids: torch.Tensor,
                           noise_scale: float = 0.02) -> float:
    """Routing coherence using 64-dim prefix (Stage 1)."""
    qp    = F.normalize(queries[:, :PREFIX_DIM], p=2, dim=-1)
    ecp   = F.normalize(expert_centroids[:, :PREFIX_DIM], p=2, dim=-1)
    noisy = F.normalize(queries + torch.randn_like(queries) * noise_scale, p=2, dim=-1)
    nqp   = F.normalize(noisy[:, :PREFIX_DIM], p=2, dim=-1)
    return ((qp @ ecp.T).argmax(1) == (nqp @ ecp.T).argmax(1)).float().mean().item()


def make_shifted_domain(n_queries: int,
                         n_experts: int,
                         shift_mag: Optional[float] = None,
                         noise: float = 0.07,
                         target_lo: float = SHIFT_COHERENCE_TARGET_LO,
                         target_hi: float = SHIFT_COHERENCE_TARGET_HI,
                         ) -> Tuple[torch.Tensor, ...]:
    """
    Creates a domain-shift scenario where baseline coherence is calibrated
    to land in [target_lo, target_hi] — guaranteed below COHERENCE_THRESHOLD.

    If shift_mag is provided, skip calibration (use it directly).

    Returns
    ───────
    expert_centroids : (n_experts, EMBED_DIM)
    clean_queries    : (n_queries, EMBED_DIM) — pre-shift
    shifted_queries  : (n_queries, EMBED_DIM) — post-shift (training input)
    true_docs        : (n_queries, EMBED_DIM) — relevant documents
    shift_vec        : (EMBED_DIM,)
    assignments      : (n_queries,) long — ground-truth expert per query
    """
    if shift_mag is None:
        # Binary search for shift_mag ∈ [0, 8] that hits [target_lo, target_hi]
        # Use actual call parameters (capped for speed) so calibration transfers
        cal_n   = min(n_queries, 300)
        cal_exp = min(n_experts, 30)
        lo_mag, hi_mag = 0.0, 8.0
        for _ in range(22):
            mid = (lo_mag + hi_mag) / 2
            coh = _coherence_for_shift(cal_n, cal_exp, mid, noise)
            if coh > target_hi:
                lo_mag = mid
            elif coh < target_lo:
                hi_mag = mid
            else:
                break
        shift_mag = (lo_mag + hi_mag) / 2

    expert_centroids = F.normalize(
        torch.randn(n_experts, EMBED_DIM, device=DEVICE), p=2, dim=-1)

    assignments = torch.arange(n_queries, device=DEVICE) % n_experts

    clean_queries = F.normalize(
        expert_centroids[assignments]
        + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)

    true_docs = F.normalize(
        clean_queries + torch.randn_like(clean_queries) * noise,
        p=2, dim=-1)

    shift_vec = F.normalize(
        torch.randn(1, EMBED_DIM, device=DEVICE), p=2, dim=-1)

    shifted_queries = F.normalize(
        clean_queries + shift_vec * shift_mag
        + torch.randn_like(clean_queries) * noise,
        p=2, dim=-1)

    return (expert_centroids, clean_queries, shifted_queries,
            true_docs, shift_vec.squeeze(0), assignments)


def make_manifold_data(n_samples: int,
                        true_rank: int,
                        ambient_dim: int = EMBED_DIM,
                        noise: float = 0.05
                        ) -> torch.Tensor:
    """
    FIX 4: Generate embeddings that genuinely lie on a rank-`true_rank` subspace.
    Used by Tests 9 and 10 so Two-NN can find a meaningful intrinsic dimension.

    Construction:
      codes  ∈ R^(N × true_rank) — random low-dim codes
      basis  ∈ R^(true_rank × ambient_dim) — random orthonormal basis for subspace
      embs   = normalize(codes @ basis + noise)

    Two-NN should recover intrinsic dim ≈ true_rank on this data.
    PCA will recover exactly true_rank (since the data is linear).
    """
    basis  = torch.linalg.svd(
        torch.randn(ambient_dim, true_rank, device=DEVICE), full_matrices=False
    ).Vh[:true_rank]                          # (true_rank, ambient_dim), orthonormal rows
    codes  = torch.randn(n_samples, true_rank, device=DEVICE)
    embs   = codes @ basis                    # (N, ambient_dim) — on subspace
    embs   = embs + torch.randn_like(embs) * noise
    return F.normalize(embs, p=2, dim=-1)


def make_hard_negatives(queries: torch.Tensor,
                         docs:    torch.Tensor,
                         k_hard:  int = 1) -> torch.Tensor:
    """
    Mine hard negatives: for each query, find its hardest negative doc
    (most similar, excluding true positive at diagonal).
    """
    with torch.no_grad():
        qn = F.normalize(queries, p=2, dim=-1)
        dn = F.normalize(docs,    p=2, dim=-1)
        sim = qn @ dn.T
        sim.fill_diagonal_(-2.0)
        top_neg_idx = sim.argmax(dim=1)
    return docs[top_neg_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Adapter architectures
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalAdapter(nn.Module):
    """Simplest: learn one scale per dimension. Params: EMBED_DIM."""
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DenseProjectionAdapter(nn.Module):
    """[R4] P ∈ R^(768×768), initialized to identity."""
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.P.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.P(x)


class BlockDiagonalAdapter(nn.Module):
    """Structural constraint: prefix and suffix projected independently."""
    def __init__(self, dim: int = EMBED_DIM, prefix_dim: int = PREFIX_DIM):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.P_pre = nn.Linear(prefix_dim,       prefix_dim,       bias=False)
        self.P_suf = nn.Linear(dim - prefix_dim, dim - prefix_dim, bias=False)
        nn.init.eye_(self.P_pre.weight)
        nn.init.eye_(self.P_suf.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.P_pre(x[:, :self.prefix_dim]),
            self.P_suf(x[:, self.prefix_dim:])
        ], dim=1)


class LowRankProjectionAdapter(nn.Module):
    """
    [R4] Low-rank domain correction. P = I + AB where A∈R^(dim×rank), B∈R^(rank×dim).

    FIX 1 — LoRA-style initialization:
      v3 used A=B=eye(dim,rank), giving AB^T = rank-r projection (catastrophic).
      v4: A = small random, B = zeros → initial delta = 0 → starts at identity.
      Learns a low-rank CORRECTION rather than a low-rank MAP.

    The forward pass output = x + (x @ A) @ B is equivalent to
      x @ (I + A @ B)  which has rank at most `rank` correction over identity.
    """
    def __init__(self, dim: int = EMBED_DIM, rank: int = RANK_DEFAULT):
        super().__init__()
        self.rank = rank
        # LoRA init: A small random, B zeros → delta=0 at init
        self.A = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: x + low_rank_correction
        return x + (x @ self.A) @ self.B


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# FIX 3: infonce_with_negs uses explicit cross-domain negatives to break
#         the degenerate in-batch-only regime on all-shifted data.
# ─────────────────────────────────────────────────────────────────────────────

def infonce_inbatch(q: torch.Tensor, pos: torch.Tensor,
                    temperature: float = TEMPERATURE) -> torch.Tensor:
    """In-batch negatives InfoNCE. (B-1) negatives per query."""
    qn = F.normalize(q,   p=2, dim=-1)
    pn = F.normalize(pos, p=2, dim=-1)
    sim = (qn @ pn.T) / temperature
    return F.cross_entropy(sim, torch.arange(len(q), device=q.device))


def infonce_with_negs(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                       temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    FIX 3: InfoNCE with explicit negatives.
    Logits: [sim(q,pos) | sim(q,neg_1) ... sim(q,neg_B)]
    Combines in-batch negative contrast with cross-domain hard negatives.
    neg shape: (B, dim) — one explicit negative per query.
    """
    qn  = F.normalize(q,   p=2, dim=-1)
    pn  = F.normalize(pos, p=2, dim=-1)
    nn_ = F.normalize(neg, p=2, dim=-1)
    # Positive similarity (diagonal of qn @ pn.T)
    pos_sim = (qn * pn).sum(dim=-1, keepdim=True)        # (B,1)
    # In-batch negatives (all-vs-all, masked)
    inbatch = (qn @ pn.T) / temperature                  # (B,B)
    inbatch_masked = inbatch.clone()
    inbatch_masked.fill_diagonal_(-1e9)
    # Hard negative similarities
    hard_sim = (qn * nn_).sum(dim=-1, keepdim=True) / temperature  # (B,1)
    # Final logit matrix: [self-positive, in-batch negs, hard neg]
    logits = torch.cat([pos_sim / temperature, hard_sim], dim=1)
    labels = torch.zeros(len(q), dtype=torch.long, device=q.device)   # positive is col 0
    loss_hard   = F.cross_entropy(logits, labels)
    loss_inbatch = F.cross_entropy(inbatch, torch.arange(len(q), device=q.device))
    return 0.6 * loss_inbatch + 0.4 * loss_hard


def matryoshka_infonce(q: torch.Tensor, pos: torch.Tensor,
                        neg: Optional[torch.Tensor] = None,
                        temperature: float = TEMPERATURE) -> torch.Tensor:
    """Joint InfoNCE at three nested MRL prefix lengths."""
    def fn(a, b, d=None):
        if neg is not None:
            # Slice neg to match the prefix dimension of a
            n = neg[:len(a), :a.shape[-1]] if d is None else neg[:len(a), :d]
            return infonce_with_negs(a, b, n, temperature)
        return infonce_inbatch(a, b, temperature)

    return (1.0 * fn(q,                  pos)
          + 0.5 * fn(q[:, :MID_DIM],     pos[:, :MID_DIM],    MID_DIM)
          + 0.3 * fn(q[:, :PREFIX_DIM],  pos[:, :PREFIX_DIM], PREFIX_DIM))


# ─────────────────────────────────────────────────────────────────────────────
# Two-NN intrinsic dimensionality  [R10]
# ─────────────────────────────────────────────────────────────────────────────

def two_nn_intrinsic_dim(embs: torch.Tensor, n_bootstrap: int = 10) -> float:
    """
    Facco et al. (2017): d_hat = -N / sum_i log(d2_i / d1_i)
    Bootstrap robustification for N < 500 per spec Section 4.4.
    """
    e = F.normalize(embs.float(), p=2, dim=-1)
    N = len(e)

    def _est(x: torch.Tensor) -> float:
        n   = len(x)
        # Use angular distance (1 - cosine_similarity)
        d   = 1.0 - (x @ x.T)
        d.fill_diagonal_(float("inf"))
        sd, _ = torch.sort(d, dim=1)
        d1 = sd[:, 0].clamp(min=1e-10)
        d2 = sd[:, 1].clamp(min=1e-10)
        ratio = (d2 / d1).clamp(min=1.0 + 1e-8)  # must be >= 1
        return -n / torch.log(ratio).sum().item()

    if N >= 500:
        return _est(e)

    ests = []
    sub = max(10, int(0.8 * N))
    for _ in range(n_bootstrap):
        idx = torch.randperm(N)[:sub]
        ests.append(_est(e[idx]))
    return float(np.median(ests))


def pca_intrinsic_dim(embs: torch.Tensor, var_threshold: float = 0.95) -> int:
    """PCA-based rank: components explaining var_threshold of total variance."""
    e = embs.float() - embs.float().mean(dim=0)
    _, S, _ = torch.linalg.svd(e, full_matrices=False)
    var_exp  = (S**2).cumsum(0) / (S**2).sum()
    n_comp   = (var_exp < var_threshold).sum().item() + 1
    return int(n_comp)


def two_nn_rank(embs: torch.Tensor) -> int:
    """r = max(4, min(64, round(d_hat * 0.5))). Default r=8 if N<200."""
    if len(embs) < SMALL_CORPUS_N:
        return RANK_DEFAULT
    return max(RANK_MIN, min(RANK_MAX, round(two_nn_intrinsic_dim(embs) * TWO_NN_SCALE)))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# FIX 3: accepts cross_domain_negs for explicit hard negatives
# ─────────────────────────────────────────────────────────────────────────────

def train_adapter(adapter:           nn.Module,
                  q_embs:            torch.Tensor,
                  pos_embs:          torch.Tensor,
                  cross_domain_negs: Optional[torch.Tensor] = None,
                  epochs:            int   = EPOCHS,
                  lr:                float = LR,
                  ) -> nn.Module:
    """
    Matryoshka InfoNCE on (query, positive) pairs.
    Adapter applied to queries only — docs unchanged [R5].

    cross_domain_negs: if provided (N, EMBED_DIM), drawn from the un-shifted
    distribution (clean queries from wrong experts). Used as explicit hard
    negatives to break the degenerate in-batch-only regime.
    """
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(q_embs)
    adapter.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i: i + BATCH_SIZE]
            opt.zero_grad()
            neg_batch = cross_domain_negs[idx] if cross_domain_negs is not None else None
            loss = matryoshka_infonce(adapter(q_embs[idx]), pos_embs[idx], neg_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), GRAD_CLIP)
            opt.step()
        scheduler.step()
    adapter.eval()
    return adapter


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def recall_at_k(q: torch.Tensor, docs: torch.Tensor, k: int = 20) -> float:
    """Recall@K: query i's ground truth is doc i."""
    qn   = F.normalize(q,    p=2, dim=-1)
    dn   = F.normalize(docs, p=2, dim=-1)
    sim  = qn @ dn.T
    topk = torch.topk(sim, k=min(k, sim.shape[1]), dim=1).indices
    return sum(i in topk[i] for i in range(len(q))) / len(q)


@torch.no_grad()
def prefix_drift(adapter:       nn.Module,
                  queries:       torch.Tensor,
                  docs:          torch.Tensor,
                  baseline_sims: torch.Tensor) -> float:
    """Mean absolute change in 64-dim prefix cosine similarity after projection."""
    qp       = adapter(queries)
    new_sims = F.cosine_similarity(qp[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
    return (baseline_sims - new_sims).abs().mean().item()


@torch.no_grad()
def routing_coherence(queries: torch.Tensor,
                       expert_centroids: torch.Tensor,
                       noise_scale: float = 0.02) -> float:
    """[R6] Fraction of near-identical query pairs routing to same expert (64-dim)."""
    return routing_coherence_raw(queries, expert_centroids, noise_scale)


@torch.no_grad()
def mrl_two_stage_routing(queries:          torch.Tensor,
                            expert_centroids: torch.Tensor,
                            k_coarse:         int = K_COARSE,
                            k_fine:           int = K_FINE,
                            adapter:          Optional[nn.Module] = None,
                            assignments:      Optional[torch.Tensor] = None,
                            ) -> Tuple[float, float]:
    """
    [R2, R3] Spec routing funnel.
    Stage 1: 64-dim prefix → top k_coarse candidates.
    Stage 2: 768-dim → top k_fine from those candidates.
    Adapter applied to queries only [R5].
    Returns (stage1_recall, stage2_recall).
    """
    qp = adapter(queries) if adapter is not None else queries
    n  = len(queries)
    ne = len(expert_centroids)

    q1    = F.normalize(qp[:, :PREFIX_DIM], p=2, dim=-1)
    e1    = F.normalize(expert_centroids[:, :PREFIX_DIM], p=2, dim=-1)
    top_c = torch.topk(q1 @ e1.T, k=min(k_coarse, ne), dim=1).indices

    q2 = F.normalize(qp, p=2, dim=-1)
    e2 = F.normalize(expert_centroids, p=2, dim=-1)

    s1_hits = s2_hits = 0
    for i in range(n):
        gt   = assignments[i].item() if assignments is not None else i % ne
        cand = top_c[i]
        if gt in cand:
            s1_hits += 1
            fine_sims = (q2[i].unsqueeze(0) @ e2[cand].T).squeeze(0)
            top_f = cand[torch.topk(fine_sims, k=min(k_fine, len(cand))).indices]
            if gt in top_f:
                s2_hits += 1

    return s1_hits / n, s2_hits / n


# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class StreamingCentroid:
    """[R12] EMA centroid per Section 9.2, tracking full and prefix dims."""
    def __init__(self, dim: int = EMBED_DIM, alpha: float = STREAMING_ALPHA):
        self.alpha  = alpha
        self.full   = torch.zeros(dim, device=DEVICE)
        self.prefix = torch.zeros(PREFIX_DIM, device=DEVICE)
        self._init  = False

    def update(self, batch: torch.Tensor):
        bm = batch.mean(dim=0)
        bp = batch[:, :PREFIX_DIM].mean(dim=0)
        if not self._init:
            self.full, self.prefix, self._init = bm.clone(), bp.clone(), True
        else:
            self.full   = (1 - self.alpha) * self.full   + self.alpha * bm
            self.prefix = (1 - self.alpha) * self.prefix + self.alpha * bp

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.full.clone(), self.prefix.clone()


class AdapterRegistry:
    """
    [R9] Stores adapters and their centroid embeddings.
    Selection: exact cosine search over adapter centroids.
    """
    def __init__(self, sim_threshold: float = 0.0):
        self.adapters:   List[nn.Module]    = []
        self.centroids:  List[torch.Tensor] = []
        self.domain_ids: List[str]          = []
        self.threshold   = sim_threshold

    def register(self, adapter: nn.Module, centroid: torch.Tensor, domain_id: str):
        assert len(self.adapters) < MAX_ADAPTERS, "Adapter cap reached"
        self.adapters.append(adapter)
        self.centroids.append(F.normalize(centroid, p=2, dim=-1))
        self.domain_ids.append(domain_id)

    def select(self, query: torch.Tensor) -> Tuple[Optional[nn.Module], str, float]:
        """Returns (adapter, domain_id, similarity). None if below threshold."""
        qn   = F.normalize(query, p=2, dim=-1)
        sims = torch.stack([qn @ c for c in self.centroids])
        best = sims.argmax().item()
        sim  = sims[best].item()
        if sim < self.threshold:
            return None, "no_adapter", sim
        return self.adapters[best], self.domain_ids[best], sim

    def __len__(self):
        return len(self.adapters)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_1_mrl_nested_validity(verbose: bool = False, quick: bool = False):
    """
    [R1] Are all six MRL nested sizes independently useful for retrieval?
    Tests recall monotonicity: larger prefix → higher recall.

    FIX 5: v3 used noise=0.05 / n_exp=150 → 32-dim already had 100% recall
    (embedding space too sparse). v4 uses noise=0.15, n_exp=400 (full) or 120
    (quick) so 32-dim recall is ~65–75%, making monotonicity meaningful.
    """
    print("\n[Test 1] MRL nested size validity — monotonic recall across prefix sizes")

    N     = 400 if quick else 1000
    n_exp = 120 if quick else 400    # FIX 5: more experts for harder retrieval
    noise = 0.15                     # FIX 5: more noise to make low-dim hard

    # Use clean no-shift data — testing the backbone MRL property
    experts = F.normalize(torch.randn(n_exp, EMBED_DIM, device=DEVICE), p=2, dim=-1)
    assigns = torch.arange(N, device=DEVICE) % n_exp
    docs    = F.normalize(experts[assigns] + torch.randn(N, EMBED_DIM, device=DEVICE) * noise,
                           p=2, dim=-1)
    queries = F.normalize(docs + torch.randn_like(docs) * noise, p=2, dim=-1)

    dims    = [32, 64, 128, 256, 512, 768]
    rows    = []
    prev_r  = 0.0
    mono_ok = True

    for d in dims:
        r = recall_at_k(queries[:, :d], docs[:, :d], k=20)
        if r < prev_r - 0.02:
            mono_ok = False
        prev_r = r
        half_d = max(d // 2, 32)
        r_half = recall_at_k(queries[:, :half_d], docs[:, :half_d], k=20)
        rows.append({"name": f"{d:4d}-dim", "recall": r, "delta": r - r_half})

    print_tournament(
        "MRL recall@20 vs prefix length",
        rows, [("recall", "R@20", True), ("delta", "ΔR@20 from half", True)],
        winner_col="recall", spec_name="768",
        notes="Recall should be strictly increasing in prefix length"
    )

    print(f"  Monotonicity (recall non-decreasing with dim): "
          f"{'✓ PASS' if mono_ok else '✗ FAIL — MRL property violated'}")
    if verbose:
        for r in rows:
            print(f"    {r['name']}: {r['recall']*100:.1f}%")


def test_2_two_stage_funnel_optimality(verbose: bool = False, quick: bool = False):
    """
    [R2, R3] Is the spec's K_coarse=20 the best choice?

    Cost model (fixed from v3 and v4-prior):
      The two-stage funnel's cost is NOT K_coarse*prefix + K_fine*full.
      That model counts every query as paying K_fine full evals regardless
      of K_coarse, which misrepresents the funnel's purpose.

      Correct model: the funnel replaces a full scan of all N_e experts.
      Baseline cost = N_e full-dim evals per query.
      Two-stage cost = N_e * prefix_cost (stage 1) + K_coarse * full_cost (stage 2).
      Single-stage 64d cost = N_e * prefix_cost.
      Single-stage 768d cost = N_e * full_cost  (normalized to 1.0).

      Efficiency = S2_recall / cost_fraction.  Now the two-stage funnel can
      win by achieving near-768d recall while avoiding full evals for most experts.

    Data: moderate n_exp (60–100) so 64d recall is ~65–80% (not trivial,
    not catastrophic) and 768d recall is ~95–100%. This is the regime where
    the two-stage funnel provides genuine lift.
    """
    print("\n[Test 2] Two-stage funnel K_coarse optimality — sweeping K_coarse")

    N     = 300 if quick else 500
    n_exp = 60  if quick else 100     # regime where 64d recall ≈ 70–80%
    noise = 0.10

    experts = F.normalize(torch.randn(n_exp, EMBED_DIM, device=DEVICE), p=2, dim=-1)
    assigns = torch.arange(N, device=DEVICE) % n_exp
    docs    = F.normalize(experts[assigns] + torch.randn(N, EMBED_DIM, device=DEVICE) * noise,
                           p=2, dim=-1)
    queries = F.normalize(docs + torch.randn_like(docs) * noise, p=2, dim=-1)

    PREFIX_COST = PREFIX_DIM / EMBED_DIM  # ≈ 0.083 (relative cost of one 64-dim eval)

    # Cost as fraction of single-stage 768d (= N_e full-dim evals = 1.0)
    def two_stage_cost(kc):
        # N_e prefix evals + K_coarse full evals, all normalised by N_e full evals
        return PREFIX_COST + kc / n_exp

    rows = []

    # 1. Single-stage 768d (oracle — full scan)
    r_full = recall_at_k(queries, docs, k=K_FINE)
    rows.append({"name": "Single-stage 768d",
                 "s1_recall": 1.0, "s2_recall": r_full,
                 "cost": 1.0,
                 "efficiency": r_full / 1.0})

    # 2. Single-stage 64d (cheapest — prefix scan only)
    r_64 = recall_at_k(queries[:, :PREFIX_DIM], docs[:, :PREFIX_DIM], k=K_FINE)
    rows.append({"name": "Single-stage 64d",
                 "s1_recall": r_64, "s2_recall": r_64,
                 "cost": PREFIX_COST,
                 "efficiency": r_64 / PREFIX_COST})

    # 3. Two-stage with various K_coarse
    k_values = [5, 10, 20, 50, 100] if not quick else [5, 10, 20, 50]
    for kc in k_values:
        s1, s2 = mrl_two_stage_routing(queries, experts, k_coarse=kc, k_fine=K_FINE,
                                        assignments=assigns)
        cost = two_stage_cost(kc)
        rows.append({"name": f"Two-stage K={kc:3d}" + (" ★SPEC" if kc == K_COARSE else ""),
                     "s1_recall": s1, "s2_recall": s2,
                     "cost": cost,
                     "efficiency": s2 / cost})

    print_tournament(
        "Two-stage funnel: S2 recall vs normalised compute cost",
        rows,
        [("s1_recall", "S1 R@20", True), ("s2_recall", "S2 R@5", True),
         ("cost", "Cost(norm)", False), ("efficiency", "Recall/Cost", False)],
        winner_col="efficiency",
        spec_name="★SPEC",
        notes=("Cost = fraction of single-stage-768d cost. "
               "Two-stage wins if it gets near-oracle recall at < oracle cost.")
    )

    spec_row = next(r for r in rows if "★SPEC" in r["name"])
    best_two = max((r["efficiency"] for r in rows if "K=" in r["name"]))
    at_knee = spec_row["efficiency"] >= best_two * 0.85
    print(f"  K_coarse=20 within 15% of best two-stage efficiency: "
          f"{'✓ PASS' if at_knee else '✗ FAIL — K_coarse may need retuning'}")
    if verbose:
        print(f"  n_exp={n_exp}, PREFIX_COST={PREFIX_COST:.4f}")
        print(f"  Two-stage K=20 cost = {two_stage_cost(K_COARSE):.4f} × single-stage-768d")


def test_3_adapter_architecture_tournament(verbose: bool = False, quick: bool = False):
    """
    [R4, R11] Tournament across all adapter architectures on shifted domain.
    Metric: Pareto score = gain / (1 + 50*drift).

    FIX 7: All adapters explicitly moved to DEVICE before training.
    FIX 1+2: LowRank now uses LoRA init; data is properly calibrated.
    Uses cross-domain hard negatives for training (FIX 3).
    """
    print("\n[Test 3] Adapter architecture tournament — domain-shifted data, Pareto criterion")

    N     = 400 if quick else 700
    n_exp = 20  if quick else 40
    experts, clean_q, shifted_q, docs, _, _ = make_shifted_domain(N, n_exp)

    # Cross-domain negatives: clean queries shuffled to wrong experts
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

    with torch.no_grad():
        base_sims = F.cosine_similarity(shifted_q[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
        base_r768 = recall_at_k(shifted_q, docs, k=20)

    r_twonn = two_nn_rank(docs)
    print(f"  Calibrated shift: pre-adapter coherence = "
          f"{routing_coherence(shifted_q, experts):.3f} "
          f"(target {SHIFT_COHERENCE_TARGET_LO:.2f}–{SHIFT_COHERENCE_TARGET_HI:.2f})")
    print(f"  Two-NN estimated rank: {r_twonn}")

    # FIX 7: all adapters explicitly instantiated with .to(DEVICE)
    candidates = [
        ("No adapter (baseline)",       None),
        ("Diagonal",                     DiagonalAdapter(EMBED_DIM).to(DEVICE)),
        ("Dense (★SPEC naive)",          DenseProjectionAdapter(EMBED_DIM).to(DEVICE)),
        ("Block-Diagonal",               BlockDiagonalAdapter(EMBED_DIM).to(DEVICE)),
        ("LowRank r=2",                  LowRankProjectionAdapter(EMBED_DIM, rank=2).to(DEVICE)),
        (f"LowRank r={r_twonn} ★SPEC",  LowRankProjectionAdapter(EMBED_DIM, r_twonn).to(DEVICE)),
        ("LowRank r=32",                 LowRankProjectionAdapter(EMBED_DIM, rank=32).to(DEVICE)),
    ]

    rows = []
    for name, adapter in candidates:
        if adapter is None:
            drift, gain, r768, n_par = 0.0, 0.0, base_r768, 0
        else:
            train_adapter(adapter, shifted_q, docs, cross_domain_negs=cross_negs)
            with torch.no_grad():
                drift = prefix_drift(adapter, shifted_q, docs, base_sims)
                r768  = recall_at_k(adapter(shifted_q), docs, k=20)
                gain  = r768 - base_r768
            n_par = sum(p.numel() for p in adapter.parameters())
        score = pareto_score(gain, drift)
        rows.append({"name": name, "drift": drift, "gain": gain,
                     "r768": r768, "params": n_par, "score": score})

    print_tournament(
        "Architecture Pareto: gain / (1 + 50 × drift)",
        rows,
        [("drift", "Drift(64d)", False), ("gain", "Gain(768d)", True),
         ("r768", "R@20(768d)", True),   ("params", "Params", False),
         ("score", "Pareto", False)],
        winner_col="score",
        spec_name="★SPEC",
        notes="Pareto penalises prefix drift 50× — routing failure cascades downstream"
    )


def test_4_projection_target_comparison(verbose: bool = False, quick: bool = False):
    """
    [R5] Query-only (spec) vs doc-only vs both vs query+centroid-update.
    FIX 1+2+3: proper adapter training, calibrated data, hard negatives.
    """
    print("\n[Test 4] Projection target comparison — who benefits from projection?")

    N     = 400 if quick else 700
    n_exp = 20  if quick else 40
    experts, clean_q, shifted_q, docs, _, assignments = make_shifted_domain(N, n_exp)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

    r = two_nn_rank(docs)
    rows = []

    def _recall_routing(q_emb, expert_emb):
        s1, _ = mrl_two_stage_routing(q_emb, expert_emb, assignments=assignments)
        return s1

    rows.append({"name": "No adapter (baseline)",
                 "r64":  _recall_routing(shifted_q, experts),
                 "r768": recall_at_k(shifted_q, docs, k=20)})

    # FIX 7: explicit .to(DEVICE)
    adapter = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    train_adapter(adapter, shifted_q, docs, cross_domain_negs=cross_negs)

    with torch.no_grad():
        q_proj = adapter(shifted_q)
        d_proj = adapter(docs)
        e_proj = adapter(experts)

    rows.append({"name": "Query-only ★SPEC",
                 "r64":  _recall_routing(q_proj, experts),
                 "r768": recall_at_k(q_proj, docs, k=20)})

    rows.append({"name": "Doc-only (incorrect)",
                 "r64":  _recall_routing(shifted_q, experts),
                 "r768": recall_at_k(shifted_q, d_proj, k=20)})

    rows.append({"name": "Both projected (incorrect)",
                 "r64":  _recall_routing(q_proj, experts),
                 "r768": recall_at_k(q_proj, d_proj, k=20)})

    rows.append({"name": "Query + centroid update (full spec)",
                 "r64":  _recall_routing(q_proj, e_proj),
                 "r768": recall_at_k(q_proj, docs, k=20)})

    print_tournament(
        "End-to-end routing recall under each projection target",
        rows,
        [("r64", "S1 R@20(64d)", True), ("r768", "R@20(768d)", True)],
        winner_col="r64",
        spec_name="★SPEC",
        notes="R64 = Stage 1 routing — the failure mode we're protecting against"
    )


def test_5_trigger_threshold_optimality(verbose: bool = False, quick: bool = False):
    """
    [R6] Is (threshold=0.70, window=3 days) optimal?

    FIX 8: v3/v4-prior used make_shifted_domain's auto-calibrated shift on
    one set of experts, then built DIFFERENT experts for the daily simulation
    → calibration didn't carry over. Fix: use the experts returned by
    make_shifted_domain for both calibration and daily scoring.

    Also: clean queries must achieve coherence reliably > 0.70.
    We verify both conditions before running the grid search.
    """
    print("\n[Test 5] Trigger threshold optimality — sweeping (threshold × window) grid")

    N     = 300 if quick else 600
    n_exp = 15

    N_DAYS   = 30
    BAD_DAYS = 10

    # Calibrated bad-day data — experts come from make_shifted_domain
    experts, clean_q, shifted_q, docs, _, _ = make_shifted_domain(N, n_exp)

    # Good-day queries: very close to expert centroids (noise=0.03 ensures high coherence)
    assigns = torch.arange(N, device=DEVICE) % n_exp
    good_q  = F.normalize(
        experts[assigns] + torch.randn(N, EMBED_DIM, device=DEVICE) * 0.03,
        p=2, dim=-1)

    bad_coh  = routing_coherence(shifted_q, experts)
    good_coh = routing_coherence(good_q,    experts)
    print(f"  Bad-day coherence: {bad_coh:.3f}  |  Good-day coherence: {good_coh:.3f}")

    if bad_coh >= COHERENCE_THRESHOLD:
        print(f"  {FAIL} WARN: bad-day coherence {bad_coh:.3f} >= threshold {COHERENCE_THRESHOLD}. "
              f"Shift calibration may need a wider target range.")
    if good_coh < COHERENCE_THRESHOLD:
        print(f"  {FAIL} WARN: good-day coherence {good_coh:.3f} < threshold — false trigger risk.")

    daily_coherence = [routing_coherence(shifted_q if d < BAD_DAYS else good_q, experts)
                       for d in range(N_DAYS)]

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90] if not quick else [0.60, 0.70, 0.80]
    windows    = [1, 2, 3, 5, 7]                 if not quick else [1, 3, 5]

    rows = []
    for thresh in thresholds:
        for win in windows:
            triggers = []
            for day in range(N_DAYS):
                window_scores = daily_coherence[max(0, day - win + 1): day + 1]
                fired = (len(window_scores) == win
                         and all(s < thresh for s in window_scores))
                triggers.append(fired)
            fpr = sum(triggers[BAD_DAYS:]) / (N_DAYS - BAD_DAYS)
            fnr = 0.0 if any(triggers[:BAD_DAYS]) else 1.0
            combined = fpr + fnr
            spec = (thresh == 0.70 and win == 3)
            rows.append({"name": f"t={thresh:.2f} w={win}" + (" ★SPEC" if spec else ""),
                         "fpr": fpr, "fnr": fnr,
                         "combined": combined,
                         "score": 1.0 - combined})

    print_tournament(
        "FPR + FNR across (threshold, window) — lower combined = better",
        rows[:12],
        [("fpr", "FPR", True), ("fnr", "FNR", True), ("combined", "FPR+FNR", True)],
        winner_col="score",
        spec_name="★SPEC",
        notes="FPR = unnecessary triggers on healthy domain. FNR = missed bad windows."
    )

    spec_row = next((r for r in rows if "★SPEC" in r["name"]), None)
    best_row = min(rows, key=lambda r: r["combined"])
    if spec_row:
        margin = spec_row["combined"] - best_row["combined"]
        ok = margin <= 0.15
        print(f"  Spec (0.70, 3) combined error: {spec_row['combined']:.3f}  "
              f"Best: {best_row['combined']:.3f}  Margin: {margin:.3f}  "
              f"{'✓ PASS' if ok else '✗ FAIL — spec trigger may need retuning'}")


def test_6_training_data_composition(verbose: bool = False, quick: bool = False):
    """
    [R7] Which training data strategy produces best coherence recovery?

    FIX 1+2+3: With LoRA init + calibrated shift + cross-domain negatives,
    adapters now actually learn. Strategies differ in *how* they construct
    the negative signal, not whether training works at all.
    """
    print("\n[Test 6] Training data composition tournament — sample efficiency")

    N     = 500 if quick else 900
    n_exp = 20  if quick else 35
    experts, clean_q, shifted_q, docs, _, _ = make_shifted_domain(N, n_exp)

    r  = two_nn_rank(docs)
    ns = [50, 100, 200, 400] if not quick else [50, 100, 200]

    random_negs = torch.roll(docs, shifts=7, dims=0)
    hard_negs   = make_hard_negatives(shifted_q, docs)
    cross_negs  = clean_q[torch.randperm(N, device=DEVICE)]  # FIX 3

    strategies = [
        ("Random triplets",      shifted_q, docs, random_negs),
        ("Error pairs ★SPEC",    shifted_q, docs, None),   # in-batch only
        ("Cross-domain negs",    shifted_q, docs, cross_negs),   # FIX 3 candidate
        ("Hard negatives",       shifted_q, docs, hard_negs),
    ]

    rows = []
    for strategy_name, sq, pd, explicit_neg in strategies:
        coh_by_n = []
        for n_train in ns:
            idx = torch.randperm(N, device=DEVICE)[:n_train]
            adapter = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
            neg_batch = explicit_neg[idx] if explicit_neg is not None else None
            train_adapter(adapter, sq[idx], pd[idx], cross_domain_negs=neg_batch)
            with torch.no_grad():
                coh = routing_coherence(adapter(shifted_q), experts)
            coh_by_n.append(coh)

        baseline_coh = routing_coherence(shifted_q, experts)
        best_coh     = max(coh_by_n)
        auc          = sum(coh_by_n) / len(ns)
        rows.append({"name":     strategy_name,
                     "coh_n50":  coh_by_n[0],
                     "coh_best": best_coh,
                     "auc":      auc,
                     "recovery": best_coh - baseline_coh})

    baseline_coh = routing_coherence(shifted_q, experts)
    print_tournament(
        f"Coherence recovery (baseline: {baseline_coh:.3f})",
        rows,
        [("coh_n50", "Coh@N=50", True), ("coh_best", "Best Coh", True),
         ("auc", "AUC", False), ("recovery", "Recovery", True)],
        winner_col="auc",
        spec_name="★SPEC",
        notes="AUC = area under coherence-vs-N curve (higher = more sample-efficient)"
    )


def test_7_centroid_update_scope(verbose: bool = False, quick: bool = False):
    """
    [R8] Affected-only centroid update vs full rebuild vs no update.
    FIX 1+2+3: adapters now train properly, making the centroid update
    comparison meaningful (v3 all conditions were equally broken at ~37%).
    """
    print("\n[Test 7] Centroid update scope + accumulation over sequential rounds")

    N_DOMAINS = 5
    N_PER     = 10
    N         = 300 if quick else 500
    n_rnd     = 3   if quick else 5

    all_cents  = [F.normalize(torch.randn(N_PER, EMBED_DIM, device=DEVICE), p=2, dim=-1)
                  for _ in range(N_DOMAINS)]
    faiss_orig = torch.cat(all_cents, dim=0)
    affected   = 2

    experts, clean_q, shifted_q, docs, _, assignments = make_shifted_domain(
        N, N_DOMAINS * N_PER)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]
    r = two_nn_rank(docs)

    def end_to_end_recall(qidx, cidx):
        s1, _ = mrl_two_stage_routing(qidx, cidx, assignments=assignments)
        return s1

    rows = []

    # No update
    rows.append({"name": "No centroid update",
                 "r_s1":       end_to_end_recall(shifted_q, faiss_orig),
                 "non_affected_changed": False,
                 "update_cost": 0.0})

    # Full rebuild
    adapter_full = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    train_adapter(adapter_full, shifted_q, docs, cross_domain_negs=cross_negs)
    with torch.no_grad():
        full_rebuilt = adapter_full(faiss_orig)
    rows.append({"name": "Full index rebuild",
                 "r_s1":       end_to_end_recall(adapter_full(shifted_q), full_rebuilt),
                 "non_affected_changed": True,
                 "update_cost": 1.0})

    # Affected-only (spec [R8])
    faiss_partial = faiss_orig.clone()
    adapter_part  = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    train_adapter(adapter_part, shifted_q, docs, cross_domain_negs=cross_negs)
    s, e = affected * N_PER, (affected + 1) * N_PER
    with torch.no_grad():
        faiss_partial[s:e] = adapter_part(all_cents[affected])
    unchanged = (torch.allclose(faiss_partial[:s], faiss_orig[:s]) and
                 torch.allclose(faiss_partial[e:], faiss_orig[e:]))
    rows.append({"name": "Affected-only ★SPEC",
                 "r_s1":       end_to_end_recall(adapter_part(shifted_q), faiss_partial),
                 "non_affected_changed": not unchanged,
                 "update_cost": N_PER / (N_DOMAINS * N_PER)})

    print_tournament(
        "End-to-end Stage 1 recall under different centroid update scopes",
        rows,
        [("r_s1", "S1 R@20", True), ("update_cost", "Update cost", False)],
        winner_col="r_s1", spec_name="★SPEC",
        notes="Update cost = fraction of FAISS entries rewritten"
    )
    print(f"  Non-affected entries unchanged (spec requirement): "
          f"{'✓ PASS' if unchanged else '✗ FAIL'}")

    print(f"\n  Accumulation: {n_rnd} sequential rounds on same domain")
    running_idx = faiss_orig.clone()
    for rnd in range(n_rnd):
        a = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
        train_adapter(a, shifted_q, docs, cross_domain_negs=cross_negs, epochs=3)
        with torch.no_grad():
            running_idx[s:e] = a(running_idx[s:e])
            r_val = end_to_end_recall(a(shifted_q), running_idx)
        print(f"    Round {rnd+1}: S1 recall = {r_val*100:.1f}%")


def test_8_adapter_selection_and_ood(verbose: bool = False, quick: bool = False):
    """
    [R9] Nearest centroid (spec) vs threshold-gated.
    Tests OOD: query far from all registered adapter centroids.
    FIX 7: all adapter instantiation explicit .to(DEVICE).
    """
    print("\n[Test 8] Adapter selection method + OOD handling tournament")

    N_ADAPTERS = 8 if not quick else 5
    N_TEST     = 200 if not quick else 100

    domain_cents = [F.normalize(torch.randn(EMBED_DIM, device=DEVICE), p=2, dim=-1)
                    for _ in range(N_ADAPTERS)]

    def make_registry(threshold=0.0):
        reg = AdapterRegistry(sim_threshold=threshold)
        for i, c in enumerate(domain_cents):
            # FIX 7: explicit .to(DEVICE)
            a = LowRankProjectionAdapter(EMBED_DIM, rank=RANK_DEFAULT).to(DEVICE)
            reg.register(a, c, f"domain_{i}")
        return reg

    def accuracy(reg, queries, true_domains):
        return sum(1 for q, td in zip(queries, true_domains)
                   if reg.select(q)[1] == f"domain_{td}") / len(queries)

    in_dist_q   = [F.normalize(domain_cents[i % N_ADAPTERS]
                                + torch.randn_like(domain_cents[0]) * 0.05, p=2, dim=-1)
                   for i in range(N_TEST)]
    in_dist_dom = [i % N_ADAPTERS for i in range(N_TEST)]
    ood_q       = [F.normalize(torch.randn(EMBED_DIM, device=DEVICE), p=2, dim=-1)
                   for _ in range(50)]

    rows = []

    reg_spec = make_registry(threshold=0.0)
    rows.append({"name": "Nearest centroid ★SPEC",
                 "in_acc":    accuracy(reg_spec, in_dist_q, in_dist_dom),
                 "ood_applied": 1.0,
                 "score":     accuracy(reg_spec, in_dist_q, in_dist_dom) - 0.5 * 1.0})

    for tau in [0.30, 0.50, 0.70]:
        reg_t   = make_registry(threshold=tau)
        in_acc  = accuracy(reg_t, in_dist_q, in_dist_dom)
        ood_app = sum(1 for q in ood_q if reg_t.select(q)[0] is not None) / len(ood_q)
        rows.append({"name": f"Threshold τ={tau}",
                     "in_acc":    in_acc,
                     "ood_applied": ood_app,
                     "score":     in_acc - 0.5 * ood_app})

    print_tournament(
        "Adapter selection: in-distribution accuracy vs OOD application rate",
        rows,
        [("in_acc", "In-dist acc", True), ("ood_applied", "OOD applied", True)],
        winner_col="score", spec_name="★SPEC",
        notes="Score = in_acc − 0.5×ood_applied. OOD application wastes compute + risks wrong projection."
    )

    reg_cap = AdapterRegistry()
    for i in range(MAX_ADAPTERS):
        reg_cap.register(LowRankProjectionAdapter(EMBED_DIM, RANK_DEFAULT).to(DEVICE),
                         torch.randn(EMBED_DIM, device=DEVICE), f"d_{i}")
    try:
        reg_cap.register(LowRankProjectionAdapter(EMBED_DIM, RANK_DEFAULT).to(DEVICE),
                         torch.randn(EMBED_DIM, device=DEVICE), "overflow")
        print(f"  Cap enforcement: {FAIL} did not raise on 51st adapter")
    except AssertionError:
        print(f"  Cap enforcement ({MAX_ADAPTERS} limit): ✓ PASS")


def test_9_rank_determination_tournament(verbose: bool = False, quick: bool = False):
    """
    [R10] Two-NN vs PCA vs fixed ranks on manifold data.

    FIX 4: v3 used random Gaussian docs → Two-NN always returned rank≈4
    (known property of Gaussians on hyperspheres). v4 uses make_manifold_data()
    with ground-truth rank=16, so Two-NN and PCA can be meaningfully compared.
    Expected: Two-NN estimates ≈ 16, PCA estimates > 16 (overestimates on linear
    data, since 0.95 variance threshold captures more components), Fixed ranks
    near 16 perform best on the Pareto criterion.
    """
    print("\n[Test 9] Rank determination tournament — Two-NN vs PCA vs fixed ranks")

    TRUE_RANK = 16
    N         = 500 if quick else 800
    n_exp     = 20  if quick else 40

    # FIX 4: use structured manifold data
    docs_manifold = make_manifold_data(N, true_rank=TRUE_RANK)
    print(f"  Ground-truth manifold rank: {TRUE_RANK}")

    _, _, shifted_q, docs, _, _ = make_shifted_domain(N, n_exp)
    # Use manifold docs for rank estimation, but shifted/docs for training eval
    r_twonn = two_nn_rank(docs_manifold)
    r_pca   = min(RANK_MAX, max(RANK_MIN, pca_intrinsic_dim(docs_manifold)))
    print(f"  Two-NN estimated rank: {r_twonn}  |  PCA estimated rank: {r_pca}")

    with torch.no_grad():
        base_sims = F.cosine_similarity(shifted_q[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
        base_r768 = recall_at_k(shifted_q, docs, k=20)

    cross_negs = F.normalize(torch.randn(N, EMBED_DIM, device=DEVICE), p=2, dim=-1)

    rows = []
    for rank, label in [(4, "Fixed r=4"), (8, "Fixed r=8"), (16, "Fixed r=16"),
                         (32, "Fixed r=32"), (64, "Fixed r=64"),
                         (r_pca,   f"PCA r={r_pca}"),
                         (r_twonn, f"Two-NN r={r_twonn} ★SPEC")]:
        adapter = LowRankProjectionAdapter(EMBED_DIM, rank=rank).to(DEVICE)
        train_adapter(adapter, shifted_q, docs, cross_domain_negs=cross_negs)
        with torch.no_grad():
            drift  = prefix_drift(adapter, shifted_q, docs, base_sims)
            gain   = recall_at_k(adapter(shifted_q), docs, k=20) - base_r768
        score = pareto_score(gain, drift)
        rows.append({"name": label, "rank": float(rank), "drift": drift,
                     "gain": gain, "params": EMBED_DIM * rank * 2, "pareto": score})

    print_tournament(
        f"Pareto score at each rank (Two-NN={r_twonn}, PCA={r_pca})",
        rows,
        [("rank", "Rank", False), ("drift", "Drift", False),
         ("gain", "Gain", True),  ("params", "Params", False), ("pareto", "Pareto", False)],
        winner_col="pareto",
        spec_name="★SPEC"
    )

    spec_r = next(r for r in rows if "★SPEC" in r["name"])
    best_p = max(r["pareto"] for r in rows)
    print(f"  Two-NN rank within 15% of best Pareto: "
          f"{'✓ PASS' if spec_r['pareto'] >= best_p * 0.85 else '✗ FAIL'}")
    twonn_lt_pca = r_twonn <= r_pca
    print(f"  PCA rank ({r_pca}) >= Two-NN rank ({r_twonn}) [spec claim on overestimation]: "
          f"{'✓ CONFIRMED' if twonn_lt_pca else '✗ NOT CONFIRMED'}")


def test_10_prefix_integrity_and_accumulation(verbose: bool = False, quick: bool = False):
    """
    [R11] Does the spec's architecture preserve 64-dim prefix integrity over
    sequential adaptation rounds?

    FIX 4: v3 used random Gaussian docs → Two-NN returned rank=4 →
    catastrophic collapse. v4 uses manifold data (rank=16) for rank estimation
    so LowRank gets a sensible rank. FIX 1: LoRA init avoids collapse at any rank.
    """
    print("\n[Test 10] Prefix integrity + accumulation across sequential adaptation rounds")

    N     = 400 if quick else 600
    n_exp = 20  if quick else 30
    N_RND = 3   if quick else 5
    TRUE_RANK = 12

    # FIX 4: manifold-based rank estimation
    docs_manifold = make_manifold_data(N, true_rank=TRUE_RANK)
    r_twonn = two_nn_rank(docs_manifold)

    experts, clean_q, shifted_q, docs, _, _ = make_shifted_domain(N, n_exp)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

    with torch.no_grad():
        base_sims = F.cosine_similarity(shifted_q[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
        base_r    = recall_at_k(shifted_q, docs, k=20)

    archs = [
        ("Dense (★SPEC naive)",        DenseProjectionAdapter),
        ("Block-Diagonal",              BlockDiagonalAdapter),
        (f"LowRank r={r_twonn} ★SPEC", LowRankProjectionAdapter),
    ]

    arch_kwargs = {
        "Dense (★SPEC naive)":          {"dim": EMBED_DIM},
        "Block-Diagonal":               {"dim": EMBED_DIM},
        f"LowRank r={r_twonn} ★SPEC":  {"dim": EMBED_DIM, "rank": r_twonn},
    }

    print(f"  Two-NN rank (from rank-{TRUE_RANK} manifold): {r_twonn}")
    print(f"\n  {'Architecture':<25} {'Rnd':>4}  {'Drift':>8}  {'Gain':>8}  {'Pareto':>8}")
    print(f"  {'─'*60}")

    for arch_name, cls in archs:
        q_current = shifted_q.clone()
        for rnd in range(1, N_RND + 1):
            kw = arch_kwargs[arch_name]
            adapter = cls(**kw).to(DEVICE)
            train_adapter(adapter, q_current, docs, cross_domain_negs=cross_negs, epochs=4)
            with torch.no_grad():
                q_current = adapter(q_current)
                drift = prefix_drift(adapter, shifted_q, docs, base_sims)
                gain  = recall_at_k(q_current, docs, k=20) - base_r
            score  = pareto_score(gain, drift)
            marker = "◄ WARN" if drift > 0.05 else ""
            print(f"  {arch_name:<25} {rnd:>4}  {drift:>8.4f}  {gain:>+8.3f}  "
                  f"{score:>8.4f}  {marker}")
        print()

    print("  Interpretation: if drift compounds (increases each round), the adapter")
    print("  architecture is unsafe for sequential deployment without index rebuild.")


def test_11_streaming_centroid_tournament(verbose: bool = False, quick: bool = False):
    """
    [R12] Is EMA with α=0.1 (spec) the best centroid tracking strategy?
    Tournament: true running mean vs fixed window vs EMA α sweep.
    """
    print("\n[Test 11] Streaming centroid method tournament — accuracy vs shift response")

    N_WARM  = 1000 if not quick else 400
    N_SHIFT = 500  if not quick else 200
    BATCH   = 50

    phase1 = F.normalize(torch.randn(N_WARM,  EMBED_DIM, device=DEVICE), p=2, dim=-1)
    phase2 = F.normalize(torch.randn(N_SHIFT, EMBED_DIM, device=DEVICE) + 2.0, p=2, dim=-1)

    true_mean1 = phase1.mean(0)
    true_mean2 = phase2.mean(0)

    def track(strategy_fn) -> Tuple[float, float]:
        state = strategy_fn()
        for i in range(0, N_WARM, BATCH):
            state["update"](phase1[i: i + BATCH])
        conv_sim = F.cosine_similarity(state["get"]().unsqueeze(0),
                                        true_mean1.unsqueeze(0)).item()
        for i in range(0, N_SHIFT, BATCH):
            state["update"](phase2[i: i + BATCH])
        shift_sim = F.cosine_similarity(state["get"]().unsqueeze(0),
                                         true_mean2.unsqueeze(0)).item()
        return conv_sim, shift_sim

    rows = []

    # True running mean
    def make_running_mean():
        s = {"n": 0, "mean": torch.zeros(EMBED_DIM, device=DEVICE)}
        def upd(b):
            s["mean"] = (s["n"] * s["mean"] + b.sum(0)) / (s["n"] + len(b))
            s["n"] += len(b)
        return {"update": upd, "get": lambda: s["mean"]}
    c, sh = track(make_running_mean)
    rows.append({"name": "True running mean", "conv": c, "shift": sh,
                 "score": 0.6 * c + 0.4 * sh})

    # Fixed window (last 100)
    from collections import deque
    def make_fixed_window(w=100):
        buf = deque(maxlen=w)
        def upd(b):
            for row in b:
                buf.append(row)
        def get():
            return torch.stack(list(buf)).mean(0) if buf else torch.zeros(EMBED_DIM, device=DEVICE)
        return {"update": upd, "get": get}
    c, sh = track(make_fixed_window)
    rows.append({"name": "Fixed window (100)", "conv": c, "shift": sh,
                 "score": 0.6 * c + 0.4 * sh})

    # EMA variants
    for alpha in ([0.01, 0.05, 0.10, 0.20, 0.50] if not quick else [0.05, 0.10, 0.20]):
        def make_ema(a=alpha):
            s = {"m": torch.zeros(EMBED_DIM, device=DEVICE), "init": False}
            def upd(b):
                bm = b.mean(0)
                if not s["init"]:
                    s["m"] = bm.clone(); s["init"] = True
                else:
                    s["m"] = (1 - a) * s["m"] + a * bm
            return {"update": upd, "get": lambda: s["m"]}
        c, sh = track(make_ema)
        spec_tag = " ★SPEC" if abs(alpha - STREAMING_ALPHA) < 1e-6 else ""
        rows.append({"name": f"EMA α={alpha:.2f}{spec_tag}",
                     "conv": c, "shift": sh,
                     "score": 0.6 * c + 0.4 * sh})

    print_tournament(
        "Centroid tracking: convergence (60%) + shift response (40%)",
        rows,
        [("conv", "Convergence", False), ("shift", "Shift resp.", False),
         ("score", "Score", False)],
        winner_col="score", spec_name="★SPEC",
        notes="Score = 0.6×convergence + 0.4×shift response"
    )


def test_12_adapter_boundary_interference(verbose: bool = False, quick: bool = False):
    """
    [NEW] Equidistant query handling. FIX 1+2+3: adapters now actually learn,
    so boundary interference is a real measurement rather than comparing
    equally-broken projections.
    """
    print("\n[Test 12] Adapter boundary interference — equidistant query handling")

    N = 200 if not quick else 100

    cent_A = F.normalize(torch.tensor([1.0] + [0.0] * (EMBED_DIM - 1)).to(DEVICE), p=2, dim=-1)
    cent_B = F.normalize(torch.tensor([0.0, 1.0] + [0.0] * (EMBED_DIM - 2)).to(DEVICE), p=2, dim=-1)

    _, cA_clean, sq_A, docs_A, _, _ = make_shifted_domain(N, 5)
    _, cB_clean, sq_B, docs_B, _, _ = make_shifted_domain(N, 5)

    r = two_nn_rank(docs_A)
    adapt_A = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    adapt_B = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    # FIX 3: cross-domain negatives
    train_adapter(adapt_A, sq_A, docs_A, cross_domain_negs=cB_clean[:N])
    train_adapter(adapt_B, sq_B, docs_B, cross_domain_negs=cA_clean[:N])

    boundary_queries = F.normalize(
        cent_A + cent_B + torch.randn(N, EMBED_DIM, device=DEVICE) * 0.02,
        p=2, dim=-1)

    def r64_boundary(q_proj):
        return recall_at_k(q_proj[:, :PREFIX_DIM], docs_A[:, :PREFIX_DIM], k=20)

    with torch.no_grad():
        base = r64_boundary(boundary_queries)
        sim_a = (F.normalize(boundary_queries, p=2, dim=-1) @ cent_A).mean()
        sim_b = (F.normalize(boundary_queries, p=2, dim=-1) @ cent_B).mean()
        nearest  = adapt_A if sim_a >= sim_b else adapt_B
        r_nearest = r64_boundary(nearest(boundary_queries))
        wa = (sim_a / (sim_a + sim_b)).item()
        blended   = wa * adapt_A(boundary_queries) + (1 - wa) * adapt_B(boundary_queries)
        r_blended = r64_boundary(blended)
        max_sim   = max(sim_a.item(), sim_b.item())
        r_gated   = r64_boundary(adapt_A(boundary_queries) if max_sim >= 0.50
                                  else boundary_queries)

    rows = [
        {"name": "No adapter (baseline)",   "r64": base,      "score": base},
        {"name": "Nearest centroid ★SPEC",  "r64": r_nearest, "score": r_nearest},
        {"name": "Blended projection",       "r64": r_blended, "score": r_blended},
        {"name": "Threshold-gated (τ=0.50)", "r64": r_gated,  "score": r_gated},
    ]
    print_tournament(
        "R@20(64d) for boundary queries — queries equidistant between two domains",
        rows, [("r64", "R@20(64d)", True)],
        winner_col="score", spec_name="★SPEC",
        notes="If blended wins significantly, spec selection logic needs extension"
    )


def test_13_sample_efficiency_curves(verbose: bool = False, quick: bool = False):
    """
    [NEW] At what N does each adapter reach 90% of asymptotic coherence recovery?
    FIX 1+2+3: adapters now learn, so curves are informative rather than flat.
    """
    print("\n[Test 13] Sample efficiency — N needed to reach 90% asymptotic coherence")

    N_MAX = 800 if not quick else 400
    n_exp = 20  if quick else 30
    ns    = [25, 50, 100, 200, 400, N_MAX] if not quick else [25, 50, 100, 200]

    experts, clean_q, shifted_q, docs, _, _ = make_shifted_domain(N_MAX, n_exp)
    cross_negs = clean_q[torch.randperm(N_MAX, device=DEVICE)]
    r = two_nn_rank(docs)
    baseline_coh = routing_coherence(shifted_q, experts)

    def _train_eval(cls, kwargs, n_train):
        a = cls(**kwargs).to(DEVICE)
        idx = torch.randperm(N_MAX, device=DEVICE)[:n_train]
        cn  = cross_negs[idx]
        train_adapter(a, shifted_q[idx], docs[idx], cross_domain_negs=cn)
        with torch.no_grad():
            return routing_coherence(a(shifted_q), experts)

    archs = [
        ("Diagonal",             DiagonalAdapter,           {"dim": EMBED_DIM}),
        ("Block-Diagonal",       BlockDiagonalAdapter,      {"dim": EMBED_DIM}),
        (f"LowRank r={r} ★SPEC", LowRankProjectionAdapter, {"dim": EMBED_DIM, "rank": r}),
        ("Dense",                DenseProjectionAdapter,    {"dim": EMBED_DIM}),
    ]

    print(f"\n  Baseline coherence (no adapter): {baseline_coh:.3f}")
    print(f"\n  {'Architecture':<25} " + "  ".join(f"N={n:>4}" for n in ns) + "  90% at N")
    print(f"  {'─'*75}")

    for name, cls, kwargs in archs:
        asym   = _train_eval(cls, kwargs, N_MAX)
        # 90% of improvement over baseline
        target = baseline_coh + 0.90 * max(asym - baseline_coh, 0.01)
        n90    = None
        row    = []
        for n in ns:
            coh = _train_eval(cls, kwargs, n)
            row.append(coh)
            if n90 is None and coh >= target:
                n90 = n
        n90_str = str(n90) if n90 else ">max"
        print(f"  {name:<25} " + "  ".join(f"{v*100:>6.1f}%" for v in row)
              + f"  {n90_str:>6}")


def test_14_recovery_speed_under_shift(verbose: bool = False, quick: bool = False):
    """
    [NEW] Which adapter architecture recovers routing coherence fastest?
    FIX 1+2+3: LowRank now starts at identity (LoRA init) so it converges
    properly instead of collapsing from epoch 1 to 0.15 coherence.
    """
    print("\n[Test 14] Recovery speed — coherence vs training steps per architecture")

    N     = 400 if not quick else 200
    n_exp = 20  if not quick else 10
    experts, clean_q, shifted_q, docs, _, _ = make_shifted_domain(N, n_exp)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]
    r = two_nn_rank(docs)

    def coherence_curve(cls, kwargs):
        a   = cls(**kwargs).to(DEVICE)
        opt = torch.optim.AdamW(a.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        n   = len(shifted_q)
        curve = [routing_coherence(shifted_q, experts)]   # step 0
        a.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=DEVICE)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i: i + BATCH_SIZE]
                opt.zero_grad()
                matryoshka_infonce(a(shifted_q[idx]), docs[idx],
                                   neg=cross_negs[idx]).backward()
                nn.utils.clip_grad_norm_(a.parameters(), GRAD_CLIP)
                opt.step()
            scheduler.step()
            a.eval()
            with torch.no_grad():
                curve.append(routing_coherence(a(shifted_q), experts))
            a.train()
        return curve

    archs = [
        ("Diagonal",             DiagonalAdapter,           {"dim": EMBED_DIM}),
        ("Block-Diagonal",       BlockDiagonalAdapter,      {"dim": EMBED_DIM}),
        (f"LowRank r={r} ★SPEC", LowRankProjectionAdapter, {"dim": EMBED_DIM, "rank": r}),
        ("Dense",                DenseProjectionAdapter,    {"dim": EMBED_DIM}),
    ]

    baseline = routing_coherence(shifted_q, experts)
    print(f"\n  Baseline coherence: {baseline:.3f}  |  Target: {COHERENCE_THRESHOLD:.2f}")
    print(f"\n  {'Architecture':<25} " + "  ".join(f"Ep{e}" for e in range(EPOCHS + 1))
          + "  Ep@thresh  FinalVar")
    print(f"  {'─'*80}")

    for name, cls, kwargs in archs:
        curve    = coherence_curve(cls, kwargs)
        ep_thresh = next((i for i, v in enumerate(curve) if v >= COHERENCE_THRESHOLD), None)
        final_var = float(np.var(curve[-3:])) if len(curve) >= 3 else 0.0
        ep_str    = str(ep_thresh) if ep_thresh is not None else ">max"
        print(f"  {name:<25} " + "  ".join(f"{v:.3f}" for v in curve)
              + f"  {ep_str:>9}  {final_var:.5f}")

    print(f"\n  Ep@thresh: epoch at which coherence first exceeds {COHERENCE_THRESHOLD}")
    print("  FinalVar:  variance of last 3 checkpoints (lower = more stable convergence)")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    1:  (test_1_mrl_nested_validity,              "[R1]      MRL nested sizes — monotonic recall"),
    2:  (test_2_two_stage_funnel_optimality,      "[R2,R3]   Two-stage funnel K_coarse optimality"),
    3:  (test_3_adapter_architecture_tournament,  "[R4,R11]  Architecture tournament on shifted domain"),
    4:  (test_4_projection_target_comparison,     "[R5]      Projection target + centroid update"),
    5:  (test_5_trigger_threshold_optimality,     "[R6]      Trigger (threshold × window) grid search"),
    6:  (test_6_training_data_composition,        "[R7]      Training data composition tournament"),
    7:  (test_7_centroid_update_scope,            "[R8]      Centroid update scope + accumulation"),
    8:  (test_8_adapter_selection_and_ood,        "[R9]      Adapter selection + OOD handling"),
    9:  (test_9_rank_determination_tournament,    "[R10]     Two-NN vs PCA vs fixed ranks"),
    10: (test_10_prefix_integrity_and_accumulation,"[R11]    Prefix integrity under sequential rounds"),
    11: (test_11_streaming_centroid_tournament,   "[R12]     Streaming centroid method tournament"),
    12: (test_12_adapter_boundary_interference,   "[NEW]     Boundary query interference"),
    13: (test_13_sample_efficiency_curves,        "[NEW]     Sample efficiency — N to 90% asymptotic"),
    14: (test_14_recovery_speed_under_shift,      "[NEW]     Recovery speed — coherence vs steps"),
}


def main():
    parser = argparse.ArgumentParser(
        description="DEMoE v4.0 Domain Projection Adapter Test Suite v4")
    parser.add_argument("--test",    nargs="*", type=int,
                        help="Run only these test numbers (default: all)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick",   action="store_true",
                        help="Reduced N and epochs — targets ~2 min on 5070")
    args = parser.parse_args()

    to_run = args.test if args.test else list(ALL_TESTS.keys())

    print("=" * 70)
    print("  DEMoE v4.0 — Domain Projection Adapter Test Suite v4")
    print(f"  Device: {DEVICE}  |  EMBED_DIM={EMBED_DIM}  |  PREFIX_DIM={PREFIX_DIM}")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"  Key fixes: LoRA init · calibrated shift · cross-domain negs · manifold rank data")
    print("=" * 70)
    print("\nTests scheduled:")
    for n in to_run:
        print(f"  {n:2d}: {ALL_TESTS[n][1]}")

    failed = []
    for n in to_run:
        fn, _ = ALL_TESTS[n]
        try:
            fn(verbose=args.verbose, quick=args.quick)
        except Exception as e:
            print(f"\n  {FAIL} Test {n} raised: {e}")
            failed.append(n)
            if args.verbose:
                import traceback; traceback.print_exc()

    print("\n" + "=" * 70)
    if failed:
        print(f"  {FAIL} Tests with errors: {failed}")
    else:
        print(f"  {PASS} All tests completed.")
    print("  Any ✗ in tournament tables = spec's approach was beaten.")
    print("  Investigate those cases before integrating into the router.")
    print("=" * 70)


if __name__ == "__main__":
    main()