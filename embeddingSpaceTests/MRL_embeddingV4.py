"""
DEMoE v4.0 — Domain Projection Adapter Test Suite v8
=====================================================
Every test answers: "Is the spec's proposed solution the best I can
think of, or does a simpler/different approach win?"

ROOT CAUSE ANALYSIS of v7 output failures
──────────────────────────────────────────────────────────────────────
  Problem A — Adapters show 0% gain (Tests 3,4,6,7,9,13,14):
    make_shifted_domain binary-searches shift_mag using a FIXED seed (42)
    in _accuracy_for_shift, producing calibrated centroids. Then it
    re-initializes expert_centroids from the current random state — NEW
    random centroids that have NO relationship to the calibrated shift_mag.
    Fix: keep the same expert_centroids used during calibration, OR
    separate calibration from data generation cleanly. v8 finds shift_mag
    by probing WITHIN the same function using temporary centroids, then
    applies that shift_mag to freshly drawn centroids with the same geometry.
    Actually the real fix: doc positives must be the EXPERT CENTROIDS, not
    near-query points. shifted_q is displaced from clean_q; docs (near
    clean_q) are almost as displaced from the clean centroid. The adapter
    needs to learn: "map shifted_q back toward the clean expert centroid".
    So pos_embs for training should be the expert centroids (or near-centroid
    docs), not near-shifted-query docs.

  Problem B — routing_accuracy always ~2-16% (Tests 4,7,13,14):
    With n_exp=25-55 and random centroids in 768-dim, the 64-dim prefix
    cosine similarity between query and its TRUE centroid is only marginally
    better than a random centroid (concentration of measure). Result: even
    clean queries route to the wrong expert most of the time at 64-dim.
    Fix: use fewer experts (n_exp ≤ 10) OR make expert centroids well-separated
    using a structured construction (e.g. scaled standard basis + noise).
    v8 uses make_well_separated_experts() to guarantee clean routing accuracy
    ≥ 0.85 before shift, and measure degradation after shift.

  Problem C — std=0.0000 in bootstrapped tests (Tests 3,4,6,9):
    bootstrap_trial sets a torch seed, but make_shifted_domain uses
    torch.randn() which respects the seed — every trial produces IDENTICAL
    data with the same noise and n_exp. The "bootstrap" does nothing.
    Fix: inside each trial, re-sample n_exp and noise from their ranges,
    and ensure different seeds produce genuinely different data.

  Problem D — Test 5 trivially zero FPR+FNR:
    good_acc=1.0, bad_acc=0.33 — every threshold between 0.33 and 1.0
    perfectly separates good/bad days. All 25 grid points score 0.
    Fix: generate good days with realistic noise (acc ≈ 0.80-0.95) and
    bad days with moderate shift (acc ≈ 0.45-0.65) so the separation is
    partial and different thresholds/windows genuinely differ.

  Problem E — Test 2 S2 recall too low (40% for K_coarse=20):
    recall_at_k measures "is query i's document among top-K documents?"
    but the two-stage funnel measures "is query i's EXPERT among top-K?"
    These are different tasks. With n_exp=57 and K_fine=5, hitting 1/57
    experts ≥ is hard at stage 1. The S2 recall is expert-recall not
    doc-recall. Fix: measure doc-level recall in Test 2 using the expert
    centroid as a proxy document — if the correct expert is in top-K at
    stage 2, recall is 1 for that query.

  Problem F — Two-NN always returns rank=4:
    Even with FIX 4 signal scaling, the Two-NN estimator on manifold data
    returns 4 instead of 16. The issue: `two_nn_rank` applies `TWO_NN_SCALE=0.5`
    giving rank = round(d_hat * 0.5). For d_hat ≈ 8-10 this gives 4-5.
    The Two-NN estimator itself is correct, it's the scaling factor that
    under-estimates. v8 increases TWO_NN_SCALE to 1.0 and verifies against
    true rank. Also uses larger N (≥ 800) for stable estimates.

RETAINED FIXES from v7
──────────────────────────────────────────────────────────────────────
  Fix 1 — routing_accuracy uses ground-truth assignments
  Fix 2 — Calibration uses exact n_exp
  Fix 3 — Test 2 n_exp capped
  Fix 4 — Manifold data signal scaling
  Fix 5 — Test 12 neg size tiled to N
  Fix 6 — Bootstrap CI (properly re-sampling per trial)

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
  python MRL_embeddingV8.py            # all tests
  python MRL_embeddingV8.py --test 3 9 # subset
  python MRL_embeddingV8.py --quick    # reduced N, fewer epochs (fastest mode)
"""

import os, math, argparse, contextlib
from typing import Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Spec constants
# ─────────────────────────────────────────────────────────────────────────────
EMBED_DIM           = 768
PREFIX_DIM          = 64
MID_DIM             = 256
K_COARSE            = 20
K_FINE              = 5
COHERENCE_THRESHOLD = 0.70
COHERENCE_WINDOW    = 3
MAX_ADAPTERS        = 50
TWO_NN_SCALE        = 1.0    # v8: increased from 0.5 — Two-NN d_hat is already conservative
RANK_MIN            = 4
RANK_MAX            = 64
RANK_DEFAULT        = 8
SMALL_CORPUS_N      = 200
STREAMING_ALPHA     = 0.1

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR          = 5e-4
BATCH_SIZE  = 128
EPOCHS      = 8            # v8: increased from 6 for more learning signal
GRAD_CLIP   = 1.0
TEMPERATURE = 0.07

N_TRIALS = 3  # set to 2 in quick mode

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter sampling
# ─────────────────────────────────────────────────────────────────────────────

_rng = np.random.default_rng()


def sample_noise(lo: float = 0.05, hi: float = 0.18) -> float:
    return float(_rng.uniform(lo, hi))


def sample_n_experts(lo: int, hi: int) -> int:
    return int(_rng.integers(lo, hi + 1))


def log_hyperparams(**kwargs):
    parts = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"  [hyper] {parts}")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6 (corrected): Bootstrap trial context manager
# Each trial re-seeds AND re-samples independent data — std will be non-zero.
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def bootstrap_trial(seed: int):
    """Deterministic seed for one trial. Restores global state after."""
    torch.manual_seed(seed)
    np_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield seed
    finally:
        np.random.set_state(np_state)
        torch.seed()


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
STAR = "\033[93m★\033[0m"
SPEC = "\033[96m[SPEC]\033[0m"


def pareto_score(gain: float, drift: float, drift_penalty: float = 50.0) -> float:
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
    print(f"\n  {'─'*72}")
    print(f"  {title}")
    print(f"  {'─'*72}")

    vals     = [r[winner_col] for r in rows]
    best_val = max(vals)
    best_name = next(r["name"] for r in rows if r[winner_col] == best_val)
    baseline  = next((r[winner_col] for r in rows if "baseline" in r["name"].lower()), None)

    name_w = max(len(r["name"]) for r in rows) + 2
    header = f"  {'Approach':<{name_w}}"
    for _, disp, _ in cols:
        header += f"  {disp:>16}"
    print(header)
    print(f"  {'─'*72}")

    for r in rows:
        is_winner = r[winner_col] == best_val
        is_spec   = spec_name and spec_name in r["name"]
        line = f"  {r['name']:<{name_w}}"
        for key, _, pct in cols:
            val = r.get(key, float("nan"))
            if isinstance(val, tuple):
                mean, std = val
                cell = f"{mean*100:.1f}±{std*100:.1f}%" if pct else f"{mean:.4f}±{std:.4f}"
                line += f"  {cell:>16}"
            else:
                line += f"  {fmt(val, pct=pct):>16}"
        suffix = ""
        if is_spec:
            suffix += f"  {SPEC}"
        if is_winner:
            suffix += f"  {STAR} WINNER"
        print(line + suffix)

    print(f"  {'─'*72}")

    spec_score    = next((r[winner_col] for r in rows if spec_name and spec_name in r["name"]), None)
    spec_row_name = next((r["name"]     for r in rows if spec_name and spec_name in r["name"]), "")

    def _scalar(x):
        return x[0] if isinstance(x, tuple) else x

    if spec_score is not None:
        sv     = _scalar(spec_score)
        bv     = _scalar(best_val)
        margin = bv - sv
        TOL    = 1e-6   # floating-point tie tolerance
        if margin > TOL:
            # Spec is genuinely behind — report gap
            print(f"  {FAIL} SPEC approach is NOT the winner. Gap: {margin:.4f}")
        else:
            # Spec is tied for first or actually first
            if baseline is not None:
                bbase = _scalar(baseline)
                if bv > bbase + TOL:
                    print(f"  {PASS} SPEC approach wins (tied for best). "
                          f"Improvement over baseline: {(bv-bbase)/max(abs(bbase),1e-9)*100:+.1f}%")
                else:
                    print(f"  {PASS} SPEC approach tied for best — "
                          f"no tested alternative beats it.")
            else:
                print(f"  {PASS} SPEC approach wins among tested alternatives.")

    if notes:
        print(f"  Note: {notes}")


# ─────────────────────────────────────────────────────────────────────────────
# Fix B: Well-separated expert construction
# Random centroids in 768-dim concentrate on the sphere — mutual cosine sims
# are ~0 ±1/√768 ≈ ±0.036 — barely distinguishable in 64-dim prefix.
# Use structured experts: e_i = normalize(e_base + α * e_i_unique) where
# e_base is shared and e_i_unique is orthogonal to all others.
# This guarantees clean routing accuracy ≥ 0.85 at 64-dim prefix.
# ─────────────────────────────────────────────────────────────────────────────

def make_well_separated_experts(n_experts: int,
                                 dim: int = EMBED_DIM,
                                 separation: float = 3.0,
                                 ) -> torch.Tensor:
    """
    Build n_experts centroids that are well-separated in PREFIX_DIM space.

    Construction:
      - Draw a random orthonormal basis of size min(n_experts, PREFIX_DIM)
        in PREFIX_DIM space (the routing space).
      - Assign each expert a unique basis direction scaled by `separation`,
        then embed into full EMBED_DIM with random suffix (irrelevant to routing).
      - After L2-normalization the prefix directions are distinct and well-separated.

    This guarantees clean queries (small noise) route to the correct expert
    with high accuracy via 64-dim nearest-centroid, giving a meaningful
    baseline to degrade with a domain shift.
    """
    assert n_experts <= PREFIX_DIM, \
        f"n_experts ({n_experts}) must be ≤ PREFIX_DIM ({PREFIX_DIM}) for guaranteed separation"

    # Random orthonormal basis in prefix space
    rand_mat  = torch.randn(PREFIX_DIM, PREFIX_DIM, device=DEVICE)
    Q, _      = torch.linalg.qr(rand_mat)           # Q: (PREFIX_DIM, PREFIX_DIM), orthonormal cols
    prefix_dirs = Q[:, :n_experts].T                 # (n_experts, PREFIX_DIM)

    # Full-dim expert: scaled prefix direction + random suffix
    suffix    = torch.randn(n_experts, dim - PREFIX_DIM, device=DEVICE) * 0.1
    full_vecs = torch.cat([prefix_dirs * separation, suffix], dim=1)
    return F.normalize(full_vecs, p=2, dim=-1)       # (n_experts, EMBED_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Routing metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def routing_accuracy(queries: torch.Tensor,
                      expert_centroids: torch.Tensor,
                      assignments: torch.Tensor) -> float:
    """
    Ground-truth routing accuracy: fraction of queries whose nearest centroid
    (by 64-dim prefix cosine similarity) matches their true assignment.
    """
    qp = F.normalize(queries[:, :PREFIX_DIM], p=2, dim=-1)
    ep = F.normalize(expert_centroids[:, :PREFIX_DIM], p=2, dim=-1)
    predicted = (qp @ ep.T).argmax(dim=1)
    return (predicted == assignments).float().mean().item()


@torch.no_grad()
def routing_accuracy_adapted(queries: torch.Tensor,
                               expert_centroids: torch.Tensor,
                               assignments: torch.Tensor,
                               adapter: Optional[nn.Module]) -> float:
    """
    Routing accuracy after adaptation.
    Both queries and expert centroids are projected (spec Section 1.3).
    """
    if adapter is None:
        return routing_accuracy(queries, expert_centroids, assignments)
    q_proj = adapter(queries)
    e_proj = adapter(expert_centroids)
    return routing_accuracy(q_proj, e_proj, assignments)


@torch.no_grad()
def routing_coherence_raw(queries: torch.Tensor,
                           expert_centroids: torch.Tensor,
                           noise_scale: float = 0.02) -> float:
    """Self-consistency coherence (retained for Test 11 streaming centroid)."""
    qp    = F.normalize(queries[:, :PREFIX_DIM], p=2, dim=-1)
    ecp   = F.normalize(expert_centroids[:, :PREFIX_DIM], p=2, dim=-1)
    noisy = F.normalize(queries + torch.randn_like(queries) * noise_scale, p=2, dim=-1)
    nqp   = F.normalize(noisy[:, :PREFIX_DIM], p=2, dim=-1)
    return ((qp @ ecp.T).argmax(1) == (nqp @ ecp.T).argmax(1)).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Fix A+B: Domain shift data generation
#
# Key design decisions:
#  1. Expert centroids are well-separated (make_well_separated_experts).
#  2. shift_mag is calibrated using THESE same centroids (not a temp set).
#  3. Training positive = expert centroid (not near-query doc) so the adapter
#     learns to map shifted_q → expert centroid direction.
#  4. true_docs are drawn near the expert centroid (independent of shift).
# ─────────────────────────────────────────────────────────────────────────────

def _acc_for_shift_with_experts(experts: torch.Tensor,
                                  n_queries: int,
                                  shift_vec: torch.Tensor,
                                  shift_mag: float,
                                  noise: float) -> float:
    """Probe routing accuracy for a given shift_mag using fixed expert centroids."""
    torch.manual_seed(99)  # local reproducibility for binary search
    n_exp   = len(experts)
    assigns = torch.arange(n_queries, device=DEVICE) % n_exp
    clean   = F.normalize(
        experts[assigns] + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)
    shifted = F.normalize(
        clean + shift_vec * shift_mag + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)
    return routing_accuracy(shifted, experts, assigns)


def make_shifted_domain(n_queries: int,
                         n_experts: int,
                         shift_mag: Optional[float] = None,
                         noise: float = 0.07,
                         target_lo: float = 0.45,
                         target_hi: float = 0.58,
                         ) -> Tuple[torch.Tensor, ...]:
    """
    Creates a domain-shift scenario where baseline routing accuracy is calibrated
    to land in [target_lo, target_hi].

    Fix A: Uses well-separated experts so clean routing accuracy is high (≥ 0.85).
    Fix B: Calibration uses the SAME expert centroids as the returned data.
    Fix A (training signal): true_docs drawn near expert centroids, NOT near
         shifted queries. The adapter must learn to pull shifted_q toward the
         right expert centroid — this gives a meaningful training signal.

    Returns
    ───────
    expert_centroids : (n_experts, EMBED_DIM)   — well-separated
    clean_queries    : (n_queries, EMBED_DIM)    — pre-shift, routes correctly
    shifted_queries  : (n_queries, EMBED_DIM)    — post-shift, misroutes
    true_docs        : (n_queries, EMBED_DIM)    — drawn near expert centroid
    shift_vec        : (EMBED_DIM,)
    assignments      : (n_queries,) long         — ground-truth expert index
    """
    experts    = make_well_separated_experts(n_experts)
    assignments = torch.arange(n_queries, device=DEVICE) % n_experts

    shift_vec = F.normalize(torch.randn(1, EMBED_DIM, device=DEVICE), p=2, dim=-1).squeeze(0)

    if shift_mag is None:
        # Binary search: find shift_mag s.t. routing accuracy ∈ [target_lo, target_hi]
        lo_mag, hi_mag = 0.0, 10.0
        for _ in range(24):
            mid = (lo_mag + hi_mag) / 2
            acc = _acc_for_shift_with_experts(experts, min(n_queries, 300),
                                               shift_vec.unsqueeze(0), mid, noise)
            if acc > target_hi:
                lo_mag = mid
            elif acc < target_lo:
                hi_mag = mid
            else:
                break
        shift_mag = (lo_mag + hi_mag) / 2

    clean_queries = F.normalize(
        experts[assignments] + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)

    shifted_queries = F.normalize(
        clean_queries + shift_vec.unsqueeze(0) * shift_mag
        + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)

    # Fix A: docs drawn near expert centroid — NOT near shifted_q.
    # Adapter training: map shifted_q → expert centroid space.
    # Positive = what the query SHOULD have been retrieving (centroid neighbourhood).
    true_docs = F.normalize(
        experts[assignments] + torch.randn(n_queries, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)

    return (experts, clean_queries, shifted_queries,
            true_docs, shift_vec, assignments)


# ─────────────────────────────────────────────────────────────────────────────
# Fix F: Manifold data with corrected Two-NN scale
# ─────────────────────────────────────────────────────────────────────────────

def make_manifold_data(n_samples: int,
                        true_rank: int,
                        ambient_dim: int = EMBED_DIM,
                        noise: float = 0.05,
                        normalize: bool = False,
                        ) -> torch.Tensor:
    """
    Embeddings on a rank-`true_rank` affine subspace with unit signal std.

    Fix 4: scale = sqrt(ambient_dim / true_rank) ensures signal std ≈ 1.0.
    Fix F: Two-NN scale increased to 1.0 (was 0.5), so Two-NN rank estimate
           is less conservative and closer to true rank.
    """
    basis = torch.linalg.svd(
        torch.randn(ambient_dim, true_rank, device=DEVICE), full_matrices=False
    ).Vh[:true_rank]                               # (true_rank, ambient_dim)
    codes = torch.randn(n_samples, true_rank, device=DEVICE)
    scale = math.sqrt(ambient_dim / true_rank)
    embs  = codes @ basis * scale
    embs  = embs + torch.randn_like(embs) * noise
    if normalize:
        return F.normalize(embs, p=2, dim=-1)
    return embs


def make_hard_negatives(queries: torch.Tensor, docs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        qn = F.normalize(queries, p=2, dim=-1)
        dn = F.normalize(docs,    p=2, dim=-1)
        sim = qn @ dn.T
        sim.fill_diagonal_(-2.0)
        return docs[sim.argmax(dim=1)]


# ─────────────────────────────────────────────────────────────────────────────
# Adapter architectures
# ─────────────────────────────────────────────────────────────────────────────

class DiagonalAdapter(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DenseProjectionAdapter(nn.Module):
    def __init__(self, dim: int = EMBED_DIM):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.P.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.P(x)


class BlockDiagonalAdapter(nn.Module):
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
    """LoRA-style residual: output = x + (x @ A) @ B. B=0 at init → identity."""
    def __init__(self, dim: int = EMBED_DIM, rank: int = RANK_DEFAULT):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.B = nn.Parameter(torch.zeros(rank, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (x @ self.A) @ self.B


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def infonce_inbatch(q: torch.Tensor, pos: torch.Tensor,
                    temperature: float = TEMPERATURE) -> torch.Tensor:
    qn  = F.normalize(q,   p=2, dim=-1)
    pn  = F.normalize(pos, p=2, dim=-1)
    sim = (qn @ pn.T) / temperature
    return F.cross_entropy(sim, torch.arange(len(q), device=q.device))


def infonce_with_negs(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor,
                       temperature: float = TEMPERATURE) -> torch.Tensor:
    qn  = F.normalize(q,   p=2, dim=-1)
    pn  = F.normalize(pos, p=2, dim=-1)
    nn_ = F.normalize(neg, p=2, dim=-1)
    pos_sim  = (qn * pn).sum(dim=-1, keepdim=True) / temperature    # (B,1)
    hard_sim = (qn * nn_).sum(dim=-1, keepdim=True) / temperature   # (B,1)
    inbatch  = (qn @ pn.T) / temperature                             # (B,B)
    logits   = torch.cat([pos_sim, hard_sim], dim=1)
    labels   = torch.zeros(len(q), dtype=torch.long, device=q.device)
    loss_hard    = F.cross_entropy(logits, labels)
    loss_inbatch = F.cross_entropy(inbatch, torch.arange(len(q), device=q.device))
    return 0.6 * loss_inbatch + 0.4 * loss_hard


def matryoshka_infonce(q: torch.Tensor, pos: torch.Tensor,
                        neg: Optional[torch.Tensor] = None,
                        temperature: float = TEMPERATURE) -> torch.Tensor:
    def fn(a, b, d=None):
        if neg is not None:
            n = neg[:len(a), :a.shape[-1]] if d is None else neg[:len(a), :d]
            return infonce_with_negs(a, b, n, temperature)
        return infonce_inbatch(a, b, temperature)
    return (1.0 * fn(q,                 pos)
          + 0.5 * fn(q[:, :MID_DIM],    pos[:, :MID_DIM],   MID_DIM)
          + 0.3 * fn(q[:, :PREFIX_DIM], pos[:, :PREFIX_DIM], PREFIX_DIM))


# ─────────────────────────────────────────────────────────────────────────────
# Two-NN intrinsic dimensionality  [R10]
# ─────────────────────────────────────────────────────────────────────────────

def two_nn_intrinsic_dim(embs: torch.Tensor, n_bootstrap: int = 15) -> float:
    """
    Facco et al. (2017): d_hat = -N / sum_i log(d2_i / d1_i)
    Uses Euclidean distance on UN-normalized embeddings (preserves subspace structure).
    """
    e = embs.float()
    N = len(e)

    def _est(x: torch.Tensor) -> float:
        n   = len(x)
        sq  = (x * x).sum(dim=1, keepdim=True)
        d2  = (sq + sq.T - 2.0 * (x @ x.T)).clamp(min=0)
        d2.fill_diagonal_(float("inf"))
        sd2, _ = torch.sort(d2, dim=1)
        d1  = sd2[:, 0].clamp(min=1e-10).sqrt()
        d2_ = sd2[:, 1].clamp(min=1e-10).sqrt()
        ratio = (d2_ / d1).clamp(min=1.0 + 1e-8)
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
    e = embs.float() - embs.float().mean(dim=0)
    _, S, _ = torch.linalg.svd(e, full_matrices=False)
    var_exp  = (S**2).cumsum(0) / (S**2).sum()
    n_comp   = (var_exp < var_threshold).sum().item() + 1
    return int(n_comp)


def two_nn_rank(embs: torch.Tensor) -> int:
    """
    Fix F: TWO_NN_SCALE=1.0 (was 0.5).
    The Two-NN estimator returns intrinsic dimension d; the spec maps this to
    adapter rank r = round(d * TWO_NN_SCALE). At 0.5 this under-estimated
    badly (rank 4 for true rank 16). At 1.0 the estimate is much closer.
    """
    if len(embs) < SMALL_CORPUS_N:
        return RANK_DEFAULT
    return max(RANK_MIN, min(RANK_MAX, round(two_nn_intrinsic_dim(embs) * TWO_NN_SCALE)))


# ─────────────────────────────────────────────────────────────────────────────
# Training
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
    Adapter applied to queries only — docs/centroids unchanged [R5].
    pos_embs should be expert-centroid-neighbourhood docs (Fix A).
    """
    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
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
    """Query i's ground truth is doc i (diagonal recall)."""
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
    """Mean absolute change in 64-dim prefix cosine similarity post-adapter."""
    qp       = adapter(queries)
    new_sims = F.cosine_similarity(qp[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
    return (baseline_sims - new_sims).abs().mean().item()


@torch.no_grad()
def mrl_two_stage_routing(queries:          torch.Tensor,
                            expert_centroids: torch.Tensor,
                            k_coarse:         int = K_COARSE,
                            k_fine:           int = K_FINE,
                            adapter:          Optional[nn.Module] = None,
                            assignments:      Optional[torch.Tensor] = None,
                            ) -> Tuple[float, float]:
    """
    [R2, R3] Two-stage MRL funnel.
    Returns (stage1_expert_recall, stage2_expert_recall).
    Both measure expert recall (not doc recall) — correct framing for routing.
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
    """[R1] Monotonic recall: larger prefix → higher recall."""
    print("\n[Test 1] MRL nested size validity — monotonic recall across prefix sizes")

    N     = 400 if quick else 1000
    n_exp = sample_n_experts(30, 60) if quick else sample_n_experts(40, 64)
    noise = sample_noise(0.10, 0.20)
    log_hyperparams(noise=f"{noise:.3f}", n_exp=n_exp, N=N)

    experts = make_well_separated_experts(n_exp)
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


def test_2_two_stage_funnel_optimality(verbose: bool = False, quick: bool = False):
    """
    [R2, R3] Is K_coarse=20 optimal?

    Fix 3: n_exp ≤ 60 so K_coarse=20 covers ≥33% of experts.
    Fix E: Two-stage routing measures EXPERT recall (correct task for a routing funnel).
           Single-stage baselines also measured as expert-recall for fair comparison.
    """
    print("\n[Test 2] Two-stage funnel K_coarse optimality — sweeping K_coarse")

    noise = sample_noise(0.10, 0.18)
    N     = 300 if quick else 600
    n_exp_max = max(K_COARSE * 3, 60)
    n_exp = sample_n_experts(20, min(n_exp_max, 50)) if quick \
            else sample_n_experts(30, n_exp_max)
    # n_exp must fit in PREFIX_DIM for well-separated experts
    n_exp = min(n_exp, PREFIX_DIM)
    log_hyperparams(noise=f"{noise:.3f}", n_exp=n_exp, N=N)

    experts = make_well_separated_experts(n_exp)
    assigns = torch.arange(N, device=DEVICE) % n_exp
    queries = F.normalize(
        experts[assigns] + torch.randn(N, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)

    PREFIX_COST = PREFIX_DIM / EMBED_DIM

    def two_stage_cost(kc):
        return PREFIX_COST + kc / n_exp

    rows = []

    # Single-stage 768d expert recall: fraction routing correctly at full dim
    r_full_expert = routing_accuracy(queries, experts, assigns)
    rows.append({"name": "Single-stage 768d",
                 "s1_recall": r_full_expert, "s2_recall": r_full_expert,
                 "cost": 1.0, "efficiency": r_full_expert / 1.0})

    # Single-stage 64d expert recall
    r_64_expert = routing_accuracy(queries, experts, assigns)   # same metric at prefix level
    # Actually measure at 64d only — clip centroids and queries to prefix
    qp = F.normalize(queries[:, :PREFIX_DIM], p=2, dim=-1)
    ep = F.normalize(experts[:, :PREFIX_DIM], p=2, dim=-1)
    r_64_expert = ((qp @ ep.T).argmax(1) == assigns).float().mean().item()
    rows.append({"name": "Single-stage 64d",
                 "s1_recall": r_64_expert, "s2_recall": r_64_expert,
                 "cost": PREFIX_COST, "efficiency": r_64_expert / PREFIX_COST})

    k_values = [5, 10, 20, 50, 100] if not quick else [5, 10, 20, 50]
    for kc in k_values:
        s1, s2 = mrl_two_stage_routing(queries, experts, k_coarse=kc, k_fine=K_FINE,
                                        assignments=assigns)
        cost = two_stage_cost(kc)
        rows.append({"name": f"Two-stage K={kc:3d}" + (" ★SPEC" if kc == K_COARSE else ""),
                     "s1_recall": s1, "s2_recall": s2,
                     "cost": cost, "efficiency": s2 / max(cost, 1e-9)})

    print_tournament(
        "Two-stage funnel: expert recall vs normalised compute cost",
        rows,
        [("s1_recall", "S1 ExpertR", True), ("s2_recall", "S2 ExpertR", True),
         ("cost", "Cost(norm)", False),      ("efficiency", "Recall/Cost", False)],
        winner_col="efficiency",
        spec_name="★SPEC",
        notes="Expert recall = fraction of queries routed to correct expert"
    )

    oracle_recall = r_full_expert
    quality_bar   = 0.90 * oracle_recall
    qualifying    = [r for r in rows if r["s2_recall"] >= quality_bar]
    print(f"\n  Pareto frontier (≥90% of oracle={quality_bar*100:.1f}%):")
    if qualifying:
        cheapest = min(qualifying, key=lambda r: r["cost"])
        for r in rows:
            meets  = r["s2_recall"] >= quality_bar
            tag    = " ← cheapest qualifying" if r is cheapest else ""
            q_mark = "✓" if meets else " "
            print(f"    [{q_mark}] {r['name']:<25} S2={r['s2_recall']*100:.1f}%  "
                  f"cost={r['cost']:.4f}{tag}")
        spec_row = next((r for r in rows if "★SPEC" in r["name"]), None)
        if spec_row:
            if spec_row["s2_recall"] >= quality_bar:
                gap = spec_row["cost"] - cheapest["cost"]
                ok  = gap <= 0.05 * cheapest["cost"]
                print(f"  K_coarse=20 qualifies, within 5% of cheapest: "
                      f"{'✓ PASS' if ok else f'✗ FAIL — cost gap {gap:.4f}'}")
            else:
                print(f"  {FAIL} K_coarse=20 does NOT reach quality bar "
                      f"({spec_row['s2_recall']*100:.1f}% < {quality_bar*100:.1f}%)")
    else:
        print(f"  {FAIL} No approach reaches {quality_bar*100:.1f}% — try smaller n_exp")

    spec_row = next((r for r in rows if "★SPEC" in r["name"]), None)
    if spec_row:
        best_two = max((r["efficiency"] for r in rows if "K=" in r["name"]))
        at_knee  = spec_row["efficiency"] >= best_two * 0.85
        print(f"  K_coarse=20 within 15% of best two-stage efficiency: "
              f"{'✓ PASS' if at_knee else '✗ FAIL'}")


def test_3_adapter_architecture_tournament(verbose: bool = False, quick: bool = False):
    """
    [R4, R11] Tournament across adapter architectures.

    Fix C (std=0): Each trial uses DIFFERENT n_exp/noise sampled INSIDE the trial.
    Fix A: docs near expert centroids → real training signal.
    Fix B: well-separated experts → meaningful routing accuracy baseline.
    """
    print("\n[Test 3] Adapter architecture tournament — domain-shifted data, Pareto criterion")

    # Outer hyperparams (fixed across trials for comparability)
    N      = 400 if quick else 700
    n_trials = 2 if quick else N_TRIALS
    log_hyperparams(N=N, n_trials=n_trials)

    # Per-trial hyperparams: n_exp and noise vary so std is non-zero
    def _make_data(trial_seed):
        with bootstrap_trial(trial_seed):
            n_exp = sample_n_experts(6, min(15, PREFIX_DIM)) if quick \
                    else sample_n_experts(8, min(20, PREFIX_DIM))
            noise = sample_noise(0.05, 0.15)
            experts, clean_q, shifted_q, docs, _, assignments = \
                make_shifted_domain(N, n_exp, noise=noise)
            cross_negs = clean_q[torch.randperm(N, device=DEVICE)]
            base_sims  = F.cosine_similarity(shifted_q[:, :PREFIX_DIM],
                                              docs[:, :PREFIX_DIM])
            base_r768  = recall_at_k(shifted_q, docs, k=20)
            pre_acc    = routing_accuracy(shifted_q, experts, assignments)
            return experts, clean_q, shifted_q, docs, assignments, \
                   cross_negs, base_sims, base_r768, pre_acc

    # Determine Two-NN rank once (using representative data)
    manifold_sample = make_manifold_data(800, true_rank=8, noise=0.05, normalize=False)
    r_twonn = two_nn_rank(manifold_sample)
    print(f"  Two-NN estimated rank: {r_twonn}")

    candidates = [
        ("No adapter (baseline)",       None,                       {}),
        ("Diagonal",                    DiagonalAdapter,            {"dim": EMBED_DIM}),
        ("Dense (★SPEC naive)",         DenseProjectionAdapter,     {"dim": EMBED_DIM}),
        ("Block-Diagonal",              BlockDiagonalAdapter,       {"dim": EMBED_DIM}),
        ("LowRank r=2",                 LowRankProjectionAdapter,   {"dim": EMBED_DIM, "rank": 2}),
        (f"LowRank r={r_twonn} ★SPEC",  LowRankProjectionAdapter,  {"dim": EMBED_DIM, "rank": r_twonn}),
        ("LowRank r=32",                LowRankProjectionAdapter,   {"dim": EMBED_DIM, "rank": 32}),
    ]

    arch_scores: Dict[str, List[float]] = {name: [] for name, _, _ in candidates}
    arch_drifts: Dict[str, List[float]] = {name: [] for name, _, _ in candidates}
    arch_gains:  Dict[str, List[float]] = {name: [] for name, _, _ in candidates}

    for trial in range(n_trials):
        (experts, clean_q, shifted_q, docs, assignments,
         cross_negs, base_sims, base_r768, pre_acc) = _make_data(trial * 37 + 5)
        if trial == 0:
            print(f"  Calibration trial 0: pre-adapter routing accuracy = {pre_acc:.3f} "
                  f"(target 0.45–0.58)")

        for name, cls, kwargs in candidates:
            if cls is None:
                arch_drifts[name].append(0.0)
                arch_gains[name].append(0.0)
                arch_scores[name].append(pareto_score(0.0, 0.0))
            else:
                adapter = cls(**kwargs).to(DEVICE)
                train_adapter(adapter, shifted_q, docs, cross_domain_negs=cross_negs)
                with torch.no_grad():
                    drift = prefix_drift(adapter, shifted_q, docs, base_sims)
                    r768  = recall_at_k(adapter(shifted_q), docs, k=20)
                    gain  = r768 - base_r768
                arch_drifts[name].append(drift)
                arch_gains[name].append(gain)
                arch_scores[name].append(pareto_score(gain, drift))

    rows = []
    for name, cls, kwargs in candidates:
        scores = arch_scores[name]
        drifts = arch_drifts[name]
        gains  = arch_gains[name]
        n_par  = sum(p.numel() for p in cls(**kwargs).parameters()) if cls else 0
        rows.append({
            "name":         name,
            "drift":        (float(np.mean(drifts)), float(np.std(drifts))),
            "gain":         (float(np.mean(gains)),  float(np.std(gains))),
            "params":       n_par,
            "_score_mean":  float(np.mean(scores)),
            "score":        (float(np.mean(scores)), float(np.std(scores))),
        })

    print_tournament(
        f"Architecture Pareto: gain/(1+50×drift) — mean±std over {n_trials} trials",
        rows,
        [("drift", "Drift(64d)", False), ("gain", "Gain(768d)", True),
         ("params", "Params", False),    ("score", "Pareto", False)],
        winner_col="_score_mean",
        spec_name="★SPEC",
        notes="Pareto penalises prefix drift 50×. Each trial uses different n_exp/noise."
    )


def test_4_projection_target_comparison(verbose: bool = False, quick: bool = False):
    """
    [R5] Query-only vs doc-only vs both vs query+centroid-update.
    Fix A+B: well-separated experts + centroid-neighbourhood docs → meaningful R64 changes.
    Fix C: per-trial n_exp/noise variation → non-zero std.
    """
    print("\n[Test 4] Projection target comparison — who benefits from projection?")

    N      = 400 if quick else 700
    n_trials = 2 if quick else N_TRIALS
    log_hyperparams(N=N, n_trials=n_trials)

    strategy_names = [
        "No adapter (baseline)",
        "Query-only ★SPEC",
        "Doc-only (incorrect)",
        "Both projected (incorrect)",
        "Query + centroid update (full spec)",
    ]
    strategy_r64:  Dict[str, List[float]] = {sn: [] for sn in strategy_names}
    strategy_r768: Dict[str, List[float]] = {sn: [] for sn in strategy_names}

    for trial in range(n_trials):
        with bootstrap_trial(trial * 41 + 3):
            n_exp = sample_n_experts(6, min(15, PREFIX_DIM)) if quick \
                    else sample_n_experts(8, min(20, PREFIX_DIM))
            noise = sample_noise(0.05, 0.15)
            manifold_s = make_manifold_data(600, true_rank=8, noise=noise*0.5, normalize=False)
            r = two_nn_rank(manifold_s)
            (experts, clean_q, shifted_q, docs, _, assignments) = \
                make_shifted_domain(N, n_exp, noise=noise)
            cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

            def _r64(q_emb, exp_emb):
                return routing_accuracy(q_emb, exp_emb, assignments)

            strategy_r64["No adapter (baseline)"].append(_r64(shifted_q, experts))
            strategy_r768["No adapter (baseline)"].append(recall_at_k(shifted_q, docs, k=20))

            adapter = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
            train_adapter(adapter, shifted_q, docs, cross_domain_negs=cross_negs)

            with torch.no_grad():
                q_proj = adapter(shifted_q)
                d_proj = adapter(docs)
                e_proj = adapter(experts)

            strategy_r64["Query-only ★SPEC"].append(_r64(q_proj, experts))
            strategy_r768["Query-only ★SPEC"].append(recall_at_k(q_proj, docs, k=20))

            strategy_r64["Doc-only (incorrect)"].append(_r64(shifted_q, experts))
            strategy_r768["Doc-only (incorrect)"].append(recall_at_k(shifted_q, d_proj, k=20))

            strategy_r64["Both projected (incorrect)"].append(_r64(q_proj, experts))
            strategy_r768["Both projected (incorrect)"].append(recall_at_k(q_proj, d_proj, k=20))

            strategy_r64["Query + centroid update (full spec)"].append(_r64(q_proj, e_proj))
            strategy_r768["Query + centroid update (full spec)"].append(
                recall_at_k(q_proj, docs, k=20))

    rows = []
    for sn in strategy_names:
        rows.append({
            "name":      sn,
            "r64":       (float(np.mean(strategy_r64[sn])),  float(np.std(strategy_r64[sn]))),
            "r768":      (float(np.mean(strategy_r768[sn])), float(np.std(strategy_r768[sn]))),
            "_r64_mean": float(np.mean(strategy_r64[sn])),
        })

    print_tournament(
        f"Routing accuracy vs projection target — mean±std over {n_trials} trials",
        rows,
        [("r64", "Acc@64d(GT)", True), ("r768", "R@20(768d)", True)],
        winner_col="_r64_mean",
        spec_name="★SPEC",
        notes="Acc@64d = ground-truth routing accuracy vs well-separated experts"
    )


def test_5_trigger_threshold_optimality(verbose: bool = False, quick: bool = False):
    """
    [R6] Is (threshold=0.70, window=3 days) optimal?

    Design: model daily routing accuracy as a random variable drawn from domain-shifted
    distributions. This gives genuine stochasticity that makes window size matter:

      Bad days:  accuracy ~ N(0.62, 0.07) — centered below threshold (0.70)
                 but with std=0.07, ~12% of bad days land above the threshold.
                 A window-1 trigger misses those days (FNR contribution).

      Good days: accuracy ~ N(0.76, 0.07) — centered above threshold (0.70)
                 but ~19% of good days land below the threshold.
                 A window-1 trigger fires on those days (FPR contribution).
                 A window-3 trigger requires 3 consecutive dips — rare for good days.

    This creates genuine trade-offs across the (threshold × window) grid:
      - Low threshold + wide window: low FPR but misses short bad spikes (high FNR)
      - High threshold + narrow window: catches all bad days but many false alarms (high FPR)
      - Spec (0.70, 3): balanced — window-3 suppresses random dips, catches sustained shifts.

    We generate N_REPS random days per run to average out per-run variance.
    Using N_REPS=1 (one season) matches the real use case but may occasionally favour/disfavour
    spec. We use N_DAYS=60 and BAD_DAYS=20 for enough statistical power to show real differences.
    """
    print("\n[Test 5] Trigger threshold optimality — sweeping (threshold × window) grid")

    N_DAYS   = 60
    BAD_DAYS = 20
    # Accuracy distributions: see module docstring for calibration rationale
    BAD_MEAN   = 0.62
    GOOD_MEAN  = 0.76
    ACC_STD    = 0.07
    N_SEASONS  = 5     # average over N_SEASONS independent 60-day seasons

    noise = sample_noise(0.05, 0.2)  # kept for log consistency
    n_exp = min(12, PREFIX_DIM)
    log_hyperparams(noise=f"{noise:.3f}", n_exp=n_exp, N_DAYS=N_DAYS, BAD_DAYS=BAD_DAYS,
                    bad_acc_dist=f"N({BAD_MEAN},{ACC_STD})", good_acc_dist=f"N({GOOD_MEAN},{ACC_STD})")

    # Verify the distributions create meaningful trigger trade-offs
    from scipy import stats as sps  # use scipy for CDF if available, else manual
    try:
        import scipy.stats
        p_bad_over_thresh  = 1 - scipy.stats.norm.cdf(COHERENCE_THRESHOLD, BAD_MEAN, ACC_STD)
        p_good_under_thresh = scipy.stats.norm.cdf(COHERENCE_THRESHOLD, GOOD_MEAN, ACC_STD)
    except ImportError:
        # Approximate with standard normal
        def _ncdf(x, mu, s): return 0.5 * (1 + math.erf((x - mu) / (s * math.sqrt(2))))
        p_bad_over_thresh  = 1 - _ncdf(COHERENCE_THRESHOLD, BAD_MEAN, ACC_STD)
        p_good_under_thresh = _ncdf(COHERENCE_THRESHOLD, GOOD_MEAN, ACC_STD)

    print(f"  Bad day:  accuracy ~ N({BAD_MEAN}, {ACC_STD})  "
          f"→ P(acc > {COHERENCE_THRESHOLD}) = {p_bad_over_thresh:.1%} (escape rate)")
    print(f"  Good day: accuracy ~ N({GOOD_MEAN}, {ACC_STD})  "
          f"→ P(acc < {COHERENCE_THRESHOLD}) = {p_good_under_thresh:.1%} (false dip rate)")
    print(f"  Window-3 reduces false alarm rate ≈ ({p_good_under_thresh:.1%})³ = "
          f"{p_good_under_thresh**3:.2%} per good window — this is why window > 1 helps.")

    rng_t5 = np.random.default_rng(seed=17)   # fixed seed for reproducibility across runs

    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80] if not quick else [0.60, 0.65, 0.70, 0.75]
    windows    = [1, 2, 3, 5, 7]              if not quick else [1, 3, 5]

    rows = []
    for thresh in thresholds:
        for win in windows:
            fpr_sum = fnr_sum = 0.0
            for season in range(N_SEASONS):
                daily = (list(rng_t5.normal(BAD_MEAN,  ACC_STD, BAD_DAYS).clip(0, 1)) +
                         list(rng_t5.normal(GOOD_MEAN, ACC_STD, N_DAYS - BAD_DAYS).clip(0, 1)))
                triggers = []
                for day in range(N_DAYS):
                    ws = daily[max(0, day - win + 1): day + 1]
                    fired = (len(ws) == win and all(s < thresh for s in ws))
                    triggers.append(fired)
                fpr_sum += sum(triggers[BAD_DAYS:]) / (N_DAYS - BAD_DAYS)
                fnr_sum += (0.0 if any(triggers[:BAD_DAYS]) else 1.0)
            fpr     = fpr_sum / N_SEASONS
            fnr     = fnr_sum / N_SEASONS
            combined = fpr + fnr
            spec     = (thresh == 0.70 and win == 3)
            rows.append({"name": f"t={thresh:.2f} w={win}" + (" ★SPEC" if spec else ""),
                         "fpr": fpr, "fnr": fnr,
                         "combined": combined,
                         "score": 1.0 - combined})

    print_tournament(
        "FPR + FNR across (threshold, window) — mean over 5 seasons, lower combined = better",
        rows[:15],
        [("fpr", "FPR", True), ("fnr", "FNR", True), ("combined", "FPR+FNR", True)],
        winner_col="score",
        spec_name="★SPEC",
        notes=f"Stochastic daily accuracy: bad~N({BAD_MEAN},{ACC_STD}), good~N({GOOD_MEAN},{ACC_STD})"
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
    [R7] Which training data strategy produces best routing accuracy recovery?
    Fix A+B: meaningful adapter learning with well-separated experts + centroid docs.
    Fix C: per-trial n_exp/noise variation.
    """
    print("\n[Test 6] Training data composition tournament — sample efficiency")

    N      = 500 if quick else 900
    n_trials = 2 if quick else N_TRIALS
    log_hyperparams(N=N, n_trials=n_trials)

    ns = [50, 100, 200, 400] if not quick else [50, 100, 200]

    strategy_aucs: Dict[str, List[float]] = {
        "Random triplets":   [],
        "Error pairs ★SPEC": [],
        "Cross-domain negs": [],
        "Hard negatives":    [],
    }

    for trial in range(n_trials):
        with bootstrap_trial(trial * 53 + 7):
            n_exp = sample_n_experts(6, min(15, PREFIX_DIM)) if quick \
                    else sample_n_experts(8, min(20, PREFIX_DIM))
            noise = sample_noise(0.05, 0.15)
            experts, clean_q, shifted_q, docs, _, assignments = \
                make_shifted_domain(N, n_exp, noise=noise)

            manifold_s = make_manifold_data(600, true_rank=8, noise=noise*0.5, normalize=False)
            r = two_nn_rank(manifold_s)

            random_negs = torch.roll(docs, shifts=7, dims=0)
            hard_negs   = make_hard_negatives(shifted_q, docs)
            cross_negs  = clean_q[torch.randperm(N, device=DEVICE)]

            baseline_acc = routing_accuracy(shifted_q, experts, assignments)

            strategies = [
                ("Random triplets",      shifted_q, docs, random_negs),
                ("Error pairs ★SPEC",    shifted_q, docs, None),
                ("Cross-domain negs",    shifted_q, docs, cross_negs),
                ("Hard negatives",       shifted_q, docs, hard_negs),
            ]

            for strategy_name, sq, pd, explicit_neg in strategies:
                coh_by_n = []
                for n_train in ns:
                    idx = torch.randperm(N, device=DEVICE)[:n_train]
                    adapter = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
                    neg_batch = explicit_neg[idx] if explicit_neg is not None else None
                    train_adapter(adapter, sq[idx], pd[idx], cross_domain_negs=neg_batch)
                    with torch.no_grad():
                        acc = routing_accuracy_adapted(shifted_q, experts, assignments, adapter)
                    coh_by_n.append(acc)
                auc = sum(coh_by_n) / len(ns)
                strategy_aucs[strategy_name].append(auc)

    rows = []
    for sn in strategy_aucs:
        aucs = strategy_aucs[sn]
        rows.append({
            "name":     sn,
            "auc":      (float(np.mean(aucs)), float(np.std(aucs))),
            "_auc_mean": float(np.mean(aucs)),
        })

    print_tournament(
        f"Routing accuracy recovery AUC — mean±std over {n_trials} trials",
        rows,
        [("auc", "AUC mean±std", False)],
        winner_col="_auc_mean",
        spec_name="★SPEC",
        notes="AUC = mean accuracy over N=[50,100,200,400]. Each trial different n_exp/noise."
    )


def test_7_centroid_update_scope(verbose: bool = False, quick: bool = False):
    """[R8] Affected-only vs full rebuild vs no update."""
    print("\n[Test 7] Centroid update scope + accumulation over sequential rounds")

    N_DOMAINS = min(5, PREFIX_DIM)
    N_PER     = 10
    N         = 300 if quick else 500
    n_rnd     = 3   if quick else 5
    noise     = sample_noise(0.05, 0.14)
    n_exp     = N_DOMAINS * N_PER
    log_hyperparams(noise=f"{noise:.3f}", N=N, N_DOMAINS=N_DOMAINS, n_exp=n_exp)

    experts, clean_q, shifted_q, docs, _, assignments = \
        make_shifted_domain(N, N_DOMAINS, noise=noise)  # N_DOMAINS well-sep experts
    # Expand to N_PER centroids per expert by adding small noise
    all_cents = []
    for d in range(N_DOMAINS):
        for _ in range(N_PER):
            all_cents.append(F.normalize(
                experts[d] + torch.randn(EMBED_DIM, device=DEVICE) * 0.01, p=2, dim=-1))
    faiss_orig = torch.stack(all_cents)    # (N_DOMAINS * N_PER, EMBED_DIM)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

    manifold_7 = make_manifold_data(600, true_rank=8, noise=noise*0.5, normalize=False)
    r = two_nn_rank(manifold_7)

    def end_to_end_acc(q_emb, cent_emb):
        """Domain-level routing accuracy: predicted centroid's domain must match true domain."""
        qp = F.normalize(q_emb[:, :PREFIX_DIM], p=2, dim=-1)
        ep = F.normalize(cent_emb[:, :PREFIX_DIM], p=2, dim=-1)
        pred_centroid = (qp @ ep.T).argmax(dim=1)       # index into cent_emb rows
        pred_domain   = pred_centroid // N_PER           # which domain block
        return (pred_domain == assignments).float().mean().item()

    rows = []
    rows.append({"name": "No centroid update",
                 "r_s1": end_to_end_acc(shifted_q, faiss_orig),
                 "non_affected_changed": False,
                 "update_cost": 0.0})

    adapter_full = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    train_adapter(adapter_full, shifted_q, docs, cross_domain_negs=cross_negs)
    with torch.no_grad():
        full_rebuilt = adapter_full(faiss_orig)
    rows.append({"name": "Full index rebuild",
                 "r_s1": end_to_end_acc(adapter_full(shifted_q), full_rebuilt),
                 "non_affected_changed": True,
                 "update_cost": 1.0})

    affected = 2
    faiss_partial = faiss_orig.clone()
    adapter_part  = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
    train_adapter(adapter_part, shifted_q, docs, cross_domain_negs=cross_negs)
    s, e = affected * N_PER, (affected + 1) * N_PER
    with torch.no_grad():
        faiss_partial[s:e] = adapter_part(faiss_orig[s:e])
    unchanged = (torch.allclose(faiss_partial[:s], faiss_orig[:s]) and
                 torch.allclose(faiss_partial[e:], faiss_orig[e:]))
    rows.append({"name": "Affected-only ★SPEC",
                 "r_s1": end_to_end_acc(adapter_part(shifted_q), faiss_partial),
                 "non_affected_changed": not unchanged,
                 "update_cost": N_PER / (N_DOMAINS * N_PER)})

    print_tournament(
        "End-to-end GT routing accuracy under different centroid update scopes",
        rows,
        [("r_s1", "Acc@64d(GT)", True), ("update_cost", "Update cost", False)],
        winner_col="r_s1", spec_name="★SPEC",
        notes="Acc@64d = ground-truth routing accuracy (well-separated experts)"
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
            r_val = end_to_end_acc(a(shifted_q), running_idx)
        print(f"    Round {rnd+1}: GT routing accuracy = {r_val*100:.1f}%")


def test_8_adapter_selection_and_ood(verbose: bool = False, quick: bool = False):
    """
    [R9] Adapter selection diagnostic: nearest-centroid (spec) vs threshold-gated.

    The spec uses nearest centroid with no OOD guard — this is intentionally simple.
    This test asks: what is the COST of that simplicity? And what τ minimizes it?

    Scoring:
      Composite = in_dist_accuracy − 0.5 × ood_apply_rate
      (penalizes applying an adapter to OOD queries — wrong domain = wrong correction)

    The spec (τ=0) will score lower because it applies adapters to OOD queries.
    The test identifies the cheapest fix. This is presented as a REFINEMENT
    opportunity, not a spec failure — the spec is aware of this trade-off and
    chose simplicity for the initial version.

    Key insight: with well-separated experts, in-distribution accuracy is always
    100% regardless of τ (until τ is so high it rejects in-distribution queries).
    The only variable is OOD rate. We measure the break-even τ where in-dist
    accuracy first degrades — that's the safe operating range for the threshold.
    """
    print("\n[Test 8] Adapter selection method + OOD handling")

    N_ADAPTERS = sample_n_experts(5, min(12, PREFIX_DIM)) if not quick \
                 else sample_n_experts(4, min(8, PREFIX_DIM))
    N_TEST     = 200 if not quick else 100
    log_hyperparams(N_ADAPTERS=N_ADAPTERS, N_TEST=N_TEST)

    domain_cents_mat = make_well_separated_experts(N_ADAPTERS)
    domain_cents     = [domain_cents_mat[i] for i in range(N_ADAPTERS)]

    def make_registry(threshold=0.0):
        reg = AdapterRegistry(sim_threshold=threshold)
        for i, c in enumerate(domain_cents):
            a = LowRankProjectionAdapter(EMBED_DIM, rank=RANK_DEFAULT).to(DEVICE)
            reg.register(a, c, f"domain_{i}")
        return reg

    def in_dist_accuracy(reg, queries, true_domains):
        return sum(1 for q, td in zip(queries, true_domains)
                   if reg.select(q)[1] == f"domain_{td}") / len(queries)

    def ood_rate(reg, ood_queries):
        return sum(1 for q in ood_queries if reg.select(q)[0] is not None) / len(ood_queries)

    # In-distribution: queries very close to their known domain centroid
    in_dist_q   = [F.normalize(domain_cents[i % N_ADAPTERS]
                                + torch.randn_like(domain_cents[0]) * 0.05, p=2, dim=-1)
                   for i in range(N_TEST)]
    in_dist_dom = [i % N_ADAPTERS for i in range(N_TEST)]

    # OOD: uniformly random unit vectors — no domain structure
    n_ood = 100
    ood_q = [F.normalize(torch.randn(EMBED_DIM, device=DEVICE), p=2, dim=-1)
             for _ in range(n_ood)]

    # Compute cosine sim between in-dist queries and their correct centroid
    # to understand what τ range is safe
    in_dist_sims = []
    for q, td in zip(in_dist_q, in_dist_dom):
        sim = (F.normalize(q, p=2, dim=-1) @ F.normalize(domain_cents[td], p=2, dim=-1)).item()
        in_dist_sims.append(sim)
    ood_sims = [(F.normalize(q, p=2, dim=-1) @
                 F.normalize(domain_cents_mat, p=2, dim=-1).T).max().item()
                for q in ood_q]

    in_dist_sim_min  = min(in_dist_sims)
    ood_sim_max      = max(ood_sims)
    print(f"  In-dist query→correct centroid sim: min={in_dist_sim_min:.3f}  "
          f"mean={sum(in_dist_sims)/len(in_dist_sims):.3f}")
    print(f"  OOD query→nearest centroid sim:     max={ood_sim_max:.3f}  "
          f"mean={sum(ood_sims)/len(ood_sims):.3f}")
    print(f"  Safe τ range: ({ood_sim_max:.3f}, {in_dist_sim_min:.3f}) "
          f"— thresholds here get 0% OOD and 100% in-dist")

    tau_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    for tau in tau_values:
        reg    = make_registry(threshold=tau)
        in_acc = in_dist_accuracy(reg, in_dist_q, in_dist_dom)
        ood_r  = ood_rate(reg, ood_q)
        score  = in_acc - 0.5 * ood_r
        spec_tag = " ★SPEC" if tau == 0.0 else ""
        rows.append({"name": f"τ={tau:.2f}{spec_tag}",
                     "in_acc": in_acc, "ood_applied": ood_r, "score": score, "tau": tau})

    print_tournament(
        "Adapter selection: in-distribution accuracy vs OOD application rate",
        rows,
        [("in_acc", "In-dist acc", True), ("ood_applied", "OOD applied", True),
         ("score", "Composite", False)],
        winner_col="score", spec_name="★SPEC",
        notes="Score = in_acc − 0.5×ood_applied. τ=0 is spec baseline."
    )

    # Identify the OOD-safe threshold range and smallest τ that achieves it
    spec_score   = next(r["score"] for r in rows if "★SPEC" in r["name"])
    ood_free     = [r for r in rows if r["ood_applied"] < 0.01 and r["in_acc"] > 0.98]
    if ood_free:
        best_τ = min(ood_free, key=lambda r: r["tau"])["tau"]
        gain   = next(r["score"] for r in rows if r["tau"] == best_τ) - spec_score
        print(f"  → Spec refinement: τ={best_τ:.2f} eliminates OOD while preserving in-dist. "
              f"Composite score gain: {gain:.3f}.")
        print(f"    Recommend adding to Section 1.3: 'apply adapter only if "
              f"max_sim(q, centroids) > τ ≈ {best_τ:.2f}'")
    else:
        print(f"  → No clean OOD-free operating point found "
              f"(in-dist and OOD sims overlap). Spec nearest-centroid is best available.")

    # Cap enforcement
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
    except AssertionError:
        print(f"  Cap enforcement ({MAX_ADAPTERS} limit): ✓ PASS")


def test_9_rank_determination_tournament(verbose: bool = False, quick: bool = False):
    """
    [R10] Two-NN vs PCA vs fixed ranks.
    Fix F: TWO_NN_SCALE=1.0 → Two-NN rank estimate closer to true rank.
    Fix C: per-trial variation → non-zero std.
    """
    print("\n[Test 9] Rank determination tournament — Two-NN vs PCA vs fixed ranks")

    TRUE_RANK = 16
    N         = 500 if quick else 800
    n_trials  = 2 if quick else N_TRIALS
    log_hyperparams(TRUE_RANK=TRUE_RANK, N=N, n_trials=n_trials)

    # Use large manifold for rank estimation (TWO_NN_SCALE=1.0 now)
    docs_manifold = make_manifold_data(1000, true_rank=TRUE_RANK, noise=0.04, normalize=False)
    r_twonn = two_nn_rank(docs_manifold)
    r_pca   = min(RANK_MAX, max(RANK_MIN, pca_intrinsic_dim(docs_manifold)))
    print(f"  Ground-truth manifold rank: {TRUE_RANK}")
    print(f"  Two-NN estimated rank: {r_twonn}  |  PCA estimated rank: {r_pca}")
    print(f"  Two-NN error: {abs(r_twonn-TRUE_RANK)}  |  PCA error: {abs(r_pca-TRUE_RANK)}")

    rank_paretos: Dict[str, List[float]] = {}
    rank_configs = [(4, "Fixed r=4"), (8, "Fixed r=8"), (16, "Fixed r=16"),
                    (32, "Fixed r=32"), (64, "Fixed r=64"),
                    (r_pca,   f"PCA r={r_pca}"),
                    (r_twonn, f"Two-NN r={r_twonn} ★SPEC")]
    for _, label in rank_configs:
        rank_paretos[label] = []

    for trial in range(n_trials):
        with bootstrap_trial(trial * 61 + 11):
            n_exp = sample_n_experts(8, min(20, PREFIX_DIM))
            noise = sample_noise(0.04, 0.12)
            (experts, _, shifted_q, docs, _, assignments) = \
                make_shifted_domain(N, n_exp, noise=noise)
            cross_negs = shifted_q[torch.randperm(N, device=DEVICE)]  # shifted queries as negs
            with torch.no_grad():
                base_sims = F.cosine_similarity(shifted_q[:, :PREFIX_DIM],
                                                 docs[:, :PREFIX_DIM])
                base_r768 = recall_at_k(shifted_q, docs, k=20)

            for rank, label in rank_configs:
                adapter = LowRankProjectionAdapter(EMBED_DIM, rank=rank).to(DEVICE)
                train_adapter(adapter, shifted_q, docs, cross_domain_negs=cross_negs)
                with torch.no_grad():
                    drift  = prefix_drift(adapter, shifted_q, docs, base_sims)
                    gain   = recall_at_k(adapter(shifted_q), docs, k=20) - base_r768
                rank_paretos[label].append(pareto_score(gain, drift))

    rows = []
    for rank, label in rank_configs:
        ps = rank_paretos[label]
        rows.append({
            "name":    label,
            "rank":    float(rank),
            "pareto":  (float(np.mean(ps)), float(np.std(ps))),
            "_p_mean": float(np.mean(ps)),
            "params":  EMBED_DIM * rank * 2,
        })

    print_tournament(
        f"Pareto score at each rank — mean±std over {n_trials} trials",
        rows,
        [("rank", "Rank", False), ("pareto", "Pareto mean±std", False),
         ("params", "Params", False)],
        winner_col="_p_mean",
        spec_name="★SPEC"
    )
    spec_r  = next(r for r in rows if "★SPEC" in r["name"])
    best_pm = max(r["_p_mean"] for r in rows)
    print(f"  Two-NN rank within 15% of best Pareto: "
          f"{'✓ PASS' if spec_r['_p_mean'] >= best_pm * 0.85 else '✗ FAIL'}")
    print(f"  PCA rank ({r_pca}) >= Two-NN rank ({r_twonn}): "
          f"{'✓ CONFIRMED' if r_twonn <= r_pca else '✗ NOT CONFIRMED'}")


def test_10_prefix_integrity_and_accumulation(verbose: bool = False, quick: bool = False):
    """[R11] Prefix integrity over sequential adaptation rounds."""
    print("\n[Test 10] Prefix integrity + accumulation across sequential adaptation rounds")

    N       = 400 if quick else 600
    N_RND   = 3   if quick else 5
    TRUE_RANK = 12
    noise   = sample_noise(0.04, 0.12)
    n_exp   = sample_n_experts(6, min(15, PREFIX_DIM))
    log_hyperparams(noise=f"{noise:.3f}", n_exp=n_exp, TRUE_RANK=TRUE_RANK)

    docs_manifold = make_manifold_data(800, true_rank=TRUE_RANK, noise=noise, normalize=False)
    r_twonn = two_nn_rank(docs_manifold)
    print(f"  Two-NN rank (from rank-{TRUE_RANK} manifold): {r_twonn}")

    experts, clean_q, shifted_q, docs, _, assignments = \
        make_shifted_domain(N, n_exp, noise=noise)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

    with torch.no_grad():
        base_sims = F.cosine_similarity(shifted_q[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
        base_r    = recall_at_k(shifted_q, docs, k=20)

    archs = [
        ("Dense (★SPEC naive)",        DenseProjectionAdapter,    {"dim": EMBED_DIM}),
        ("Block-Diagonal",              BlockDiagonalAdapter,      {"dim": EMBED_DIM}),
        (f"LowRank r={r_twonn} ★SPEC", LowRankProjectionAdapter,  {"dim": EMBED_DIM, "rank": r_twonn}),
    ]

    print(f"\n  {'Architecture':<25} {'Rnd':>4}  {'Drift':>8}  {'Gain':>8}  {'Pareto':>8}")
    print(f"  {'─'*60}")

    for arch_name, cls, kwargs in archs:
        q_current = shifted_q.clone()
        for rnd in range(1, N_RND + 1):
            adapter = cls(**kwargs).to(DEVICE)
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

    print("  Interpretation: increasing drift across rounds → unsafe for sequential deployment.")


def test_11_streaming_centroid_tournament(verbose: bool = False, quick: bool = False):
    """[R12] EMA vs running mean vs fixed window. Fix C 3-phase scenario preserved."""
    print("\n[Test 11] Streaming centroid method tournament — accuracy vs shift response")

    N_WARM  = 1000 if not quick else 400
    N_SHIFT = 500  if not quick else 200
    N_DRIFT = 300  if not quick else 150
    BATCH   = 50

    phase1 = F.normalize(torch.randn(N_WARM,  EMBED_DIM, device=DEVICE), p=2, dim=-1)
    phase2 = F.normalize(torch.randn(N_SHIFT, EMBED_DIM, device=DEVICE) + 2.0, p=2, dim=-1)

    drift_batches = []
    drift_vec = F.normalize(torch.randn(EMBED_DIM, device=DEVICE), p=2, dim=-1)
    for step in range(0, N_DRIFT, BATCH):
        mag = 2.0 + (step / N_DRIFT) * 1.5
        b   = F.normalize(torch.randn(BATCH, EMBED_DIM, device=DEVICE)
                          + drift_vec * mag, p=2, dim=-1)
        drift_batches.append(b)
    drift_target = drift_batches[-1].mean(0)

    true_mean1 = phase1.mean(0)
    true_mean2 = phase2.mean(0)

    def track(strategy_fn) -> Tuple[float, float, float]:
        state = strategy_fn()
        for i in range(0, N_WARM, BATCH):
            state["update"](phase1[i: i + BATCH])
        conv_sim = F.cosine_similarity(state["get"]().unsqueeze(0),
                                        true_mean1.unsqueeze(0)).item()
        for i in range(0, N_SHIFT, BATCH):
            state["update"](phase2[i: i + BATCH])
        shift_sim = F.cosine_similarity(state["get"]().unsqueeze(0),
                                         true_mean2.unsqueeze(0)).item()
        for b in drift_batches:
            state["update"](b)
        drift_sim = F.cosine_similarity(state["get"]().unsqueeze(0),
                                         drift_target.unsqueeze(0)).item()
        return conv_sim, shift_sim, drift_sim

    rows = []

    def make_running_mean():
        s = {"n": 0, "mean": torch.zeros(EMBED_DIM, device=DEVICE)}
        def upd(b):
            s["mean"] = (s["n"] * s["mean"] + b.sum(0)) / (s["n"] + len(b))
            s["n"] += len(b)
        return {"update": upd, "get": lambda: s["mean"]}
    c, sh, dr = track(make_running_mean)
    rows.append({"name": "True running mean", "conv": c, "shift": sh, "drift": dr,
                 "score": 0.4 * c + 0.3 * sh + 0.3 * dr})

    from collections import deque
    def make_fixed_window(w=100):
        buf = deque(maxlen=w)
        def upd(b):
            for row in b:
                buf.append(row)
        def get():
            return torch.stack(list(buf)).mean(0) if buf else \
                   torch.zeros(EMBED_DIM, device=DEVICE)
        return {"update": upd, "get": get}
    c, sh, dr = track(make_fixed_window)
    rows.append({"name": "Fixed window (100)", "conv": c, "shift": sh, "drift": dr,
                 "score": 0.4 * c + 0.3 * sh + 0.3 * dr})

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
        c, sh, dr = track(make_ema)
        spec_tag = " ★SPEC" if abs(alpha - STREAMING_ALPHA) < 1e-6 else ""
        rows.append({"name": f"EMA α={alpha:.2f}{spec_tag}",
                     "conv": c, "shift": sh, "drift": dr,
                     "score": 0.4 * c + 0.3 * sh + 0.3 * dr})

    print_tournament(
        "Centroid tracking: convergence (40%) + step shift (30%) + drift (30%)",
        rows,
        [("conv", "Convergence", False), ("shift", "Step shift", False),
         ("drift", "Grad. drift", False), ("score", "Score", False)],
        winner_col="score", spec_name="★SPEC",
        notes="Score = 0.4×convergence + 0.3×step_shift + 0.3×gradual_drift"
    )
    print("  Note: True running mean cannot track gradual drift — Phase 3 reveals this.")


def test_12_adapter_boundary_interference(verbose: bool = False, quick: bool = False):
    """
    [NEW] How do different adapter selection strategies handle boundary queries?

    Boundary queries sit near the midpoint between two domain centroids — both
    domains are equally valid answers. This tests:
      1. In-distribution accuracy: does the strategy correctly route clear in-dist
         queries to their known domain?
      2. Boundary valid-domain coverage: for boundary queries between domains 0 and 1,
         does the strategy route them to EITHER domain 0 OR domain 1? (Both are valid.)
         Strategies that suppress routing (high τ) score 0% here — they fail to route.
         Strategies that route to a completely wrong domain (domain 2,3,...) also score 0%.

    Score = 0.6 × in_dist_accuracy + 0.4 × valid_domain_coverage
    (In-dist accuracy weighted higher — boundary handling is secondary.)

    This correctly penalizes τ=0.5 which achieves "consistency" by suppressing
    all boundary routing — consistent, but useless for the downstream retrieval task.

    Fix 5: negatives tiled to exactly N rows.
    """
    print("\n[Test 12] Adapter boundary interference — routing validity under ambiguity")

    N     = 200 if not quick else 100
    noise = sample_noise(0.04, 0.10)
    n_dom = sample_n_experts(3, min(6, PREFIX_DIM))
    log_hyperparams(noise=f"{noise:.3f}", N=N, n_domains=n_dom)

    domain_cents_mat = make_well_separated_experts(n_dom)
    domain_cents     = [domain_cents_mat[i] for i in range(n_dom)]

    adapters = []
    manifold_r_data = make_manifold_data(600, true_rank=8, noise=0.05, normalize=False)
    r = max(RANK_MIN, min(RANK_MAX, round(two_nn_intrinsic_dim(manifold_r_data) * TWO_NN_SCALE)))

    for d in range(n_dom):
        exp_d, clean_d, sq_d, doc_d, _, _ = make_shifted_domain(N, min(5, PREFIX_DIM),
                                                                   noise=noise)
        # Fix 5: build exactly N other-domain negative rows by tiling
        other_parts = []
        for k in range(1, n_dom):
            base = domain_cents[(d + k) % n_dom].unsqueeze(0).expand(N, -1)
            part = F.normalize(base + torch.randn(N, EMBED_DIM, device=DEVICE) * noise,
                               p=2, dim=-1)
            other_parts.append(part)
        other_queries = torch.cat(other_parts, dim=0)
        perm = torch.randperm(len(other_queries), device=DEVICE)[:N]
        neg  = other_queries[perm]

        adapt = LowRankProjectionAdapter(EMBED_DIM, rank=r).to(DEVICE)
        train_adapter(adapt, sq_d, doc_d, cross_domain_negs=neg)
        adapters.append((adapt, domain_cents[d]))

    def select_with_threshold(q, tau=0.0):
        """Returns (domain_idx or None, sim)."""
        sims = torch.stack([F.normalize(q, p=2, dim=-1) @ c for _, c in adapters])
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()
        if best_sim < tau:
            return None, best_sim
        return best_idx, best_sim

    # In-distribution queries: unambiguously close to their domain
    in_dist_q    = []
    in_dist_true = []
    for i in range(N):
        d = i % n_dom
        q = F.normalize(domain_cents[d] + torch.randn(EMBED_DIM, device=DEVICE) * 0.02,
                         p=2, dim=-1)
        in_dist_q.append(q)
        in_dist_true.append(d)
    in_dist_q = torch.stack(in_dist_q)

    # Boundary queries: midpoint between domain 0 and domain 1 (both valid)
    boundary_q = F.normalize(
        domain_cents[0] + domain_cents[1]
        + torch.randn(N, EMBED_DIM, device=DEVICE) * noise,
        p=2, dim=-1)
    valid_boundary_domains = {0, 1}

    print(f"  Boundary queries: midpoint of domains 0 and 1 → both are valid answers")
    print(f"  Key: strategies that SUPPRESS boundary routing score 0% valid-coverage")
    print(f"  (consistent-but-absent is worse than making a valid choice)")

    # τ values chosen to straddle the expected boundary sim value.
    # For orthonormal experts, the midpoint query has sim ≈ 1/sqrt(2) ≈ 0.707 to each
    # of the two adjacent experts. Choosing τ ∈ {0.0, 0.60, 0.80}:
    #   τ=0.0:  always routes → 100% valid coverage for boundary queries
    #   τ=0.60: still below 0.707 → still routes → still 100% valid coverage
    #   τ=0.80: above 0.707 → suppresses boundary queries → 0% valid coverage
    # This shows the meaningful break-point at the midpoint sim value.
    tau_values_12 = [0.0, 0.60, 0.80]

    rows = []
    for tau in tau_values_12:
        # In-dist accuracy
        correct_in = sum(
            1 for q, td in zip(in_dist_q, in_dist_true)
            if select_with_threshold(q, tau)[0] == td
        )
        in_acc = correct_in / N

        # Valid-domain coverage: boundary queries routed to domain 0 or 1
        valid_cnt = sum(
            1 for q in boundary_q
            if select_with_threshold(q, tau)[0] in valid_boundary_domains
        )
        valid_cov = valid_cnt / N

        # Also track: suppressed (None), wrong domain (not 0 or 1)
        suppressed = sum(1 for q in boundary_q if select_with_threshold(q, tau)[0] is None)
        wrong_dom  = N - valid_cnt - suppressed

        score   = 0.6 * in_acc + 0.4 * valid_cov
        spec_tag = " ★SPEC" if tau == 0.0 else ""
        print(f"  τ={tau:.2f}{spec_tag}: in-dist={in_acc*100:.1f}%  "
              f"boundary→valid={valid_cov*100:.1f}%  "
              f"suppressed={suppressed/N*100:.1f}%  wrong={wrong_dom/N*100:.1f}%")
        rows.append({"name": f"τ={tau:.2f}{spec_tag}",
                     "in_acc": in_acc, "valid_cov": valid_cov,
                     "suppressed": suppressed / N, "score": score})

    print_tournament(
        f"Boundary routing validity (domains 0 & 1 both correct) — n_domains={n_dom}, r={r}",
        rows,
        [("in_acc", "In-dist acc", True), ("valid_cov", "Valid coverage", True),
         ("suppressed", "Suppressed %", True)],
        winner_col="score", spec_name="★SPEC",
        notes="Score = 0.6×in_acc + 0.4×valid_coverage. Suppression ≠ correct routing."
    )
    print(f"  FIX 5: other-domain negatives tiled to exactly N={N} rows.")
    print("  Insight: nearest-centroid routes boundary queries to the geometrically "
          "closer valid domain — threshold rejection provides no benefit here.")


def test_13_sample_efficiency_curves(verbose: bool = False, quick: bool = False):
    """[NEW] N to 90% asymptotic routing accuracy recovery."""
    print("\n[Test 13] Sample efficiency — N needed to reach 90% asymptotic routing accuracy")

    N_MAX = 800 if not quick else 400
    n_exp = sample_n_experts(8, min(20, PREFIX_DIM))
    noise = sample_noise(0.05, 0.14)
    ns    = [25, 50, 100, 200, 400, N_MAX] if not quick else [25, 50, 100, 200]
    log_hyperparams(noise=f"{noise:.3f}", n_exp=n_exp, N_MAX=N_MAX)

    experts, clean_q, shifted_q, docs, _, assignments = \
        make_shifted_domain(N_MAX, n_exp, noise=noise)
    cross_negs = clean_q[torch.randperm(N_MAX, device=DEVICE)]

    r = two_nn_rank(make_manifold_data(800, true_rank=10, noise=noise*0.5, normalize=False))
    baseline_acc = routing_accuracy(shifted_q, experts, assignments)

    def _train_eval(cls, kwargs, n_train):
        a = cls(**kwargs).to(DEVICE)
        idx = torch.randperm(N_MAX, device=DEVICE)[:n_train]
        cn  = cross_negs[idx]
        ep  = EPOCHS + max(0, (200 - n_train) // 50)
        train_adapter(a, shifted_q[idx], docs[idx], cross_domain_negs=cn, epochs=ep)
        with torch.no_grad():
            return routing_accuracy_adapted(shifted_q, experts, assignments, a)

    archs = [
        ("Diagonal",             DiagonalAdapter,           {"dim": EMBED_DIM}),
        ("Block-Diagonal",       BlockDiagonalAdapter,      {"dim": EMBED_DIM}),
        (f"LowRank r={r} ★SPEC", LowRankProjectionAdapter, {"dim": EMBED_DIM, "rank": r}),
        ("Dense",                DenseProjectionAdapter,    {"dim": EMBED_DIM}),
    ]

    print(f"\n  Baseline routing accuracy (no adapter): {baseline_acc:.3f}")
    print(f"\n  {'Architecture':<25} " + "  ".join(f"N={n:>4}" for n in ns)
          + "  Asym    90% at N")
    print(f"  {'─'*90}")

    for name, cls, kwargs in archs:
        asym   = _train_eval(cls, kwargs, N_MAX)
        improvement = asym - baseline_acc
        target = baseline_acc + 0.90 * max(improvement, 0.01)
        n90    = None
        row    = []
        for n in ns:
            acc = _train_eval(cls, kwargs, n)
            row.append(acc)
            if n90 is None and acc >= target:
                n90 = n
        n90_str = str(n90) if n90 else ">max"
        if improvement < 0.05:
            n90_str = "N/A(<5% gain)"
        print(f"  {name:<25} " + "  ".join(f"{v*100:>6.1f}%" for v in row)
              + f"  {asym*100:>5.1f}%  {n90_str:>12}")


def test_14_recovery_speed_under_shift(verbose: bool = False, quick: bool = False):
    """[NEW] Routing accuracy recovery speed per epoch per architecture."""
    print("\n[Test 14] Recovery speed — routing accuracy vs training steps per architecture")

    N     = 400 if not quick else 200
    n_exp = sample_n_experts(8, min(20, PREFIX_DIM))
    noise = sample_noise(0.05, 0.14)
    log_hyperparams(noise=f"{noise:.3f}", n_exp=n_exp, N=N)

    experts, clean_q, shifted_q, docs, _, assignments = \
        make_shifted_domain(N, n_exp, noise=noise)
    cross_negs = clean_q[torch.randperm(N, device=DEVICE)]

    r = two_nn_rank(make_manifold_data(600, true_rank=8, noise=noise*0.5, normalize=False))

    def accuracy_curve(cls, kwargs):
        a   = cls(**kwargs).to(DEVICE)
        opt = torch.optim.AdamW(a.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        n   = len(shifted_q)
        with torch.no_grad():
            curve = [routing_accuracy(shifted_q, experts, assignments)]
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
                curve.append(routing_accuracy_adapted(shifted_q, experts, assignments, a))
            a.train()
        return curve

    archs = [
        ("Diagonal",             DiagonalAdapter,           {"dim": EMBED_DIM}),
        ("Block-Diagonal",       BlockDiagonalAdapter,      {"dim": EMBED_DIM}),
        (f"LowRank r={r} ★SPEC", LowRankProjectionAdapter, {"dim": EMBED_DIM, "rank": r}),
        ("Dense",                DenseProjectionAdapter,    {"dim": EMBED_DIM}),
    ]

    baseline = routing_accuracy(shifted_q, experts, assignments)
    print(f"\n  Baseline routing accuracy (GT): {baseline:.3f}  |  Target: {COHERENCE_THRESHOLD:.2f}")
    print(f"\n  {'Architecture':<25} " + "  ".join(f"Ep{e}" for e in range(EPOCHS + 1))
          + "  Ep@thresh  FinalVar")
    print(f"  {'─'*90}")

    for name, cls, kwargs in archs:
        curve     = accuracy_curve(cls, kwargs)
        ep_thresh = next((i for i, v in enumerate(curve) if v >= COHERENCE_THRESHOLD), None)
        final_var = float(np.var(curve[-3:])) if len(curve) >= 3 else 0.0
        ep_str    = str(ep_thresh) if ep_thresh is not None else ">max"
        print(f"  {name:<25} " + "  ".join(f"{v:.3f}" for v in curve)
              + f"  {ep_str:>9}  {final_var:.5f}")

    print(f"\n  Ep@thresh: first epoch with GT routing accuracy ≥ {COHERENCE_THRESHOLD}")
    print("  FinalVar:  variance over last 3 checkpoints (lower = more stable)")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    1:  (test_1_mrl_nested_validity,               "[R1]    MRL nested sizes — monotonic recall"),
    2:  (test_2_two_stage_funnel_optimality,       "[R2,R3] Two-stage funnel K_coarse optimality"),
    3:  (test_3_adapter_architecture_tournament,   "[R4,R11] Architecture tournament (bootstrapped)"),
    4:  (test_4_projection_target_comparison,      "[R5]    Projection target comparison (bootstrapped)"),
    5:  (test_5_trigger_threshold_optimality,      "[R6]    Trigger (threshold × window) grid search"),
    6:  (test_6_training_data_composition,         "[R7]    Training data composition (bootstrapped)"),
    7:  (test_7_centroid_update_scope,             "[R8]    Centroid update scope + accumulation"),
    8:  (test_8_adapter_selection_and_ood,         "[R9]    Adapter selection + OOD handling"),
    9:  (test_9_rank_determination_tournament,     "[R10]   Two-NN vs PCA vs fixed ranks (bootstrapped)"),
    10: (test_10_prefix_integrity_and_accumulation,"[R11]   Prefix integrity under sequential rounds"),
    11: (test_11_streaming_centroid_tournament,    "[R12]   Streaming centroid method tournament"),
    12: (test_12_adapter_boundary_interference,    "[NEW]   Boundary query routing consistency"),
    13: (test_13_sample_efficiency_curves,         "[NEW]   Sample efficiency — N to 90% asymptotic"),
    14: (test_14_recovery_speed_under_shift,       "[NEW]   Recovery speed — GT accuracy vs steps"),
}


def main():
    parser = argparse.ArgumentParser(
        description="DEMoE v4.0 Domain Projection Adapter Test Suite v8")
    parser.add_argument("--test",    nargs="*", type=int,
                        help="Run only these test numbers (default: all)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick",   action="store_true",
                        help="Reduced N, fewer epochs, 2 trials — fastest mode")
    args = parser.parse_args()

    global N_TRIALS
    if args.quick:
        N_TRIALS = 2

    to_run = args.test if args.test else list(ALL_TESTS.keys())

    print("=" * 72)
    print("  DEMoE v4.0 — Domain Projection Adapter Test Suite v8")
    print(f"  Device: {DEVICE}  |  EMBED_DIM={EMBED_DIM}  |  PREFIX_DIM={PREFIX_DIM}")
    print(f"  Mode: {'QUICK' if args.quick else 'FULL'}  |  N_TRIALS={N_TRIALS}")
    print(f"  v8 fixes: well-sep experts · centroid-pos training · per-trial variation ·")
    print(f"            Two-NN scale=1.0 · good-day realistic noise · expert-recall metric")
    print("=" * 72)
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

    print("\n" + "=" * 72)
    if failed:
        print(f"  {FAIL} Tests with errors: {failed}")
    else:
        print(f"  {PASS} All tests completed.")
    print("  Any ✗ in tournament tables = spec's approach was beaten.")
    print("  Bootstrapped tests (3,4,6,9) show mean±std across independent trials.")
    print("  Investigate ✗ cases before integrating into the router.")
    print("=" * 72)


if __name__ == "__main__":
    main()