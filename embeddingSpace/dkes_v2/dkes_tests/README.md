# DKES Evaluation Suite

## Methodology

This test suite proves the Dynamic Kanerva-Enhanced Embedding Space (DKES) works correctly
across five evaluation dimensions, drawn from how similar systems were validated:

| Source Paper | What they tested | Translated to DKES |
|---|---|---|
| Kanerva Machine (Wu et al. 2018) | One-shot generation fidelity, iterative retrieval convergence | KDM read quality, composite embedding accuracy |
| Dynamic Kanerva Machine (NIPS 2018) | Attractor dynamics, pattern completion under noise | Memory robustness, noisy query handling |
| BEIR Benchmark (Thakur et al. 2021) | Zero-shot retrieval NDCG@k / Recall@k across 18 domains | Routing accuracy across domain-shifted queries |
| PMoE/MoLE continual learning papers | Forgetting error (F), backward/forward transfer | Memory retention across sequential domain injection |
| MTEB Embedding Benchmark | Semantic similarity, clustering, retrieval | Composite embedding quality vs backbone baseline |

## Test Dimensions

### 1. Unit Tests (`unit/`)
Verify each module in isolation with synthetic data.
- KDM read/write/evict correctness
- Composite embedding LayerNorm and gamma arithmetic
- ColdStore archive/reactivation cycle
- Domain projection InfoNCE training
- MRL funnel routing logic
- Serialization round-trips

### 2. Integration Tests (`integration/`)
Verify the full pipeline end-to-end with realistic data flows.
- Write → Read → Composite roundtrip fidelity
- Multi-expert routing with EASM domain labels
- Eviction → ColdStore → Reactivation cycle
- Checkpoint → Restore equivalence

### 3. Benchmark Tests (`benchmarks/`)
Measure DKES against quantitative baselines using standard datasets.
- **Retrieval accuracy**: NDCG@1/5/10, Recall@10, MRR on synthetic and STS datasets
- **Memory capacity utilization**: slot efficiency as concept count grows
- **Routing precision**: fraction of queries routed to correct domain
- **Composite embedding quality**: improvement over backbone-only baseline

### 4. Stress Tests (`stress/`)
Prove behavior under adversarial / edge-case conditions.
- **Near-capacity memory**: behavior at 90%, 99%, 100% slot utilization
- **Cold domain injection**: routing for completely unseen domains
- **Noisy query robustness**: degraded input embeddings
- **Continual learning**: backward transfer (forgetting) and forward transfer metrics
- **Concurrent reads/writes**: thread-safety under load

### 5. Ablation Tests (`benchmarks/ablation.py`)
Quantify contribution of each component.
- DKES full vs backbone-only
- DKES full vs backbone + KDM (no meta-concepts)
- DKES full vs DKES without domain projection
- DKES full vs DKES without two-stage MRL funnel

## Datasets Used

| Dataset | Use | Format |
|---|---|---|
| Synthetic semantic clusters | Unit / integration tests | Generated on-the-fly |
| STS-B (Semantic Textual Similarity) | Composite embedding quality | sentence-pairs with float similarity scores |
| NF-Corpus subsets (simplified BEIR-style) | Domain retrieval accuracy | query → relevant_doc_ids |
| Custom multi-domain probe set | Domain shift / continual learning | queries across 8 named domains |
| Noise augmentation of above | Robustness testing | Gaussian noise + random dropout |

## Running

```bash
# Install dependencies
pip install pytest numpy scipy scikit-learn datasets sentence-transformers --break-system-packages

# Run all tests
cd /path/to/dkes_tests
pytest -v

# Run specific dimension
pytest unit/ -v
pytest benchmarks/ -v --tb=short

# Run full benchmark with report
python run_evaluation.py --report reports/evaluation_report.json
```

## Pass Criteria (Published Targets)

| Metric | Minimum Pass | Target |
|---|---|---|
| KDM read cosine fidelity | ≥ 0.90 | ≥ 0.95 |
| Routing Recall@1 (in-distribution) | ≥ 0.75 | ≥ 0.85 |
| Routing Recall@3 | ≥ 0.90 | ≥ 0.95 |
| NDCG@10 (retrieval benchmark) | ≥ 0.55 | ≥ 0.65 |
| Composite vs backbone gain (NDCG) | ≥ +0.03 | ≥ +0.08 |
| Memory retention (forgetting error F) | ≤ 0.15 | ≤ 0.08 |
| Thread safety (0 race conditions) | 100% | 100% |
| Cold reactivation success rate | ≥ 0.80 | ≥ 0.95 |
| Checkpoint roundtrip max deviation | ≤ 1e-6 | ≤ 1e-7 |
