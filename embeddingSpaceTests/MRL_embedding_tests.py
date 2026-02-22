import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Hardware & Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FULL_DIM = 1024  # BAAI/bge-m3 hidden size
PREFIX_DIM = 64
BATCH_SIZE = 32
TRIPLETS_PER_DOMAIN = 500  # Scale up for production runs
LEARNING_RATE = 1e-4
EPOCHS = 3

# We use BGE-M3 as the frozen backbone as recommended in the spec
MODEL_NAME = "BAAI/bge-m3" 

class DenseProjectionAdapter(nn.Module):
    """The standard P in R^(dxd) proposed by the DEMoE v4 spec."""
    def __init__(self, dim):
        super().__init__()
        self.P = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.P.weight) # Initialized to identity per spec

    def forward(self, x):
        return self.P(x)

class BlockDiagonalProjectionAdapter(nn.Module):
    """Our proposed fix: isolated blocks to protect the MRL prefix."""
    def __init__(self, full_dim, prefix_dim):
        super().__init__()
        self.prefix_dim = prefix_dim
        self.P_prefix = nn.Linear(prefix_dim, prefix_dim, bias=False)
        self.P_suffix = nn.Linear(full_dim - prefix_dim, full_dim - prefix_dim, bias=False)
        
        # Initialize to identity
        nn.init.eye_(self.P_prefix.weight)
        nn.init.eye_(self.P_suffix.weight)

    def forward(self, x):
        prefix = self.P_prefix(x[:, :self.prefix_dim])
        suffix = self.P_suffix(x[:, self.prefix_dim:])
        return torch.cat([prefix, suffix], dim=1)

def build_contrastive_triplets(domain_name, num_triplets):
    """Pulls BEIR datasets and constructs (query, pos_doc, neg_doc) triplets."""
    print(f"Loading {domain_name} dataset from HuggingFace...")
    
    # Load corpus and queries from the main repository
    queries = load_dataset(f"BeIR/{domain_name}", "queries", split="queries")
    corpus = load_dataset(f"BeIR/{domain_name}", "corpus", split="corpus")
    
    # Load qrels from the dedicated -qrels repository
    # Note: Depending on what you are doing, you might want split="train" or split="test"
    # SciFact specifically has "train" (919 rows) and "test" (339 rows) splits.
    qrels = load_dataset(f"BeIR/{domain_name}-qrels", split="train")

    # Map IDs to text
    query_dict = {row['_id']: row['text'] for row in queries}
    corpus_dict = {row['_id']: row['text'] for row in corpus}
    
    triplets = []
    all_doc_ids = list(corpus_dict.keys())
    
    print("Generating positive/negative triplets...")
    for qrel in qrels:
        # BEIR qrels columns are usually query-id, corpus-id, score
        q_id = str(qrel['query-id'])
        doc_id = str(qrel['corpus-id'])
        score = qrel['score']
        
        # Only use positive matches (usually score > 0)
        if score > 0 and q_id in query_dict and doc_id in corpus_dict:
            query_text = query_dict[q_id]
            pos_text = corpus_dict[doc_id]
            
            # Simple random negative sampling (pick a doc that isn't the positive one)
            neg_id = random.choice(all_doc_ids)
            while neg_id == doc_id:
                neg_id = random.choice(all_doc_ids)
            
            neg_text = corpus_dict[neg_id]
            
            # Append the full 3-part tuple
            triplets.append((query_text, pos_text, neg_text))
            
            if len(triplets) >= num_triplets:
                break
                
    print(f"Successfully built {len(triplets)} triplets.")
    return triplets

def infonce_loss(q_emb, pos_emb, neg_emb, temperature=0.05):
    """InfoNCE contrastive loss over triplets."""
    pos_sim = F.cosine_similarity(q_emb, pos_emb) / temperature
    neg_sim = F.cosine_similarity(q_emb, neg_emb) / temperature
    
    logits = torch.stack([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=DEVICE)
    return F.cross_entropy(logits, labels)

def evaluate_prefix_drift(adapter, queries, docs, original_sims):
    """Measures how much the 64-dim prefix similarity changes after projection."""
    adapter.eval()
    with torch.no_grad():
        proj_queries = adapter(queries)
        new_sims = F.cosine_similarity(proj_queries[:, :PREFIX_DIM], docs[:, :PREFIX_DIM])
        drift = torch.mean(torch.abs(original_sims - new_sims)).item()
    return drift

def run_domain_audit(domain_name, model):
    print(f"\n{'='*50}\nTesting Domain: {domain_name.upper()}\n{'='*50}")
    
    triplets = build_contrastive_triplets(domain_name, TRIPLETS_PER_DOMAIN)
    
    # Pre-compute embeddings (frozen backbone per spec)
    print("Pre-computing BGE-M3 MRL embeddings...")
    q_texts = [t[0] for t in triplets]
    pos_texts = [t[1] for t in triplets]
    neg_texts = [t[2] for t in triplets]
    
    with torch.no_grad():
        q_embs = torch.tensor(model.encode(q_texts), device=DEVICE)
        pos_embs = torch.tensor(model.encode(pos_texts), device=DEVICE)
        neg_embs = torch.tensor(model.encode(neg_texts), device=DEVICE)

    # Baseline Stage 1 Prefix Similarity
    baseline_prefix_sims = F.cosine_similarity(q_embs[:, :PREFIX_DIM], pos_embs[:, :PREFIX_DIM])

    # Initialize Adapters
    dense_adapter = DenseProjectionAdapter(FULL_DIM).to(DEVICE)
    block_adapter = BlockDiagonalProjectionAdapter(FULL_DIM, PREFIX_DIM).to(DEVICE)
    
    opt_dense = torch.optim.Adam(dense_adapter.parameters(), lr=LEARNING_RATE)
    opt_block = torch.optim.Adam(block_adapter.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("Training Domain Projection Adapters (InfoNCE)...")
    for epoch in range(EPOCHS):
        dense_adapter.train()
        block_adapter.train()
        
        # Batching
        for i in range(0, len(q_embs), BATCH_SIZE):
            q_batch = q_embs[i:i+BATCH_SIZE]
            pos_batch = pos_embs[i:i+BATCH_SIZE]
            neg_batch = neg_embs[i:i+BATCH_SIZE]
            
            # --- Dense Update ---
            opt_dense.zero_grad()
            q_proj_dense = dense_adapter(q_batch)
            loss_dense = infonce_loss(q_proj_dense, pos_batch, neg_batch)
            loss_dense.backward()
            opt_dense.step()
            
            # --- Block Update ---
            opt_block.zero_grad()
            q_proj_block = block_adapter(q_batch)
            loss_block = infonce_loss(q_proj_block, pos_batch, neg_batch)
            loss_block.backward()
            opt_block.step()

    # Post-Run Evaluation
    dense_drift = evaluate_prefix_drift(dense_adapter, q_embs, pos_embs, baseline_prefix_sims)
    block_drift = evaluate_prefix_drift(block_adapter, q_embs, pos_embs, baseline_prefix_sims)
    
    print("\n--- Audit Results ---")
    print(f"Dense Adapter Prefix Drift: {dense_drift:.4f}")
    print(f"Block Adapter Prefix Drift: {block_drift:.4f}")
    
    if dense_drift > block_drift * 5:
        print(">> FATAL FLAW CONFIRMED: Dense projection severely destroys the MRL 64-dim prefix.")
    
    return dense_drift, block_drift

if __name__ == "__main__":
    print("Loading Frozen MRL Backbone (BGE-M3)...")
    bge_m3 = SentenceTransformer("BAAI/bge-m3", device=DEVICE)
    
    # Test across multiple disparate domains
    domains = ["scifact", "nfcorpus", "fiqa"]
    
    results = {}
    for domain in domains:
        d_drift, b_drift = run_domain_audit(domain, bge_m3)
        results[domain] = {"Dense": d_drift, "Block": b_drift}
        
    print("\n" + "="*50)
    print("FINAL SYSTEM-WIDE AUDIT REPORT")
    print("="*50)
    for dom, metrics in results.items():
        print(f"Domain: {dom:10} | Dense Drift: {metrics['Dense']:.4f} | Block Drift: {metrics['Block']:.4f}")