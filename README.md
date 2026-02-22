# DEMoE-Research
This repository contains the research code and documentation for a proposed ML system architecture; Dynamically Expanding Mixture of Experts.

Seven Defining Properties of DEMoE:

Self-Expansion: The system creates new expert models and adapters autonomously in response to detected knowledge gaps, without human intervention.

Geometric Routing: All routing decisions are made by nearest-neighbor search in a unified MRL embedding space. No trained router network exists. Routing generalizes to new experts the moment they are indexed.

Accurate Bayesian Uncertainty: BLoB with MAP initialization and cyclical KL annealing provides end-to-end calibrated epistemic uncertainty — the foundational signal driving all expansion decisions. Laplace-LoRA is reserved exclusively for bootstrapped external adapters where BLoB retraining is not possible.

Two-Level Specialization: Broad domain coverage in independent base expert models; fine-grained sub-domain coverage in BLoB adapter layers on top of those base models.
Pre-Trained Bootstrapping: The expert library is seeded from the existing ecosystem of open-source domain-specific models, eliminating cold-start training costs for common domains.

Frozen Encoder Stability: The shared MRL encoder backbone is permanently frozen. Domain-specific projection adapters handle new domain accommodation without FAISS index rebuilds.

Expert-Aware Synthesis Model: A general model fine-tuned on current expert outputs provides query planning before routing and cross-expert semantic alignment after activation, with lightweight LoRA updates whenever new experts are registered.