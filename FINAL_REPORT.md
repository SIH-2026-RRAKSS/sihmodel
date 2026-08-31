# Final Architecture and Benchmark Report

## 1. Problem & Architecture
This system implements Anti-Money Laundering (AML) detection across three distinct graph datasets: Dataset A (synthetic subgraphs), Dataset B (IBM AML subgraphs), and Dataset C (Elliptic chronological transaction graph). The core architectural evaluation compares a dual-head GraphSAGE network (root-centric + global pooling) against a baseline XGBoost classifier utilizing engineered tabular features.

## 2. Final Locked Baselines
The following table represents the verified, 10-seed (Datasets A/B) and 5-seed (Dataset C) cross-validated baselines. Ensembling XGBoost with GraphSAGE embeddings was tested and does not outperform the standalone models; ensembling is not recommended for production.

| Dataset | Best Model | F1 Score (mean ± std) |
| :--- | :--- | :--- |
| **Dataset A** | Standalone GraphSAGE | 90.71% ± 2.02% |
| **Dataset B** | Standalone GraphSAGE | 77.81% ± 3.07% |
| **Dataset C** | Standalone XGBoost | 76.88% ± 1.20% |

*(Note: Standalone GraphSAGE on Dataset C achieved 44.00% ± 1.84%)*

## 3. Methodology Audit Trail

**a) Threshold calibration audit**
- **Hypothesis:** Defaulting to F1@0.50 is misleading on heavily imbalanced AML data.
- **Test:** Ran a 5-seed baseline audit for XGBoost and GNN across datasets (`final_threshold_audit.py`).
- **Bug found:** XGBoost seed was not actually varying (yielding 0.00% standard deviation) due to fixed sampling parameters. Pipeline mismatch corrected.
- **Result:** Threshold tuning was neutral on Datasets A/B. On Dataset C, val-optimized thresholding increased GraphSAGE performance from 26.92% to 44.00%. Thresholds were stable only on Dataset C (~0.95), while highly volatile on XGBoost.

**b) Ensemble stacking (GNN embeddings → XGBoost)**
- **Hypothesis:** Stacking GNN representations with tabular features improves F1.
- **First attempt:** Initially showed large gains, but was suspected of target leakage (embeddings were extracted from a GNN trained on the same fold).
- **Diagnostic:** Feature importance (`diagnostic_c.py`) revealed 84.54% of split gain went to the GNN dimensions.
- **OOF fix v1 (raw 128-dim):** Implemented k-fold Out-Of-Fold (OOF) extraction (`run_oof_ab.py`). The ensemble collapsed on A (75.47% ± 25.81%) and B (64.12% ± 11.78%). Root cause: latent space misalignment across independently initialized fold-models.
- **OOF fix v2 (1D logits):** Extracted alignment-invariant 1D pre-sigmoid logits (`run_oof_ab_1d.py`). 
- **Result:** Standalone GraphSAGE strictly outperformed the ensemble on A (90.71% vs 89.53%) and B (77.81% vs 74.69%). On Dataset C, the ensemble failed to beat standalone XGBoost under both default thresholding (76.28% ± 1.14%) and validation-optimized thresholding (69.79% ± 5.65%) (`run_oof_ensemble_c.py`).
- **Verdict:** Ensembling adds no generalizable value on any dataset.

**c) Edge-feature investigation (Dataset C)**
- **Hypothesis:** GraphSAGE fails on Dataset C because it is blind to transaction amounts/timing stored as edge features.
- **Falsified:** Inspection of the PyTorch Geometric `EllipticBitcoinDataset` object confirmed no `edge_attr` exists. All 165 features are node-level. Edges act strictly as structural pointers.
- **Pivot:** Shifted investigation from missing edge attributes to aggregation capacity and graph directionality.

**d) Capacity vs. directionality ablation (Dataset C)**
- **Hypothesis:** GraphSAGE fails on Dataset C either because linear aggregation (`W * x`) lacks capacity compared to XGBoost trees, or because standard directed message passing discards outgoing fan-out flows.
- **Test:** Ran a 5-way, 5-seed ablation on Dataset C (`test_elliptic_archs.py`).
- **Result:**
  - `BaseSAGE`: 43.47% ± 3.10%
  - `FatSAGE` (wider dims + MLP head): 41.55% ± 2.00%
  - `GIN` (MLP inside aggregation): 38.24% ± 2.78%
  - `Undirected_BaseSAGE` (Bidirectional): 47.04% ± 1.24%
  - `Undirected_GIN` (Bidirectional + MLP): 31.23% ± 1.37%
- **Verdict:** Increasing capacity (`FatSAGE`, `GIN`) strictly hurt generalization by overfitting. Modifying directionality (`Undirected_BaseSAGE`) provided the only measurable gain (+3.5%) and tightened variance. Combining both collapsed entirely.
- **Conclusion:** These results are consistent with a low-pass filter effect, where message-passing (averaging neighbor features) blurs Elliptic's sharp tabular signatures. XGBoost's non-linear tree splits isolate these signatures better than GNN neighborhood aggregation.

## 4. Known Limitations / Explicitly Untested
- **Aggregator Types:** PNA and GAT architectures (non-averaging aggregators utilizing max pooling or attention) were not tested on Dataset C. Given the conclusion that averaging dilutes the sharp tabular signal, these represent the most mathematically promising lever if the GNN track is revisited. They were excluded in this cycle due to compute constraints.
- **Ensemble Tuning:** XGBoost ensemble hyperparameters were not re-tuned post-OOF correction. This was deprioritized once the unbiased OOF logits demonstrated ensembling failed to clear the standalone baselines.

## 5. Reproduction
The final locked baselines can be independently reproduced using the following scripts:
- **Datasets A & B Baseline:** Run `run_oof_ab_1d.py`. Configured for 10 seeds, extracts Model 6 standalone evaluations at 0.50 threshold.
- **Dataset C Baseline:** Run `run_oof_ensemble_c.py` (which produces the 76.88% XGBoost result over 5 chronological seeds) and `test_elliptic_archs.py` (which produces the comparative GNN results). Threshold is validation-optimized (`get_opt` over timesteps 30-34).
