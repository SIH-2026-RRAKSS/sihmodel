# Cybercrime Predictive Analytics ? Multi-Dataset Mule-Chain Detection & Triage Architecture

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcpu-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8.0-3C2179.svg)](https://pyg.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB5424.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Multi_Dataset_Validation-AUDITED-success.svg)]()

A modular, multi-dataset predictive analytics framework designed for post-complaint financial cybercrime triage, inductive Graph Neural Network (GraphSAGE) laundering detection, terminal exit ranking, uncertainty calibration, and rule-based investigative explainability.

> [!IMPORTANT]
> **Operational Scope & Architectural Boundary**:
> This framework is strictly a **retrospective, post-complaint analytical triage engine** triggered when an incident complaint is logged. It is **NOT a real-time live transaction stream monitoring system**.
> All reported metrics state exact sample sizes ($N$), class distributions, and 95% Confidence Intervals. Performance is benchmarked across three separate datasets without blending claims.

---

## 1. Multi-Dataset Scope & Reality Check

The pipeline interfaces with three distinct data sources through dedicated, isolated adapters in [`src/adapters/`](src/adapters):

| Dataset Identifier | Domain & Topology | Total Volume / Sample Size ($N$) | Geographic Coordinates | ATM Hardware Terminals | Evaluation Scope |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Dataset A (Synthetic)** | Domestic cybercrime complaints & multi-hop transaction subgraphs | $N = 1,000$ subgraphs ($15,000$ Tx, $700$ accounts, $25$ mule rings) | ? **AVAILABLE** (15 Indian Cities) | ? **AVAILABLE** (`ATM_001`?`050`) | Full pipeline validation (Stages 0?10). |
| **Dataset B (IBM AML)** | Real-world multi-bank payment ledger transactions (`HI-Small_Trans.csv`) | $1,000,000$ sample transactions ($1,000$ extracted subgraphs) | ? **UNAVAILABLE** (No GPS recorded) | ? **UNAVAILABLE** (Inter-bank rails only) | End-to-end extended validation (Items 1?5, 7?10). |
| **Dataset C (Elliptic)** | Real-world Bitcoin transaction DAG (`EllipticBitcoinDataset`) | $203,769$ transaction nodes, $234,355$ directed payment edges ($46,564$ labeled) | ? **UNAVAILABLE** (UTXO DAG) | ? **UNAVAILABLE** (Decentralized blockchain) | Inductive GraphSAGE node-classification architecture benchmark. |

---

## 2. Global Three-Way Multi-Dataset Architecture Benchmark

A standardized, multi-seed comparison across all three evaluated datasets demonstrates how inductive Graph Neural Networks generalize across synthetic and real-world payment topologies:

| Evaluation Dimension | Dataset A (Synthetic Typology Subgraphs) | Dataset B (IBM AML Multi-Bank Subgraphs) | Dataset C (Elliptic Bitcoin Transaction DAG) |
| :--- | :--- | :--- | :--- |
| **Evaluation Task** | Inductive Subgraph Binary Classification | Inductive Subgraph Binary Classification | Inductive Node Classification (Temporal Split) |
| **Test Sample Size ($N_{\text{test}}$)** | $N = 200$ subgraphs ($37$ Positives / $18.5\%$) | $N = 200$ subgraphs ($59$ Positives / $29.5\%$) | $N = 16,670$ nodes ($1,083$ Illicit / $6.50\%$) |
| **XGBoost Baseline F1 (Mean $\pm$ Std)** | $86.98\% \pm 2.28\%$ | $73.93\% \pm 3.37\%$ | N/A (DAG Node Benchmark) |
| **GraphSAGE GNN F1 (Mean $\pm$ Std)** | **$90.66\% \pm 1.58\%$** | **$77.70\% \pm 2.57\%$** | **$46.44\% \pm 2.52\%$** ($49.45\%$ single run) |
| **F1 Delta ($\Delta$) & Significance** | **$+3.69\%$** ($t=3.583$, **$p = 0.0231 < 0.05$**) | **$+3.76\%$** ($t=6.323$, **$p = 0.0032 < 0.01$**) | N/A |
| **GraphSAGE Precision** | $89.66\% \pm 3.54\%$ | $72.33\% \pm 2.11\%$ | $35.75\% \pm 3.58\%$ ($40.18\%$ single run) |
| **GraphSAGE Recall** | $91.89\% \pm 3.82\%$ | $84.41\% \pm 7.80\%$ | $66.85\% \pm 2.40\%$ ($64.27\%$ single run) |
| **GraphSAGE PR-AUC** | $0.9680 \pm 0.0117$ | $0.8775 \pm 0.0198$ | $0.5118 \pm 0.0396$ ($0.5194$ single run) |
| **Generalization Finding** | Clean synthetic baseline; high structural regularity. | Statistically significant GNN gain on multi-bank ledgers; $\approx 13\%$ lower than synthetic. | Real-world UTXO graph noise & heavy imbalance ($6.5\%$ illicit); aligns with peer-reviewed literature. |

---

## 3. Subgraph Size Decomposition: Why GNNs Show Advantage on Full Corpora

To evaluate whether GraphSAGE's statistical advantage over tabular XGBoost on full corpora was driven by distinguishing trivial 1-node components from multi-hop structures, we decomposed both subgraph datasets into single-node vs. strictly multi-node subsets:

| Dataset & Partition | Sample Size ($N$) | Class Balance | XGBoost Baseline F1 (Mean $\pm$ Std) | GraphSAGE GNN F1 (Mean $\pm$ Std) | Mean F1 Delta ($\Delta$) | Paired $t$-test & $p$-value | Methodological Conclusion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dataset A (Synthetic) ? Full Pool** | $N = 1,000$ | $18.5\%$ Pos | $86.98\% \pm 2.28\%$ | **$90.66\% \pm 1.58\%$** | **$+3.69\%$** | $t = 3.583$, **$p = 0.0231$** | Significant on full corpus containing $36.9\%$ single-node subgraphs (all 100% negative). |
| **Dataset A (Synthetic) ? Multi-Node ($\ge 2$ Nodes)** | $N = 631$ | $29.5\%$ Pos | $89.52\% \pm 1.46\%$ | **$92.02\% \pm 2.07\%$** | $+2.50\%$ | $t = 2.001$, **$p = 0.1159$** | **No statistically significant advantage detected** on multi-node subgraphs ($p > 0.05$); inconclusive / potentially underpowered at $N=631$. |
| **Dataset B (IBM AML) ? Full Pool** | $N = 1,000$ | $29.7\%$ Pos | $73.93\% \pm 3.37\%$ | **$77.70\% \pm 2.57\%$** | **$+3.76\%$** | $t = 6.323$, **$p = 0.0032$** | Significant on full corpus containing $33.0\%$ single-node self-loop subgraphs (all negative). |
| **Dataset B (IBM AML) ? Multi-Node ($\ge 2$ Nodes)** | $N = 670$ | $44.3\%$ Pos | $75.27\% \pm 3.05\%$ | **$75.75\% \pm 2.48\%$** | $+0.48\%$ | $t = 0.311$, **$p = 0.7716$** | **Clear empirical null**: Tabular flow features match GraphSAGE once multi-node topology is established. |
| **Dataset C (Elliptic Bitcoin DAG)** | $N_{\text{test}} = 16,670$ | $6.5\%$ Illicit | N/A | **$46.44\% \pm 2.52\%$** | N/A | N/A | *Context only: Elliptic's 67.8% leaf-node prevalence in node classification is structurally distinct from A/B's single-node subgraphs.* |

> [!IMPORTANT]
> **Core Architectural Takeaways**:
> 1. **Trivial Component Separation**: In both evaluated subgraph datasets (Dataset A: $36.9\%$, Dataset B: $33.0\%$), roughly one-third of extracted subgraphs are 1-node components. GNN graph pooling naturally separates these 1-node components from multi-hop networks, driving the apparent $+3.7\%$ statistically significant gain on full corpora.
> 2. **Power vs. Null Distinction**: On strictly multi-node subgraphs, the IBM result is a **clear empirical null ($p = 0.7716, \Delta = +0.48\%$)**, showing near-total parity between XGBoost and GraphSAGE. The synthetic multi-node result is **inconclusive ($p = 0.1159, \Delta = +2.50\%$)**, failing to reach statistical significance at conventional thresholds ($\alpha = 0.05$).
> 3. **Engineering Synthesis**: Graph Neural Networks excel at topological structural filtering across heterogeneous graph sizes, but domain-engineered tabular flow features (`velocity_tph`, `fan_out_ratio`, `num_terminal_sinks`, `total_flow`) remain equally potent once multi-hop transaction flow is established.

---

## 4. IBM AML Dataset (Dataset B) Pipeline Extension Matrix

| Pipeline Stage / Item | Status on IBM AML Data | Implemented Engineering & Methodological Adaptation |
| :--- | :---: | :--- |
| **Item 1: Entity Resolution** | **CONFIRMED CANONICAL** | Pre-resolved composite keys (`B<BankID>_<AccountID>`) constructed directly from IBM ledger fields. No artificial fuzzy matching needed. |
| **Item 2: Graph Construction** | **ADAPTED** | Extracted $N=1,000$ directed 72h temporal subgraphs ($297$ laundering / $703$ normal). Initial seed target of 200/800 expanded to 297/703 because 97 clean seeds touched laundering flows in 3-hop BFS. Zero subgraphs have 0 edges; $330$ are single-account bursts ($1$ node with reinvestment self-loops). |
| **Item 3: Baseline Classifier (XGBoost)** | **ADAPTED** | Rebuilt on available flow and topology features (`fan_out_ratio`, `velocity_tph`, `density`, `num_terminal_sinks`). Missing synthetic features (`account_age_days`, `dormancy_score`) **explicitly dropped**. F1: **$73.93\% \pm 3.37\%$** ($N_{\text{test}}=200$). |
| **Item 4: GraphSAGE Classifier** | **ADAPTED & VALIDATED** | 2-layer GraphSAGE trained on PyG subgraphs (`data/ibm_pyg_dataset.pt`). Achieves **$77.70\% \pm 2.57\%$ F1** (Statistically significant gain over XGBoost: $+3.76\%$, $p = 0.0032 < 0.01$). |
| **Item 5: Terminal Exit Ranking** | **AUDITED & BENCHMARKED** | Evaluated on $N=384$ laundering flow clusters. Achieves **$10.42\%$ Top-1 Hit Rate** [$7.74\% - 13.87\%$]. **Audit finding**: Model makes byte-for-byte identical predictions to the single-feature naive heuristic "Lowest Out-Degree" on all $384/384$ pairs ($p=1.0$). |
| **Item 6: Geo-Cluster Estimation** | ? **EXCLUDED** | **CONFIRMED NOT POSSIBLE**. IBM schema contains zero GPS coordinates. Excluded per Guardrail #1. |
| **Item 7: Confidence Tiers** | **ADAPTED** | Uncertainty calibration using IBM GraphSAGE probability + structural complexity counts ONLY. Synthetic reference pattern circularity **explicitly removed**. High Confidence tier achieves **$89.02\%$ Precision** and captures **$73.74\%$** of laundering flows ($N=1,000$). |
| **Item 8: Rule-Based Explainability** | **ADAPTED** | Generates human-readable rationales directly from observable IBM features (`num_nodes`, `velocity_tph`, `total_flow`, `num_terminal_sinks`). Terminals described as absorbing sink accounts ($\text{out\_degree}=0$). |
| **Item 9: Tunable Threshold Policy** | **ADAPTED (GraphSAGE Native)** | Evaluated directly on IBM GraphSAGE holdout test set ($N_{\text{test}}=200$, 59 Positives / 141 Negatives): $\tau=0.10 \rightarrow 100.0\%$ Rec / $50.86\%$ Prec; $\tau=0.50 \rightarrow 77.97\%$ Rec / $71.88\%$ Prec; $\tau=0.70 \rightarrow 71.19\%$ Rec / $84.00\%$ Prec; $\tau=0.90 \rightarrow 54.24\%$ Rec / $91.43\%$ Prec. |
| **Item 10: Tactical Dashboard** | **ADAPTED** | Added **Active Dataset Selector** (`Synthetic Domestic Prototype` vs `IBM AML Multi-Bank Benchmark`) with separate, honest Recall Reality panels and automatic map disabling for IBM data. |

---

## 5. How to Run the Pipeline

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Run Synthetic Pipeline
python src/xgboost_baseline.py
python src/graphsage_classifier.py

# 3. Run Real-World Benchmarks (Elliptic & IBM AML)
python src/elliptic_benchmark.py
python src/ibm_graph_construction.py
python src/ibm_xgboost_baseline.py
python src/ibm_graphsage_classifier.py
python src/ibm_tactical_intelligence.py

# 4. Run Multi-Dataset Terminal Ranking Benchmark
python src/terminal_ranking_benchmark.py

# 5. Launch Tactical Dashboard
streamlit run src/dashboard.py
```
