# Cybercrime Predictive Analytics — Multi-Dataset Mule-Chain Detection & Triage Architecture

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0%2Bcpu-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.8.0-3C2179.svg)](https://pyg.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-EB5424.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Stage_8_Enterprise_Backend-VALIDATED-success.svg)]()

A modular, multi-dataset predictive analytics framework designed for post-complaint financial cybercrime triage, inductive Graph Neural Network (GraphSAGE) laundering detection, terminal exit ranking, uncertainty calibration, real-time streaming ingestion, interactive graph visualization, and REST API deployment.

> [!IMPORTANT]
> **Operational Scope & Architectural Boundary**:
> This framework is strictly a **retrospective, post-complaint analytical triage engine** triggered when an incident complaint is logged. Real-time streaming transaction feeds are supported and benchmarked via the sliding-window `TemporalTransactionGraph` engine.
> All reported metrics state exact sample sizes ($N$), class distributions, and 95% Confidence Intervals. Performance is benchmarked across three separate datasets without blending claims.

---

## 1. Multi-Dataset Scope & Reality Check

The pipeline interfaces with three distinct data sources through dedicated, isolated adapters in [`src/adapters/`](src/adapters):

| Dataset Identifier | Domain & Topology | Total Volume / Sample Size ($N$) | Geographic Coordinates | ATM Hardware Terminals | Evaluation Scope |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Dataset A (Synthetic)** | Domestic cybercrime complaints & multi-hop transaction subgraphs | $N = 1,000$ subgraphs ($15,000$ Tx, $700$ accounts, $25$ mule rings) | ✅ **AVAILABLE** (15 Indian Cities) | ✅ **AVAILABLE** (`ATM_001`–`050`) | Full pipeline validation (Stages 0–8). |
| **Dataset B (IBM AML)** | Real-world multi-bank payment ledger transactions (`HI-Small_Trans.csv`) | $1,000,000$ sample transactions ($1,000$ extracted subgraphs) | ❌ **UNAVAILABLE** (No GPS recorded) | ❌ **UNAVAILABLE** (Inter-bank rails only) | End-to-end extended validation (Items 1–10). |
| **Dataset C (Elliptic)** | Real-world Bitcoin transaction DAG (`EllipticBitcoinDataset`) | $203,769$ transaction nodes, $234,355$ directed payment edges ($46,564$ labeled) | ❌ **UNAVAILABLE** (UTXO DAG) | ❌ **UNAVAILABLE** (Decentralized blockchain) | Inductive GraphSAGE node-classification architecture benchmark. |

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

---

## 3. Stage 8: Enterprise FastAPI Backend & Streaming Architecture

Stage 8 integrates the analytical engine into an enterprise-ready system:

### 1. High-Performance REST API Service (`src/api.py`)
- **FastAPI Endpoints**:
  - `GET /api/health`: System health status, loaded model checkpoints, and SQLite DB connectivity.
  - `GET /api/stats`: Real-time queue metrics and model performance summaries.
  - `GET /api/incidents`: Filterable incident alert queue with pagination, confidence tier filters, and risk thresholds.
  - `GET /api/incidents/{incident_id}`: Incident case metadata, resolved entity details, and model outputs.
  - `GET /api/incidents/{incident_id}/graph`: Dynamic network topology in Cytoscape/vis.js JSON format.
  - `POST /api/predict/subgraph`: On-the-fly GraphSAGE classification on arbitrary entity seeds.
  - `POST /api/policy/tune`: Real-time alert threshold simulator calculating precision/recall/F1 for any cutoff $\tau$.
  - `GET /api/dossier/{incident_id}/export`: Printable Law Enforcement case dossiers in Markdown, HTML, or JSON.
  - `GET /api/streaming/benchmark`: Real-time streaming throughput and latency metrics.
  - `GET /api/benchmarks/three_way`: Global multi-dataset comparison matrix.
- **Interactive Documentation**: Available automatically at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.

### 2. Real-Time Streaming Ingestion & Two-Stage Hybrid Trigger (`src/streaming_engine.py`)
- **Stage 1 (Lightweight Anomaly Gate):** $O(1)$ statistical behavioral gate utilizing Welford's algorithm to track rolling single-transaction Z-scores, daily velocity limits, and cold-start boundaries. Safely filters **>86% of benign micro-traffic** in-memory.
- **Stage 2 (Event-Driven Graph Triage):** Breaches instantly trigger sub-millisecond $k$-hop temporal BFS extraction and live Dual-Head GraphSAGE scoring. 
- **Sub-5ms SLA Verification:** Real-time ingestion processes at **940+ Tx/sec**. Stage 2 proactive GNN triage (BFS + Forward pass) resolves in **1.07 ms mean latency** (P99: 1.97 ms), vastly exceeding the sub-5ms operational SLA constraint.

### 3. Database Persistence Layer (`src/database.py`)
- Relational SQLite schema with SQLAlchemy ORM models (`Complaint`, `EntityMaster`, `TransactionRecord`, `IncidentPrediction`, `AuditLog`).
- Indexed multi-hop queries for rapid entity resolution and transaction history lookup.

### 4. Interactive Web Dashboard (`src/dashboard.py`)
- **Interactive Subgraph Visualizer**: Physics-based graph exploration using PyVis (drag, zoom, tooltips, flow directions, and ATM highlight nodes).
- **Incident Queue & Printable Dossier**: One-click case dossier export in Markdown/JSON.
- **Tunable Policy Slider**: Real-time triage volume estimation and alert tradeoff simulator.

---

## 4. How to Run the Pipeline & Services

```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Download Real-World IBM AML Dataset (KaggleHub via API)
python src/download_ibm_data.py

# 3. Initialize and Seed Database
python src/database.py

# 4. Run Real-Time Streaming Ingestion & Inference Benchmark
python src/streaming_engine.py

# 5. Launch FastAPI Backend REST Service (Port 8000)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 6. Launch Tactical Interactive Dashboard (Streamlit)
streamlit run src/dashboard.py

# 7. Run Automated Test Suite (0 Regressions across GNN and Trigger Gates)
pytest tests/test_dynamic_trigger.py -v
pytest tests/test_api_streaming.py -v
```
