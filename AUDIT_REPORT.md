# SIH System Audit & Remediation Report
**Date:** August 2026
**Scope:** `sihmodel` (Backend/ML) and `sihweb` (Frontend/UI)
**Branch Status:** Updates pushed to `origin/kush` (sihmodel) and `origin/Kush` (sihweb)

## Executive Summary
A comprehensive end-to-end audit was conducted on both the machine learning backend (`sihmodel`) and the React frontend (`sihweb`). The audit uncovered severe data leakage in the ML evaluation pipeline that artificially inflated GraphSAGE performance on real-world data, and a pervasive "mock data fallback" architecture in the frontend that suppressed critical system failures. All identified vulnerabilities have been remediated, structurally fixed, and verified via live execution.

---

## 1. Backend & Machine Learning Remediations (`sihmodel`)

### 1.1 Test-Set Leakage Eradicated
- **Finding:** The original training scripts (`graphsage_classifier.py` and `ibm_graphsage_classifier.py`) suffered from test-set leakage. They dynamically selected the `best_epoch` model state by continuously evaluating on the held-out `test_loader` during training, rather than a dedicated validation set. This artificially inflated GraphSAGE's F1 score on the IBM dataset to 77.70% (a +3.76% delta over XGBoost).
- **Remediation:** Both scripts were rewritten to enforce a strict `70% Train / 12.5% Val / 17.5% Test` split.
- **Verification:** Live execution confirms that early stopping is now driven strictly by `val_loader`. The true test F1 for GraphSAGE on the IBM dataset converged to ~73.27%, effectively tying the XGBoost baseline and eliminating the false GNN advantage. 

### 1.2 "64% Bug" (Batch Normalization & Saturation) Fixed
- **Finding:** Initial calibration issues in the streaming pipeline led to a persistent 64% sub-optimal F1 bound. This was traced to unnormalized tensors in PyG batching and a silent fallback to dummy graphs in `streaming_engine.py`.
- **Remediation:** Data normalizers were decoupled and correctly integrated into the streaming engine's inference loop. The PyG graph conversion utilities were hardened to ensure batching maintains tensor fidelity.

### 1.3 `three_way_benchmark` API Cleaned
- **Finding:** The backend endpoint serving the benchmark comparison to the UI (`/api/benchmarks/three_way`) was serving the old, leaky 77.70% metrics via a statically loaded CSV file.
- **Remediation:** `data/three_way_benchmark_comparison.csv` was patched with the clean, 5-seed benchmark numbers. The endpoint now honestly reports the ~73.57% GNN F1 (p=0.50) for the IBM dataset, and ~89.77% GNN F1 for Dataset A.

### 1.4 Missing Geographical Route Implemented
- **Finding:** The `/api/entities/locations` endpoint, required by the map UI, was returning a 404. 
- **Remediation:** A robust SQL outer-join was added to `api.py` to aggregate geographical data across `EntityMaster`, `Complaint`, and `IncidentPrediction`. Verified returning HTTP 200 with valid schema.

---

## 2. Frontend Architecture Remediations (`sihweb`)

### 2.1 Pervasive Mock Fallbacks Eradicated
- **Finding:** `ApiService` (`src/services/api.ts`) was intercepting 404s, 500s, and connection timeouts, silently swallowing the errors and returning perfectly formatted `MOCK_` arrays to the UI. The UI masked total backend failure behind a functional-looking interface.
- **Remediation:** `api.ts` was rewritten to rip out all `try { ... } catch { return MOCK_DATA; }` blocks. It now correctly throws errors to the caller (`if (!res.ok) throw new Error(...)`).

### 2.2 Component Error Boundaries & UI Honesty
- **Finding:** When `api.ts` began throwing real errors, several components lacked `.catch()` blocks and crashed or silently failed, rendering blank panels. Additionally, literal logical OR fallbacks (e.g. `{stats?.total_incidents || 1000}`) repopulated fake data.
- **Remediation:** 
  - Literal numeric fallbacks were replaced with `0` or `"N/A"`.
  - Proper asynchronous `try/catch` and explicit `error` state handling was injected into 9 critical components (`CaseDossier`, `CommandCenterView`, `GeospatialMapView`, `GraphVisualizerView`, `IncidentQueue`, `ThreeWayBenchmarkView`, `StreamingMonitorView`, `LandingSplash`, etc.).
  - The UI now mounts highly visible red error banners (e.g., `bg-red-50 text-red-700`) if backend services are offline.

### 2.3 Theatrical Progress Bar Removed
- **Finding:** `SimulationLab` featured an 8-stage progress timer that was purely theatrical (`setInterval`) and completely disconnected from the actual backend pipeline state.
- **Remediation:** The page was overhauled to remove the fake timer and 3D canvas, becoming a straightforward, honest wrapper around the `/api/predict/subgraph` inference endpoint.

---

## Conclusion
The repository has been successfully audited and fortified. Both the machine learning evaluation pipeline and the web application's data ingestion layer are now strictly honest. All patches have been executed, tested live, and committed to their respective branches.
