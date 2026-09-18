# Production Guide: Rules of Engagement

This guide documents how to safely make future changes to the SIH AML Predictive Intelligence codebase (both `sihmodel` backend and `sihweb` frontend). 
**This is not a generic best-practices doc.** Every rule below traces back to a specific, critical failure mode found and fixed during the initial architectural audit.

---

## 1. Non-Negotiable Rules

* **Never let early-stopping/checkpoint-selection evaluate against the test set.**
  * *Citation:* `graphsage_classifier.py` and `ibm_graphsage_classifier.py` leakage. Test-set leakage drove the early-stopping metric, causing a severe metric distortion. Fixing it dropped GraphSAGE's apparent performance from a falsely inflated 77.70% down to 73.27% (effectively tying the baseline) on the IBM dataset.
* **Never add a silent fallback on an API or model failure path.**
  * *Citation:* The 64% dummy-graph fallback bug in `streaming_engine.py` (which hard-capped model performance by silently substituting dummy data during tensor conversion errors), and the pervasive `try/catch -> return MOCK_DATA` pattern found in `sihweb/src/services/api.ts` (which lied to the user by showing 100% healthy dashboards when the backend was entirely offline).
* **Never use a bare `||` / `??` numeric/string literal default on live system state.**
  * *Citation:* The `stats?.total_incidents_monitored || 1000` pattern in the frontend. Replacing it with `|| 0` was still unsafe without a paired error state, as it masked critical backend outages by presenting an empty operational state rather than a failure state.
* **Any new API method must throw on `!res.ok`, and callers must render a distinct, visible error state.**
  * *Citation:* The 5 "silent crasher" frontend components that failed to render when data was missing, and the `IncidentQueue` dead-code error state that was never actually triggered because `api.ts` was swallowing the errors.
* **Any new relative file path must resolve via absolute constants (e.g., `ROOT_DIR / MODELS_DIR`).**
  * *Citation:* The CWD-dependent path bug in `ibm_graphsage_classifier.py` and `streaming_partial_window_evaluation.py` that caused "file not found" crashes when the scripts were executed from outside their immediate directory.
* **Any new user input field must be validated both client-side and server-side.**
  * *Citation:* The entity-ID vs search-query validator split (`InputValidator.ts`). A generic validator failed because valid search queries permit different character sets than strict backend Entity IDs. One generic validator does not fit all.
* **Never bulk-edit TSX/Python files with a single regex script without an immediate build/execution check.**
  * *Citation:* Two separate incidents: Regex patching broke the ML training loops (`val_loader` undefined scope error) and caused Vite build crashes (missing template literal backticks `` ` `` stripped by the shell in `api.ts`).
* **Any script that writes to a file also read by `api.py` or the dashboard needs that dependency documented, or the write should be removed/redirected.**
  * *Citation:* The benchmark CSV clobbering bug. The master presentation file `data/three_way_benchmark_comparison.csv` was silently overwritten with stale numbers by a side-effect output block at the end of `ibm_graphsage_classifier.py` during a routine training run, causing the API to serve regressed benchmark metrics. Files must not be treated as a source of truth by one part of the system while another treats them as disposable intermediate output.
* **Never report a fix as "done" without running it and pasting raw output.**
  * *Citation:* Describing what code "should" do is not verification. This audit required live curls, raw build logs, and confusion matrices to expose discrepancies between the code's intent and reality.
* **Never push code to origin or deployment branches without explicitly presenting the changes and confirming the outcomes with the user first.**
  * *Citation:* A previous automated push merged architectural fixes directly into the live `Deployment` pipeline immediately after a training run finished, without allowing the user to review the final benchmark outcomes, F1 jumps, or test the live behavior. This bypasses the mandatory user-approval checkpoint.

---

## 2. Pre-Merge Checklist for Any Model Change

- [ ] **Data Splits:** Does the change introduce or modify a train/val/test split? 
  * *Action:* Show raw TP/FP/FN/TN for at least 2 seeds. Confirm split ID overlap with any existing reference split, and confirm *no* evaluation loop touches the `test_loader` before final scoring.
- [ ] **Checkpoints:** Does it touch checkpoint saving/loading?
  * *Action:* Confirm weight fingerprints (hash + first N values) are distinct across seeds/runs to ensure it isn't repeatedly loading the same stale weights.
- [ ] **Benchmarks:** Does it change a benchmark number that any frontend endpoint serves?
  * *Action:* Confirm `/api/benchmarks/three_way` (or equivalent) is actually re-synced. Check whether the endpoint reads from a live computation, a CSV, or a hardcoded dict, and update the specific source of truth.
- [ ] **Significance:** Does it claim a model performance improvement?
  * *Action:* Run a paired significance test (p-value calculation, not just a point-estimate comparison) before claiming one model/method beats another.

---

## 3. Pre-Merge Checklist for Any Web App Change

- [ ] **Fetching:** Does this component fetch data?
  * *Action:* Confirm the API method throws on failure, the calling component has a *visible* (not just state-tracked) error UI, and the error UI is explicitly distinguishable from a legitimate empty/zero state.
- [ ] **Placeholders:** Does this introduce a new hardcoded/demo/placeholder value?
  * *Action:* It must be either removed, wired to a real endpoint, or explicitly labeled in the UI as static/reference data. (*Citation:* Follow the `PolicySimulatorView` amber disclaimer banner pattern).
- [ ] **Build Check:** Did you modify TSX/TS files?
  * *Action:* Run `npm run build` and confirm zero errors before considering the change complete.
- [ ] **Backend Sync:** Are you adding a new backend-dependent feature?
  * *Action:* Confirm the corresponding `api.py` route *actually exists*. Do not assume; grep for it and curl it live from the terminal.

---

## 4. API Contract Process (Mitigating Silent Drift)

Given the current lack of end-to-end contract testing, we must mitigate silent frontend/backend drift (which caused the `/api/entities/locations` missing-route bug and the leaky-benchmark-CSV format mismatch).

**The Protocol:**
Create a lightweight Python smoke-test script (`tests/smoke_test_api.py`) that boots the `uvicorn` server, curls every endpoint defined in `src/services/api.ts`, and asserts two things:
1. HTTP 200 OK.
2. The top-level JSON keys returned exactly match a hardcoded list of expected keys derived from the TypeScript interfaces.

This is the lowest-effort safety net that requires zero complex shared schema tooling (like OpenAPI/tRPC generators) while catching 99% of naming drift and missing route errors.

---

## 5. Known Open Items (Resolve Before Major Feature Work)

1. **API Contract Drift:** Currently unmonitored. Needs the smoke-test script implemented.
2. **Missing Test Infrastructure:** Vitest (frontend) and PyTest (backend) are not configured. 
3. **404/Entity-Eviction Logging Gap:** The backend silently handles some entity evictions without robust telemetry.
4. **Live-Flagging Extension (Partial-Window Benchmark):** Needs deeper verification to ensure partial graph snapshots don't degrade the GNN latency.
5. **Dataset C (Elliptic Dataset):** Has *never* been audited for leakage in this session. Treat its benchmark metrics with high suspicion until subjected to the same strict train/val/test rewrite as the IBM dataset.
6. **0.5-Default Risk Masking:** Entities without calculated scores default to `0.5` risk in `/api/entities/locations`, effectively hiding unscored/failed nodes in the middle of the spectrum rather than flagging them.

---

## 6. How to Verify a Fix Is Actually Fixed (The Meta-Rule)

A fix is not a fix until it is proven in reality. Use the following verification methods:
* **Model Metric Claims:** $\rightarrow$ Raw counts (TP/FP/FN/TN) + P-value significance test.
* **API Endpoint Changes:** $\rightarrow$ Live `curl` output showing the raw JSON payload.
* **Frontend Error Handling:** $\rightarrow$ Forced failure (kill the backend) + visual inspection of the DOM error boundary.
* **Path/Config Changes:** $\rightarrow$ Execute the script from a different working directory (e.g., `cd /tmp && python /path/to/script.py`) to prove relative path safety.
