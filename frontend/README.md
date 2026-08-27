# I4C Cybercrime Predictive Interception Portal — React Frontend
## Smart India Hackathon 2026 | Problem Statement ID: 26184
**Ministry of Home Affairs — Indian Cyber Crime Coordination Centre (I4C)**

---

### How to Run the Frontend

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies** (already installed):
   ```bash
   npm install
   ```

3. **Start the Vite development server**:
   ```bash
   npm run dev
   ```
   > The React app will run at `http://localhost:5173`.

---

### Features & Capabilities

1. **🚨 Live NCRP / 1930 Complaint Ingestion & Instant Advance Forecast**:
   - Interactive intake form to log complaints with presets (Digital Arrest, Investment Fraud).
   - Ingests into FastAPI and outputs GraphSAGE risk probability, forecasted ATM exit terminal, and withdrawal window.
2. **🕸️ Multi-Hop Mule Chain Interactive Visualizer**:
   - Physics-based interactive network graph (Vis-Network) displaying Victim Account ➔ Layer 1 Mule ➔ Layer 2 Mule ➔ Physical ATM Exit with transfer amounts in ₹.
3. **🗺️ National Tactical Cash-Out Heatmap**:
   - Real-time surveillance grid across 15 Indian metropolitan cybercrime hubs.
4. **📋 Incident Triage Queue**:
   - 1,000 incident subgraphs with confidence tier badges and search.
5. **📑 Law Enforcement Actionable Freeze Dossier**:
   - Auto-generated Section 102 BNSS / Section 91 CrPC Bank Freeze Notice with 1-click Markdown and JSON export.
6. **⚙️ Decision Policy Tuner**:
   - Dynamic threshold ($\tau$) slider to simulate investigator caseload and precision-recall trade-offs.
