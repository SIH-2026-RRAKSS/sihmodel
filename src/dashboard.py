"""
Stage 10: Dual-Dataset Tactical Intelligence Dashboard
======================================================
Interactive Streamlit UI supporting both:
1. Dataset A: Synthetic Domestic Cybercrime Subgraphs (with GPS/ATMs)
2. Dataset B: IBM AML Multi-Bank Ledger Benchmark (Pure Bank-to-Bank Rails)

GUARDRAIL COMPLIANCE:
- Explicit Dataset Selector prevents blending metrics.
- Geospatial Map is enabled ONLY for Synthetic data (explicitly disabled with notice for IBM).
- Separate, honest Recall Reality panels rendered per dataset.
- Auth / RBAC explicitly labeled as STUBBED.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Page config
st.set_page_config(
    page_title="Mule-Chain Predictive Intelligence Triage",
    page_icon="???",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Data Loaders (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_all_datasets():
    data_dir = Path("data")
    
    # Dataset A (Synthetic)
    df_comp_a = pd.read_csv(data_dir / "complaints.csv") if (data_dir / "complaints.csv").exists() else pd.DataFrame()
    df_tiers_a = pd.read_csv(data_dir / "confidence_tiers.csv") if (data_dir / "confidence_tiers.csv").exists() else pd.DataFrame()
    df_locs_a = pd.read_csv(data_dir / "entity_locations.csv") if (data_dir / "entity_locations.csv").exists() else pd.DataFrame()
    df_policy_a = pd.read_csv(data_dir / "threshold_policy_analysis.csv") if (data_dir / "threshold_policy_analysis.csv").exists() else pd.DataFrame()
    df_tier_eval_a = pd.read_csv(data_dir / "confidence_tier_evaluation.csv") if (data_dir / "confidence_tier_evaluation.csv").exists() else pd.DataFrame()
    
    exp_a = {}
    if (data_dir / "explainability_examples.json").exists():
        with open(data_dir / "explainability_examples.json", "r", encoding="utf-8") as f:
            exp_a = json.load(f)
            
    # Dataset B (IBM AML)
    df_summary_b = pd.read_csv(data_dir / "ibm_graph_summary.csv") if (data_dir / "ibm_graph_summary.csv").exists() else pd.DataFrame()
    df_tiers_b = pd.read_csv(data_dir / "ibm_confidence_tiers.csv") if (data_dir / "ibm_confidence_tiers.csv").exists() else pd.DataFrame()
    df_policy_b = pd.read_csv(data_dir / "ibm_threshold_policy_analysis.csv") if (data_dir / "ibm_threshold_policy_analysis.csv").exists() else pd.DataFrame()
    df_tier_eval_b = pd.read_csv(data_dir / "ibm_confidence_tier_evaluation.csv") if (data_dir / "ibm_confidence_tier_evaluation.csv").exists() else pd.DataFrame()
    
    exp_b = {}
    if (data_dir / "ibm_explainability_examples.json").exists():
        with open(data_dir / "ibm_explainability_examples.json", "r", encoding="utf-8") as f:
            exp_b = json.load(f)
            
    # Three-way comparison
    df_three_way = pd.read_csv(data_dir / "three_way_benchmark_comparison.csv") if (data_dir / "three_way_benchmark_comparison.csv").exists() else pd.DataFrame()
    
    return {
        "A": (df_comp_a, df_tiers_a, df_locs_a, df_policy_a, df_tier_eval_a, exp_a),
        "B": (df_summary_b, df_tiers_b, pd.DataFrame(), df_policy_b, df_tier_eval_b, exp_b),
        "three_way": df_three_way
    }

datasets = load_all_datasets()

# -----------------------------------------------------------------------------
# Top Banner & System Scope Disclaimer
# -----------------------------------------------------------------------------
st.title("??? Multi-Dataset Cybercrime Triage & Laundering Intelligence Platform")
st.warning(
    "?? **Operational Scope Notice**: This dashboard is a **retrospective, post-complaint analytical triage prototype**. "
    "It is **NOT a real-time live transaction stream monitoring system**. "
    "Authentication and Role-Based Access Control (RBAC) are **STUBBED** for development demonstration."
)

# -----------------------------------------------------------------------------
# Sidebar: Dataset Selector & Auth Stub
# -----------------------------------------------------------------------------
st.sidebar.header("??? Active Dataset Selection")
active_dataset = st.sidebar.radio(
    "Choose Evaluation Corpus:",
    ["Dataset A: Synthetic Domestic Prototype (1,000 Incidents)",
     "Dataset B: IBM AML Multi-Bank Benchmark (1,000 Subgraphs)"],
    index=0
)
is_synthetic = "Dataset A" in active_dataset

st.sidebar.markdown("---")
st.sidebar.header("?? User Access & Configuration")
st.sidebar.info("?? **User Role**: `Investigator / FIU Analyst`\n\n??? **RBAC Status**: `STUBBED (Dev Mode)`")

# -----------------------------------------------------------------------------
# Render Active Dataset View
# -----------------------------------------------------------------------------
if is_synthetic:
    df_comp, df_tiers, df_locations, df_policy, df_tier_eval, exp_dict = datasets["A"]
    
    st.sidebar.markdown("---")
    st.sidebar.header("?? Alert Threshold Policy Dial (Item 9 - Synthetic)")
    threshold_val = st.sidebar.slider("Decision Cutoff (tau)", 0.10, 0.90, 0.50, 0.05)
    
    # Reality Panel Synthetic
    with st.container():
        st.markdown("### ?? Dataset A (Synthetic) Recall & Tier Health Audit (N = 1,000)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Suspicious Recall", "48.92%", "91 / 186 Flagged")
        c2.metric("NORMAL Miss Rate", "51.08%", "95 FNs in NORMAL Tier", delta_color="inverse")
        c3.metric("HIGH_CONF Precision", "91.00%", "91 TP / 9 FP")
        c4.metric("MEDIUM_CONF Precision", "0.00%", "?? 18 FP / 0 TP (Noise)", delta_color="inverse")
        
        st.info("?? **Synthetic Tradeoff**: Classifier achieves 97.3% raw recall, but conservative multi-signal tiering discards 51% to guarantee 91% alert precision.")

else:
    df_summary, df_tiers, _, df_policy, df_tier_eval, exp_dict = datasets["B"]
    
    st.sidebar.markdown("---")
    st.sidebar.header("?? Alert Threshold Policy Dial (Item 9 - IBM AML)")
    threshold_val = st.sidebar.slider("Decision Cutoff (tau)", 0.10, 0.90, 0.50, 0.05)
    
    # Reality Panel IBM
    with st.container():
        st.markdown("### ?? Dataset B (IBM AML) Recall & Tier Health Audit (N = 1,000 Subgraphs)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HIGH_CONF Capture", "53.87%", "160 / 297 Laundering")
        c2.metric("NORMAL Miss Rate", "7.74%", "23 FNs in NORMAL Tier", delta_color="inverse")
        c3.metric("HIGH_CONF Precision", "63.49%", "160 TP / 92 FP")
        c4.metric("MEDIUM_CONF Precision", "49.35%", "114 TP / 117 FP")
        
        st.info("?? **IBM AML Reality Check**: Multi-bank payment rails exhibit higher structural complexity. Zero reference-pattern circularity.")

st.markdown("---")

# -----------------------------------------------------------------------------
# Tabs: 1. Incident Queue | 2. Geospatial Map (Synthetic Only) | 3. Policy & 3-Way Benchmark
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["?? Incident Triage Queue & Dossier", "??? Geospatial Cash-Out Map", "?? Threshold Policy & Three-Way Benchmark"])

# TAB 1: Incident Queue & Drill-Down
with tab1:
    st.subheader(f"?? Retrospective Incident Queue ({'Dataset A: Synthetic' if is_synthetic else 'Dataset B: IBM AML'})")
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        tier_filter = st.selectbox(
            "Filter by Confidence Tier:",
            ["ALL", "HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "NORMAL"]
        )
    with col_f2:
        id_col = "complaint_id" if is_synthetic else "subgraph_id"
        search_query = st.text_input("Search by ID:", "")
        
    df_view = df_tiers.copy() if not df_tiers.empty else pd.DataFrame()
    if tier_filter != "ALL" and not df_view.empty:
        df_view = df_view[df_view["confidence_tier"] == tier_filter]
    if search_query and not df_view.empty:
        df_view = df_view[df_view[id_col].str.contains(search_query, case=False, na=False)]
        
    st.dataframe(df_view.head(50), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("?? Case Dossier & Rule-Based Evidence Panel (Item 8)")
    
    selected_id = st.selectbox(
        "Select Incident to Inspect Dossier:",
        df_view[id_col].head(20).tolist() if not df_view.empty else []
    )
    
    if selected_id and selected_id in exp_dict:
        case_data = exp_dict[selected_id]
        
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            st.markdown(f"**Incident/Subgraph ID**: `{selected_id}`")
            st.markdown(f"**Root Financial Account**: `{case_data.get('incident_entity_id') or case_data.get('seed_account')}`")
        with cm2:
            st.markdown(f"**Model Risk Probability**: `{case_data.get('graphsage_risk_probability') or case_data.get('risk_probability'):.4f}`")
            st.info(f"**Confidence Tier**: `{case_data['confidence_tier']}`")
        with cm3:
            term_details = case_data.get("top_terminal_details", {})
            st.markdown(f"**Terminal Exit Type**: `{term_details.get('terminal_type', 'Physical ATM')}`")
            
        st.markdown("#### ?? Executive Summary")
        st.info(f"?? *\"{case_data['executive_summary']}\"*")
        
        st.markdown("#### ?? Observable Graph Evidence Bullets")
        for idx, bullet in enumerate(case_data["investigative_evidence_bullets"], 1):
            st.markdown(f"**{idx}.** {bullet}")
            
        if term_details:
            st.markdown("#### ?? Terminal Flow Exit Rationale")
            st.caption(term_details.get("rationale", ""))

# TAB 2: Geospatial Map
with tab2:
    st.subheader("🗺️ Physical Cash-Out Terminals & Coordinate Mapping")
    if is_synthetic and not df_locations.empty:
        df_map = df_locations.copy().rename(columns={"latitude": "lat", "longitude": "lon"}).dropna(subset=["lat", "lon"])
        st.map(df_map, latitude="lat", longitude="lon", size=20, color="#FF4B4B")
        
        st.markdown("##### 📍 Monitored Financial Entity Coordinates (Sample Table)")
        st.dataframe(df_locations.head(15), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ **Geospatial Mapping Disabled for IBM AML Dataset**: IBM ledger data contains bank-to-bank electronic transfers without physical GPS coordinates or ATM hardware IDs. Per Guardrail #1, no synthetic coordinates are fabricated.")

# TAB 3: Policy & 3-Way Benchmark
with tab3:
    st.subheader("?? Operational Precision-Recall Policy Curve")
    if not df_policy.empty:
        st.dataframe(df_policy, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    st.subheader("?? Global Three-Way Multi-Dataset Architecture Benchmark")
    st.caption("Standardized GraphSAGE vs XGBoost comparison across all three evaluated datasets.")
    
    if not datasets["three_way"].empty:
        st.dataframe(datasets["three_way"], use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Mule-Chain Detection Framework | Multi-Dataset Scoped Rebuild Architecture")
