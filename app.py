import streamlit as st
import pandas as pd
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="BE 2026 QC System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Custom CSS for Modern UI
# -----------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #1a237e, #283593);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    .card-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Metrics styling */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #1a237e;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a237e;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #546e7a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #283593 0%, #3949ab 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 35, 126, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a237e !important;
        color: white !important;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    
    .status-good {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #c8e6c9;
    }
    
    .status-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffe0b2;
    }
    
    .status-error {
        background-color: #ffebee;
        color: #c62828;
        border: 1px solid #ffcdd2;
    }
    
    /* Progress bar */
    .progress-container {
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 8px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .progress-good {
        background: linear-gradient(90deg, #4caf50, #8bc34a);
    }
    
    .progress-warning {
        background: linear-gradient(90deg, #ff9800, #ffb74d);
    }
    
    .progress-error {
        background: linear-gradient(90deg, #f44336, #ef5350);
    }
    
    /* Icon styling */
    .icon-large {
        font-size: 2rem;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load prediction library
# -----------------------------------------------------
# For Streamlit Cloud compatibility
try:
    # First try relative path
    sys.path.append(".")
    from be_qc_lib_saved import predict_new
    MODEL_DIR = "./be_qc_models"
    LOOKUP = "./lookup"
except ImportError:
    # Fallback to original paths
    sys.path.append(r"D:\app ML QC2\app ML QC")
    from be_qc_lib_saved import predict_new
    MODEL_DIR = r"D:\app ML QC2\app ML QC\be_qc_models"
    LOOKUP = r"D:\app ML QC2\app ML QC\lookup"

# -----------------------------------------------------
# Load lookups (dependency)
# -----------------------------------------------------
try:
    df_hierarchy = pd.read_csv(f"{LOOKUP}/lookup_sektor_subsektor_msic.csv")
    df_nd = pd.read_csv(f"{LOOKUP}/lookup_negeri_daerah.csv")
except:
    st.warning("Lookup files not found. Using sample data.")
    # Create sample data for demo
    df_hierarchy = pd.DataFrame({
        'SEKTOR': ['Manufacturing', 'Services', 'Construction', 'Agriculture'],
        'SUBSEKTOR': ['Food Processing', 'IT Services', 'Residential', 'Crops'],
        'MSIC_5D': ['10101', '62010', '41001', '01111']
    })
    
    df_nd = pd.DataFrame({
        'NEGERI': ['Selangor', 'Kuala Lumpur', 'Johor', 'Penang'],
        'DAERAH': ['Petaling', 'Kuala Lumpur', 'Johor Bahru', 'Timur Laut']
    })

# -----------------------------------------------------
# Targets + features
# -----------------------------------------------------
TARGETS = ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "JUMLAH_PEKERJA"]

FEATURES = {
    "OUTPUT": {"num": ["JUMLAH_PEKERJA","HARTA_TETAP","GAJI_UPAH","OUTPUT"]},
    "INPUT": {"num": ["JUMLAH_PEKERJA","HARTA_TETAP","OUTPUT","INPUT"]},
    "NILAI_DITAMBAH": {"num": ["OUTPUT","INPUT","JUMLAH_PEKERJA","HARTA_TETAP","NILAI_DITAMBAH"]},
    "GAJI_UPAH": {"num": ["JUMLAH_PEKERJA","OUTPUT","HARTA_TETAP","GAJI_UPAH"]},
    "JUMLAH_PEKERJA": {"num": ["OUTPUT","INPUT","NILAI_DITAMBAH","GAJI_UPAH","HARTA_TETAP","JUMLAH_PEKERJA"]}
}

# -----------------------------------------------------
# UI Header
# -----------------------------------------------------
st.markdown('<h1 class="main-header">📊 BE 2026 Quality Control System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Powered Data Validation & Anomaly Detection</p>', unsafe_allow_html=True)

# Status bar
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-label">System Status</div><div class="metric-value" style="color: #4caf50;">● Online</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-label">Models Loaded</div><div class="metric-value">5</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-label">Ready for Analysis</div><div class="metric-value">✓</div></div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------
# MODE SELECTOR in a Card
# -----------------------------------------------------
st.markdown('<div class="card"><div class="card-header">🔧 Analysis Configuration</div></div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    mode = st.radio("**Select Analysis Mode**", ["Single Input", "Batch (CSV Upload)"], horizontal=False)
with col2:
    selected = st.selectbox("**Select Target Variable**", TARGETS, index=0)

st.markdown("---")

# =======================================================================
# MODE 1 — SINGLE INPUT
# =======================================================================
if mode == "Single Input":
    
    # Create two columns layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown('<div class="card"><div class="card-header">📝 Input Parameters</div></div>', unsafe_allow_html=True)
        
        user_input = {}
        feats = FEATURES[selected]

        # -------------------------------
        # DEPENDENCY DROPDOWNS
        # -------------------------------
        sektor_list = sorted(df_hierarchy["SEKTOR"].unique())
        sektor = st.selectbox("**SEKTOR**", sektor_list, key=f"{selected}_sektor")
        user_input["SEKTOR"] = sektor

        sub_opts = sorted(df_hierarchy[df_hierarchy["SEKTOR"] == sektor]["SUBSEKTOR"].unique())
        subsektor = st.selectbox("**SUBSEKTOR**", sub_opts, key=f"{selected}_subsektor")
        user_input["SUBSEKTOR"] = subsektor

        msic_opts = sorted(df_hierarchy[
            (df_hierarchy["SEKTOR"] == sektor) &
            (df_hierarchy["SUBSEKTOR"] == subsektor)
        ]["MSIC_5D"].unique())
        msic = st.selectbox("**MSIC 5D**", msic_opts, key=f"{selected}_msic")
        user_input["MSIC_5D"] = msic

        negeri_list = sorted(df_nd["NEGERI"].unique())
        negeri = st.selectbox("**NEGERI**", negeri_list, key=f"{selected}_negeri")
        user_input["NEGERI"] = negeri

        daerah_opts = sorted(df_nd[df_nd["NEGERI"] == negeri]["DAERAH"].unique())
        daerah = st.selectbox("**DAERAH**", daerah_opts, key=f"{selected}_daerah")
        user_input["DAERAH"] = daerah

        # numeric inputs
        st.markdown("---")
        st.markdown("#### 📊 Numeric Values")
        for col in feats["num"]:
            key = f"{selected}_num_{col}"
            if col == "JUMLAH_PEKERJA":
                user_input[col] = st.number_input(col, min_value=0, step=1, key=key)
            else:
                user_input[col] = st.number_input(col, min_value=0.0, format="%.2f", key=key)

        # Run button
        run = st.button(f"🚀 Run QC Analysis for {selected}", key=f"run_{selected}", type="primary")

    with right_col:
        # Results display area
        st.markdown('<div class="card"><div class="card-header">📈 Results Dashboard</div></div>', unsafe_allow_html=True)
        
        if run:
            with st.spinner("🔍 Analyzing data..."):
                df_input = pd.DataFrame([user_input])
                result = predict_new(df_input, out_dir=MODEL_DIR)

                # Display metrics in cards
                st.markdown("### 📊 Analysis Results")
                
                # Extract boundaries
                low_col = next((c for c in result.columns if "low" in c.lower() and selected.lower() in c.lower()), None)
                med_col = next((c for c in result.columns if "med" in c.lower() and selected.lower() in c.lower()), None)
                up_col  = next((c for c in result.columns if "up"  in c.lower() and selected.lower() in c.lower()), None)

                if low_col and med_col and up_col:
                    lb = float(result[low_col].iloc[0])
                    mb = float(result[med_col].iloc[0])
                    ub = float(result[up_col].iloc[0])
                    actual = float(user_input.get(selected, 0))

                    # Create metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Actual Value</div>
                            <div class="metric-value">{actual:,.2f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Predicted Median</div>
                            <div class="metric-value">{mb:,.2f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Lower Bound</div>
                            <div class="metric-value">{lb:,.2f}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'''
                        <div class="metric-card">
                            <div class="metric-label">Upper Bound</div>
                            <div class="metric-value">{ub:,.2f}</div>
                        </div>
                        ''', unsafe_allow_html=True)

                    # Status indicator
                    st.markdown("---")
                    st.markdown("### 📋 Quality Status")
                    
                    if actual < lb:
                        status = "UNDER-REPORTING"
                        status_class = "status-error"
                        explanation = "Value falls below expected range"
                        progress_class = "progress-error"
                        progress_width = 20
                    elif actual > ub:
                        status = "OVER-REPORTING"
                        status_class = "status-warning"
                        explanation = "Value exceeds expected range"
                        progress_class = "progress-warning"
                        progress_width = 80
                    else:
                        status = "WITHIN RANGE"
                        status_class = "status-good"
                        explanation = "Value is within expected parameters"
                        progress_class = "progress-good"
                        progress_width = 50

                    # Status display
                    col_status, col_explanation = st.columns([1, 3])
                    with col_status:
                        st.markdown(f'<div class="status-badge {status_class}">{status}</div>', unsafe_allow_html=True)
                    with col_explanation:
                        st.info(f"**Analysis:** {explanation}")

                    # Progress bar visualization
                    st.markdown(f'''
                    <div class="progress-container">
                        <div class="progress-bar {progress_class}" style="width: {progress_width}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666; margin-top: 5px;">
                        <span>Lower Bound</span>
                        <span>Expected Range</span>
                        <span>Upper Bound</span>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Visualization
                    fig = go.Figure()
                    
                    # Add range
                    fig.add_vrect(
                        x0=lb, x1=ub,
                        fillcolor="rgba(0, 150, 136, 0.2)",
                        layer="below",
                        line_width=0,
                        annotation_text="Expected Range",
                        annotation_position="top left"
                    )
                    
                    # Add lines
                    fig.add_vline(x=lb, line_dash="dash", line_color="#009688", 
                                annotation_text=f"Lower: {lb:,.2f}", 
                                annotation_position="top")
                    fig.add_vline(x=mb, line_color="#3f51b5", line_width=3,
                                annotation_text=f"Median: {mb:,.2f}")
                    fig.add_vline(x=ub, line_dash="dash", line_color="#009688",
                                annotation_text=f"Upper: {ub:,.2f}")
                    
                    # Add actual value
                    fig.add_trace(go.Scatter(
                        x=[actual], y=[0],
                        mode="markers+text",
                        marker=dict(
                            color="#ff4081" if actual < lb or actual > ub else "#4caf50",
                            size=20,
                            symbol="diamond"
                        ),
                        text=[f"Actual: {actual:,.2f}"],
                        textposition="top center",
                        name="Actual Value",
                        hovertemplate=f"<b>Actual Value</b><br>{actual:,.2f}<extra></extra>"
                    ))

                    fig.update_layout(
                        title=f"QC Analysis for {selected}",
                        xaxis_title=f"{selected} Value",
                        yaxis=dict(
                            showticklabels=False,
                            range=[-0.5, 0.5]
                        ),
                        height=300,
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Arial, sans-serif")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                    # Input values visualization
                    st.markdown("---")
                    st.markdown("### 📋 Input Summary")
                    
                    bar_df = pd.DataFrame({
                        "Parameter": feats["num"],
                        "Value": [user_input[v] for v in feats["num"]]
                    })
                    
                    fig_bar = px.bar(
                        bar_df, 
                        x="Parameter", 
                        y="Value", 
                        text="Value",
                        color="Value",
                        color_continuous_scale="Blues",
                        title="Input Values Used in Analysis"
                    )
                    fig_bar.update_traces(
                        texttemplate='%{y:,.2f}',
                        textposition='outside',
                        marker_line_color='rgb(8,48,107)',
                        marker_line_width=1.5
                    )
                    fig_bar.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Arial, sans-serif")
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Detailed results in expander
                    with st.expander("📄 View Detailed Results"):
                        selected_cols = [c for c in result.columns if selected.lower() in c.lower()]
                        st.dataframe(
                            result[selected_cols].style
                            .background_gradient(subset=[low_col, med_col, up_col], cmap='Blues')
                            .format("{:,.2f}")
                        )

# =======================================================================
# MODE 2 — BATCH INPUT (CSV UPLOAD)
# =======================================================================
else:
    st.markdown('<div class="card"><div class="card-header">📁 Batch Processing</div></div>', unsafe_allow_html=True)
    
    # Upload section
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "**Upload CSV File**", 
            type=["csv"],
            help="Upload your data file for batch analysis"
        )
    
    with col_info:
        st.markdown("""
        <div style="background: #f5f7fa; padding: 15px; border-radius: 10px; border-left: 4px solid #1a237e;">
        <h4 style="margin-top: 0; color: #1a237e;">📋 File Requirements</h4>
        <ul style="margin-bottom: 0;">
        <li>CSV format</li>
        <li>Include <code>NO_SIRI</code> column</li>
        <li>Include target variable</li>
        <li>Maximum size: 200MB</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        
        # Preview in a card
        st.markdown('<div class="card"><div class="card-header">🔍 Data Preview</div></div>', unsafe_allow_html=True)
        
        col_preview, col_stats = st.columns([2, 1])
        
        with col_preview:
            st.dataframe(df_batch.head(10).style.background_gradient(cmap='Blues'), use_container_width=True)
        
        with col_stats:
            st.markdown(f'''
            <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                <h4 style="color: #1a237e; margin-top: 0;">📊 File Statistics</h4>
                <p><strong>Records:</strong> {len(df_batch):,}</p>
                <p><strong>Columns:</strong> {len(df_batch.columns)}</p>
                <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                <p><strong>Target:</strong> {selected}</p>
            </div>
            ''', unsafe_allow_html=True)

        if st.button("🚀 Run Batch Analysis", type="primary"):
            with st.spinner("Processing batch data..."):
                # Run prediction
                result_batch = predict_new(df_batch, out_dir=MODEL_DIR)

                # ALWAYS include NO_SIRI
                if "NO_SIRI" in df_batch.columns:
                    result_batch["NO_SIRI"] = df_batch["NO_SIRI"]

                # Identify prediction columns
                low_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "low" in c.lower()), None)
                med_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "med" in c.lower()), None)
                up_col  = next((c for c in result_batch.columns if selected.lower() in c.lower() and "up"  in c.lower()), None)

                # Construct output table
                clean_df = pd.DataFrame()
                clean_df["NO_SIRI"] = df_batch["NO_SIRI"]
                clean_df[selected] = df_batch[selected]
                clean_df[f"{selected}_PRED_LOW"] = result_batch[low_col]
                clean_df[f"{selected}_PRED_MED"] = result_batch[med_col]
                clean_df[f"{selected}_PRED_UP"] = result_batch[up_col]

                # Compute flags
                flags = []
                for i in range(len(clean_df)):
                    actual = clean_df.iloc[i][selected]
                    lb = clean_df.iloc[i][f"{selected}_PRED_LOW"]
                    ub = clean_df.iloc[i][f"{selected}_PRED_UP"]
                    flags.append(actual < lb or actual > ub)
                
                clean_df[f"{selected}_FLAG"] = flags

                # Split data
                df_issue = clean_df[clean_df[f"{selected}_FLAG"] == True]
                df_ok = clean_df[clean_df[f"{selected}_FLAG"] == False]

                # Calculate statistics
                total = len(clean_df)
                total_issue = len(df_issue)
                total_ok = len(df_ok)
                pct_issue = round((total_issue / total) * 100, 2) if total > 0 else 0
                pct_ok = round((total_ok / total) * 100, 2) if total > 0 else 0

                # Results Dashboard
                st.markdown('<div class="card"><div class="card-header">📈 Batch Results Dashboard</div></div>', unsafe_allow_html=True)
                
                # Summary metrics
                st.markdown("### 📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Total Records</div>
                        <div class="metric-value">{total:,}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Issues Found</div>
                        <div class="metric-value" style="color: {"#f44336" if total_issue > 0 else "#4caf50"}">{total_issue}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">OK Records</div>
                        <div class="metric-value" style="color: #4caf50">{total_ok}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f'''
                    <div class="metric-card">
                        <div class="metric-label">Issue Rate</div>
                        <div class="metric-value">{pct_issue}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

                # Visualization
                st.markdown("---")
                st.markdown("### 📋 Distribution Overview")
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Pie chart
                    fig_pie = px.pie(
                        values=[total_issue, total_ok],
                        names=['With Issues', 'OK'],
                        title='Record Distribution',
                        color=['With Issues', 'OK'],
                        color_discrete_map={'With Issues': '#f44336', 'OK': '#4caf50'},
                        hole=0.4
                    )
                    fig_pie.update_traces(textinfo='percent+label', pull=[0.1, 0])
                    fig_pie.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_viz2:
                    # Bar chart
                    fig_bar = px.bar(
                        x=['Issues', 'OK'],
                        y=[total_issue, total_ok],
                        title='Record Count by Status',
                        color=['Issues', 'OK'],
                        color_discrete_map={'Issues': '#f44336', 'OK': '#4caf50'},
                        text=[total_issue, total_ok]
                    )
                    fig_bar.update_traces(texttemplate='%{y:,}', textposition='outside')
                    fig_bar.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Data tabs
                st.markdown("---")
                st.markdown("### 📄 Detailed Results")
                
                tab1, tab2, tab3 = st.tabs(["⚠️ Issues Detected", "✅ OK Records", "📋 All Results"])
                
                with tab1:
                    if df_issue.empty:
                        st.success("🎉 Excellent! No issues detected in your data.")
                        st.balloons()
                    else:
                        st.markdown(f"**Found {len(df_issue)} records with potential issues:**")
                        # Highlight issues
                        styled_df = df_issue.style.apply(
                            lambda x: ['background-color: #ffebee' if x.name == selected else '' for _ in x], 
                            axis=1
                        )
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Show issue breakdown
                        st.markdown("#### 📊 Issue Breakdown")
                        under_count = len(df_issue[df_issue[selected] < df_issue[f"{selected}_PRED_LOW"]])
                        over_count = len(df_issue[df_issue[selected] > df_issue[f"{selected}_PRED_UP"]])
                        
                        col_under, col_over = st.columns(2)
                        with col_under:
                            st.markdown(f'''
                            <div style="background: #fff3e0; padding: 15px; border-radius: 8px; border-left: 4px solid #ff9800;">
                                <h4 style="margin: 0 0 10px 0; color: #ef6c00;">⬇️ Under-Reporting</h4>
                                <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #ef6c00;">{under_count}</p>
                                <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #666;">Below lower bound</p>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        with col_over:
                            st.markdown(f'''
                            <div style="background: #ffebee; padding: 15px; border-radius: 8px; border-left: 4px solid #f44336;">
                                <h4 style="margin: 0 0 10px 0; color: #c62828;">⬆️ Over-Reporting</h4>
                                <p style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #c62828;">{over_count}</p>
                                <p style="margin: 5px 0 0 0; font-size: 0.9rem; color: #666;">Above upper bound</p>
                            </div>
                            ''', unsafe_allow_html=True)
                
                with tab2:
                    st.markdown(f"**{len(df_ok)} records are within expected ranges:**")
                    st.dataframe(df_ok.style.background_gradient(subset=[selected], cmap='Greens'), use_container_width=True)
                
                with tab3:
                    st.markdown("**Complete analysis results:**")
                    st.dataframe(clean_df, use_container_width=True)

                # Download section
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    st.download_button(
                        label=f"📥 Download Issues Only ({len(df_issue)} records)",
                        data=df_issue.to_csv(index=False).encode('utf-8'),
                        file_name=f"qc_issues_{selected}.csv",
                        mime="text/csv",
                        help="Download only records with quality issues"
                    )
                
                with col_dl2:
                    st.download_button(
                        label=f"📥 Download All Results ({total} records)",
                        data=clean_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"batch_analysis_{selected}.csv",
                        mime="text/csv",
                        help="Download complete analysis results"
                    )

                # Success message
                st.success(f"✅ Batch analysis completed successfully! Processed {total:,} records.")
                
                # Performance metrics
                with st.expander("📊 Performance Metrics"):
                    st.markdown(f"""
                    - **Records processed:** {total:,}
                    - **Processing time:** Instant
                    - **Issue detection rate:** {pct_issue}%
                    - **Data quality score:** {100 - pct_issue}%
                    - **Target variable:** {selected}
                    """)

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666; background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%); border-radius: 10px;">
    <h3 style="color: #1a237e; margin-bottom: 1rem;">BE 2026 Quality Control System</h3>
    <p style="margin-bottom: 0.5rem;">Powered by Machine Learning & Advanced Analytics</p>
    <p style="font-size: 0.9rem; margin: 0;">Version 2.0 • © 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
