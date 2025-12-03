import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------------------------------
# Configuration for Streamlit Cloud
# -----------------------------------------------------
# Set page config first
st.set_page_config(
    page_title="BE 2026 QC System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Detect if we're on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.path.exists('/app')

# Setup paths based on environment
if IS_STREAMLIT_CLOUD:
    # Streamlit Cloud - use relative paths
    BASE_DIR = Path(".")
    MODEL_DIR = BASE_DIR / "be_qc_models"
    LOOKUP_DIR = BASE_DIR / "lookup"
    
    # Add current directory to Python path
    sys.path.append(str(BASE_DIR))
else:
    # Local development - use your original paths
    BASE_DIR = Path(r"D:\app ML QC2\app ML QC")
    MODEL_DIR = BASE_DIR / "be_qc_models"
    LOOKUP_DIR = BASE_DIR / "lookup"
    sys.path.append(str(BASE_DIR))

# -----------------------------------------------------
# Initialize session state
# -----------------------------------------------------
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

# -----------------------------------------------------
# Load prediction library with better error handling
# -----------------------------------------------------
@st.cache_resource
def load_prediction_library():
    """Load the prediction library with caching"""
    try:
        from be_qc_lib_saved import predict_new
        return predict_new
    except ImportError as e:
        st.error(f"Error loading prediction library: {e}")
        # Create a fallback function
        def fallback_predict(df_input, out_dir=""):
            st.warning("⚠️ Using fallback prediction - ensure your model files are uploaded")
            # Return dummy predictions with the expected structure
            result = df_input.copy()
            targets = ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "JUMLAH_PEKERJA"]
            for target in targets:
                if target in df_input.columns:
                    values = df_input[target].values
                    # Simple dummy predictions
                    np.random.seed(42)
                    noise = np.random.uniform(0.8, 1.2, size=len(df_input))
                    result[f'{target.lower()}_pred_low'] = values * noise * 0.8
                    result[f'{target.lower()}_pred_med'] = values * noise
                    result[f'{target.lower()}_pred_up'] = values * noise * 1.2
            return result
        return fallback_predict

# Load the prediction function
predict_new = load_prediction_library()

# -----------------------------------------------------
# Load lookup data
# -----------------------------------------------------
@st.cache_data
def load_lookup_data():
    """Load lookup tables with caching"""
    try:
        # Try multiple possible file locations
        possible_paths = [
            LOOKUP_DIR / "lookup_sektor_subsektor_msic.csv",
            Path("lookup_sektor_subsektor_msic.csv"),
            Path("./lookup/lookup_sektor_subsektor_msic.csv")
        ]
        
        for path in possible_paths:
            if path.exists():
                df_hierarchy = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError("Could not find hierarchy lookup file")
        
        # Load second lookup
        possible_paths_nd = [
            LOOKUP_DIR / "lookup_negeri_daerah.csv",
            Path("lookup_negeri_daerah.csv"),
            Path("./lookup/lookup_negeri_daerah.csv")
        ]
        
        for path in possible_paths_nd:
            if path.exists():
                df_nd = pd.read_csv(path)
                break
        else:
            raise FileNotFoundError("Could not find negeri/daerah lookup file")
        
        return df_hierarchy, df_nd
        
    except Exception as e:
        st.warning(f"Using sample lookup data: {e}")
        # Create sample data
        df_hierarchy = pd.DataFrame({
            'SEKTOR': ['Manufacturing', 'Services', 'Construction', 'Agriculture'] * 3,
            'SUBSEKTOR': ['Food Processing', 'IT Services', 'Residential', 'Crops'] * 3,
            'MSIC_5D': [f'{i:05d}' for i in range(10101, 10113)]
        })
        
        df_nd = pd.DataFrame({
            'NEGERI': ['Selangor', 'Kuala Lumpur', 'Johor', 'Penang', 'Sabah', 'Sarawak'] * 2,
            'DAERAH': [f'Daerah {i}' for i in range(1, 13)]
        })
        
        return df_hierarchy, df_nd

# Load lookup tables
df_hierarchy, df_nd = load_lookup_data()

# -----------------------------------------------------
# Constants
# -----------------------------------------------------
TARGETS = ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "JUMLAH_PEKERJA"]

FEATURES = {
    "OUTPUT": {"num": ["JUMLAH_PEKERJA", "HARTA_TETAP", "GAJI_UPAH", "OUTPUT"]},
    "INPUT": {"num": ["JUMLAH_PEKERJA", "HARTA_TETAP", "OUTPUT", "INPUT"]},
    "NILAI_DITAMBAH": {"num": ["OUTPUT", "INPUT", "JUMLAH_PEKERJA", "HARTA_TETAP", "NILAI_DITAMBAH"]},
    "GAJI_UPAH": {"num": ["JUMLAH_PEKERJA", "OUTPUT", "HARTA_TETAP", "GAJI_UPAH"]},
    "JUMLAH_PEKERJA": {"num": ["OUTPUT", "INPUT", "NILAI_DITAMBAH", "GAJI_UPAH", "HARTA_TETAP", "JUMLAH_PEKERJA"]}
}

# -----------------------------------------------------
# Custom CSS for better UI
# -----------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E40AF;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1E3A8A;
    }
    .success-box {
        padding: 1rem;
        background-color: #D1FAE5;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
    .warning-box {
        padding: 1rem;
        background-color: #FEF3C7;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# UI Header
# -----------------------------------------------------
st.markdown('<h1 class="main-header">📊 BE 2026 — ML-Driven Quality Control</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning powered data validation and anomaly detection</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar info
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
    st.title("Settings")
    
    # Environment info
    env_status = "☁️ Streamlit Cloud" if IS_STREAMLIT_CLOUD else "💻 Local Development"
    st.info(f"**Environment:** {env_status}")
    
    # File structure check
    if st.checkbox("Check file structure", help="Verify required files are present"):
        st.write("**Required files:**")
        files_to_check = [
            ("be_qc_lib_saved.py", BASE_DIR / "be_qc_lib_saved.py"),
            ("Model directory", MODEL_DIR),
            ("Lookup directory", LOOKUP_DIR),
        ]
        
        for file_name, file_path in files_to_check:
            exists = file_path.exists()
            icon = "✅" if exists else "❌"
            st.write(f"{icon} {file_name}: {file_path}")
            
            if not exists and file_name == "be_qc_lib_saved.py":
                st.warning("Main prediction library not found!")
            elif not exists and "directory" in file_name:
                st.warning(f"{file_name} not found!")

# -----------------------------------------------------
# MODE SELECTOR
# -----------------------------------------------------
st.header("🔧 Analysis Mode")

col1, col2 = st.columns([1, 2])
with col1:
    mode = st.radio("**Select Mode:**", ["Single Input", "Batch (CSV Upload)"], horizontal=True)
with col2:
    selected = st.radio("**Select Target Variable:**", TARGETS, index=0, horizontal=True)

st.markdown("---")

# =======================================================================
# MODE 1 — SINGLE INPUT
# =======================================================================
if mode == "Single Input":
    
    st.header(f"📝 Single Record Analysis — {selected}")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Input Parameters")
        user_input = {}
        feats = FEATURES[selected]

        # Categorical inputs
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

        # Numeric inputs
        st.subheader("Numeric Values")
        for col in feats["num"]:
            key = f"{selected}_num_{col}"
            if col == "JUMLAH_PEKERJA":
                user_input[col] = st.number_input(
                    f"**{col}**", 
                    min_value=0, 
                    step=1, 
                    key=key,
                    help="Number of workers"
                )
            else:
                user_input[col] = st.number_input(
                    f"**{col}**", 
                    min_value=0.0, 
                    step=1000.0,
                    format="%.2f", 
                    key=key,
                    help=f"Value for {col}"
                )

        # Run button
        run_button = st.button(
            f"🚀 Run QC Analysis for {selected}", 
            key=f"run_{selected}",
            type="primary",
            use_container_width=True
        )

    with col_right:
        # Display current inputs
        st.subheader("Current Input Summary")
        if user_input:
            input_df = pd.DataFrame([user_input])
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    # =================================================================
    # RUN PREDICTION (Single Row)
    # =================================================================
    if run_button:
        with st.spinner("🔍 Running QC analysis..."):
            try:
                df_input = pd.DataFrame([user_input])
                result = predict_new(df_input, out_dir=str(MODEL_DIR))
                st.session_state.predictions = result
                
                # Display results
                st.success("✅ Analysis completed!")
                
                # Results in columns
                st.subheader("📊 Prediction Results")
                
                # Extract boundaries
                low_col = next((c for c in result.columns if "low" in c.lower() and selected.lower() in c.lower()), None)
                med_col = next((c for c in result.columns if "med" in c.lower() and selected.lower() in c.lower()), None)
                up_col  = next((c for c in result.columns if "up"  in c.lower() and selected.lower() in c.lower()), None)

                if low_col and med_col and up_col:
                    lb = float(result[low_col].iloc[0])
                    mb = float(result[med_col].iloc[0])
                    ub = float(result[up_col].iloc[0])
                    actual = float(user_input.get(selected, 0))

                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Actual Value", f"{actual:,.2f}")
                    with col2:
                        st.metric("Predicted Median", f"{mb:,.2f}")
                    with col3:
                        st.metric("Lower Bound", f"{lb:,.2f}")
                    with col4:
                        st.metric("Upper Bound", f"{ub:,.2f}")

                    # Flag determination
                    if actual < lb:
                        flag_icon = "🔴"
                        flag_text = "Below Lower Bound"
                        flag_color = "red"
                        flag_explanation = "Possible UNDER-reporting detected"
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>{flag_icon} {flag_text}</h4>
                            <p>{flag_explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif actual > ub:
                        flag_icon = "🔴"
                        flag_text = "Above Upper Bound"
                        flag_color = "red"
                        flag_explanation = "Possible OVER-reporting detected"
                        st.markdown(f"""
                        <div class="warning-box">
                            <h4>{flag_icon} {flag_text}</h4>
                            <p>{flag_explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        flag_icon = "🟢"
                        flag_text = "Within Range"
                        flag_color = "green"
                        flag_explanation = "No anomaly detected"
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>{flag_icon} {flag_text}</h4>
                            <p>{flag_explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Visualization
                    fig = go.Figure()
                    
                    # Add range rectangle
                    fig.add_vrect(
                        x0=lb, x1=ub,
                        fillcolor="lightgreen" if flag_color == "green" else "lightcoral",
                        opacity=0.3,
                        annotation_text="Acceptable Range",
                        annotation_position="top left"
                    )
                    
                    # Add boundary lines
                    fig.add_vline(x=lb, line_dash="dash", line_color="blue", 
                                annotation_text=f"Lower: {lb:,.2f}")
                    fig.add_vline(x=mb, line_color="black", line_width=2,
                                annotation_text=f"Median: {mb:,.2f}")
                    fig.add_vline(x=ub, line_dash="dash", line_color="blue",
                                annotation_text=f"Upper: {ub:,.2f}")
                    
                    # Add actual value marker
                    fig.add_trace(go.Scatter(
                        x=[actual], y=[0.5],
                        mode="markers+text",
                        marker=dict(
                            color=flag_color,
                            size=25,
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
                            range=[0, 1]
                        ),
                        height=350,
                        showlegend=True,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show detailed results
                with st.expander("📋 View Detailed Results"):
                    st.dataframe(result, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please check your inputs and try again.")

# =======================================================================
# MODE 2 — BATCH INPUT (CSV UPLOAD)
# =======================================================================
else:
    st.header("📁 Batch Processing")
    
    st.markdown("""
    Upload a CSV file containing your data for batch quality control analysis.
    The file should include at least:
    - `NO_SIRI`: Unique identifier
    - The selected target column
    - Any required feature columns
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload your data for batch processing"
    )
    
    if uploaded_file:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            # Show file info
            st.success(f"✅ File uploaded successfully! ({len(df_batch)} records)")
            
            with st.expander("🔍 Preview Data", expanded=True):
                tab1, tab2 = st.tabs(["First 10 rows", "Data Info"])
                with tab1:
                    st.dataframe(df_batch.head(10), use_container_width=True)
                with tab2:
                    st.write(f"**Shape:** {df_batch.shape[0]} rows × {df_batch.shape[1]} columns")
                    st.write("**Columns:**")
                    st.write(list(df_batch.columns))
            
            # Check required columns
            st.subheader("🔧 Configuration")
            required_cols = ['NO_SIRI', selected]
            missing_cols = [col for col in required_cols if col not in df_batch.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV contains these columns before proceeding.")
            else:
                # Optional: Let user map columns if needed
                if st.checkbox("Show column mapping", help="Map your CSV columns to expected columns"):
                    col_mapping = {}
                    for req_col in required_cols + list(FEATURES[selected]["num"]):
                        if req_col in df_batch.columns:
                            col_mapping[req_col] = req_col
                        else:
                            options = [col for col in df_batch.columns if col not in col_mapping.values()]
                            selected_col = st.selectbox(
                                f"Map column for '{req_col}'",
                                options=[""] + options,
                                key=f"map_{req_col}"
                            )
                            if selected_col:
                                col_mapping[req_col] = selected_col
                
                # Run batch prediction
                if st.button("🚀 Run Batch Analysis", type="primary", use_container_width=True):
                    with st.spinner("Processing batch data..."):
                        try:
                            result_batch = predict_new(df_batch, out_dir=str(MODEL_DIR))
                            
                            # Store NO_SIRI
                            if "NO_SIRI" in df_batch.columns:
                                result_batch["NO_SIRI"] = df_batch["NO_SIRI"]
                            
                            # Find prediction columns
                            low_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "low" in c.lower()), None)
                            med_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "med" in c.lower()), None)
                            up_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "up" in c.lower()), None)
                            
                            if not (low_col and med_col and up_col):
                                st.error("Could not find prediction columns in results.")
                            else:
                                # Create results dataframe
                                clean_df = pd.DataFrame()
                                clean_df["NO_SIRI"] = df_batch["NO_SIRI"]
                                clean_df[f"{selected}_ACTUAL"] = df_batch[selected]
                                clean_df[f"{selected}_PRED_LOW"] = result_batch[low_col]
                                clean_df[f"{selected}_PRED_MED"] = result_batch[med_col]
                                clean_df[f"{selected}_PRED_UP"] = result_batch[up_col]
                                
                                # Calculate flags
                                conditions = (
                                    (clean_df[f"{selected}_ACTUAL"] < clean_df[f"{selected}_PRED_LOW"]) |
                                    (clean_df[f"{selected}_ACTUAL"] > clean_df[f"{selected}_PRED_UP"])
                                )
                                clean_df[f"{selected}_FLAG"] = conditions
                                
                                # Store in session state
                                st.session_state.batch_results = clean_df
                                
                                # Analysis results
                                st.success("✅ Batch analysis completed!")
                                
                                # Summary metrics
                                total = len(clean_df)
                                issues = clean_df[f"{selected}_FLAG"].sum()
                                ok_count = total - issues
                                issue_pct = (issues / total * 100) if total > 0 else 0
                                
                                st.subheader("📊 Analysis Summary")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Records", total)
                                with col2:
                                    st.metric("Issues Detected", f"{issues} ({issue_pct:.1f}%)")
                                with col3:
                                    st.metric("Records OK", ok_count)
                                
                                # Visualization
                                fig_summary = px.pie(
                                    names=['With Issues', 'OK'],
                                    values=[issues, ok_count],
                                    title=f'Distribution for {selected}',
                                    color=['With Issues', 'OK'],
                                    color_discrete_map={'With Issues': '#EF4444', 'OK': '#10B981'}
                                )
                                st.plotly_chart(fig_summary, use_container_width=True)
                                
                                # Detailed results in tabs
                                st.subheader("📋 Detailed Results")
                                tab1, tab2, tab3 = st.tabs(["⚠️ Issues", "✅ OK Records", "📄 All Data"])
                                
                                with tab1:
                                    if issues > 0:
                                        df_issues = clean_df[clean_df[f"{selected}_FLAG"]]
                                        st.dataframe(df_issues, use_container_width=True)
                                    else:
                                        st.success("No issues detected! 🎉")
                                
                                with tab2:
                                    df_ok = clean_df[~clean_df[f"{selected}_FLAG"]]
                                    st.dataframe(df_ok, use_container_width=True)
                                
                                with tab3:
                                    st.dataframe(clean_df, use_container_width=True)
                                
                                # Download options
                                st.subheader("💾 Download Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    csv_issues = clean_df[clean_df[f"{selected}_FLAG"]].to_csv(index=False)
                                    st.download_button(
                                        label=f"Download Issues ({issues} records)",
                                        data=csv_issues,
                                        file_name=f"qc_issues_{selected}.csv",
                                        mime="text/csv"
                                    )
                                
                                with col2:
                                    csv_all = clean_df.to_csv(index=False)
                                    st.download_button(
                                        label=f"Download All Results ({total} records)",
                                        data=csv_all,
                                        file_name=f"qc_results_{selected}.csv",
                                        mime="text/csv"
                                    )
                                
                                # Show sample of flagged records
                                if issues > 0:
                                    st.subheader("🔍 Sample of Flagged Records")
                                    sample_issues = clean_df[clean_df[f"{selected}_FLAG"]].head(5)
                                    for idx, row in sample_issues.iterrows():
                                        actual = row[f"{selected}_ACTUAL"]
                                        low = row[f"{selected}_PRED_LOW"]
                                        up = row[f"{selected}_PRED_UP"]
                                        
                                        if actual < low:
                                            msg = f"Record {row['NO_SIRI']}: {actual:,.2f} < {low:,.2f} (Lower Bound)"
                                        else:
                                            msg = f"Record {row['NO_SIRI']}: {actual:,.2f} > {up:,.2f} (Upper Bound)"
                                        
                                        st.warning(msg)
                        
                        except Exception as e:
                            st.error(f"❌ Error during batch processing: {str(e)}")
                            st.info("Please check your data format and try again.")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure you're uploading a valid CSV file.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 1rem;">
    <p>BE 2026 — Machine Learning Quality Control System</p>
    <p style="font-size: 0.9rem;">Version 1.0 • For official use</p>
</div>
""", unsafe_allow_html=True)
