import streamlit as st
import pandas as pd
import sys
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------
# Configuration for Streamlit Cloud
# -----------------------------------------------------
# For Streamlit Cloud, we need to handle paths differently
# Assuming you'll upload your model files to Streamlit Cloud

# Check if we're running on Streamlit Cloud
is_streamlit_cloud = os.path.exists('/app')

# Set base paths based on environment
if is_streamlit_cloud:
    # Streamlit Cloud paths
    BASE_DIR = "."
    MODEL_DIR = "./be_qc_models"
    LOOKUP_DIR = "./lookup"
else:
    # Local development paths
    BASE_DIR = r"D:\app ML QC2\app ML QC"
    MODEL_DIR = r"D:\app ML QC2\app ML QC\be_qc_models"
    LOOKUP_DIR = r"D:\app ML QC2\app ML QC\lookup"

# -----------------------------------------------------
# Load prediction library - need to handle differently for cloud
# -----------------------------------------------------
try:
    # Try to import from a local module first
    sys.path.append(BASE_DIR)
    from be_qc_lib_saved import predict_new
    st.success("✅ Prediction library loaded successfully!")
except ImportError as e:
    st.error(f"❌ Error loading prediction library: {e}")
    st.info("Please ensure 'be_qc_lib_saved.py' is in the correct location.")
    
    # Create a dummy function for demo purposes
    def predict_new(df_input, out_dir=""):
        st.warning("⚠️ Using dummy prediction function - replace with actual model")
        # Create dummy predictions for demo
        import numpy as np
        result = df_input.copy()
        for col in ['OUTPUT', 'INPUT', 'NILAI_DITAMBAH', 'GAJI_UPAH', 'JUMLAH_PEKERJA']:
            if col in df_input.columns:
                # Create dummy predictions
                value = df_input[col].values[0] if len(df_input) == 1 else df_input[col].values
                noise = np.random.uniform(0.8, 1.2, size=len(df_input))
                result[f'{col.lower()}_pred_low'] = value * noise * 0.8
                result[f'{col.lower()}_pred_med'] = value * noise
                result[f'{col.lower()}_pred_up'] = value * noise * 1.2
        return result

# -----------------------------------------------------
# Load lookups (dependency)
# -----------------------------------------------------
try:
    # Try to load from local files first
    df_hierarchy = pd.read_csv(f"{LOOKUP_DIR}/lookup_sektor_subsektor_msic.csv")
    df_nd = pd.read_csv(f"{LOOKUP_DIR}/lookup_negeri_daerah.csv")
    st.success("✅ Lookup tables loaded successfully!")
except FileNotFoundError as e:
    st.warning(f"⚠️ Lookup files not found at {LOOKUP_DIR}. Creating sample data.")
    
    # Create sample lookup data for demonstration
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
# Session state initialization
# -----------------------------------------------------
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# -----------------------------------------------------
# Targets + features
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
# UI Header
# -----------------------------------------------------
st.title("📊 BE 2026 — ML-Driven Quality Control")
st.markdown("---")

# Add environment indicator
env_status = "☁️ Streamlit Cloud" if is_streamlit_cloud else "💻 Local Development"
st.sidebar.markdown(f"**Environment:** {env_status}")

# -----------------------------------------------------
# MODE SELECTOR
# -----------------------------------------------------
mode = st.radio("Select Mode:", ["Single Input", "Batch (CSV Upload)"], horizontal=True)

selected = st.radio("Select Target Variable:", TARGETS, index=0, horizontal=True)

# =======================================================================
# MODE 1 — SINGLE INPUT
# =======================================================================
if mode == "Single Input":
    
    st.sidebar.header(f"📝 Input Data — {selected}")
    user_input = {}
    feats = FEATURES[selected]

    # -------------------------------
    # DEPENDENCY DROPDOWNS
    # -------------------------------
    sektor_list = sorted(df_hierarchy["SEKTOR"].unique())
    sektor = st.sidebar.selectbox("SEKTOR", sektor_list, key=f"{selected}_sektor")
    user_input["SEKTOR"] = sektor

    sub_opts = sorted(df_hierarchy[df_hierarchy["SEKTOR"] == sektor]["SUBSEKTOR"].unique())
    subsektor = st.sidebar.selectbox("SUBSEKTOR", sub_opts, key=f"{selected}_subsektor")
    user_input["SUBSEKTOR"] = subsektor

    msic_opts = sorted(df_hierarchy[
        (df_hierarchy["SEKTOR"] == sektor) &
        (df_hierarchy["SUBSEKTOR"] == subsektor)
    ]["MSIC_5D"].unique())
    msic = st.sidebar.selectbox("MSIC 5D", msic_opts, key=f"{selected}_msic")
    user_input["MSIC_5D"] = msic

    negeri_list = sorted(df_nd["NEGERI"].unique())
    negeri = st.sidebar.selectbox("NEGERI", negeri_list, key=f"{selected}_negeri")
    user_input["NEGERI"] = negeri

    daerah_opts = sorted(df_nd[df_nd["NEGERI"] == negeri]["DAERAH"].unique())
    daerah = st.sidebar.selectbox("DAERAH", daerah_opts, key=f"{selected}_daerah")
    user_input["DAERAH"] = daerah

    # numeric inputs
    st.sidebar.markdown("### Numeric Inputs")
    for col in feats["num"]:
        key = f"{selected}_num_{col}"
        if col == "JUMLAH_PEKERJA":
            user_input[col] = st.sidebar.number_input(
                col, 
                min_value=0, 
                step=1, 
                key=key,
                help=f"Enter value for {col}"
            )
        else:
            user_input[col] = st.sidebar.number_input(
                col, 
                min_value=0.0, 
                format="%.2f", 
                key=key,
                help=f"Enter value for {col}"
            )

    run = st.sidebar.button(f"🚀 Run QC Analysis for {selected}", key=f"run_{selected}")

    # =================================================================
    # RUN PREDICTION (Single Row)
    # =================================================================
    if run:
        with st.spinner("Running QC analysis..."):
            df_input = pd.DataFrame([user_input])
            
            try:
                result = predict_new(df_input, out_dir=MODEL_DIR)
                st.session_state.predictions = result
                
                # filter selection
                selected_cols = [c for c in result.columns if selected.lower() in c.lower()]
                
                st.subheader("📈 Prediction Results")
                st.dataframe(result[selected_cols], use_container_width=True)

                # Extract boundaries
                low_col = next((c for c in result.columns if "low" in c.lower() and selected.lower() in c.lower()), None)
                med_col = next((c for c in result.columns if "med" in c.lower() and selected.lower() in c.lower()), None)
                up_col  = next((c for c in result.columns if "up"  in c.lower() and selected.lower() in c.lower()), None)

                if low_col and med_col and up_col:
                    lb = float(result[low_col].iloc[0])
                    mb = float(result[med_col].iloc[0])
                    ub = float(result[up_col].iloc[0])
                    actual = float(user_input.get(selected, 0))

                    # Create result card
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual Value", f"{actual:,.2f}")
                    with col2:
                        st.metric("Predicted Median", f"{mb:,.2f}")
                    with col3:
                        st.metric("Range", f"{lb:,.2f} - {ub:,.2f}")

                    # Determine flag status
                    if actual < lb:
                        flag_color = "red"
                        explanation = "🔴 **Below Lower Bound** — Possible UNDER-reporting"
                        st.error(explanation)
                    elif actual > ub:
                        flag_color = "red"
                        explanation = "🔴 **Above Upper Bound** — Possible OVER-reporting"
                        st.error(explanation)
                    else:
                        flag_color = "green"
                        explanation = "🟢 **Within Model Range** — No anomaly detected"
                        st.success(explanation)

                    # Create visualization
                    fig = go.Figure()
                    fig.add_vrect(x0=lb, x1=ub, fillcolor="lightblue", opacity=0.3, 
                                annotation_text="Acceptable Range", annotation_position="top left")
                    fig.add_vline(x=lb, line_dash="dash", line_color="blue", 
                                annotation_text=f"Lower: {lb:,.2f}")
                    fig.add_vline(x=mb, line_color="black", 
                                annotation_text=f"Median: {mb:,.2f}")
                    fig.add_vline(x=ub, line_dash="dash", line_color="blue", 
                                annotation_text=f"Upper: {ub:,.2f}")

                    fig.add_trace(go.Scatter(
                        x=[actual], y=[0],
                        mode="markers+text",
                        marker=dict(color=flag_color, size=20),
                        text=[f"Actual: {actual:,.2f}"],
                        textposition="top center",
                        name="Actual Value"
                    ))

                    fig.update_layout(
                        title=f"QC Analysis for {selected}",
                        xaxis_title=f"{selected} Value",
                        yaxis=dict(showticklabels=False, range=[-0.5, 1]),
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show numeric inputs as bar chart
                bar_df = pd.DataFrame({
                    "Category": feats["num"],
                    "Value": [user_input[v] for v in feats["num"]]
                })
                st.subheader("📊 Input Values Used")
                fig_bar = px.bar(bar_df, x="Category", y="Value", text="Value",
                               title=f"Input Values for {selected}")
                fig_bar.update_traces(texttemplate='%{y:,.2f}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check your model files and input data.")

# =======================================================================
# MODE 2 — BATCH INPUT (CSV UPLOAD)
# =======================================================================
else:
    st.subheader("📁 Batch Processing with CSV Upload")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your CSV file", 
        type=["csv"],
        help="Upload a CSV file containing your data for batch processing"
    )

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        
        # Show file info
        st.success(f"✅ File uploaded successfully! Shape: {df_batch.shape}")
        
        with st.expander("🔍 Preview uploaded data"):
            st.dataframe(df_batch.head(), use_container_width=True)
            st.caption(f"Total rows: {len(df_batch)} | Columns: {', '.join(df_batch.columns.tolist())}")

        # Check for required columns
        required_cols = ['NO_SIRI', selected]
        missing_cols = [col for col in required_cols if col not in df_batch.columns]
        
        if missing_cols:
            st.warning(f"⚠️ Missing required columns: {', '.join(missing_cols)}")
            st.info("Your CSV should contain at least 'NO_SIRI' and the selected target column.")
        else:
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch data..."):
                    try:
                        # Run prediction on entire dataset
                        result_batch = predict_new(df_batch, out_dir=MODEL_DIR)

                        # ALWAYS include NO_SIRI
                        if "NO_SIRI" in df_batch.columns:
                            result_batch["NO_SIRI"] = df_batch["NO_SIRI"]

                        # Identify prediction columns for selected target
                        low_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "low" in c.lower()), None)
                        med_col = next((c for c in result_batch.columns if selected.lower() in c.lower() and "med" in c.lower()), None)
                        up_col  = next((c for c in result_batch.columns if selected.lower() in c.lower() and "up"  in c.lower()), None)

                        if not (low_col and med_col and up_col):
                            st.error("Could not find prediction columns in the results.")
                        else:
                            # Construct clean output table
                            clean_df = pd.DataFrame()
                            clean_df["NO_SIRI"] = df_batch["NO_SIRI"]
                            clean_df[f"{selected}_ACTUAL"] = df_batch[selected]
                            clean_df[f"{selected}_PRED_LOW"] = result_batch[low_col]
                            clean_df[f"{selected}_PRED_MED"] = result_batch[med_col]
                            clean_df[f"{selected}_PRED_UP"] = result_batch[up_col]

                            # Compute flag for selected target
                            flags = []
                            for i in range(len(clean_df)):
                                actual = clean_df.iloc[i][f"{selected}_ACTUAL"]
                                lb = clean_df.iloc[i][f"{selected}_PRED_LOW"]
                                ub = clean_df.iloc[i][f"{selected}_PRED_UP"]

                                if actual < lb or actual > ub:
                                    flags.append(True)
                                else:
                                    flags.append(False)

                            clean_df[f"{selected}_FLAG"] = flags

                            # Store in session state
                            st.session_state.batch_results = clean_df

                            # Split into issue / OK
                            df_issue = clean_df[clean_df[f"{selected}_FLAG"] == True]
                            df_ok    = clean_df[clean_df[f"{selected}_FLAG"] == False]

                            # Summary statistics
                            total = len(clean_df)
                            total_issue = len(df_issue)
                            total_ok = len(df_ok)
                            pct_issue = round((total_issue / total) * 100, 2) if total > 0 else 0
                            pct_ok = round((total_ok / total) * 100, 2) if total > 0 else 0

                            # Display summary
                            st.subheader("📊 Batch Results Summary")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Records", total)
                            with col2:
                                st.metric("Records with Issues", total_issue, f"{pct_issue}%")
                            with col3:
                                st.metric("Records OK", total_ok, f"{pct_ok}%")

                            # Create pie chart for visualization
                            fig_pie = px.pie(
                                values=[total_issue, total_ok],
                                names=['With Issues', 'OK'],
                                title=f"Distribution of QC Flags for {selected}",
                                color=['With Issues', 'OK'],
                                color_discrete_map={'With Issues':'red', 'OK':'green'}
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                            # Display tables in tabs
                            tab1, tab2, tab3 = st.tabs(["⚠️ Issues", "✅ OK Records", "📋 All Results"])

                            with tab1:
                                st.subheader(f"Records with Issues ({selected})")
                                if df_issue.empty:
                                    st.success("🎉 No issues detected for this target!")
                                else:
                                    st.dataframe(df_issue, use_container_width=True)
                                    st.caption(f"Showing {len(df_issue)} records with potential issues")

                            with tab2:
                                st.subheader(f"Records without Issues ({selected})")
                                st.dataframe(df_ok, use_container_width=True)
                                st.caption(f"Showing {len(df_ok)} records without issues")

                            with tab3:
                                st.subheader(f"All Results ({selected})")
                                st.dataframe(clean_df, use_container_width=True)

                            # Download buttons
                            st.subheader("📥 Download Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                csv_issues = df_issue.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"Download Issues Only ({len(df_issue)} records)",
                                    data=csv_issues,
                                    file_name=f"qc_issues_{selected}.csv",
                                    mime="text/csv",
                                    help="Download only records with QC issues"
                                )
                            
                            with col2:
                                csv_all = clean_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label=f"Download All Results ({total} records)",
                                    data=csv_all,
                                    file_name=f"qc_results_{selected}.csv",
                                    mime="text/csv",
                                    help="Download all results including flags"
                                )

                            st.success("✅ Batch processing completed successfully!")

                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")
                        st.info("Please check your input data format and model compatibility.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>BE 2026 — ML-Driven Quality Control System</p>
    <p>For issues or questions, please contact the development team.</p>
</div>
""", unsafe_allow_html=True)
