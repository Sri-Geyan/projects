import streamlit as st
import pandas as pd
import time
import os

from engine.data_generator import generate_pandas_data, generate_spark_data

st.set_page_config(
    page_title="Credit Risk Scoring Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Injection
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0A0F1A 0%, #111827 100%); }
    [data-testid="stSidebar"] {
        background: rgba(21, 26, 40, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .stMetric, .stDataFrame, div[data-testid="stForm"], div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .stMetric:hover, div[data-testid="stForm"]:hover, div[data-testid="stExpander"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
    }
    h1, h2, h3 {
        background: linear-gradient(90deg, #00E5FF, #0077FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00E5FF, #0077FF) !important;
        color: #0A0F1A !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)


st.title("⚡ Large-Scale Credit Risk Scoring Engine")
st.markdown("""
This application demonstrates an institutional-style credit risk scoring system. 
It uses **synthetic borrower data** to compute credit risk scores using **Logistic Regression**.
""")

st.sidebar.header("Configuration")
engine_choice = st.sidebar.radio("Execution Engine", ["Pandas (Local)", "PySpark (Distributed)"])

st.sidebar.header("Data Upload (CSV)")
loan_file = st.sidebar.file_uploader("Upload Loan Data", type=["csv"])
bureau_file = st.sidebar.file_uploader("Upload Bureau Data", type=["csv"])
txn_file = st.sidebar.file_uploader("Upload Transaction Data", type=["csv"])

run_button = st.sidebar.button("Run Pipeline ▶️", type="primary")

if run_button:
    if engine_choice == "PySpark (Distributed)":
        st.warning("⚠️ Warning: Initializing Spark context. This might fail if your Java environment (e.g., Java 21+) is incompatible with local PySpark.")
        try:
            from pyspark.sql import SparkSession
            from engine.pipeline_spark import SparkRiskScoringPipeline
            
            with st.spinner("Initializing Spark..."):
                spark = SparkSession.builder.master('local[*]').appName('RiskScoring').getOrCreate()
            
            if not (loan_file and bureau_file and txn_file):
                st.error("Please upload all three CSV files for Spark processing.")
                st.stop()
            
            with st.spinner("Loading user data into Spark..."):
                import tempfile
                import os
                
                # Write uploaded files to temp space so Spark can read them
                temp_dir = tempfile.mkdtemp()
                p1, p2, p3 = os.path.join(temp_dir, 'l.csv'), os.path.join(temp_dir, 'b.csv'), os.path.join(temp_dir, 't.csv')
                with open(p1, 'wb') as f: f.write(loan_file.getvalue())
                with open(p2, 'wb') as f: f.write(bureau_file.getvalue())
                with open(p3, 'wb') as f: f.write(txn_file.getvalue())
                
                loan_df = spark.read.csv(p1, header=True, inferSchema=True)
                bureau_df = spark.read.csv(p2, header=True, inferSchema=True)
                txn_df = spark.read.csv(p3, header=True, inferSchema=True)
                
            with st.spinner("Running Spark Pipeline..."):
                pipeline = SparkRiskScoringPipeline(spark)
                base_df = pipeline.join_data(loan_df, bureau_df, txn_df)
                features_df = pipeline.feature_engineering(base_df)
                predictions, model = pipeline.train_model(features_df)
                auc = pipeline.evaluate_model(predictions)
                scored_df = pipeline.calculate_risk_scores(predictions)
                
                # Fetch a sample to display (since displaying full DataFrame takes too long)
                st.success(f"Pipeline executed successfully in Spark! AUC-ROC: **{auc:.4f}**")
                display_df = scored_df.limit(100).toPandas()
                
            spark.stop()
            
        except Exception as e:
            st.error(f"Spark Engine Failed: {e}")
            st.info("💡 Try using the Pandas engine instead (recommended for local Macs with newer Java versions).")
            st.stop()
            
    else:
        # Pandas Branch
        from engine.pipeline_pandas import PandasRiskScoringPipeline
        
        start_time = time.time()
        
        if not (loan_file and bureau_file and txn_file):
            st.error("Please upload all three CSV files.")
            st.stop()
            
        with st.spinner("Loading user data..."):
            loan_df = pd.read_csv(loan_file)
            bureau_df = pd.read_csv(bureau_file)
            txn_df = pd.read_csv(txn_file)
            
        with st.spinner("Running Pandas Pipeline & 5-Fold CV..."):
            pipeline = PandasRiskScoringPipeline()
            base_df = pipeline.join_data(loan_df, bureau_df, txn_df)
            features_df = pipeline.feature_engineering(base_df)
            predictions, model = pipeline.train_model(features_df)
            auc = pipeline.evaluate_model(predictions)
            
            # 5-Fold CV
            cv_auc = pipeline.cross_validate_model(features_df)
            
            scored_df = pipeline.calculate_risk_scores(predictions)
            display_df = scored_df.head(100)
            
        duration = time.time() - start_time
        st.success(f"Pipeline executed! AUC: **{auc:.4f}** | 5-Fold Mean AUC: **{cv_auc:.4f}**")

    # Shared UI outputs
    col1, col2, col3 = st.columns(3)
    
    bucket_counts = display_df['risk_bucket'].value_counts()
    
    col1.metric("Low Risk Borrowers", int(bucket_counts.get('Low', 0)))
    col2.metric("Medium Risk Borrowers", int(bucket_counts.get('Medium', 0)))
    col3.metric("High Risk Borrowers", int(bucket_counts.get('High', 0)))
    
    st.markdown("### 📊 Sample Output (100 Rows)")
    st.dataframe(display_df[['borrower_id', 'risk_score', 'risk_bucket']], use_container_width=True)
    
    st.markdown("### 🔍 Risk Bucket Distribution")
    st.bar_chart(bucket_counts, color="#ff4b4b")
    
else:
    st.info("👈 Please select your configuration and click 'Run Pipeline'.")
