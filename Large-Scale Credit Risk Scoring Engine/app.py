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

st.title("⚡ Large-Scale Credit Risk Scoring Engine")
st.markdown("""
This application demonstrates an institutional-style credit risk scoring system. 
It uses **synthetic borrower data** to compute credit risk scores using **Logistic Regression**.
""")

st.sidebar.header("Configuration")
engine_choice = st.sidebar.radio("Execution Engine", ["Pandas (Local)", "PySpark (Distributed)"])

try:
    N = int(st.sidebar.text_input("Number of Borrowers (N)", "1000"))
except ValueError:
    N = 1000
    st.sidebar.error("Please enter a valid integer for N.")

run_button = st.sidebar.button("Run Pipeline ▶️", type="primary")

if run_button:
    if engine_choice == "PySpark (Distributed)":
        st.warning("⚠️ Warning: Initializing Spark context. This might fail if your Java environment (e.g., Java 21+) is incompatible with local PySpark.")
        try:
            from pyspark.sql import SparkSession
            from engine.pipeline_spark import SparkRiskScoringPipeline
            
            with st.spinner("Initializing Spark..."):
                spark = SparkSession.builder.master('local[*]').appName('RiskScoring').getOrCreate()
            
            with st.spinner("Generating data via Spark..."):
                loan_df, bureau_df, txn_df = generate_spark_data(spark, N)
                
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
        
        with st.spinner("Generating data via Pandas..."):
            loan_df, bureau_df, txn_df = generate_pandas_data(N)
            
        with st.spinner("Running Pandas Pipeline..."):
            pipeline = PandasRiskScoringPipeline()
            base_df = pipeline.join_data(loan_df, bureau_df, txn_df)
            features_df = pipeline.feature_engineering(base_df)
            predictions, model = pipeline.train_model(features_df)
            auc = pipeline.evaluate_model(predictions)
            scored_df = pipeline.calculate_risk_scores(predictions)
            display_df = scored_df.head(100)
            
        duration = time.time() - start_time
        st.success(f"Pipeline executed successfully in Pandas! Duration: {duration:.2f}s | AUC-ROC: **{auc:.4f}**")

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
