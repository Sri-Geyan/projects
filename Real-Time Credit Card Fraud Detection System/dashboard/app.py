import json, time, random
from datetime import datetime
import pandas as pd
import streamlit as st
from kafka import KafkaConsumer, KafkaProducer

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="🚨")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .big-font { font-size: 24px !important; font-weight: bold; }
    .fraud-text { color: #ff4b4b; font-weight: bold; }
    .normal-text { color: #00cc96; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Real-Time Credit Card Fraud Detection Center")
st.markdown("Monitoring live transaction stream for fraudulent patterns using **Kafka** and **XGBoost**.")

# --- Kafka Setup ---
@st.cache_resource
def get_consumer():
    return KafkaConsumer(
        "fraud_predictions",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=500  # unblock every 0.5s to keep UI responsive
    )

@st.cache_resource
def get_producer():
    return KafkaProducer(
        bootstrap_servers="localhost:9092",
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

consumer = get_consumer()
producer = get_producer()

# --- Sidebar: Interactive Hybrid Interface ---
st.sidebar.header("🛠️ Manual Transaction Override")
st.sidebar.markdown("Test the XGBoost ML model by injecting a custom transaction directly into the live Kafka stream.")

with st.sidebar.form("manual_txn_form", clear_on_submit=False):
    st.subheader("Simulate Transaction")
    custom_card_id = st.number_input("Card ID", min_value=1000, max_value=9999, value=1010)
    custom_amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=100000.0, value=35000.0)
    
    submitted = st.form_submit_button("🔥 Send Live to Kafka", use_container_width=True)
    if submitted:
        txn = {
            "transaction_id": random.randint(100000, 999999),
            "card_id": int(custom_card_id),
            "amount": float(custom_amount),
            "timestamp": datetime.utcnow().isoformat()
        }
        producer.send("credit_card_transactions", txn)
        producer.flush()
        st.sidebar.success("Transaction sent! Watch the alerts...")

st.sidebar.divider()
st.sidebar.info("💡 **Pro-tip**: Trying sending an extraordinarily high amount (e.g., $35,000) to explicitly trigger a fraud alert in the Recent Alerts window.")

# --- Main Dashboard Polling ---
if "data" not in st.session_state:
    st.session_state.data = []

# Fetch new messages
msgs = consumer.poll(timeout_ms=500)
for tp, messages in msgs.items():
    for msg in messages:
         st.session_state.data.append(msg.value)

# Keep only the last 1000 transactions for memory reasons
if len(st.session_state.data) > 1000:
    st.session_state.data = st.session_state.data[-1000:]

df = pd.DataFrame(st.session_state.data)

# Real-time Metrics Layer
st.subheader("📊 Live Streaming Metrics")
col1, col2, col3, col4 = st.columns(4)

total_txns = len(df)
fraud_txns = len(df[df["is_fraud"] == True]) if not df.empty else 0
normal_txns = total_txns - fraud_txns
fraud_rate = (fraud_txns / total_txns * 100) if total_txns > 0 else 0.0

with col1:
    st.metric(label="Total Transactions Monitored", value=total_txns)
with col2:
    st.metric(label="✅ Normal Transactions", value=normal_txns)
with col3:
    st.metric(label="🚨 Fraud Alerts", value=fraud_txns, delta=f"{fraud_txns} detected" if fraud_txns > 0 else None, delta_color="inverse")
with col4:
    st.metric(label="Current Fraud Rate", value=f"{fraud_rate:.2f}%")

st.divider()

col_charts, col_alerts = st.columns([2, 1])

with col_charts:
    st.subheader("📈 Transaction Amount Stream")
    if not df.empty:
        # Create a line chart with 'timestamp' and 'amount'
        chart_df = df.copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
        chart_df.set_index("timestamp", inplace=True)
        # We can color fraud vs normal optionally, but simple line is fast
        st.line_chart(chart_df['amount'], use_container_width=True)
    else:
        st.info("Waiting for transactions...")

with col_alerts:
    st.subheader("🚨 Recent Fraud Alerts")
    if not df.empty:
        fraud_df = df[df["is_fraud"] == True].tail(10) # last 10 frauds
        if not fraud_df.empty:
            display_df = fraud_df[["card_id", "amount", "fraud_score"]].copy()
            display_df["amount"] = display_df["amount"].apply(lambda x: f"${x:,.2f}")
            display_df["fraud_score"] = display_df["fraud_score"].apply(lambda x: f"{x*100:.1f}%")
            
            # Show as dataframe
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.success("No fraud detected yet!")
    else:
        st.info("Waiting for transactions...")

# Show raw stream snippet
with st.expander("Raw Live Stream (Last 5)"):
    if not df.empty:
        st.dataframe(df.tail(5), use_container_width=True)

# Auto-refresh loop
time.sleep(1)
st.rerun()
