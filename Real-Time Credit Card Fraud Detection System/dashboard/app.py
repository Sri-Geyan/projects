import json, pandas as pd, streamlit as st
from kafka import KafkaConsumer

st.set_page_config(layout="wide")
st.title("🚨 Real-Time Fraud Detection Dashboard")

@st.cache_resource
def get_consumer():
    return KafkaConsumer(
        "fraud_predictions",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        consumer_timeout_ms=1000
    )

consumer = get_consumer()

if "data" not in st.session_state:
    st.session_state.data = []

for msg in consumer:
    st.session_state.data.append(msg.value)
df = pd.DataFrame(st.session_state.data)
if not df.empty:
    total_txns = len(df)
    fraud_txns = df[df["is_fraud"] == True].shape[0]

    st.metric("Total Transactions", total_txns)
    st.metric("Fraud Alerts", fraud_txns)
else:
    st.metric("Total Transactions", 0)
    st.metric("Fraud Alerts", 0)
