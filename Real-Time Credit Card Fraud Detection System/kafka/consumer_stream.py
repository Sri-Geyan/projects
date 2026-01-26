import os, json, joblib
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from collections import deque
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")

model = joblib.load(MODEL_PATH)

consumer = KafkaConsumer(
    "credit_card_transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

windows = {}
WINDOW_SIZE = timedelta(minutes=5)
FRAUD_THRESHOLD = 0.95

def extract_features(card_id, txn):
    now = datetime.fromisoformat(txn["timestamp"])
    windows.setdefault(card_id, deque())
    window = windows[card_id]
    window.append(txn)

    while window and datetime.fromisoformat(window[0]["timestamp"]) < now - WINDOW_SIZE:
        window.popleft()

    amounts = [t["amount"] for t in window]
    return {
        "amount": txn["amount"],
        "txn_count_5m": len(amounts),
        "avg_amount_5m": sum(amounts) / len(amounts)
    }

for msg in consumer:
    txn = msg.value
    feats = extract_features(txn["card_id"], txn)
    df = pd.DataFrame([feats])
    score = model.predict_proba(df)[0][1]

    alert = {
        **txn,
        "fraud_score": float(score),
        "is_fraud": bool(score > FRAUD_THRESHOLD)
    }

    producer.send("fraud_predictions", alert)
    print("Alert:", alert)