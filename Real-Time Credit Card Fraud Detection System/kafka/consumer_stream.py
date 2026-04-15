import os, json, joblib, logging
import pandas as pd
from kafka import KafkaConsumer, KafkaProducer
from collections import deque
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")

# We handle the model load cleanly
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Loaded XGBoost fraud detection model successfully.")
except Exception as e:
    logger.error(f"Could not load model: {e}")
    exit(1)

consumer = KafkaConsumer(
    "credit_card_transactions",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    consumer_timeout_ms=5000  # Added explicit timeout to not block indefinitely
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

windows = {}
WINDOW_SIZE = timedelta(minutes=5)
# Set a more aggressive threshold given our synthetic data
FRAUD_THRESHOLD = 0.85

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

logger.info("Starting Consumer and waiting for transactions...")

try:
    for msg in consumer:
        txn = msg.value
        feats = extract_features(txn["card_id"], txn)
        
        # Predict
        df = pd.DataFrame([feats])
        score = model.predict_proba(df)[0][1]

        is_fraud = bool(score > FRAUD_THRESHOLD)
        
        alert = {
            **txn,
            "fraud_score": float(score),
            "is_fraud": is_fraud
        }

        # Send to the alert topic
        producer.send("fraud_predictions", alert)
        
        if is_fraud:
            logger.warning(f"🚨 FRAUD ALERT! | Card {txn['card_id']} | ${txn['amount']} | Score: {score:.2f}")
        else:
            logger.info(f"✅ Normal Txn | Card {txn['card_id']} | ${txn['amount']} | Score: {score:.2f}")
except KeyboardInterrupt:
    logger.info("Consumer stopped.")