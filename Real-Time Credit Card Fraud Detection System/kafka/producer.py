import json, time, random
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_transaction():
    return {
        "transaction_id": random.randint(100000, 999999),
        "card_id": random.randint(1000, 1020),
        "amount": round(random.uniform(50, 20000), 2),
        "timestamp": datetime.utcnow().isoformat()
    }

while True:
    txn = generate_transaction()
    producer.send("credit_card_transactions", txn)
    print("Sent:", txn)
    time.sleep(1)
