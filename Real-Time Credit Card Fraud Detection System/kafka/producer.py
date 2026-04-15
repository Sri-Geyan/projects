import json, time, random, logging
from datetime import datetime
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def generate_transaction():
    # 10% chance of a fraudulent burst (high amount)
    is_fraud = random.random() < 0.10
    
    if is_fraud:
        amount = round(random.uniform(5000, 25000), 2)
    else:
        amount = round(random.uniform(10, 500), 2)
        
    return {
        "transaction_id": random.randint(100000, 999999),
        "card_id": random.randint(1000, 1020),
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat()
    }

logger.info("Starting Credit Card Transaction Producer...")

while True:
    txn = generate_transaction()
    producer.send("credit_card_transactions", txn)
    logger.info(f"Sent: Transaction {txn['transaction_id']} | Card {txn['card_id']} | ${txn['amount']}")
    time.sleep(0.5)  # Slightly faster frequency for a better real-time feel
