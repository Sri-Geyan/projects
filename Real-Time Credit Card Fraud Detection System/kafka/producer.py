import json, time, random, logging
from datetime import datetime
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

logger.info("Starting Credit Card Transaction Producer...")

while True:
    print("\n--- Enter New Transaction ---")
    try:
        card_input = input("Card ID (default 1010): ")
        card_id = int(card_input) if card_input.strip() else 1010
        amount_input = input("Amount: ")
        amount = float(amount_input)
    except Exception:
        print("Invalid input, please enter numeric values.")
        continue

    txn = {
        "transaction_id": random.randint(100000, 999999),
        "card_id": card_id,
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    producer.send("credit_card_transactions", txn)
    logger.info(f"Sent: Transaction {txn['transaction_id']} | Card {txn['card_id']} | ${txn['amount']}")
