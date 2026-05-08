import os, random
import pandas as pd

os.makedirs("data", exist_ok=True)

data = []
try:
    num_records = int(input("Enter number of records to generate (e.g., 5000): "))
except ValueError:
    num_records = 5000
for _ in range(num_records):
    is_fraud = random.random() < 0.1
    if is_fraud:
        amount = random.uniform(5000, 25000)
        txn_count = random.randint(5, 15)
        avg_amt = random.uniform(2000, 10000)
    else:
        amount = random.uniform(10, 500)
        txn_count = random.randint(1, 4)
        avg_amt = random.uniform(50, 300)
    
    data.append({
        "amount": amount,
        "txn_count_5m": txn_count,
        "avg_amount_5m": avg_amt,
        "is_fraud": int(is_fraud)
    })

df = pd.DataFrame(data)
df.to_csv("data/transactions.csv", index=False)
print("Dummy data generated.")
