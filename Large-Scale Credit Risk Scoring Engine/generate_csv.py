import pandas as pd
import numpy as np

n = 5000
# Loan Applications
loan_df = pd.DataFrame({
    'borrower_id': np.arange(n),
    'loan_amount': np.random.uniform(100000, 1000000, n).astype(int),
    'loan_tenure': np.random.uniform(12, 60, n).astype(int),
    'interest_rate': np.random.uniform(8, 18, n),
    'default': np.random.choice([0, 1], p=[0.8, 0.2], size=n)
})

# Credit Bureau
bureau_df = pd.DataFrame({
    'borrower_id': np.arange(n),
    'credit_score': np.random.uniform(550, 850, n).astype(int),
    'total_credit': np.random.uniform(200000, 1000000, n).astype(int),
    'used_credit': np.random.uniform(0, 500000, n).astype(int),
    'delinquency_count': np.random.uniform(0, 5, n).astype(int)
})

# Transactions
txn_df = pd.DataFrame({
    'borrower_id': np.arange(n),
    'monthly_income': np.random.uniform(15000, 135000, n).astype(int),
    'emi_outflow': np.random.uniform(0, 40000, n).astype(int)
})

loan_df.to_csv("loan_data.csv", index=False)
bureau_df.to_csv("bureau_data.csv", index=False)
txn_df.to_csv("txn_data.csv", index=False)
print("CSV files generated successfully!")
