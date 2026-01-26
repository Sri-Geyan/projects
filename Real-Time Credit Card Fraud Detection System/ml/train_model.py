import pandas as pd, joblib
from xgboost import XGBClassifier

df = pd.read_csv("data/transactions.csv")

X = df[["amount", "txn_count_5m", "avg_amount_5m"]]
y = df["is_fraud"]

model = XGBClassifier(scale_pos_weight=10, eval_metric="logloss")
model.fit(X, y)

joblib.dump(model, "ml/model.pkl")
print("Model trained & saved")
