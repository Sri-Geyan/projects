# ⚡ Large-Scale Credit Risk Scoring Engine (Apache Spark + Pandas)

An **institutional-style credit risk scoring system** initially built using **Apache Spark** for Databricks, and now augmented with a **Local Pandas Engine** and **Streamlit UI**.

This project simulates how banks and NBFCs score **millions of borrowers** by leveraging synthetic data generation, feature engineering, and machine learning.

---

## 🏗️ Architecture

You can now run this pipeline in two modes via the new Streamlit Dashboard:

1. **Pandas (Local Mode)**: Ideal for local laptops/Macs running newer Java versions where PySpark setup is problematic. This engine translates the logic into standard Scikit-Learn logic.
2. **PySpark (Distributed Mode)**: The original execution mode intended for Databricks Serverless, demonstrating distributed table joins and VectorAssembler logic.

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| Core Engine 1 (Local) | Pandas / Scikit-Learn |
| Core Engine 2 (Distributed) | Apache Spark |
| User Interface | Streamlit |
| Model | Logistic Regression |

---

## 📂 Project Structure

```
large-scale-credit-risk-scoring-engine/
├── app.py                      # Main Streamlit Dashboard
├── engine/
│   ├── data_generator.py       # Generators for synthetic borrowers
│   ├── pipeline_pandas.py      # Local execution engine module
│   └── pipeline_spark.py       # Distributed execution engine module
├── Output Dataframe.csv
└── readME.md
```

---

## ▶️ How to Run

### Local Execution (Streamlit)
We added a UI layer for rapid visual debugging! Check it out:
```bash
pip install streamlit pandas scikit-learn pyspark
streamlit run app.py
```
*(We recommend running the dashboard using the "Pandas Engine" for the best local experience).*

### Databricks Execution
To run the PySpark pipelines natively on Databricks Serverless, simply import the contents of `pipeline_spark.py` into a Databricks Notebook environment.

---

## 👨‍💻 Author

**Sri Geyan**  
Financial Data Science / Risk Analytics
