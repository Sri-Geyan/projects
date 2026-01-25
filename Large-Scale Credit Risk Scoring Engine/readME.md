# ⚡ Large-Scale Credit Risk Scoring Engine (Apache Spark – Databricks Serverless)

An **institutional-style credit risk scoring system** built using **Apache Spark** and executed on **Databricks Serverless (Community Edition)**.

This project simulates how banks and NBFCs score **millions of borrowers** by leveraging distributed data processing, feature engineering, and machine learning — while adapting to **real-world platform constraints**.

---

## 🧠 Problem Statement

Banks need to evaluate **loan default risk** for millions of borrowers efficiently.

The goal of this project is to:

- Generate large-scale borrower data
- Engineer credit risk features at scale
- Train a distributed ML model
- Produce **probability-based risk scores**
- Bucket customers into actionable risk segments

---

## 🏗️ Architecture (Logical)

```
Synthetic Data
      ↓
Spark DataFrames (Distributed)
      ↓
Feature Engineering
      ↓
Spark ML (Logistic Regression)
      ↓
Risk Scores + Buckets
      ↓
Notebook Output (Serverless-safe)
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| Distributed Engine | Apache Spark |
| Execution Platform | Databricks Serverless (Community Edition) |
| Storage Format | Parquet (logical) |
| ML Framework | Spark MLlib |
| Model | Logistic Regression |
| Interface | Databricks Notebook |

---

## ⚠️ Platform Constraints & Design Decisions (IMPORTANT)

This project intentionally adapts to **Databricks Free / Serverless limitations**, mirroring real enterprise constraints.

### ❌ Constraints Encountered

- Java 21 incompatibility with local Spark
- No local Spark clusters
- No Docker access
- No DBFS root write permissions
- No table persistence (`saveAsTable`, `CACHE`, `PERSIST`)
- No Delta writes in `/tmp`

### ✅ Design Adaptations (Why They Matter)

| Change | Reason |
|--------|--------|
| Databricks Serverless | Avoid JVM & Hadoop conflicts |
| Notebook-first execution | Serverless-compatible Spark |
| In-memory DataFrames | Persistence not allowed |
| TEMP VIEWS instead of tables | Serverless restriction |
| No `.cache()` / `.persist()` | Not supported |
| `vector_to_array()` | Spark ML probability is a Vector UDT |
| Explicit ID retention | Spark ML drops business keys |

> **Key takeaway:**  
> The logic is **production-ready**, while execution is adapted to platform limits.

---

## 📘 Notebook Execution Flow

### 1️⃣ Spark Environment Check
Verify Spark availability using Databricks-managed session.

### 2️⃣ Synthetic Data Generation
- Loan applications
- Credit bureau data
- Transactional data  

Generated at scale using Spark primitives.

### 3️⃣ Data Ingestion & Joins
Borrower-level analytical base table created via distributed joins.

### 4️⃣ Feature Engineering
Key risk features:
- Debt-to-Income (DTI)
- Credit Utilization
- Delinquency Flag
- Credit Score

### 5️⃣ ML Pipeline
- VectorAssembler
- Logistic Regression (Spark MLlib)

### 6️⃣ Model Evaluation
- AUC-ROC calculated at scale

### 7️⃣ Risk Scoring
- Extract default probability
- Convert ML vector → array
- Generate risk buckets (Low / Medium / High)

### 8️⃣ Final Output
Results displayed directly in notebook (serverless-safe).

---

## 📊 Risk Bucketing Logic

| Probability of Default | Risk Bucket |
|------------------------|-------------|
| ≤ 0.40 | Low |
| 0.40 – 0.70 | Medium |
| ≥ 0.70 | High |

---

## 📦 Final Output Schema

| Column | Description |
|--------|-------------|
| `borrower_id` | Unique borrower identifier |
| `risk_score` | Probability of default (0–1) |
| `risk_bucket` | Credit risk category |

---

## 🧠 Why Logistic Regression?

- Industry-standard baseline model
- Interpretable coefficients
- Stable at scale
- Common in regulated financial environments

---

## 🚀 Production Readiness

In a **production environment**, this same pipeline would:

- Persist outputs to **Delta Lake / S3 / ADLS**
- Use **Databricks All-Purpose Compute**
- Add **MLflow tracking**
- Integrate **real-time data via Kafka**
- Include **PSI & drift monitoring**

**No business logic changes required.**

---

## 🧑‍💼 Interview-Ready Explanation

> *"I implemented a large-scale credit risk scoring engine using Apache Spark on Databricks Serverless. Due to platform restrictions, the pipeline executes fully in-memory using temporary views, while maintaining production-grade Spark ML logic that can be persisted to Delta tables in enterprise environments."*

---

## ✅ Key Skills Demonstrated

- ✔️ Distributed data processing
- ✔️ Feature engineering at scale
- ✔️ Spark ML pipelines
- ✔️ Probability-based risk modeling
- ✔️ Platform-aware system design
- ✔️ Debugging real-world infra constraints

---

## 📂 Project Structure

```
credit-risk-engine/
│
├── notebooks/
│   └── credit_risk_scoring.ipynb    # Main Databricks notebook
│
├── docs/
│   └── architecture.md               # Detailed architecture docs
│
├── README.md                          # This file
└── requirements.txt                   # Python dependencies (if any)
```

---

## ▶️ How to Run

### On Databricks Serverless (Community Edition)

1. **Create a Databricks account** (Community Edition)
2. **Import the notebook:**
   - Upload `credit_risk_scoring.ipynb` to your workspace
3. **Attach to a Serverless cluster**
4. **Run all cells** sequentially

### Local Execution (Not Recommended)

Due to Java 21 incompatibility and Hadoop setup complexity, local execution is **not supported**. Use Databricks Serverless instead.

---

## 🏁 Conclusion

This project reflects **real-world data engineering and ML workflows**, where engineers must adapt to platform constraints without compromising system design.

It prioritizes:
- ✅ Correct architecture
- ✅ Scalable logic
- ✅ Production realism
- ✅ Interview credibility

---

## 📌 Future Enhancements

- [ ] Random Forest / XGBoost models
- [ ] Feature importance analysis
- [ ] PSI & data drift monitoring
- [ ] MLflow experiment tracking
- [ ] SQL-based dashboards
- [ ] Streaming risk scoring with Kafka
- [ ] Integration with Delta Lake for persistence
- [ ] Model explainability using SHAP

---

## 🧾 Resume-Ready Description

*Built a large-scale credit risk scoring engine using Apache Spark on Databricks Serverless, implementing distributed feature engineering, Spark MLlib-based logistic regression, and probability-based risk bucketing for millions of borrowers, demonstrating production-grade ML pipeline design under platform constraints.*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/yourusername/credit-risk-engine/issues).

---

## 👨‍💻 Author

**Sri Geyan**  
Financial Data Science / Risk Analytics  
Execution: Databricks Serverless (Apache Spark)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/yourusername)

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you understand large-scale ML on Spark!

---

## 📚 References

- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Databricks Documentation](https://docs.databricks.com/)
- [Credit Risk Modeling Best Practices](https://www.risk.net/)

---

## ⚖️ Disclaimer

This project is for **educational and demonstration purposes only**.  
It does **not** constitute financial, credit, or lending advice.
