# 🚨 Real-Time Credit Card Fraud Detection System (Kafka KRaft)

A **real-time fraud detection pipeline** that streams credit card transactions through **Apache Kafka (KRaft mode)**, performs low-latency ML inference, and visualizes alerts on a **Streamlit dashboard**.

This project mirrors **production-grade streaming systems** used in fintech and banking environments.

---

## 🧠 System Overview

```
Transaction Producer
        ↓
Kafka (KRaft, no ZooKeeper)
        ↓
Streaming ML Consumer (XGBoost)
        ↓
Kafka (fraud_predictions topic)
        ↓
Streamlit Fraud Dashboard
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| Streaming | Apache Kafka 4.x (KRaft mode) |
| Language | Python 3.11 |
| ML Model | XGBoost (classification) |
| Serialization | JSON |
| Dashboard | Streamlit |
| OS | macOS / Linux |

---

## 📁 Project Structure

```
Real-Time-Credit-Card-Fraud-Detection/
│
├── kafka/
│   ├── producer.py              # Streams synthetic transactions
│   └── consumer_stream.py       # Real-time ML inference + alerting
│
├── ml/
│   ├── train_model.py           # Trains fraud detection model
│   └── model.pkl                # Saved XGBoost model
│
├── dashboard/
│   └── app.py                   # Streamlit dashboard
│
├── data/
│   └── transactions.csv         # Synthetic training data
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ✅ Prerequisites (MANDATORY)

### 1️⃣ Java (for Kafka)

Kafka 4.x requires **Java 17+**.

```bash
java -version
```

✔ **Tested with:**
- Java 17
- Java 21

### 2️⃣ Kafka (KRaft mode – NO ZooKeeper)

Install via Homebrew:

```bash
brew install kafka
```

**Kafka home (Apple Silicon):**
```
/opt/homebrew/opt/kafka
```

### 3️⃣ Python (CRITICAL)

⚠️ **Python 3.14 is NOT supported**

You must use:
- **Python 3.11**

Install:

```bash
brew install python@3.11
```

---

## 🚀 Setup & Run Instructions (KRaft)

### 1️⃣ Clone / Extract Project

```bash
unzip Real-Time-Credit-Card-Fraud-Detection.zip
cd Real-Time-Credit-Card-Fraud-Detection
```

### 2️⃣ Create Python Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Verify:**

```bash
python --version
# Python 3.11.x
```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Configure Kafka (KRaft)

**Create config:**

```bash
cd /opt/homebrew/opt/kafka
mkdir -p config/kraft
nano config/kraft/server.properties
```

**Paste:**

```properties
process.roles=broker,controller
node.id=1

controller.quorum.voters=1@localhost:9094
controller.listener.names=CONTROLLER

listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9094
advertised.listeners=PLAINTEXT://localhost:9092
listener.security.protocol.map=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT

log.dirs=/tmp/kafka-logs

num.partitions=3
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1

group.initial.rebalance.delay.ms=0
```

### 5️⃣ Initialize KRaft Metadata (ONCE)

```bash
rm -rf /tmp/kafka-logs
bin/kafka-storage random-uuid
bin/kafka-storage format \
  -t <UUID> \
  -c config/kraft/server.properties
```

### 6️⃣ Start Kafka

```bash
bin/kafka-server-start config/kraft/server.properties
```

**Verify:**

```bash
bin/kafka-topics --bootstrap-server localhost:9092 --list
```

### 7️⃣ Create Topics

```bash
bin/kafka-topics --create \
  --topic credit_card_transactions \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

bin/kafka-topics --create \
  --topic fraud_predictions \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1
```

### 8️⃣ Train Model (ONCE)

```bash
cd ~/Real-Time-Credit-Card-Fraud-Detection
source venv/bin/activate
python ml/train_model.py
```

### 9️⃣ Run the Pipeline (ORDER MATTERS)

**Terminal 1 – Consumer:**

```bash
python kafka/consumer_stream.py
```

**Terminal 2 – Producer:**

```bash
python kafka/producer.py
```

**Terminal 3 – Dashboard:**

```bash
streamlit run dashboard/app.py
```

**Open:**
```
http://localhost:8501
```
<img width="1377" height="782" alt="Output" src="https://github.com/user-attachments/assets/4f998e8c-5847-4d85-9dee-44335d08b995" />

---

## 📊 Dashboard Metrics Logic

- **Total Transactions** → total records consumed
- **Fraud Alerts** → rows where `is_fraud == True`

Explicit logic avoids boolean miscounts.

---

## 🧯 Troubleshooting (REAL ERRORS ENCOUNTERED)

This section documents **actual issues faced during development** and how they were fixed.

### ❌ Missing required configuration "process.roles"

**Cause:**  
Kafka 4.x does not support ZooKeeper

**Fix:**
- Switched fully to KRaft
- Added `process.roles=broker,controller`

### ❌ No such file or directory: config/server.properties

**Cause:**  
Kafka 4.x no longer ships ZooKeeper configs

**Fix:**  
Created `config/kraft/server.properties`

### ❌ Invalid cluster.id during kafka-storage format

**Cause:**  
Old metadata in `/tmp/kafka-logs`

**Fix:**

```bash
rm -rf /tmp/kafka-logs
```

Reformatted storage with a single cluster ID.

### ❌ Address already in use :9093

**Cause:**  
Controller port conflict on macOS

**Fix:**  
Moved controller to port `9094`

### ❌ KafkaTimeoutError: Failed to update metadata

**Cause:**  
Incorrect `advertised.listeners`

**Fix:**

```properties
advertised.listeners=PLAINTEXT://localhost:9092
```

### ❌ ImportError: cannot import name KafkaProducer

**Cause:**  
Wrong Python package (`kafka` instead of `kafka-python`)

**Fix:**

```bash
pip uninstall kafka
pip install kafka-python==2.0.2
```

### ❌ Object of type bool is not JSON serializable

**Cause:**  
NumPy boolean in alert payload

**Fix:**

```python
"is_fraud": bool(score > 0.95)
"fraud_score": float(score)
```

### ❌ Dashboard shows Total Transactions = Fraud Alerts

**Cause:**  
Fraud threshold too low

**Fix:**
- Increased threshold
- Explicit dashboard filtering

---

## 🧠 Interview-Ready Summary

> *"I built a real-time fraud detection system using Kafka in KRaft mode with Python-based streaming ML inference. I debugged real-world issues like metadata corruption, port conflicts, client incompatibilities, and NumPy serialization — exactly the kind of problems faced in production streaming systems."*

---

## ✅ Key Skills Demonstrated

- ✔️ Real-time stream processing with Kafka
- ✔️ KRaft mode configuration (modern Kafka, no ZooKeeper)
- ✔️ Low-latency ML inference
- ✔️ Producer-Consumer architecture
- ✔️ Production debugging and troubleshooting
- ✔️ Real-time data visualization
- ✔️ Event-driven architecture

---

## 🚀 Possible Extensions

- [ ] Precision–Recall tuning
- [ ] Isolation Forest for anomaly detection
- [ ] Spark Structured Streaming integration
- [ ] Docker Compose deployment
- [ ] Alerting via Slack / Email
- [ ] Kubernetes deployment
- [ ] Model monitoring and drift detection
- [ ] Multi-model ensemble approach
- [ ] A/B testing framework for models

---

## 🧾 Resume-Ready Description

*Built a real-time credit card fraud detection pipeline using Apache Kafka (KRaft mode) and XGBoost for streaming ML inference, processing transactions with sub-second latency and visualizing fraud alerts on a Streamlit dashboard, demonstrating production-grade event-driven architecture and stream processing capabilities.*

---

## 🏁 Final Notes

This project demonstrates:

- ✅ Real Kafka operations (KRaft)
- ✅ Streaming ML engineering
- ✅ Production debugging skills
- ✅ Dashboard-driven monitoring

**This is NOT a toy project.**  
It's **portfolio-ready** and **interview-strong**.

---


## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/Sri-Geyan/projects/issues).

---

## 👨‍💻 Author

**Sri Geyan**  
Real-Time ML & Streaming Systems  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/sri-geyan-558769252/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Sri-Geyan)

---

## 📚 References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [KRaft Mode Migration Guide](https://kafka.apache.org/documentation/#kraft)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## ⚖️ Disclaimer

This project is for **educational and demonstration purposes only**.  
It does **not** constitute fraud detection advice or production-ready fraud prevention systems.

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you understand real-time streaming ML!
