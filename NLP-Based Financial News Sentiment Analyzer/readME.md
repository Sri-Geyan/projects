# 📊 NLP-Based Financial News Sentiment Analyzer (FinBERT)

An **end-to-end financial NLP system** that quantifies market sentiment from unstructured news using **FinBERT (Transformer-based NLP)** and contextualizes it with stock price movements.

This project demonstrates how textual market information can be transformed into **actionable quantitative signals**, a technique widely used in hedge funds, trading desks, and financial research teams.

---

## 🚀 Why This Project Matters

Financial markets are **information-driven**. Prices react not just to numbers, but to **language**—earnings reports, analyst commentary, macroeconomic news, and forward guidance.

This system answers a key question:

> **Can financial news sentiment be quantified and analyzed alongside stock price movements?**

---

## 🧠 Key Capabilities

- 📰 Financial news sentiment analysis using domain-specific BERT (**FinBERT**)
- 🔢 Conversion of unstructured text into numeric sentiment signals
- 📈 Stock price ingestion and visualization
- 🔗 Alignment of sentiment with next-day market returns
- 🌐 Interactive Streamlit dashboard
- 🧪 Batch pipeline for quantitative analysis
- 🛠️ Production-safe architecture with robust debugging

---

## 🏗️ System Architecture

```
Financial News Headline
        ↓
Text Preprocessing
        ↓
FinBERT (Transformer NLP)
        ↓
Sentiment Score (-1 → +1)
        ↓
Stock Price Data (CSV / API)
        ↓
Correlation & Visualization
```

---

## 📁 Project Structure

```
financial-news-sentiment/
│
├── app.py                      # Streamlit dashboard
├── run_pipeline.py             # Batch NLP + correlation pipeline
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── news.csv                # Financial news data
│   └── prices.csv              # Stock OHLC data
│
└── src/
    ├── __init__.py             # Python package marker
    ├── load_data.py            # Data loaders
    ├── preprocess.py           # Text preprocessing
    ├── sentiment.py            # FinBERT inference logic
    └── analysis.py             # Returns & correlation analysis
```

---

## 🧰 Tech Stack

- **Python** 3.10 / 3.11
- **Transformers** (Hugging Face)
- **PyTorch**
- **FinBERT** (ProsusAI/finbert)
- **pandas**, **numpy**
- **Streamlit**
- **yfinance** (optional)
- **nltk**

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/financial-news-sentiment.git
cd financial-news-sentiment
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏳ **First run downloads the FinBERT model (~400MB).**

---

## ▶️ How to Run

### 🔹 Batch Pipeline (NLP + Correlation)

```bash
python run_pipeline.py
```

This executes:
- News ingestion
- FinBERT sentiment scoring
- Daily sentiment aggregation
- Next-day return alignment
- Correlation analysis

### 🔹 Streamlit Dashboard

```bash
streamlit run app.py
```

**Open in browser:**
```
http://localhost:8501
```

---

## 🌐 Streamlit App Overview

The dashboard allows you to:

- ✍️ Enter a financial news headline
- 🧠 View FinBERT sentiment score with confidence
- 📊 Visualize recent stock price movement
- 🔍 Interpret sentiment signals in market context

---

## 📊 Sentiment Scoring Methodology

FinBERT outputs a **label** and **confidence score**.

| Label | Numeric Mapping |
|-------|----------------|
| Positive | +confidence |
| Neutral | 0.0 |
| Negative | -confidence |

**Example:**

**Sentiment Score: -0.93** → **Strongly Bearish**

---

## 📈 Important Interpretation Note

**Sentiment is a signal, not a prediction.**

Negative sentiment does **not** guarantee immediate price declines.

Markets may:
- Price in information earlier
- React with a lag
- Ignore expected news

This project intentionally **separates signal generation from market reaction**, reflecting **real-world quantitative research practices**.

---

## 🧪 Mock Data

Synthetic but **market-realistic** datasets are provided:

- **news.csv** — Financial headlines with dates & tickers
- **prices.csv** — OHLC stock price data

These allow **validation of the pipeline** before connecting to live data sources.

---

## 🛠️ Troubleshooting & Engineering Fixes

This project documents **real production issues and resolutions**, including:

### ✅ Python Import Errors
- Fixed by structuring `src/` as a package (`__init__.py`)
- Cleared stale bytecode (`__pycache__`, `.pyc`)

### ✅ Silent Pipeline Execution
- Added explicit logging
- Handled long FinBERT load times

### ✅ Blank Streamlit Screen
- Fixed by lazy-loading FinBERT
- Added UI spinners and health checks

### ✅ Module State Mismatch
- Rebuilt corrupted modules
- Verified runtime symbols programmatically

**This reflects real-world debugging, not toy examples.**

---

## 🧠 Resume-Ready Description

*Built an NLP-based financial news sentiment analyzer using FinBERT (Transformer-based model) to quantify market sentiment from unstructured text and evaluated its relationship with next-day stock returns through correlation analysis, deployed as an interactive Streamlit dashboard.*

---

## 🎯 Skills Demonstrated

- ✔️ Transformer-based NLP (FinBERT)
- ✔️ Financial text understanding
- ✔️ Quantitative time-series alignment
- ✔️ Python package design
- ✔️ Debugging complex runtime issues
- ✔️ Streamlit deployment
- ✔️ Production-oriented thinking

---

## 🚀 Future Enhancements

- [ ] Real-time news ingestion with Kafka
- [ ] Multi-stock sentiment index
- [ ] Lagged backtesting (T+1, T+3, T+5)
- [ ] TF-IDF fallback model
- [ ] Dockerized deployment
- [ ] REST API for sentiment scoring
- [ ] MLflow experiment tracking
- [ ] A/B testing framework for models
- [ ] Integration with Bloomberg/Reuters feeds

---

## 📊 Sample Output

<img width="464" height="166" alt="Output" src="https://github.com/user-attachments/assets/c87c9add-a71c-40a3-b4d2-25447f643955" />

<img width="812" height="791" alt="Output" src="https://github.com/user-attachments/assets/364921b6-53c5-409c-84cf-3faff1b87806" />

<img width="1470" height="956" alt="Output" src="https://github.com/user-attachments/assets/5349c45d-3cd7-4c7a-afa7-20f1c0ad4f88" />


```
=== Financial News Sentiment Analysis ===

Headline: "Apple reports record Q4 earnings, beats analyst expectations"
Sentiment: Positive (+0.87)
Interpretation: Strong bullish signal

Headline: "Fed signals potential rate hikes amid inflation concerns"
Sentiment: Negative (-0.72)
Interpretation: Bearish market sentiment

Average Daily Sentiment: +0.23
Next-Day Return Correlation: 0.34
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/Sri-Geyan/projects/issues).

---

## 👨‍💻 Author

**Sri Geyan**  
Financial NLP & Quantitative Analysis

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/sri-geyan-558769252/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Sri-Geyan)

---

## 📚 References

- [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Financial Sentiment Analysis Literature](https://www.tandfonline.com/toc/rquf20/current)

---

## ⚖️ Disclaimer

This project is for **educational and research purposes only**.  
It does **not** constitute financial or investment advice.

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you understand financial NLP and sentiment analysis!

---

## 🔗 Related Projects

- [Portfolio Optimization System](https://github.com/Sri-Geyan/projects/tree/main/Portfolio%20Optimization%20%26%20Risk%20Monitoring%20System)
- [Credit Risk Scoring Engine](https://github.com/Sri-Geyan/projects/tree/main/Large-Scale%20Credit%20Risk%20Scoring%20Engine)
- [Real-Time Fraud Detection](https://github.com/Sri-Geyan/projects/tree/main/Real-Time%20Credit%20Card%20Fraud%20Detection%20System)
