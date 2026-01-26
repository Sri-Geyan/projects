# Portfolio Optimization & Risk Monitoring System

An **institutional-grade portfolio optimization and risk monitoring system** for Indian equity markets (NSE), built using modern quantitative finance techniques and deployed as an interactive web application using **Streamlit**.

This system helps investors **optimize asset allocation**, **control sector exposure**, and **quantify downside risk** under real-world market uncertainty.

---

## 🎯 Problem Statement

Retail and professional investors often struggle to:

- Balance return vs risk  
- Avoid sector over-concentration  
- Understand downside risk and worst-case scenarios  

This project addresses these challenges by combining **portfolio optimization**, **risk analytics**, and **Monte Carlo stress testing** into a single decision-support system.

---

## 🚀 Key Features

### 📈 Portfolio Optimization
- Mean-Variance Optimization (Modern Portfolio Theory)
- Risk-adjusted allocation using Sharpe Ratio
- RBI 10-Year G-Sec based risk-free rate

### 🏭 Sector-Wise Constraints (India-Focused)
- Banking, IT, FMCG exposure controls
- Prevents over-concentration
- Mirrors real PMS / AMC allocation rules

### ⚠️ Risk Monitoring
- Annualized Volatility
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown

### 🔮 Monte Carlo Stress Testing
- Thousands of simulated future price paths
- Visualizes uncertainty, tail risk, and upside potential
- Helps evaluate worst-case scenarios

### 🖥️ Interactive Dashboard
- Built using Streamlit
- Real-time recalculation on asset selection
- Clean, dark-mode friendly UI

---

## 🏗️ System Architecture

```
Market Data (NSE Stocks)
        ↓
Return & Covariance Engine
        ↓
Risk Engine (VaR / CVaR)
        ↓
Optimization Engine
(Mean-Variance + Sector Constraints)
        ↓
Monte Carlo Simulation
        ↓
Interactive Decision Dashboard
```

---

## 🧠 Financial Concepts Used

- Risk vs Return Trade-off
- Volatility & Correlation
- Diversification Benefits
- Tail Risk Measurement
- Scenario-Based Stress Testing

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Data | yFinance (NSE) |
| Computation | NumPy, Pandas |
| Optimization | SciPy |
| Risk Analytics | Statistical methods |
| Visualization | Matplotlib |
| UI | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📂 Project Structure

```
portfolio_system/
│
├── app.py                 # Streamlit dashboard
├── data.py                # Market data ingestion
├── optimization.py        # Portfolio optimization logic
├── risk.py                # Risk metrics & sector exposure
├── simulation.py          # Monte Carlo simulation
├── sectors.py             # Sector definitions & constraints
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ▶️ How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 📊 Sample Output Metrics

- **Expected Annual Return:** ~13–15%
- **Annualized Volatility:** ~16–18%
- **95% VaR:** ~-1.5%
- **95% CVaR:** ~-2.5%
- Sector-balanced asset allocation

*Exact values depend on selected stocks and historical time period.*

![Output](https://github.com/user-attachments/assets/9d97c5f3-a51e-43cd-ba51-309158e3fe50)

![Output](https://github.com/user-attachments/assets/ff9cd522-2ecc-46a7-9019-2d9323444814)



---

## 🧪 Example Use Cases

- Retail investors seeking diversified Indian equity portfolios
- Risk profiling for PMS / RIA clients
- Educational tool for quantitative finance
- Resume-grade portfolio project for Data Science / Quant roles

---

## 📌 Future Enhancements

- [ ] NIFTY benchmark alpha & beta analysis
- [ ] SEBI risk classification (Low / Moderate / High)
- [ ] Sector rotation strategies
- [ ] Historical crisis replay (COVID-19 / 2008)
- [ ] PDF-based risk report generation
- [ ] Spark-based large stock universe optimization

---

## 🧾 Resume-Ready Description

*Built an Indian equity portfolio optimization and risk monitoring system using Mean-Variance Optimization, sector-wise exposure constraints, VaR/CVaR risk metrics, and Monte Carlo stress testing, deployed as an interactive Streamlit dashboard.*

---

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**.  
It does **not** constitute financial or investment advice.

---

## 👨‍💻 Author

**Sri Geyan**  
Data Science & Quantitative Finance Enthusiast

---

