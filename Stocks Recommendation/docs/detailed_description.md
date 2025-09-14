# 🔬 Detailed Description of Investment Recommendation System

## 1. Workflow Overview
The app follows an **ETL + Analytics pipeline**:

```mermaid
flowchart TD
    A[User enters tickers] --> B[Fetch price data from Yahoo Finance]
    B --> C[Compute features (Return, Volatility, Sharpe, Drawdown)]
    C --> D[Risk bucketing with K-Means clustering]
    D --> E[Content-based similarity (cosine similarity)]
    E --> F[Collaborative boost (synthetic users)]
    F --> G[Ranking system with adjustable weights]
    G --> H[Visualizations: scatter, bar chart]
```

---

## 2. Feature Engineering
For each asset, we calculate:

- **Annualized Return** = `(1 + mean_daily)^252 - 1`  
- **Volatility** = `std(daily_returns) * sqrt(252)`  
- **Sharpe Ratio** = `(annual_return - risk_free_rate) / volatility`  
- **Max Drawdown** = `(min(cumulative_return) - 1)`  
- **Mean Daily Return** = `avg(daily_returns)`

---

## 3. Ranking Method
We normalize each metric → assign weights → build a composite score:

**Score = (Return × W1) + (Sharpe × W2) + (Volatility × W3) + (Drawdown × W4) + (Mean Daily × W5)**

Users adjust **weights via sliders**.

---

## 4. Visual Representation

### Risk vs Return
- X-axis → Volatility  
- Y-axis → Annual Return  
- Color → Risk bucket (Low/Medium/High)  

### Ranking Bar Chart
- Y-axis → Tickers  
- X-axis → Composite Score  
- Color → Risk Level  

---

## 5. Example Use Case
👩‍💻 User enters: `AAPL, MSFT, TSLA, SPY, QQQ`  
- App shows **AAPL and MSFT are safer**  
- **TSLA has high return but high volatility**  
- Recommends **balanced portfolio** with optimized weights.
