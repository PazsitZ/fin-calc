# Portfolio Optimization: on a Main ETF and it's Short version

## Project Description

This portfolio optimization tool analyzes the optimal allocation strategy between two ETFs: **Main** - i.e. a standard S&P 500 tracker, and **Short** - i.e. a -3x leveraged inverse S&P 500 ETF. The tool uses mean-variance optimization to maximize returns from a €3,000 initial investment with quarterly rebalancing. It downloads live market data, performs statistical analysis, simulates multiple allocation strategies, and generates comprehensive performance reports with visualizations.

---

## Key Functions

### **Data Management**
- `download_etf_data()` - Downloads live historical price data from Yahoo Finance
- `load_data_from_csv()` - Loads data from static CSV files as fallback

### **Statistical Analysis**
- `calculate_statistics()` - Computes daily/annualized returns, volatility, and correlation
- `optimize_portfolio()` - Finds optimal weight allocation using mean-variance optimization

### **Portfolio Simulation**
- `simulate_strategy()` - Simulates portfolio performance with quarterly rebalancing
- Accounts for transaction fees (€7 per rebalancing)
- Tests 6 different allocation strategies (optimal, 100% long, 100% short, balanced, weighted)

### **Visualization & Reporting**
- `create_visualizations()` - Generates 4-panel chart with performance comparisons
- `run_analysis()` - Orchestrates complete analysis workflow and output generation

---

## Outputs

### **CSV Files (8 files)**
1. **{main}_{short}_data.csv** - Raw downloaded market data (Date, Open, Close prices)
2. **portfolio_comparison.csv** - Summary table comparing all 6 strategies (returns, fees, profit/loss)
3. **portfolio_performance_optimal.csv** - Daily portfolio values for optimal allocation
4. **portfolio_performance_{main}_only.csv** - 100% Main strategy performance
5. **portfolio_performance_{short}_only.csv** - 100% Short strategy performance
6. **portfolio_performance_balanced.csv** - 50/50 allocation performance
7. **portfolio_performance_{main}_heavy.csv** - 70% Main / 30% Short performance
8. **portfolio_performance_{short}_heavy.csv** - 30% Main / 70% Short performance

### **Visualization (1 PNG file)**
**portfolio_analysis.png** - 4-panel dashboard containing:
- Panel 1: Portfolio performance over time (all strategies)
- Panel 2: Total return comparison bar chart
- Panel 3: Optimal vs pure strategies (head-to-head comparison)
- Panel 4: Asset allocation comparison across strategies

### **Console Output**
- Statistical summary (mean returns, volatility, correlation)
- Optimal allocation percentages with euro amounts
- Strategy comparison table with returns and profit/loss
- Progress messages and file save confirmations

---

## Mathematical Foundation

**Optimization Objective:**  
Maximize: `E[Rp] = w₁μ₁ + w₂μ₂`  
Subject to: `w₁ + w₂ = 1, w₁,w₂ ≥ 0`

Where:
- `w₁, w₂` = portfolio weights for SXR8 and 3USS
- `μ₁, μ₂` = expected returns
- Portfolio variance: `σ²p = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂`

**Rebalancing:** Quarterly (every 65 trading days) with €7 transaction fee per rebalancing event.

---

## Use Cases

✓ **Long-term investment strategy** - Determine optimal allocation for 5-year horizon  
✓ **Risk-return analysis** - Compare leveraged vs traditional ETF performance  
✓ **Hedging strategy evaluation** - Assess inverse ETF effectiveness for portfolio protection  
✓ **Transaction cost impact** - Understand how rebalancing fees affect returns  
✓ **Historical backtesting** - Test allocation strategies on 12+ years of market data
