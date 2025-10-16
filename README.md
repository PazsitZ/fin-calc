# 📈 fin-calc — Financial Data Analysis \& Visualization Toolkit

**fin-calc** is a collection of Python scripts for quantitative analysis, visualization, and research on**stocks and ETFs**, designed to support**investment strategy development** and**data-driven decision-making**.

The project focuses on:

- Building reproducible workflows for fetching, cleaning, and combining financial data.

- Creating insightful visualizations (e.g. normalized growth, valuation trends, return correlations).

- Exploring valuation metrics such as**Shiller CAPE** and their predictive power on future returns.

---

## 📊 Main Modules

### **1. indexes_vs_cape.py**

Extends the CAPE analysis to a**diverse ETF set** (e.g. `VOO, SPY, QQQ, VTI, VUG, BND, GLD, IEFA, VWO`), combining**broad market**,**sector**, and**international** funds.

**Functions:**

- Downloads ETF price data via `yfinance`.

- Merges with**Shiller’s TR-CAPE** data (`ie_data_with_TRCAPE.xls`).

- Normalizes and plots ETF growth alongside CAPE.

- Computes and visualizes correlation between CAPE and**forward annualized ETF returns**.

**Outputs:**

- `etfs_vs_cape_timeseries.png`: All ETF normalized growth vs CAPE.

- Individual scatter plots per ETF showing CAPE vs forward returns (e.g. `QQQ_cape_vs_fwdreturn.png`).

### **2. main_vs_short_etf_analysis.py**

This portfolio optimization tool analyzes the optimal allocation strategy between two ETFs

**Functions:**

- Downloads ETF price data via `yfinance` or fetches from local file.

- Statistical: Computes daily/annualized returns, volatility, and correlation, Finds optimal weight allocation using mean-variance optimization

- Portfolio Simulation: Simulates portfolio performance with quarterly rebalancing

- Generates Visualization & Reporting

**Outputs:**

- *CSV Files* :
 Raw data (Date, Open, Close prices), Summary table, Daily portfolio values for optimal allocation,  5 strategy performance files

- *Visualization*: 4-panel dashboard PNG

---


## ⚙️ Installation

Clone and install dependencies:

```bash

git clone https://github.com/<yourname>/fin-calc.git

cd fin-calc

python -m venv .venv

source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

pip install -r requirements.txt
```

---

📘 Data Sources

- Shiller CAPE / TR-CAPE Data:
Robert Shiller’s Yale Data
- [yFinance](https://github.com/ranaroussi/yfinance) - Yahoo! Finance’s API
