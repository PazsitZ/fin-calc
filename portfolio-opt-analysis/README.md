\# Portfolio Optimization on a main ETF and it's short version

## Features:
### Data Analysis:

- Loads data from yfinance or local file
- Calculates daily returns, volatility, and correlation
- Performs mean-variance portfolio optimization
- Simulates 6 different allocation strategies

### Quarterly Rebalancing:

- Simulates real-world portfolio management
- Accounts for €7 transaction fees
- Rebalances every 65 trading days (~3 months) to maintain target allocation
- Annual return calculation uses 252 trading days (market standard) instead of 52 weeks

### Output Files (CSV):

- portfolio_comparison.csv - Summary of all strategies (returns, fees, profit/loss)
- portfolio_performance_optimal.csv - Detailed performance data for optimal strategy
- portfolio_performance_{main}_only.csv - 100% "main" performance
- portfolio_performance_{short}_only.csv - 100% "short" performance
vPlus performance files for balanced and weighted strategies

### Visualizations (PNG):

- Portfolio Performance Over Time - Line chart showing all strategies
- Strategy Comparison - Bar chart of total returns
- Optimal vs Pure Strategies - Head-to-head comparison
- Asset Allocation Comparison - Pie/bar chart of allocations

### Console Output:

- Detailed statistics (mean returns, volatility, correlation)
- Optimal allocation percentages
- Strategy comparison table
- File save confirmations



