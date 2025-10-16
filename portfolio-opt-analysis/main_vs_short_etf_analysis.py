import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

MAIN_TICKER = "CSPX.AS"
SHORT_TICKER = "3USS.MI"
MAIN = "CSPX"
SHORT = "3USS"

START = "2014-02-25"
END = None  # till today

def download_etf_data(tickers):
    """
    Download adjusted close and open prices for a list of tickers using yfinance.
    Returns a single DataFrame with proper column structure.
    """
    print(f"Downloading data for {tickers}...")
    
    # Download both tickers at once
    data = yf.download(tickers, start=START, end=END, progress=True, auto_adjust=False)
    
    if data.empty:
        raise RuntimeError("No ETF data could be downloaded – check internet or ticker symbols.")
    
    # Extract the data we need
    # yfinance returns multi-index columns when downloading multiple tickers
    main_ticker = tickers[0]
    short_ticker = tickers[1]
    
    # Create a clean DataFrame with the structure we need
    df = pd.DataFrame({
        'Date': data.index,
        f"{MAIN}_Open": data['Open'][main_ticker].values,
        f"{MAIN}_Close": data['Adj Close'][main_ticker].values,
        f"{SHORT}_Open": data['Open'][short_ticker].values,
        f"{SHORT}_Close": data['Adj Close'][short_ticker].values
    })
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save raw downloaded data
    df.to_csv(f"{MAIN}_{SHORT}_data.csv", index=False)
    print(f"✓ Raw data saved to: {MAIN}_{SHORT}_data.csv")
    print(f"✓ Downloaded {len(df)} trading days")
    print(f"✓ Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    return df

def load_data_from_csv(filename):
    """Load stock data from CSV file (for static file usage)"""
    df = pd.read_csv(filename, skiprows=1)
    # Parse columns: Date, {MAIN} Open, {MAIN} Close, Date, {SHORT} Open, {SHORT} Close
    df = df.iloc[:, [0, 2, 4, 5]]
    df.columns = ['Date', f"{MAIN}_Close", f"{SHORT}_Open", f"{SHORT}_Close"]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def calculate_statistics(df):
    """Calculate mean return and volatility for both assets"""
    df[f"{MAIN}_Return"] = df[f"{MAIN}_Close"].pct_change()
    df[f"{SHORT}_Return"] = df[f"{SHORT}_Close"].pct_change()
    
    main_returns = df[f"{MAIN}_Return"].dropna()
    short_returns = df[f"{SHORT}_Return"].dropna()
    
    stats = {
        f"{MAIN}_mean": main_returns.mean(),
        f"{MAIN}_std": main_returns.std(),
        f"{SHORT}_mean": short_returns.mean(),
        f"{SHORT}_std": short_returns.std(),
        'correlation': main_returns.corr(short_returns),
        'data_points': len(main_returns),
        'period': f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    }
    return stats, df

def optimize_portfolio(stats):
    """Find optimal weights to maximize return"""
    best_weight = 0
    best_return = -np.inf
    
    for w in np.arange(0, 1.01, 0.01):
        portfolio_return = w * stats[f"{MAIN}_mean"] + (1 - w) * stats[f"{SHORT}_mean"]
        if portfolio_return > best_return:
            best_return = portfolio_return
            best_weight = w
    
    return {f"{MAIN}": best_weight, f"{SHORT}": 1 - best_weight}

def simulate_strategy(df, weight_main, initial_capital=3000, transaction_fee=7, rebalance_days=65):
    """
    Simulate portfolio performance with given weights.
    rebalance_days: approximately 13 weeks = 65 trading days (assuming ~5 trading days/week)
    """
    
    capital = initial_capital - transaction_fee
    main_shares = (capital * weight_main) / df[f"{MAIN}_Close"].iloc[0]
    short_shares = (capital * (1 - weight_main)) / df[f"{SHORT}_Close"].iloc[0]
    
    portfolio_values = []
    total_fees = transaction_fee
    days_since_rebalance = 0
    
    for idx, row in df.iterrows():
        days_since_rebalance += 1
        
        main_value = main_shares * row[f"{MAIN}_Close"]
        short_value = short_shares * row[f"{SHORT}_Close"]
        total_value = main_value + short_value
        
        portfolio_values.append({
            'Date': row['Date'],
            'Total_Value': total_value,
            f"{MAIN}_Value": main_value,
            f"{SHORT}_Value": short_value
        })
        
        # Quarterly rebalancing (approximately 65 trading days = ~3 months)
        if days_since_rebalance >= rebalance_days:
            days_since_rebalance = 0
            total_fees += transaction_fee
            
            target_main = total_value * weight_main
            target_short = total_value * (1 - weight_main)
            
            main_shares = target_main / row[f"{MAIN}_Close"]
            short_shares = target_short / row[f"{SHORT}_Close"]
    
    portfolio_df = pd.DataFrame(portfolio_values)
    final_value = portfolio_df['Total_Value'].iloc[-1]
    total_return = ((final_value - total_fees) / initial_capital - 1) * 100
    
    # Calculate annual return based on actual days
    years = len(df) / 252  # 252 trading days per year
    annual_return = (((final_value - total_fees) / initial_capital) ** (1 / years) - 1) * 100
    profit = final_value - initial_capital
    
    return {
        'portfolio': portfolio_df,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'total_fees': total_fees,
        'profit': profit
    }

def run_analysis(use_csv=False, csv_filename='input_data.csv'):
    """Run complete portfolio optimization analysis"""
    
    print("="*60)
    print(f"PORTFOLIO OPTIMIZATION: {MAIN} vs {SHORT}")
    print("="*60)
    
    if use_csv:
        print(f"\nLoading data from CSV file: {csv_filename}...")
        df = load_data_from_csv(csv_filename)
    else:
        print("\nDownloading live data from Yahoo Finance...")
        try:
            df = download_etf_data([MAIN_TICKER, SHORT_TICKER])
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Falling back to CSV file...")
            df = load_data_from_csv(csv_filename)
    
    print(f"\n✓ Data loaded successfully")
    print(f"✓ Total records: {len(df)}")
    
    print("\nCalculating statistics...")
    stats, df = calculate_statistics(df)
    
    print("Optimizing portfolio...")
    optimal_weights = optimize_portfolio(stats)
    
    # Simulate different strategies
    print("Simulating strategies...")
    strategies = {
        'optimal': simulate_strategy(df, optimal_weights[MAIN]),
        f"{MAIN}_only": simulate_strategy(df, 1.0),
        f"{SHORT}_only": simulate_strategy(df, 0.0),
        'balanced': simulate_strategy(df, 0.5),
        f"{MAIN}_heavy": simulate_strategy(df, 0.7),
        f"{SHORT}_heavy": simulate_strategy(df, 0.3),
    }
    
    # Print statistics
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Period: {stats['period']}")
    print(f"Data Points: {stats['data_points']} trading days")
    print(f"\{MAIN} ETF:")
    print(f"  Mean Daily Return: {stats[f"{MAIN}_mean"]*100:.3f}%")
    print(f"  Daily Volatility: {stats[f"{MAIN}_std"]*100:.3f}%")
    print(f"  Annualized Return: {stats[f"{MAIN}_mean"]*252*100:.2f}%")
    print(f"  Annualized Volatility: {stats[f"{MAIN}_std"]*np.sqrt(252)*100:.2f}%")
    print(f"\n{SHORT} ETF:")
    print(f"  Mean Daily Return: {stats[f"{SHORT}_mean"]*100:.3f}%")
    print(f"  Daily Volatility: {stats[f"{SHORT}_std"]*100:.3f}%")
    print(f"  Annualized Return: {stats[f"{SHORT}_mean"]*252*100:.2f}%")
    print(f"  Annualized Volatility: {stats[f"{SHORT}_std"]*np.sqrt(252)*100:.2f}%")
    print(f"\nCorrelation: {stats['correlation']:.4f}")
    
    print("\n" + "="*60)
    print("OPTIMAL ALLOCATION (Pure Return Maximization)")
    print("="*60)
    print(f"{MAIN} : {optimal_weights[f"{MAIN}"]*100:.2f}% (€{3000 * optimal_weights[f"{MAIN}"]:.2f})")
    print(f"{SHORT} : {optimal_weights[f"{SHORT}"]*100:.2f}% (€{3000 * optimal_weights[f"{SHORT}"]:.2f})")
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    comparison_data = []
    for name, data in strategies.items():
        if name == 'optimal':
            main_pct = optimal_weights[f"{MAIN}"] * 100
            short_pct = optimal_weights[f"{SHORT}"] * 100
        elif name == f"{MAIN}_only":
            main_pct, short_pct = 100, 0
        elif name == f"{SHORT}_only":
            main_pct, short_pct = 0, 100
        elif name == 'balanced':
            main_pct, short_pct = 50, 50
        elif name == f"{MAIN}_heavy":
            main_pct, short_pct = 70, 30
        else:  # {SHORT}_heavy
            main_pct, short_pct = 30, 70
        
        comparison_data.append({
            'Strategy': name.replace('_', ' ').title(),
            f"{MAIN}_%": f"{main_pct:.0f}%",
            f"{SHORT}_%": f"{short_pct:.0f}%",
            'Final_Value': f"€{data['final_value']:.2f}",
            'Total_Return_%': f"{data['total_return']:.2f}%",
            'Annual_Return_%': f"{data['annual_return']:.2f}%",
            'Profit_Loss': f"€{data['profit']:.2f}",
            'Total_Fees': f"€{data['total_fees']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save results to CSV
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    comparison_df.to_csv('portfolio_comparison.csv', index=False)
    print("✓ Strategy comparison saved to: portfolio_comparison.csv")
    
    # Save detailed portfolio performance
    for name, data in strategies.items():
        portfolio_df = data['portfolio'].copy()
        portfolio_df['Date'] = portfolio_df['Date'].dt.strftime('%Y-%m-%d')
        filename = f"portfolio_performance_{name}.csv"
        portfolio_df.to_csv(filename, index=False)
        print(f"✓ Performance data saved to: {filename}")
    
    # Create visualization
    create_visualizations(strategies, stats, optimal_weights)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

def create_visualizations(strategies, stats, optimal_weights):
    """Create visualization charts"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Portfolio Optimization Analysis: {MAIN} vs {SHORT} ", 
                 fontsize=16, fontweight='bold')
    
    # Chart 1: Portfolio Performance Comparison
    ax1 = axes[0, 0]
    for name, data in strategies.items():
        portfolio_df = data['portfolio']
        ax1.plot(portfolio_df['Date'], portfolio_df['Total_Value'], 
                label=name.replace('_', ' ').title(), linewidth=2)
    ax1.set_title('Portfolio Performance Over Time', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value (€)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=45)
    
    # Chart 2: Strategy Comparison Bar Chart
    ax2 = axes[0, 1]
    strategy_names = []
    total_returns = []
    for name, data in strategies.items():
        strategy_names.append(name.replace('_', ' ').title())
        total_returns.append(data['total_return'])
    
    colors = ['green' if x > 0 else 'red' for x in total_returns]
    ax2.bar(range(len(strategy_names)), total_returns, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(strategy_names)))
    ax2.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax2.set_title('Total Return by Strategy', fontweight='bold')
    ax2.set_ylabel('Total Return (%)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Optimal vs 100% Strategies
    ax3 = axes[1, 0]
    optimal_portfolio = strategies['optimal']['portfolio']
    main_portfolio = strategies[f"{MAIN}_only"]['portfolio']
    short_portfolio = strategies[f"{SHORT}_only"]['portfolio']
    
    ax3.plot(optimal_portfolio['Date'], optimal_portfolio['Total_Value'], 
            label='Optimal', linewidth=2.5, color='green')
    ax3.plot(main_portfolio['Date'], main_portfolio['Total_Value'], 
            label="100% {MAIN}", linewidth=2, linestyle='--', color='blue')
    ax3.plot(short_portfolio['Date'], short_portfolio['Total_Value'], 
            label="100% {SHORT}", linewidth=2, linestyle='--', color='red')
    ax3.set_title('Optimal vs Pure Strategies', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Portfolio Value (€)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Asset Allocation Comparison
    ax4 = axes[1, 1]
    allocations = {
        'Optimal': [optimal_weights[f"{MAIN}"]*100, optimal_weights[f"{SHORT}"]*100],
        '50/50': [50, 50],
        '70% {MAIN}': [70, 30],
        '30% {MAIN}': [30, 70]
    }
    
    x_pos = np.arange(len(allocations))
    main_vals = [v[0] for v in allocations.values()]
    short_vals = [v[1] for v in allocations.values()]
    
    width = 0.35
    ax4.bar(x_pos - width/2, main_vals, width, label=f"{MAIN}", color='blue', alpha=0.7)
    ax4.bar(x_pos + width/2, short_vals, width, label=f"{SHORT}", color='red', alpha=0.7)
    ax4.set_ylabel('Allocation (%)')
    ax4.set_title('Asset Allocation Comparison', fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(allocations.keys())
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('portfolio_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: portfolio_analysis.png")
    plt.show()

if __name__ == "__main__":
    # Set use_csv=True to use a static CSV file instead of downloading
    # Set use_csv=False to download live data from Yahoo Finance
    run_analysis(use_csv=False)
