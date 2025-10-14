"""
ETF vs Shiller TR-CAPE Analysis
-------------------------------

This script downloads and merges Shiller’s Total Return CAPE (valuation metric)
with monthly prices of major ETFs (e.g. VOO, SPY, QQQ, VTI, etc.) from Yahoo Finance.
It produces:

1. A time-series plot comparing normalized ETF growth vs. CAPE.
2. Scatter plots showing the relationship between CAPE and forward (e.g. 5-year) returns.

The outputs help visualize how valuation levels (CAPE) have historically related
to future returns across key passive index ETFs.
"""

import io
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.stats import pearsonr

ETF_TICKERS = [ # largest asset value ETFs https://etfdb.com/compare/market-cap/
    "VOO","IVV","SPY","VTI","QQQ","VUG","VEA","IEFA","VTV","BND","AGG","GLD","IWF","VGT","IEMG","VXUS","VWO"
    #"VOO"
]

#START = "2000-01-01"
START = "2012-10-01" # all datapoint available from this time for the above ETF_TICKERS
END = None  # till today
OUTPUT_DIR = "indexes_vs_cape_outputs"

def download_etf_data(tickers):
    """
    Download monthly adjusted close prices for a list of tickers using yfinance.
    Returns a single DataFrame with columns = tickers.
    """
    frames = []
    for t in tickers:
        print(f"Downloading {t} ...")
        df = yf.download(t, start=START, end=END, progress=False, auto_adjust=False)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{t}_out.csv"))
        if df.empty:
            print(f"⚠️ Warning: No data for {t}, skipping.")
            continue
        df_m = df["Adj Close"].resample("ME").last() #.to_frame(name=t)
        df_m.to_csv(os.path.join(OUTPUT_DIR, f"{t}_adjclose_out.csv"))
        frames.append(df_m)
        #df_m.info()
    if not frames:
        raise RuntimeError("No ETF data could be downloaded — check internet or ticker symbols.")
    df_all = pd.concat(frames, axis=1)
    df_all.to_csv(os.path.join(OUTPUT_DIR, f"all_adjclose_out.csv"))
    return df_all


#SHILLER_TRCAPE_URL = "http://www.econ.yale.edu/~shiller/data/ie_data_with_TRCAPE.xls"
#SHILLER_TRCAPE_URL = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/4a0629e8-3f7f-4884-817b-81cf106d8f88/ie_data.xls" # from: https://shillerdata.com/
SHILLER_TRCAPE_URL = "https://pazsitz.hu/etc/ie_data.xls"
CAPE_TR_XLS_COL = "CAPE_TR" # "CAPE"
cape_tr_col = CAPE_TR_XLS_COL

def download_shiller_trcape_df():
    global cape_tr_col
    """
    Download the Shiller total-return CAPE Excel file and parse it to extract the TRCAPE series.
    Returns a DataFrame indexed by month, with column 'CAPE_TR' (or alternate name).
    """
    resp = requests.get(SHILLER_TRCAPE_URL, timeout=30)
    resp.raise_for_status()
    xls = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data", header=7, index_col=None)
    # Explicitly rename columns to unique names
    xls.columns = [
        'Date', 'P', 'D', 'E', 'CPI', 'Fraction', 'Rate_GS10', 'Price', 'Dividend', 'Price_1', 
        'Earnings', 'Earnings_1', 'CAPE', 'unnamed 13', 'CAPE_TR', 'Unnamed: 15', 'Yield', 'Returns', 'Returns.1',
        'Real Return', 'Real Return.1', 'Returns.2'
    ]
    
    df = xls[['Date', 'CAPE_TR']].copy()
    
    # convert Shiller YYYY.MM to datetime
    def shiller_to_datetime(x):
        if pd.isna(x):
            return pd.NaT
        year = int(x)
        month = round((x - year) * 100)
        month = max(1, min(month, 12))
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    
    df['Date'] = df['Date'].apply(shiller_to_datetime)
    df = df.set_index('Date').resample('ME').last()
    return df[['CAPE_TR']].dropna()

def prepare_combined():
    df_etf = download_etf_data(ETF_TICKERS)
    df_cape = download_shiller_trcape_df()  # make sure this returns a column named 'CAPE_TR'

    common_start = max(df_etf.index.min(), df_cape.index.min())
    common_end   = min(df_etf.index.max(), df_cape.index.max())
    print(f"common start: {common_start} ; commond end: {common_end}")

    df_etf = df_etf.loc[common_start:common_end]
    df_cape = df_cape.loc[common_start:common_end]

    df = df_etf.join(df_cape, how='inner')

    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="any")

    # Normalize ETFs safely
    #for t in [c for c in ETF_TICKERS if c in df.columns]:
    for t in ETF_TICKERS:
        if t not in df.columns:
            print(f"⚠️ Skipping {t} — not in DataFrame")
            continue
        first_valid = df[t].first_valid_index()
        if first_valid is None:
            print(f"⚠️ Skipping {t} — no valid data")
            continue
   
        base_value = df.loc[first_valid, t]
        if pd.isna(base_value):
            print(f"⚠️ Skipping {t} — base value is NaN.")
            continue
        df[f"{t}_norm"] = df[t] / base_value
        
        # forward 5y return
        df[f"{t}_fwd5y"] = df[t].shift(-60) / df[t]
        df[f"{t}_fwd5y_ann"] = df[f"{t}_fwd5y"] ** (1/5) - 1
   
    return df #.dropna()

def plot_summary(df):   
    plt.figure(figsize=(14,7))
    for t in ETF_TICKERS:
        col_norm = f"{t}_norm"
        if col_norm not in df.columns:
            print(f"⚠️ Skipping plot for {t} — normalized column not found")
            continue
        plt.plot(df.index, df[col_norm], label=t, alpha=0.7)
        
    plt.ylabel("Normalized ETF price")
    plt.twinx()
    plt.plot(df.index, df[cape_tr_col], color="black", linestyle="--", label=cape_tr_col)
    plt.title("ETF Performance vs Shiller CAPE")
    plt.legend()
    plt.show()

def correlation_table(df):
    corr = {}
    for t in ETF_TICKERS:
        corr[t] = pearsonr(df[cape_tr_col], df[f"{t}_fwd5y_ann"])[0]
    return pd.DataFrame.from_dict(corr, orient="index", columns=["CAPE correlation (5y fwd)"]).sort_values(by="CAPE correlation (5y fwd)")

def plot_etfs_vs_cape(df, etf_tickers, cape_col=CAPE_TR_XLS_COL, outdir=OUTPUT_DIR):
    """
    Time series plot: normalized ETFs vs Shiller TR-CAPE.
    """
    os.makedirs(outdir, exist_ok=True)
    
    plt.figure(figsize=(14,7))
    ax = plt.gca()
    
    # plot normalized ETFs
    for t in etf_tickers:
        col = f"{t}_norm"
        if col in df.columns:
            ax.plot(df.index, df[col], label=t, alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Normalized ETF prices')
    ax.legend(loc='upper left', fontsize=8)
    
    # twin axis for CAPE
    ax2 = ax.twinx()
    if cape_col in df.columns:
        ax2.plot(df.index, df[cape_col], color='black', linestyle='--', label=cape_col)
        ax2.set_ylabel('TR-CAPE')
        ax2.legend(loc='upper right')
    
    plt.title('ETF growth (normalized) vs Shiller TR-CAPE')
    plt.tight_layout()
    
    outpng = os.path.join(outdir, 'etfs_vs_cape_timeseries.png')
    plt.savefig(outpng, dpi=150)
    plt.close()
    return outpng


def plot_cape_vs_forward_return(df, etf_tickers, cape_col=CAPE_TR_XLS_COL, forward_months=60, outdir=OUTPUT_DIR):
    """
    Scatter plot of TR-CAPE vs forward returns for multiple ETFs.
    forward_months: e.g., 60 for 5-year monthly forward return
    """
    os.makedirs(outdir, exist_ok=True)
    
    results = {}
    
    for t in etf_tickers:
        if t not in df.columns:
            continue
        # forward return
        dfplot = df.copy()
        dfplot[f"{t}_fwd"] = dfplot[t].shift(-forward_months) / dfplot[t]
        dfplot[f"{t}_fwd_ann"] = dfplot[f"{t}_fwd"] ** (12/forward_months) - 1
        dfplot = dfplot.replace('NaN', pd.NA).dropna(axis=0, how="any")
        
        x = dfplot[cape_col]
        y = dfplot[f"{t}_fwd_ann"]
        corr, pval = pearsonr(x, y)
        results[t] = (corr, pval)
        
        plt.figure(figsize=(8,6))
        plt.scatter(x, y, s=10, alpha=0.6)
        plt.xlabel('TR-CAPE')
        plt.ylabel(f'{t} forward {forward_months//12}y annualized return')
        plt.title(f'{t}: TR-CAPE vs Forward Return\nPearson r={corr:.3f}, p={pval:.3g}')
        plt.grid(True)
        plt.tight_layout()
        outpng = os.path.join(outdir, f'{t}_cape_vs_fwdreturn.png')
        plt.savefig(outpng, dpi=150)
        plt.close()
        
    return results



if __name__ == "__main__":
    df = prepare_combined()
    
    df.to_csv(os.path.join(OUTPUT_DIR, f"indexes_vs_cape_summary.csv"))
    
    plot_etfs_vs_cape(df, ETF_TICKERS)
    plot_cape_vs_forward_return(df, ETF_TICKERS)
    
    
    
    
    #plot_summary(df)
    #corr = correlation_table(df)
    #print(corr)
