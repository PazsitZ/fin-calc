📊 Script Overview — ETF vs Shiller TR-CAPE Analysis



This Python script downloads and combines Shiller’s Total Return CAPE (Cyclically Adjusted P/E) data with a set of major ETFs (such as VOO, SPY, QQQ, VTI, etc.) to explore long-term valuation and return relationships.



It performs three main tasks:



📥 Data Preparation



Fetches the Shiller TR-CAPE dataset from Yale’s public Excel file (ie\_data\_with\_TRCAPE.xls).



Downloads monthly adjusted closing prices for selected ETFs via Yahoo Finance.



Aligns and merges these datasets into a consistent monthly time series.



📈 Time-Series Visualization



Normalizes ETF prices (rebased to 1.0 at the starting point) to compare their relative growth.



Plots all ETFs alongside Shiller’s TR-CAPE on a twin-axis chart.



Saves the output as etfs\_vs\_cape\_timeseries.png.



📉 Valuation vs. Forward Return Analysis



Calculates forward annualized returns for each ETF over a configurable horizon (e.g., 5 years = 60 months).



Plots scatter plots of TR-CAPE versus forward returns for each ETF.



Computes and reports the Pearson correlation between valuation (CAPE) and future performance, highlighting potential predictive relationships.



Saves one scatter PNG per ETF (e.g., QQQ\_cape\_vs\_fwdreturn.png).



🎯 Purpose



The goal of this analysis is to visualize and quantify how valuation levels (CAPE) relate to future long-term returns across different asset classes and ETFs — helping to assess whether high CAPE values have historically corresponded to lower subsequent performance.

