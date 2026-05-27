"""
Day 1: Data Pipeline.

Pulls two complementary datasets and aligns them for downstream factor work:

    1. Fama-French 3-factor + Momentum daily returns from Ken French's library
       (the canonical academic factor returns, used in every paper since 1993).
    2. Daily prices for a universe of ~30 large-cap US stocks from yfinance,
       converted to daily simple returns.

Both datasets are aligned on a common date index and saved to parquet for fast
reload on later days.

Run with: python3 data.py
Produces: data/factors.parquet, data/stock_returns.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf


# ----------------------------- CONFIG (yours to set) -----------------------------


# We want ~30 large-cap US stocks spanning multiple sectors so factor exposures
# vary meaningfully across the cross-section. The names below are a reasonable
# starter set — tech, finance, healthcare, consumer, industrials, energy.
# You can adjust, but keep ~30 tickers. More creates noise; fewer creates
# undertrained cross-sectional regressions.
UNIVERSE = [
    # Tech / Comms
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "ORCL", "CRM",
    # Finance
    "JPM", "BAC", "GS", "MS", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK",
    # Consumer
    "WMT", "HD", "KO", "PG", "PEP", "MCD", "NKE",
    # Industrials / Materials
    "BA", "CAT", "GE",
    # Energy
    "XOM", "CVX",
    # Utility / Telecom
    "T", "VZ",
]


# We want enough history for stable factor estimation, but not so much that you're
# pulling stocks that didn't exist at the start (e.g., GOOGL IPO'd in 2004).
# 2015-01-01 to today gives ~10 years and avoids most pre-IPO issues for the
# universe above. Adjust if you want.
START_DATE = "2015-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")


# ----------------------------- FAMA-FRENCH FACTOR DATA (provided) -----------------------------
def fetch_ff_factors() -> pd.DataFrame:
    """Download Fama-French 3-factor + Momentum daily returns from Ken French.

    Returns
    -------
    pd.DataFrame with columns: Mkt-RF, SMB, HML, RF, MOM
        Index is daily dates. Values are decimal daily returns
        (e.g., 0.0123 = 1.23% return that day, NOT 1.23 in percent form).
    """
    # Three-factor + Momentum tables from Ken French's library, daily frequency
    ff3 = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                         start=START_DATE, end=END_DATE)[0]
    mom = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench",
                         start=START_DATE, end=END_DATE)[0]

    # Ken French returns values in percent (1.23 means 1.23%, not 123%).
    # Convert to decimal so they match yfinance's convention.
    ff3 = ff3 / 100
    mom = mom / 100

    # Momentum factor has whitespace in column name; standardize.
    mom.columns = ["MOM"]

    # Merge on date index
    factors = ff3.join(mom, how="inner")
    factors.index = pd.to_datetime(factors.index)

    return factors


# ----------------------------- STOCK PRICE DATA (provided) -----------------------------
def fetch_stock_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily adjusted close prices for a universe.

    Returns
    -------
    pd.DataFrame
        Index: dates. Columns: tickers. Values: adjusted close prices.
    """
    # yfinance's batch download: one call for the whole universe.
    # auto_adjust=True returns prices already adjusted for splits and dividends,
    # so the "Close" column is what you want for return computation.
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # When you pass multiple tickers, yfinance returns a MultiIndex on columns.
    # We want just the adjusted close prices.
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    # Drop any tickers with too many missing values (delisted, late IPO, etc.)
    valid_threshold = 0.9
    valid_tickers = prices.columns[prices.notna().mean() > valid_threshold]
    prices = prices[valid_tickers]

    # Drop dates where ANY stock has NaN — keeps the panel balanced
    prices = prices.dropna()

    return prices


# ----------------------------- RETURNS COMPUTATION -----------------------------
def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to daily simple returns.

    Returns
    -------
    pd.DataFrame
        Same index/columns as input, but values are returns
        and the first row is dropped (no return on day 0).
    """
    
    # Simple return on day t: R_t = (P_t / P_{t-1}) - 1
    
    # pandas method `pct_change()` does exactly this — for each column,
    # computes (current value / previous value) - 1.
    
    # After pct_change, the first row will be NaN (no previous day).
    
    returns = prices.pct_change().dropna()

    return returns


# ----------------------------- ALIGNMENT -----------------------------
def align_data(factors: pd.DataFrame, stock_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align factor and stock returns on a common date index.

    Returns the two DataFrames reindexed to their date intersection.
    """
    common_dates = factors.index.intersection(stock_returns.index)
    factors_aligned = factors.loc[common_dates]
    stocks_aligned = stock_returns.loc[common_dates]

    return factors_aligned, stocks_aligned


# ----------------------------- SANITY CHECKS -----------------------------
def report_summary(factors: pd.DataFrame, stock_returns: pd.DataFrame):
    """Print summary statistics so you can eyeball the data before saving."""
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(f"\nDate range: {factors.index.min().date()} to {factors.index.max().date()}")
    print(f"Trading days: {len(factors)}")
    print(f"Universe size: {len(stock_returns.columns)} stocks")

    print("\nFama-French factor returns — annualized statistics:")
    print("  (mean × 252 = annualized return; std × √252 = annualized vol)")
    summary_ff = pd.DataFrame({
        "Annualized Return (%)": factors.mean() * 252 * 100,
        "Annualized Volatility (%)": factors.std() * np.sqrt(252) * 100,
    })
    print(summary_ff.round(2))

    print("\nStock universe summary:")
    print(f"  Stocks retained: {list(stock_returns.columns)}")
    print(f"\n  Top 5 highest average daily return:")
    top5 = stock_returns.mean().nlargest(5) * 252 * 100
    for ticker, ret in top5.items():
        print(f"    {ticker}: {ret:.1f}% annualized")

    
    # A daily return greater than 50% or less than -50% almost certainly means
    # bad data (a stock split mishandled, a corporate event, etc.).
    
    extreme = (stock_returns.abs() > 0.5).any(axis=None)

    if extreme:
        print("WARNING: extreme daily returns detected — investigate")
    else:
        print("OK: no extreme returns detected")

# ----------------------------- MAIN PIPELINE -----------------------------
if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    print("Pulling Fama-French factor returns from Ken French Data Library...")
    factors = fetch_ff_factors()
    print(f"  Got {len(factors)} days of factor returns: {list(factors.columns)}")

    print(f"\nPulling {len(UNIVERSE)} stocks from yfinance...")
    prices = fetch_stock_prices(UNIVERSE)
    print(f"  Got {len(prices)} days of price data for {len(prices.columns)} stocks")

    print("\nComputing daily returns from prices...")
    stock_returns = compute_returns(prices)
    print(f"  Got {len(stock_returns)} days of returns")

    print("\nAligning factor and stock dates...")
    factors, stock_returns = align_data(factors, stock_returns)
    print(f"  Aligned to {len(factors)} common trading days")

    report_summary(factors, stock_returns)

    # Save to parquet for fast reload on later days
    factors.to_parquet(out_dir / "factors.parquet")
    stock_returns.to_parquet(out_dir / "stock_returns.parquet")
    print(f"\nSaved to data/factors.parquet and data/stock_returns.parquet")
