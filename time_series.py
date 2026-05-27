"""
Day 2: Time-Series Regressions.

For each stock in our universe, regress its excess returns on the
Fama-French + Momentum factors:

    R_i - R_f = alpha_i + beta_i*(R_m - R_f) + s_i*SMB + h_i*HML + m_i*MOM + eps_i

We use Newey-West standard errors (HAC, maxlags=5) because daily return
residuals are heteroskedastic and autocorrelated. Without this correction,
t-stats are systematically too high.


Run with: python3 time_series.py
Produces: data/factor_loadings.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ----------------------------- LOAD DATA -----------------------------
def load_aligned_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the parquet files saved by data.py."""
    factors = pd.read_parquet("data/factors.parquet")
    stock_returns = pd.read_parquet("data/stock_returns.parquet")
    return factors, stock_returns


# ----------------------------- EXCESS RETURNS -----------------------------
def compute_excess_returns(stock_returns: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Subtract the daily risk-free rate from each stock's daily return.

    Both inputs are aligned to the same date index.
    """

    # pandas broadcasts a Series against a DataFrame on the column axis
    # by default, so the cleanest version is:
    #
    #   excess = stock_returns.sub(rf, axis=0)
    #
    # `axis=0` says "align on the row index (dates), subtract rf from each column."
    excess_returns = stock_returns.sub(rf, axis=0)

    return excess_returns


# ----------------------------- SINGLE-STOCK REGRESSION -----------------------------
def regress_stock(y: pd.Series, X: pd.DataFrame, maxlags: int = 5) -> dict:
    """Run a single time-series regression with Newey-West standard errors.

    Parameters
    ----------
    y : pd.Series
        The stock's excess returns (the dependent variable).
    X : pd.DataFrame
        Factor returns (Mkt-RF, SMB, HML, MOM). A constant will be added.
    maxlags : int
        Number of lags for the Newey-West HAC correction.

    Returns
    -------
    dict with keys: alpha, beta, s, h, m, t_alpha, t_beta, t_s, t_h, t_m, r_squared
    """
    # Add a constant column for the intercept (alpha)
    X_with_const = sm.add_constant(X)

   
    # The statsmodels API:
    #   model = sm.OLS(y, X_with_const)
    #   results = model.fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    #
    # The result object has .params (coefficients) and .tvalues (t-statistics)
    # and .rsquared. We extract them below.
    model = sm.OLS(y, X_with_const)

    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )
    # Extract coefficients and t-stats
    # Order in X_with_const after add_constant: const, Mkt-RF, SMB, HML, MOM
    coefs = results.params
    tvals = results.tvalues

    return {
        "alpha": coefs["const"],
        "beta": coefs["Mkt-RF"],
        "s": coefs["SMB"],
        "h": coefs["HML"],
        "m": coefs["MOM"],
        "t_alpha": tvals["const"],
        "t_beta": tvals["Mkt-RF"],
        "t_s": tvals["SMB"],
        "t_h": tvals["HML"],
        "t_m": tvals["MOM"],
        "r_squared": results.rsquared,
    }


# ----------------------------- LOOP OVER UNIVERSE -----------------------------
def estimate_all_loadings(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    maxlags: int = 5,
) -> pd.DataFrame:
    """Run the factor regression for every stock in the universe.

    Returns a DataFrame indexed by ticker, columns are the result keys.
    """
    # Factor matrix: Mkt-RF, SMB, HML, MOM (RF is the risk-free; we already used it)
    factor_cols = ["Mkt-RF", "SMB", "HML", "MOM"]
    X = factors[factor_cols]

    results = {}

    
    # For each ticker, run regress_stock on its excess return series
    # and store the dict in results[ticker].
    
    for ticker in excess_returns.columns:
        y = excess_returns[ticker]
        results[ticker] = regress_stock(y, X, maxlags=maxlags)

    # Convert dict-of-dicts to a DataFrame; rows are tickers, columns are result keys
    loadings = pd.DataFrame.from_dict(results, orient="index")
    loadings.index.name = "ticker"

    return loadings


# ----------------------------- REPORT -----------------------------
def report_loadings(loadings: pd.DataFrame):
    """Pretty-print the factor loadings sorted by R²."""
    print("\n" + "=" * 100)
    print("FACTOR LOADINGS BY STOCK")
    print("=" * 100)
    print("Sorted by R² (descending). Higher R² = factor model explains more of the stock's variation.\n")

    # Sort by R² descending
    sorted_loadings = loadings.sort_values("r_squared", ascending=False)

    # Round for display
    display = sorted_loadings.round(3)

    # Print with t-stats in parentheses for the key coefficients
    print(f"{'Ticker':<8} {'R²':>6}  {'alpha':>8} {'(t)':>7}  {'beta':>7} {'(t)':>7}  "
          f"{'s':>7} {'(t)':>7}  {'h':>7} {'(t)':>7}  {'m':>7} {'(t)':>7}")
    print("-" * 100)

    for ticker, row in display.iterrows():
        print(
            f"{ticker:<8} "
            f"{row['r_squared']:>6.2f}  "
            f"{row['alpha']:>8.4f} ({row['t_alpha']:>5.2f})  "
            f"{row['beta']:>7.3f} ({row['t_beta']:>5.2f})  "
            f"{row['s']:>7.3f} ({row['t_s']:>5.2f})  "
            f"{row['h']:>7.3f} ({row['t_h']:>5.2f})  "
            f"{row['m']:>7.3f} ({row['t_m']:>5.2f})  "
        )

    
    # - Most stocks should have |t_beta| > 5 (highly significant market exposure)
    # - Most alphas should have |t_alpha| < 2 (not statistically significant)
    # - R² should mostly be between 0.20 and 0.70
    
    print(f"\nSanity checks:")
    print(f"  Stocks with significant β (|t|>2):     {(loadings['t_beta'].abs() > 2).sum()} of {len(loadings)}")
    print(f"  Stocks with significant α (|t|>2):     {(loadings['t_alpha'].abs() > 2).sum()} of {len(loadings)}")
    print(f"  Median R²:                              {loadings['r_squared'].median():.2f}")

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    out_dir = Path("data")

    print("Loading data from Day 1...")
    factors, stock_returns = load_aligned_data()
    print(f"  Factors: {len(factors)} days, columns {list(factors.columns)}")
    print(f"  Stocks:  {len(stock_returns)} days, {len(stock_returns.columns)} tickers")

    print("\nComputing excess returns (R_stock - R_f)...")
    excess_returns = compute_excess_returns(stock_returns, factors["RF"])
    print(f"  Mean excess return across universe: {excess_returns.mean().mean() * 252 * 100:.1f}% annualized")

    print("\nRunning time-series regressions (Fama-French 3 + Momentum)...")
    print("  Using Newey-West (HAC) standard errors with maxlags=5")
    loadings = estimate_all_loadings(excess_returns, factors, maxlags=5)
    print(f"  Estimated loadings for {len(loadings)} stocks")

    report_loadings(loadings)

    # Save for Day 3
    loadings.to_parquet(out_dir / "factor_loadings.parquet")
    print(f"\nSaved to data/factor_loadings.parquet")
