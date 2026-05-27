"""
Day 3b: Portfolio-Based Fama-MacBeth.

Fixes the small-universe pathology from Day 3 by sorting 31 stocks into 9
portfolios on (β, h) and running Fama-MacBeth on the portfolios.

Why this works:
    - Portfolio loadings are much more stable than individual stock loadings
      (idiosyncratic noise averages out within each basket).
    - The cross-sectional spread in loadings widens because we deliberately
      sorted on them.
    - The errors-in-variables problem on the right-hand side collapses.

We use a 3×3 sort: tercile on β × tercile on h → 9 portfolios.
Each portfolio is an equal-weighted basket of ~3 stocks.
Cross-sectional regression uses 3 factors (Mkt, SMB, HML) — momentum dropped
because portfolio momentum loadings aren't sorted on and would be incidental.

This is a small-universe version of Fama-French (1993)'s 25-portfolio sort
on size × book-to-market.


Run with: python3 portfolio_fm.py
Produces: data/portfolio_premia.parquet, data/portfolio_fm_summary.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ----------------------------- LOAD DATA -----------------------------
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factors = pd.read_parquet("data/factors.parquet")
    stock_returns = pd.read_parquet("data/stock_returns.parquet")
    loadings = pd.read_parquet("data/factor_loadings.parquet")
    return factors, stock_returns, loadings


# ----------------------------- SORT STOCKS INTO PORTFOLIOS -----------------------------
def assign_portfolios(loadings: pd.DataFrame) -> pd.DataFrame:
    """Sort stocks into a 3×3 grid: β tercile × h tercile = 9 portfolios.

    Returns a DataFrame with one row per stock, columns: beta_bucket, h_bucket,
    portfolio (the bucket label like "P11", "P12", ..., "P33").
    """
    # Rank each stock's β into one of 3 buckets (low / mid / high).
    # pd.qcut splits into roughly equal-sized buckets by quantile.
    beta_bucket = pd.qcut(loadings["beta"], q=3, labels=[1, 2, 3])
    h_bucket = pd.qcut(loadings["h"], q=3, labels=[1, 2, 3])

    portfolio_id = beta_bucket.astype(str) + h_bucket.astype(str)

    return pd.DataFrame({
        "beta_bucket": beta_bucket,
        "h_bucket": h_bucket,
        "portfolio": "P" + portfolio_id,
    }, index=loadings.index)


# ----------------------------- BUILD PORTFOLIO RETURNS -----------------------------
def build_portfolio_returns(
    stock_returns: pd.DataFrame,
    portfolio_assignment: pd.DataFrame,
) -> pd.DataFrame:
    """For each day, compute equal-weighted average return per portfolio.

    Returns
    -------
    pd.DataFrame indexed by date, columns are portfolio IDs (e.g., P11, P12, ...).
    """
    # Group stocks by portfolio assignment
    portfolios = {}
    for port_id, group in portfolio_assignment.groupby("portfolio", observed=True):
        tickers_in_port = group.index.tolist()

        # compute the equal-weighted daily return of this portfolio.
        # We have stock_returns (daily returns indexed by date, columns are tickers).
        # We want the average across the tickers in this portfolio, per day.
        #  stock_returns[tickers_in_port].mean(axis=1)
        # `axis=1` says "average across columns (stocks) on each row (date)".

        portfolios[port_id] = stock_returns[tickers_in_port].mean(axis=1)

    return pd.DataFrame(portfolios)


# ----------------------------- TIME-SERIES REGRESSION ON PORTFOLIOS -----------------------------
def estimate_portfolio_loadings(
    portfolio_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Run a 3-factor time-series regression for each portfolio.

    Returns a DataFrame indexed by portfolio ID with columns:
    alpha, beta, s, h, r_squared.
    """
    rf = factors["RF"]
    portfolio_excess = portfolio_returns.sub(rf, axis=0)
    X = factors[["Mkt-RF", "SMB", "HML"]]
    X_with_const = sm.add_constant(X)

    results = {}
    for port_id in portfolio_returns.columns:
        y = portfolio_excess[port_id]

        #   fit OLS with Newey-West HAC standard errors (maxlags=5),
        #   res = sm.OLS(y, X_with_const).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

        res = sm.OLS(y, X_with_const).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 5},
        )

        results[port_id] = {
            "alpha":    res.params["const"],
            "beta":     res.params["Mkt-RF"],
            "s":        res.params["SMB"],
            "h":        res.params["HML"],
            "r_squared": res.rsquared,
        }

    return pd.DataFrame.from_dict(results, orient="index")


# ----------------------------- CROSS-SECTIONAL FM -----------------------------
def fama_macbeth_on_portfolios(
    portfolio_excess: pd.DataFrame,
    portfolio_loadings: pd.DataFrame,
) -> pd.DataFrame:
    """For each day, run a cross-sectional regression of portfolio returns
    on portfolio loadings. Returns daily premia.
    """
    # Portfolio loadings: 9 rows (portfolios) × 3 columns (β, s, h)
    X = portfolio_loadings[["beta", "s", "h"]].values
    X_with_const = sm.add_constant(X, has_constant="add")
    port_ids = portfolio_loadings.index.tolist()

    daily_results = {}
    for date, row in portfolio_excess.iterrows():
        y = row[port_ids].values

        if np.isnan(y).any():
            continue

        # fit a plain OLS (no HAC needed cross-sectionally).
        # 9 observations × 4 parameters → 5 df.
        
        results = sm.OLS(y, X_with_const).fit()

        daily_results[date] = {
            "gamma_0":    results.params[0],
            "lambda_mkt": results.params[1],
            "lambda_smb": results.params[2],
            "lambda_hml": results.params[3],
        }

    premia_df = pd.DataFrame.from_dict(daily_results, orient="index")
    premia_df.index.name = "date"
    return premia_df


# ----------------------------- SUMMARIZE -----------------------------
def summarize(daily_premia: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Time-series average + SEs + comparison to raw factor returns."""
    mean_daily = daily_premia.mean()
    std_daily = daily_premia.std()
    se_daily = std_daily / np.sqrt(len(daily_premia))
    t_stat = mean_daily / se_daily

    annualized_mean = mean_daily * 252 * 100
    annualized_vol = std_daily * np.sqrt(252) * 100
    sharpe = annualized_mean / annualized_vol

    factor_avg = factors[["Mkt-RF", "SMB", "HML"]].mean() * 252 * 100

    summary = pd.DataFrame({
        "FM Premium (% annual)":      annualized_mean.values,
        "FM t-stat":                  t_stat.values,
        "FM Vol (% annual)":          annualized_vol.values,
        "FM Sharpe":                  sharpe.values,
        "Raw Factor Avg (% annual)":  [np.nan] + list(factor_avg.values),
    }, index=annualized_mean.index)

    return summary


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data...")
    factors, stock_returns, loadings = load_all()

    print("\nSorting stocks into 9 portfolios (3×3 on β × h)...")
    portfolio_assignment = assign_portfolios(loadings)
    print("\nPortfolio composition:")
    for port_id, group in portfolio_assignment.groupby("portfolio", observed=True):
        tickers = ", ".join(group.index.tolist())
        print(f"  {port_id}: {tickers}")

    print("\nBuilding daily portfolio returns (equal-weighted)...")
    portfolio_returns = build_portfolio_returns(stock_returns, portfolio_assignment)
    print(f"  {len(portfolio_returns)} days × {len(portfolio_returns.columns)} portfolios")

    print("\nEstimating portfolio-level factor loadings (Pass 1)...")
    portfolio_loadings = estimate_portfolio_loadings(portfolio_returns, factors)
    print("\nPortfolio loadings:")
    print(portfolio_loadings.round(3).to_string())

    print("\nRunning cross-sectional Fama-MacBeth on portfolios (Pass 2)...")
    portfolio_excess = portfolio_returns.sub(factors["RF"], axis=0)
    daily_premia = fama_macbeth_on_portfolios(portfolio_excess, portfolio_loadings)
    print(f"  {len(daily_premia)} valid daily regressions")

    print("\nSummarizing...")
    summary = summarize(daily_premia, factors)

    print("\n" + "=" * 90)
    print("PORTFOLIO-BASED FAMA-MACBETH FACTOR RISK PREMIA")
    print("=" * 90)
    print("(9 portfolios sorted on β × h tercile; 3-factor model)\n")
    print(summary.round(3).to_string())

    # print a short interpretation comparing to the individual-stock
    # version from Day 3. The portfolio premia should be much closer to the
    # raw factor return averages (much less inflation).
   
    print("\nInterpretation:")

    for factor in ["lambda_mkt", "lambda_smb", "lambda_hml"]:
        t = summary.loc[factor, "FM t-stat"]
        prem = summary.loc[factor, "FM Premium (% annual)"]
        raw = summary.loc[factor, "Raw Factor Avg (% annual)"]

        diff = prem - raw

        print(
            f"  {factor:14s}: "
            f"FM={prem:+6.2f}%  "
            f"Raw={raw:+6.2f}%  "
            f"Diff={diff:+5.2f}pp  "
            f"(t={t:+.2f})"
        )

    # Save
    out_dir = Path("data")
    daily_premia.to_parquet(out_dir / "portfolio_premia.parquet")
    summary.to_parquet(out_dir / "portfolio_fm_summary.parquet")
    print(f"\nSaved to data/portfolio_premia.parquet and data/portfolio_fm_summary.parquet")
