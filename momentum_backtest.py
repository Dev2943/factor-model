"""
Day 4: Long-Short Momentum Portfolio Backtest.

Builds a dollar-neutral long-short momentum strategy and backtests it over
the full 2015-2026 sample.

Strategy:
    1. At each rebalance date, compute the 12-1 month momentum signal for
       every stock: cumulative return from t-12mo to t-2mo (skipping t-1).
    2. Rank stocks by signal. Top tercile (~10 stocks) → long; bottom tercile
       → short; middle ignored.
    3. Equal-weight within each leg.
    4. Daily portfolio return = mean(long-leg returns) - mean(short-leg returns).
    5. Rebalance monthly.

Critical: signal at end of day t is used for holdings on day t+1 onward.
The signal.shift(1) is the single most important line for avoiding look-ahead bias.


Run with: python3 momentum_backtest.py
Produces: data/momentum_returns.parquet, data/momentum_performance.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------- CONFIG -----------------------------
SIGNAL_LOOKBACK = 252   # trading days ≈ 12 months for total signal window
SIGNAL_SKIP = 21        # trading days ≈ 1 month to skip (short-term reversal)
SIGNAL_WINDOW = SIGNAL_LOOKBACK - SIGNAL_SKIP  # 231 days of returns to sum
REBALANCE_FREQ = 21     # rebalance every 21 trading days (monthly)
TOP_FRACTION = 1/3      # top tercile goes long


# ----------------------------- LOAD -----------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = pd.read_parquet("data/factors.parquet")
    stock_returns = pd.read_parquet("data/stock_returns.parquet")
    return factors, stock_returns


# ----------------------------- MOMENTUM SIGNAL -----------------------------
def compute_momentum_signal(stock_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the 12-1 month momentum signal for every stock at every date.

    For each date t, the signal is:
        cumulative log-return from day (t - SIGNAL_LOOKBACK) to day (t - SIGNAL_SKIP)

    We use a sum of simple returns as a fast log-return proxy — for signals
    used only for cross-sectional ranking, the two are essentially identical.

    Returns
    -------
    pd.DataFrame
        Same index/columns as stock_returns. Values are signals.
        Early dates will be NaN until enough history accumulates.
    """
    # implement the 12-1 momentum signal.
    # Step A: shift returns forward by SIGNAL_SKIP days, so that "today" sees
    # only returns from at least SIGNAL_SKIP days ago.
    # Step B: take a rolling sum over the next SIGNAL_WINDOW days.
    # That gives, on date t, the sum of stock returns from t-SIGNAL_LOOKBACK
    # to t-SIGNAL_SKIP — exactly the 12-1 month return.


    signal = stock_returns.shift(SIGNAL_SKIP).rolling(SIGNAL_WINDOW).sum()

    return signal


# ----------------------------- LAG SIGNAL FOR LOOK-AHEAD PROTECTION -----------------------------
def lag_signal(signal: pd.DataFrame) -> pd.DataFrame:
    """Shift the signal forward by one day so today's holdings only use
    information available at end of yesterday.

    This is THE line that prevents look-ahead bias. Forget it and your
    backtest will print fake Sharpe ratios.
    """
    # shift the signal forward by 1 day.
    
    return signal.shift(1)


# ----------------------------- BUILD LONG-SHORT WEIGHTS -----------------------------
def build_weights(signal: pd.DataFrame, top_frac: float = TOP_FRACTION) -> pd.DataFrame:
    """For each date, rank stocks by signal. Top tercile → long, bottom → short.

    Each leg is equal-weighted to sum to $1 (long leg) and -$1 (short leg).

    Returns
    -------
    pd.DataFrame
        Same shape as signal. Each row sums to ~0 (long $1 − short $1).
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    # For each date, rank within the cross-section
    for date in signal.index:
        row = signal.loc[date].dropna()
        n_stocks = len(row)
        if n_stocks < 10:
            continue  # need enough stocks to form deciles meaningfully

        n_long = max(1, int(n_stocks * top_frac))
        n_short = max(1, int(n_stocks * top_frac))

        # rank stocks by signal value.
        # Top n_long stocks (highest signals) get long weights of +1/n_long each.
        # Bottom n_short stocks (lowest signals) get short weights of -1/n_short each.
        # Middle stocks stay at 0.
        
        
        sorted_tickers = row.sort_values(ascending=False).index

        long_tickers = sorted_tickers[:n_long]

        short_tickers = sorted_tickers[-n_short:]

        weights.loc[date, long_tickers] = 1.0 / n_long

        weights.loc[date, short_tickers] = -1.0 / n_short

    return weights


# ----------------------------- PORTFOLIO RETURNS -----------------------------
def compute_portfolio_returns(
    weights: pd.DataFrame,
    stock_returns: pd.DataFrame,
) -> pd.Series:
    """Daily portfolio return = sum over stocks of weight × return.

    Weights are dollar amounts (long $1 / short $1 split across stocks);
    returns are percentages. The product, summed, gives daily P&L per $1 NAV.
    """
    # Align weights and returns on date and ticker
    common_dates = weights.index.intersection(stock_returns.index)
    common_tickers = weights.columns.intersection(stock_returns.columns)
    w = weights.loc[common_dates, common_tickers]
    r = stock_returns.loc[common_dates, common_tickers]

    # portfolio daily return = sum across columns of (w * r)
    
    portfolio_returns = (w * r).sum(axis=1)

    return portfolio_returns


# ----------------------------- PERFORMANCE METRICS -----------------------------
def compute_performance(returns: pd.Series) -> dict:
    """Standard performance metrics for a daily return series."""
    # Drop any leading NaNs
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    # Annualized return
    ann_return = returns.mean() * 252

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio (assumes near-zero risk-free rate at the daily level)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Cumulative returns and drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1  # always ≤ 0
    max_drawdown = drawdown.min()

    # Calmar
    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # Hit rate
    hit_rate = (returns > 0).mean()

    return {
        "n_days": int(len(returns)),
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_vol_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "calmar_ratio": round(calmar, 3),
        "hit_rate_pct": round(hit_rate * 100, 2),
        "best_day_pct": round(returns.max() * 100, 2),
        "worst_day_pct": round(returns.min() * 100, 2),
    }


# ----------------------------- VALIDATION AGAINST KEN FRENCH MOM -----------------------------
def correlation_with_french_mom(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
) -> float:
    """Our momentum strategy should correlate substantially with French's MOM.

    Both capture the same underlying factor; ours just uses a 31-stock universe
    while French uses thousands of stocks. Expect correlation of ~0.4-0.7.
    """
    # compute correlation between portfolio_returns and factors["MOM"]
    # over their common date range.
   
    common = portfolio_returns.index.intersection(factors.index)

    return portfolio_returns.loc[common].corr(
           factors.loc[common, "MOM"]
    )


# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    print("Loading data...")
    factors, stock_returns = load_data()
    print(f"  {len(stock_returns)} days × {len(stock_returns.columns)} stocks")

    print("\nComputing 12-1 momentum signal...")
    signal = compute_momentum_signal(stock_returns)
    print(f"  First non-NaN signal: {signal.dropna().index.min().date()}")

    print("\nLagging signal by 1 day (look-ahead protection)...")
    signal_lagged = lag_signal(signal)

    print("\nBuilding long-short weights (top vs bottom tercile)...")
    weights = build_weights(signal_lagged)
    nonzero_dates = (weights != 0).any(axis=1).sum()
    print(f"  Non-zero positions on {nonzero_dates} days")

    print("\nComputing portfolio returns...")
    portfolio_returns = compute_portfolio_returns(weights, stock_returns)

    # Trim leading NaNs (no signal until enough history)
    portfolio_returns = portfolio_returns.loc[portfolio_returns.first_valid_index():]
    print(f"  {len(portfolio_returns)} trading days of returns")

    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)
    perf = compute_performance(portfolio_returns)
    for k, v in perf.items():
        print(f"  {k:30s}: {v}")

    print("\nValidation: correlation with Ken French MOM factor...")
    corr_mom = correlation_with_french_mom(portfolio_returns, factors)
    print(f"  Correlation: {corr_mom:.3f}")
    print(f"  (Expect 0.4-0.7 — our 31-stock universe is much smaller than French's)")

    # ----- Sanity check warnings -----
    if perf["sharpe_ratio"] > 1.5:
        print("\nWARNING: Sharpe > 1.5 is suspicious for vanilla momentum.")
        print("Check that signal_lagged.shift(1) is in place (look-ahead bias).")
    if perf["sharpe_ratio"] < -0.5:
        print("\nNOTE: negative Sharpe means momentum lost money in this sample.")
        print("This is real — momentum had several poor stretches over 2015-2026.")

    # Save outputs
    out_dir = Path("data")
    portfolio_returns.to_frame("momentum_return").to_parquet(out_dir / "momentum_returns.parquet")
    with open(out_dir / "momentum_performance.json", "w") as f:
        json.dump(perf, f, indent=2)
    print("\nSaved to data/momentum_returns.parquet and data/momentum_performance.json")
