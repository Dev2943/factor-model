"""
Test suite for the factor model project.

Five categories:
    1. DATA INTEGRITY — factors and stock returns are well-formed
    2. LOOK-AHEAD BIAS DETECTION — the backtest doesn't use future information
    3. WEIGHT CONSTRAINTS — long-short portfolio is dollar-neutral by construction
    4. STRUCTURAL SANITY — momentum strategy correlates with the benchmark factor
    5. PERFORMANCE METRICS — Sharpe, drawdown, hit rate fall in plausible ranges

Run with: pytest test_factor_model.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momentum_backtest import (
    compute_momentum_signal,
    lag_signal,
    build_weights,
    compute_portfolio_returns,
    compute_performance,
    SIGNAL_SKIP,
    SIGNAL_WINDOW,
)


# ----- Fixtures -----

@pytest.fixture(scope="module")
def factors():
    return pd.read_parquet("data/factors.parquet")


@pytest.fixture(scope="module")
def stock_returns():
    return pd.read_parquet("data/stock_returns.parquet")


@pytest.fixture(scope="module")
def loadings():
    return pd.read_parquet("data/factor_loadings.parquet")


@pytest.fixture(scope="module")
def momentum_returns():
    return pd.read_parquet("data/momentum_returns.parquet")["momentum_return"]


# ========== 1. DATA INTEGRITY ==========

def test_factors_have_required_columns(factors):
    """Fama-French daily file should provide Mkt-RF, SMB, HML, RF, MOM."""
    required = {"Mkt-RF", "SMB", "HML", "RF", "MOM"}
    assert required.issubset(factors.columns), \
        f"Missing columns: {required - set(factors.columns)}"


def test_factors_in_decimal_not_percent(factors):
    """Factor returns should be decimals (0.01 = 1%) — typical daily values < 0.1."""
    assert factors["Mkt-RF"].abs().max() < 0.5, \
        "Mkt-RF daily moves > 50% suggest data is in % rather than decimal form"


def test_no_duplicated_dates(stock_returns, factors):
    """Both panels should have unique date indices."""
    assert stock_returns.index.is_unique
    assert factors.index.is_unique


def test_universe_size_reasonable(stock_returns):
    """We expect ~30 stocks. <10 means filtering went wrong; >50 wasn't requested."""
    n = stock_returns.shape[1]
    assert 10 <= n <= 50, f"Got {n} stocks — expected ~30"


# ========== 2. LOOK-AHEAD BIAS DETECTION ==========

def test_signal_lag_shifts_one_day(stock_returns):
    """lag_signal(s) should equal s.shift(1)."""
    signal = compute_momentum_signal(stock_returns)
    lagged = lag_signal(signal)

    # The values of `lagged` at date t should equal `signal` at date t-1
    common = signal.dropna().index.intersection(lagged.dropna().index)[:100]
    for d in common[10:20]:  # spot-check a few
        prior_date_loc = signal.index.get_loc(d) - 1
        if prior_date_loc < 0:
            continue
        prior_date = signal.index[prior_date_loc]
        # Pick a stock that has data on both dates
        for ticker in signal.columns[:3]:
            sig_prior = signal.loc[prior_date, ticker]
            lagged_today = lagged.loc[d, ticker]
            if pd.notna(sig_prior) and pd.notna(lagged_today):
                assert abs(sig_prior - lagged_today) < 1e-12, \
                    f"Lag mismatch on {d} for {ticker}: prior={sig_prior}, lagged={lagged_today}"
                break


def test_signal_does_not_use_future_data(stock_returns):
    """Signal at date t must only depend on returns up to t-SIGNAL_SKIP."""
    signal = compute_momentum_signal(stock_returns)

    # Pick a mid-sample date
    test_date = stock_returns.index[1000]
    test_loc = stock_returns.index.get_loc(test_date)
    cutoff_loc = test_loc - SIGNAL_SKIP

    # Mutate returns AFTER the cutoff and verify the signal is unchanged
    mutated = stock_returns.copy()
    mutated.iloc[cutoff_loc + 1:] = mutated.iloc[cutoff_loc + 1:] * 100  # crazy values

    mutated_signal = compute_momentum_signal(mutated)

    diff = (signal.loc[test_date] - mutated_signal.loc[test_date]).abs().max()
    assert diff < 1e-10, \
        "Mutating future returns changed today's signal — look-ahead bias present"


# ========== 3. WEIGHT CONSTRAINTS ==========

def test_weights_are_dollar_neutral(stock_returns):
    """Each non-empty row of weights should sum to ~0 (long $1 - short $1)."""
    signal = compute_momentum_signal(stock_returns)
    signal_lagged = lag_signal(signal)
    weights = build_weights(signal_lagged)

    # Examine rows that have non-zero weights (post-warmup)
    nonzero_mask = (weights != 0).any(axis=1)
    row_sums = weights[nonzero_mask].sum(axis=1)

    assert row_sums.abs().max() < 1e-10, \
        f"Some rows are not dollar-neutral: max abs sum = {row_sums.abs().max()}"


def test_weights_have_long_and_short_sides(stock_returns):
    """Each non-empty row should have both positive and negative weights."""
    signal = compute_momentum_signal(stock_returns)
    signal_lagged = lag_signal(signal)
    weights = build_weights(signal_lagged)

    nonzero_mask = (weights != 0).any(axis=1)
    has_long = (weights[nonzero_mask] > 0).any(axis=1)
    has_short = (weights[nonzero_mask] < 0).any(axis=1)

    assert has_long.all(), "Some non-empty rows have no long position"
    assert has_short.all(), "Some non-empty rows have no short position"


def test_long_leg_sums_to_one(stock_returns):
    """The long leg of each non-empty row should sum to exactly +1."""
    signal = compute_momentum_signal(stock_returns)
    signal_lagged = lag_signal(signal)
    weights = build_weights(signal_lagged)

    nonzero_mask = (weights != 0).any(axis=1)
    long_sums = weights[nonzero_mask].where(weights > 0, 0).sum(axis=1)

    assert (long_sums - 1.0).abs().max() < 1e-10, \
        f"Long leg doesn't sum to 1: max deviation = {(long_sums - 1.0).abs().max()}"


# ========== 4. STRUCTURAL SANITY ==========

def test_momentum_correlates_with_french_mom(momentum_returns, factors):
    """Our momentum strategy should correlate ≥0.4 with Ken French's MOM."""
    common = momentum_returns.index.intersection(factors.index)
    corr = momentum_returns.loc[common].corr(factors.loc[common, "MOM"])
    assert corr > 0.4, \
        f"Correlation with French MOM is only {corr:.3f} — likely a signal construction bug"


def test_factor_loadings_have_significant_betas(loadings):
    """Every stock should have statistically significant market β (|t|>2)."""
    significant_betas = (loadings["t_beta"].abs() > 2).sum()
    assert significant_betas >= len(loadings) * 0.9, \
        f"Only {significant_betas}/{len(loadings)} stocks have significant β — unusual"


def test_alpha_rarely_significant(loadings):
    """Under efficient markets, |t_alpha| > 2 should be rare (<25% of stocks)."""
    significant_alphas = (loadings["t_alpha"].abs() > 2).sum()
    assert significant_alphas <= len(loadings) * 0.25, \
        f"{significant_alphas}/{len(loadings)} stocks have significant alpha — too many"


# ========== 5. PERFORMANCE METRICS ==========

def test_momentum_performance_in_reasonable_range(momentum_returns):
    """Sharpe should be in the typical academic range; max DD reasonable."""
    perf = compute_performance(momentum_returns)

    # Sharpe in plausible range for vanilla momentum
    assert -1.0 <= perf["sharpe_ratio"] <= 1.5, \
        f"Sharpe {perf['sharpe_ratio']} is outside the plausible vanilla-momentum range"

    # Max drawdown should be negative and bounded
    assert -80 <= perf["max_drawdown_pct"] <= 0, \
        f"Max drawdown {perf['max_drawdown_pct']}% is implausible"

    # Vol should be in equity-like range
    assert 5 <= perf["annualized_vol_pct"] <= 40, \
        f"Annualized vol {perf['annualized_vol_pct']}% is outside the plausible range"


def test_hit_rate_near_fifty_percent(momentum_returns):
    """Momentum has slightly-below-50 hit rate (negative skew strategy)."""
    perf = compute_performance(momentum_returns)
    assert 40 <= perf["hit_rate_pct"] <= 60, \
        f"Hit rate {perf['hit_rate_pct']}% is outside typical momentum range"
