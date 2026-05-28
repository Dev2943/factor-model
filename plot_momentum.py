"""
Momentum Strategy Performance Visualization.

Two-panel chart:
    Top: cumulative return curve for our momentum strategy vs Ken French's MOM
    Bottom: drawdown chart showing the famous momentum crashes

Run with: python3 plot_momentum.py
Produces: momentum_performance.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ----- Load data -----
mom_returns = pd.read_parquet("data/momentum_returns.parquet")["momentum_return"]
factors = pd.read_parquet("data/factors.parquet")

# Trim leading NaNs
mom_returns = mom_returns.dropna()

# Align Ken French's MOM with our returns
common_dates = mom_returns.index.intersection(factors.index)
mom_returns = mom_returns.loc[common_dates]
french_mom = factors.loc[common_dates, "MOM"]

# Cumulative returns (compounded)
our_cumulative = (1 + mom_returns).cumprod()
french_cumulative = (1 + french_mom).cumprod()

# Drawdown for our strategy
running_max = our_cumulative.cummax()
drawdown = our_cumulative / running_max - 1

# Find the worst drawdown date (for annotation)
worst_dd_date = drawdown.idxmin()
worst_dd_value = drawdown.min()
worst_day_date = mom_returns.idxmin()
worst_day_return = mom_returns.min()

# ----- Plot -----
fig = plt.figure(figsize=(11, 8))
gs = GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.15)

# Top panel: cumulative returns
ax1 = fig.add_subplot(gs[0])
ax1.plot(our_cumulative.index, our_cumulative.values,
         color="#C44E52", linewidth=1.8, label="Our 12-1 Momentum (31 stocks)")
ax1.plot(french_cumulative.index, french_cumulative.values,
         color="#4C72B0", linewidth=1.5, label="Ken French MOM (universe of US stocks)", alpha=0.7)
ax1.axhline(1.0, color="gray", linewidth=0.5, alpha=0.5)

ax1.set_ylabel("Cumulative return ($1 invested)", fontsize=11)
ax1.set_title("Long-Short Momentum Strategy — Backtest 2016-2026\n"
              "Built on 31-stock universe; validated against Ken French's MOM factor",
              fontsize=12.5)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate worst day
ax1.annotate(
    f"Worst day: {worst_day_return*100:.1f}%\non {worst_day_date.strftime('%b %Y')}",
    xy=(worst_day_date, our_cumulative.loc[worst_day_date]),
    xytext=(-80, -50), textcoords="offset points",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    arrowprops=dict(arrowstyle="->", color="gray", connectionstyle="arc3,rad=0.2"),
)

# Bottom panel: drawdown
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                 color="#C44E52", alpha=0.4)
ax2.plot(drawdown.index, drawdown.values * 100, color="#C44E52", linewidth=1)
ax2.set_ylabel("Drawdown (%)", fontsize=11)
ax2.set_xlabel("Date", fontsize=11)
ax2.grid(True, alpha=0.3)

# Annotate the worst drawdown
ax2.annotate(
    f"Max drawdown: {worst_dd_value*100:.1f}%",
    xy=(worst_dd_date, worst_dd_value * 100),
    xytext=(20, 20), textcoords="offset points",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
    arrowprops=dict(arrowstyle="->", color="gray", connectionstyle="arc3,rad=-0.2"),
)

# Correlation annotation
correlation = mom_returns.corr(french_mom)
ax1.text(
    0.98, 0.05,
    f"Correlation with French MOM: {correlation:.3f}\n"
    f"Sharpe: {(mom_returns.mean()/mom_returns.std()*np.sqrt(252)):.2f}\n"
    f"Annualized return: {mom_returns.mean()*252*100:+.1f}%\n"
    f"Max drawdown: {worst_dd_value*100:.1f}%",
    transform=ax1.transAxes,
    fontsize=9.5,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"),
)

plt.tight_layout()
plt.savefig("momentum_performance.png", dpi=150, bbox_inches="tight")
print(f"Saved momentum_performance.png")
print(f"\nKey statistics:")
print(f"  Correlation with French MOM: {correlation:.3f}")
print(f"  Annualized return: {mom_returns.mean()*252*100:+.2f}%")
print(f"  Annualized vol: {mom_returns.std()*np.sqrt(252)*100:.2f}%")
print(f"  Sharpe ratio: {(mom_returns.mean()/mom_returns.std()*np.sqrt(252)):.3f}")
print(f"  Max drawdown: {worst_dd_value*100:.2f}%")
print(f"  Worst single day: {worst_day_return*100:.2f}% on {worst_day_date.date()}")

try:
    plt.show()
except Exception:
    pass
