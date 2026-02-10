"""
Reporting module for the equity strategy.

Generates:
- Performance visualizations (equity curve, drawdown, etc.)
- PDF/HTML reports
- Summary tables
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from .backtest import BacktestResults
from .metrics import (
    compute_metrics,
    compute_drawdown_series,
    compute_trade_stats,
    get_monthly_returns,
    get_rolling_sharpe,
    PerformanceMetrics,
)


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_equity_curve(
    results: BacktestResults,
    benchmark: Optional[pd.Series] = None,
    figsize: tuple = (14, 6),
    title: str = "Portfolio Equity Curve",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot equity curve with optional benchmark.
    
    Args:
        results: Backtest results
        benchmark: Optional benchmark equity curve
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    equity = results.get_equity_curve()
    
    # Normalize to starting value
    equity_norm = equity / equity.iloc[0] * 100
    ax.plot(equity_norm.index, equity_norm.values, label='Strategy', linewidth=2)
    
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * 100
        ax.plot(bench_norm.index, bench_norm.values, label='Benchmark', 
                linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value (Starting = 100)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drawdown(
    results: BacktestResults,
    figsize: tuple = (14, 4),
    title: str = "Drawdown",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot drawdown chart.
    
    Args:
        results: Backtest results
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    equity = results.get_equity_curve()
    drawdown = compute_drawdown_series(equity)
    
    ax.fill_between(drawdown.index, 0, drawdown.values * 100, 
                    alpha=0.5, color='red')
    ax.plot(drawdown.index, drawdown.values * 100, color='red', linewidth=1)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Add max drawdown annotation
    min_idx = drawdown.idxmin()
    min_dd = drawdown.min() * 100
    ax.annotate(f'Max: {min_dd:.1f}%', 
                xy=(min_idx, min_dd),
                xytext=(min_idx, min_dd - 2),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_monthly_returns(
    results: BacktestResults,
    figsize: tuple = (14, 8),
    title: str = "Monthly Returns Heatmap",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot monthly returns heatmap.
    
    Args:
        results: Backtest results
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    monthly = get_monthly_returns(results)
    
    # Create year-month pivot table
    monthly_df = pd.DataFrame(monthly)
    monthly_df.columns = ['return']
    monthly_df['year'] = monthly_df.index.year
    monthly_df['month'] = monthly_df.index.month
    
    pivot = monthly_df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
    
    ax.set_title(title)
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rolling_sharpe(
    results: BacktestResults,
    window: int = 252,
    figsize: tuple = (14, 4),
    title: str = "Rolling Sharpe Ratio (1-Year)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio.
    
    Args:
        results: Backtest results
        window: Rolling window in days
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    rolling_sharpe = get_rolling_sharpe(results, window=window)
    
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_positions_and_exposure(
    results: BacktestResults,
    figsize: tuple = (14, 6),
    title: str = "Positions and Exposure",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot number of positions and gross exposure over time.
    
    Args:
        results: Backtest results
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    positions = results.get_positions_count()
    exposure = pd.Series({r.date: r.gross_exposure for r in results.daily_results})
    
    # Positions
    ax1.fill_between(positions.index, 0, positions.values, alpha=0.5)
    ax1.set_ylabel('# Positions')
    ax1.set_title(title)
    
    # Exposure
    ax2.fill_between(exposure.index, 0, exposure.values * 100, alpha=0.5, color='orange')
    ax2.set_ylabel('Gross Exposure (%)')
    ax2.set_xlabel('Date')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_turnover(
    results: BacktestResults,
    figsize: tuple = (14, 4),
    title: str = "Daily Turnover",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot daily turnover.
    
    Args:
        results: Backtest results
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    turnover = results.get_turnover()
    
    ax.bar(turnover.index, turnover.values * 100, alpha=0.7, width=1)
    ax.axhline(y=turnover.mean() * 100, color='red', linestyle='--', 
               label=f'Avg: {turnover.mean()*100:.1f}%')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Turnover (%)')
    ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_equity_and_drawdown(
    results: BacktestResults,
    benchmark: Optional[pd.Series] = None,
    figsize: tuple = (14, 10),
    title: str = "Portfolio Performance",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot combined equity curve and drawdown on same figure.
    Shows when money was made and lost.
    
    Args:
        results: Backtest results
        benchmark: Optional benchmark series
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    equity = results.get_equity_curve()
    drawdown = compute_drawdown_series(equity)
    
    # Top panel: Equity curve
    ax1.plot(equity.index, equity.values / 1e6, label='Portfolio Value', 
             linewidth=2, color='#2E86AB')
    
    if benchmark is not None:
        # Scale benchmark to start at same value as portfolio
        bench_scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax1.plot(bench_scaled.index, bench_scaled.values / 1e6, 
                 label='Benchmark (SPY)', linewidth=1.5, alpha=0.7, color='#A23B72')
    
    ax1.set_ylabel('Portfolio Value ($M)')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add shading for profitable periods
    returns = results.get_returns()
    positive = returns > 0
    for i, (date, is_pos) in enumerate(positive.items()):
        if is_pos:
            ax1.axvspan(date, date + pd.Timedelta(days=1), 
                       alpha=0.1, color='green', linewidth=0)
    
    # Bottom panel: Drawdown
    ax2.fill_between(drawdown.index, 0, drawdown.values * 100, 
                     alpha=0.7, color='#E94F37', label='Drawdown')
    ax2.plot(drawdown.index, drawdown.values * 100, color='#E94F37', linewidth=0.5)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add max drawdown annotation
    min_idx = drawdown.idxmin()
    min_dd = drawdown.min() * 100
    ax2.annotate(f'Max DD: {min_dd:.1f}%', 
                xy=(min_idx, min_dd),
                xytext=(10, -10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trading_signals(
    results: BacktestResults,
    figsize: tuple = (14, 10),
    title: str = "Trading Signals and Portfolio Value",
    max_markers_per_side: int = 2000,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot trading signals (buys/sells) overlaid on portfolio value.
    
    Args:
        results: Backtest results
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                         gridspec_kw={'height_ratios': [2, 1, 1]})
    
    equity = results.get_equity_curve()
    trades = results.trades
    
    # Top panel: Portfolio value with buy/sell markers
    ax1.plot(equity.index, equity.values / 1e6, linewidth=2, color='#2E86AB', label='Portfolio')
    
    if not trades.empty and 'date' in trades.columns:
        # Separate buys and sells
        trades['date'] = pd.to_datetime(trades['date'])
        buys = trades[trades['side'] == 'BUY']
        sells = trades[trades['side'] == 'SELL']

        if len(buys) > max_markers_per_side:
            buys = buys.sample(n=max_markers_per_side, random_state=42).sort_values('date')
        if len(sells) > max_markers_per_side:
            sells = sells.sample(n=max_markers_per_side, random_state=42).sort_values('date')

        buy_dates = pd.to_datetime(buys['date']).drop_duplicates()
        sell_dates = pd.to_datetime(sells['date']).drop_duplicates()
        buy_dates = buy_dates[buy_dates.isin(equity.index)]
        sell_dates = sell_dates[sell_dates.isin(equity.index)]

        if len(buy_dates) > 0:
            buy_vals = (equity.reindex(buy_dates).dropna() / 1e6)
            ax1.scatter(buy_vals.index, buy_vals.values, marker='^', color='green', s=20, alpha=0.5)
        if len(sell_dates) > 0:
            sell_vals = (equity.reindex(sell_dates).dropna() / 1e6)
            ax1.scatter(sell_vals.index, sell_vals.values, marker='v', color='red', s=20, alpha=0.5)
        
        ax1.scatter([], [], marker='^', color='green', s=50, label='Buy')
        ax1.scatter([], [], marker='v', color='red', s=50, label='Sell')
    
    ax1.set_ylabel('Portfolio ($M)')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Middle panel: Number of trades per day
    if not trades.empty and 'date' in trades.columns:
        daily_trades = trades.groupby('date').size()
        ax2.bar(daily_trades.index, daily_trades.values, alpha=0.7, color='#F18F01', width=1)
    ax2.set_ylabel('# Trades')
    ax2.grid(True, alpha=0.3)
    
    # Bottom panel: Positions count
    positions = results.get_positions_count()
    ax3.fill_between(positions.index, 0, positions.values, alpha=0.5, color='#C73E1D')
    ax3.plot(positions.index, positions.values, linewidth=1, color='#C73E1D')
    ax3.set_ylabel('# Positions')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_daily_pnl(
    results: BacktestResults,
    figsize: tuple = (14, 6),
    title: str = "Daily Profit & Loss",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot daily P&L as bar chart.
    
    Args:
        results: Backtest results
        figsize: Figure size
        title: Chart title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    equity = results.get_equity_curve()
    daily_pnl = equity.diff()
    
    # Color bars based on positive/negative
    colors = ['green' if x >= 0 else 'red' for x in daily_pnl.values]
    
    ax.bar(daily_pnl.index, daily_pnl.values / 1000, color=colors, alpha=0.7, width=1)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily P&L ($K)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add summary stats
    avg_pnl = daily_pnl.mean() / 1000
    ax.axhline(y=avg_pnl, color='blue', linestyle='--', alpha=0.5, 
               label=f'Avg: ${avg_pnl:.1f}K')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_returns_distribution(
    results: BacktestResults,
    figsize: tuple = (12, 5),
    title: str = "Distribution of Daily Returns",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot histogram of daily returns with mean and median markers.
    """
    fig, ax = plt.subplots(figsize=figsize)
    returns = results.get_returns().dropna()

    if returns.empty:
        ax.text(0.5, 0.5, "No return data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(returns * 100, bins=50, alpha=0.75, color="#2E86AB", label="Daily returns")
        mean_val = returns.mean() * 100
        med_val = returns.median() * 100
        ax.axvline(mean_val, color="#E94F37", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}%")
        ax.axvline(med_val, color="#6A994E", linestyle="-.", linewidth=1.5, label=f"Median: {med_val:.2f}%")
        ax.set_xlabel("Daily Return (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_negative_tail_distribution(
    results: BacktestResults,
    percentile: float = 5.0,
    figsize: tuple = (12, 5),
    title: str = "Negative Tail Distribution (Below 5th Percentile)",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot return distribution for negative tail events below selected percentile.
    """
    fig, ax = plt.subplots(figsize=figsize)
    returns = results.get_returns().dropna()

    if returns.empty:
        ax.text(0.5, 0.5, "No return data available", ha="center", va="center")
        ax.set_axis_off()
    else:
        threshold = returns.quantile(percentile / 100.0)
        tail = returns[returns <= threshold]

        if tail.empty:
            ax.text(0.5, 0.5, "No tail observations available", ha="center", va="center")
            ax.set_axis_off()
        else:
            ax.hist(
                tail * 100,
                bins=max(10, min(40, len(tail))),
                alpha=0.8,
                color="#E94F37",
                label=f"Tail returns (<= {percentile:.0f}th pct)",
            )
            ax.axvline(
                threshold * 100,
                color="#222222",
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold: {threshold*100:.2f}%",
            )
            ax.axvline(
                tail.mean() * 100,
                color="#6A994E",
                linestyle="-.",
                linewidth=1.5,
                label=f"Tail mean: {tail.mean()*100:.2f}%",
            )
            ax.set_xlabel("Daily Return (%)")
            ax.set_ylabel("Frequency")
            ax.set_title(title)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _aligned_strategy_benchmark_returns(
    results: BacktestResults,
    benchmark: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Align strategy and benchmark daily returns on shared dates.
    """
    if benchmark is None or len(benchmark.dropna()) < 2:
        return pd.DataFrame(columns=["strategy", "benchmark"])

    strategy_returns = results.get_returns().sort_index()
    benchmark_returns = benchmark.sort_index().pct_change()
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    return aligned


def _benchmark_diagnostics_payload(
    results: BacktestResults,
    benchmark: Optional[pd.Series],
) -> dict[str, object]:
    """
    Compute benchmark-relative diagnostics and decomposition terms.
    """
    aligned = _aligned_strategy_benchmark_returns(results, benchmark)
    if aligned.empty:
        return {}

    strategy_cagr = compute_metrics(results).cagr
    benchmark_cagr = float((1.0 + aligned["benchmark"]).prod() ** (252.0 / len(aligned)) - 1.0)
    active_cagr_gap = strategy_cagr - benchmark_cagr

    cov = np.cov(aligned["strategy"].values, aligned["benchmark"].values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 1e-12 else np.nan

    mean_s = float(aligned["strategy"].mean())
    mean_b = float(aligned["benchmark"].mean())
    alpha_daily = mean_s - beta * mean_b if not pd.isna(beta) else np.nan
    alpha_annual = float((1.0 + alpha_daily) ** 252 - 1.0) if not pd.isna(alpha_daily) else np.nan

    up = aligned[aligned["benchmark"] > 0]
    down = aligned[aligned["benchmark"] < 0]
    up_capture = (
        float(up["strategy"].mean() / up["benchmark"].mean())
        if len(up) > 0 and abs(up["benchmark"].mean()) > 1e-12
        else np.nan
    )
    down_capture = (
        float(down["strategy"].mean() / down["benchmark"].mean())
        if len(down) > 0 and abs(down["benchmark"].mean()) > 1e-12
        else np.nan
    )

    equity = results.get_equity_curve().dropna()
    daily_df = pd.DataFrame(
        [
            {
                "date": r.date,
                "gross_exposure": r.gross_exposure,
                "cash": r.cash,
                "portfolio_value": r.portfolio_value,
                "turnover": r.turnover,
                "costs": r.costs,
            }
            for r in results.daily_results
        ]
    ).sort_values("date")
    daily_df["cash_ratio"] = daily_df["cash"] / daily_df["portfolio_value"]

    total_costs = float(daily_df["costs"].sum())
    years = max(1e-9, len(aligned) / 252.0)
    avg_equity = float(equity.mean()) if len(equity) else np.nan
    annual_cost_drag = (
        float(-(total_costs / avg_equity) / years)
        if not pd.isna(avg_equity) and avg_equity > 0
        else np.nan
    )

    beta_gap = (
        float((beta - 1.0) * benchmark_cagr)
        if not pd.isna(beta)
        else np.nan
    )
    residual = (
        float(active_cagr_gap - beta_gap - annual_cost_drag)
        if not (pd.isna(beta_gap) or pd.isna(annual_cost_drag))
        else np.nan
    )

    rolling_window = min(252, len(aligned))
    rolling_active = pd.Series(dtype=float)
    if rolling_window >= 20:
        roll_s = (1.0 + aligned["strategy"]).rolling(rolling_window).apply(np.prod, raw=True) - 1.0
        roll_b = (1.0 + aligned["benchmark"]).rolling(rolling_window).apply(np.prod, raw=True) - 1.0
        rolling_active = (roll_s - roll_b).dropna()

    return {
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "active_cagr_gap": active_cagr_gap,
        "beta": beta,
        "alpha_annual": alpha_annual,
        "up_capture": up_capture,
        "down_capture": down_capture,
        "avg_gross_exposure": float(daily_df["gross_exposure"].mean()),
        "median_gross_exposure": float(daily_df["gross_exposure"].median()),
        "avg_cash_ratio": float(daily_df["cash_ratio"].mean()),
        "median_cash_ratio": float(daily_df["cash_ratio"].median()),
        "daily_turnover": float(daily_df["turnover"].mean()),
        "total_costs": total_costs,
        "annual_cost_drag": annual_cost_drag,
        "beta_gap": beta_gap,
        "selection_residual": residual,
        "rolling_active": rolling_active,
        "window_days": int(len(aligned)),
    }


def compute_benchmark_diagnostics(
    results: BacktestResults,
    benchmark: Optional[pd.Series],
) -> pd.DataFrame:
    """
    Build benchmark-relative diagnostics table for report and W&B logging.
    """
    payload = _benchmark_diagnostics_payload(results, benchmark)
    if not payload:
        return pd.DataFrame()

    rows = [
        ("Strategy CAGR", payload["strategy_cagr"]),
        ("Benchmark CAGR", payload["benchmark_cagr"]),
        ("Active CAGR Gap", payload["active_cagr_gap"]),
        ("Beta to Benchmark", payload["beta"]),
        ("Annual Alpha (CAPM residual)", payload["alpha_annual"]),
        ("Up Capture", payload["up_capture"]),
        ("Down Capture", payload["down_capture"]),
        ("Avg Gross Exposure", payload["avg_gross_exposure"]),
        ("Median Gross Exposure", payload["median_gross_exposure"]),
        ("Avg Cash Ratio", payload["avg_cash_ratio"]),
        ("Median Cash Ratio", payload["median_cash_ratio"]),
        ("Avg Daily Turnover", payload["daily_turnover"]),
        ("Total Costs ($)", payload["total_costs"]),
        ("Annual Cost Drag (approx)", payload["annual_cost_drag"]),
        ("Beta Gap Contribution (approx)", payload["beta_gap"]),
        ("Selection Residual (approx)", payload["selection_residual"]),
        ("Aligned Return Days", float(payload["window_days"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def plot_benchmark_gap_decomposition(
    results: BacktestResults,
    benchmark: Optional[pd.Series],
    figsize: tuple = (14, 8),
    title: str = "Benchmark Gap Decomposition",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot active CAGR decomposition and rolling active 1Y return.
    """
    payload = _benchmark_diagnostics_payload(results, benchmark)
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1, 1.2]},
    )

    if not payload:
        ax1.text(0.5, 0.5, "No benchmark diagnostics available", ha="center", va="center")
        ax1.set_axis_off()
        ax2.set_axis_off()
    else:
        labels = [
            "Active CAGR Gap",
            "Beta Gap",
            "Cost Drag",
            "Selection Residual",
        ]
        values = np.array(
            [
                payload["active_cagr_gap"],
                payload["beta_gap"],
                payload["annual_cost_drag"],
                payload["selection_residual"],
            ],
            dtype=float,
        ) * 100.0
        colors = ["#E94F37" if v < 0 else "#2E86AB" for v in values]
        ax1.bar(labels, values, color=colors, alpha=0.85)
        ax1.axhline(0.0, color="black", linewidth=0.8)
        ax1.set_ylabel("Contribution (percentage points)")
        ax1.set_title(f"{title} (Approximate Annualized Terms)")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=12)

        rolling_active = payload.get("rolling_active", pd.Series(dtype=float))
        if isinstance(rolling_active, pd.Series) and not rolling_active.empty:
            ax2.plot(
                rolling_active.index,
                rolling_active.values * 100.0,
                color="#6A994E",
                linewidth=1.8,
                label="Rolling Active Return",
            )
            ax2.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax2.set_ylabel("Active Return (%)")
            ax2.set_xlabel("Date")
            ax2.set_title("Rolling 1Y Active Return (Strategy - Benchmark)")
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for rolling active return", ha="center", va="center")
            ax2.set_axis_off()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_benchmark_capture_ratio(
    results: BacktestResults,
    benchmark: Optional[pd.Series],
    figsize: tuple = (8, 5),
    title: str = "Benchmark Capture Ratios",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot up/down capture ratio bars.
    """
    payload = _benchmark_diagnostics_payload(results, benchmark)
    fig, ax = plt.subplots(figsize=figsize)

    if not payload:
        ax.text(0.5, 0.5, "No benchmark diagnostics available", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = ["Up Capture", "Down Capture"]
        values = [payload.get("up_capture", np.nan), payload.get("down_capture", np.nan)]
        bars = ax.bar(labels, values, color=["#2E86AB", "#E94F37"], alpha=0.85)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9, label="Parity = 1.0")
        ax.set_ylabel("Capture Ratio")
        ax.set_xlabel("Capture Type")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(loc="best")
        for bar, val in zip(bars, values):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    float(val),
                    f"{float(val):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_turnover_performance_by_year(results: BacktestResults) -> pd.DataFrame:
    """
    Compute yearly return vs yearly turnover statistics.
    """
    returns = results.get_returns().dropna()
    turnover = results.get_turnover().dropna()
    aligned = pd.concat(
        [returns.rename("return"), turnover.rename("turnover")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    yearly = aligned.resample("YE").agg(
        year_return=("return", lambda s: float((1.0 + s).prod() - 1.0)),
        avg_daily_turnover=("turnover", "mean"),
        annual_turnover=("turnover", "sum"),
        trading_days=("return", "count"),
    )
    yearly["year"] = yearly.index.year.astype(int)
    yearly["year_return_pct"] = yearly["year_return"] * 100.0
    yearly["avg_daily_turnover_pct"] = yearly["avg_daily_turnover"] * 100.0
    yearly["annual_turnover_x"] = yearly["annual_turnover"]
    cols = [
        "year",
        "year_return",
        "year_return_pct",
        "avg_daily_turnover",
        "avg_daily_turnover_pct",
        "annual_turnover",
        "annual_turnover_x",
        "trading_days",
    ]
    return yearly[cols].reset_index(drop=True)


def plot_turnover_vs_performance_by_year(
    yearly_df: pd.DataFrame,
    figsize: tuple = (10, 6),
    title: str = "Turnover vs Performance by Year",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Scatter chart of annual performance vs average daily turnover.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if yearly_df is None or yearly_df.empty:
        ax.text(0.5, 0.5, "No yearly turnover/performance data", ha="center", va="center")
        ax.set_axis_off()
    else:
        pos = yearly_df[yearly_df["year_return_pct"] >= 0]
        neg = yearly_df[yearly_df["year_return_pct"] < 0]
        if not pos.empty:
            ax.scatter(
                pos["avg_daily_turnover_pct"],
                pos["year_return_pct"],
                color="#2E86AB",
                alpha=0.8,
                label="Positive Return Year",
            )
        if not neg.empty:
            ax.scatter(
                neg["avg_daily_turnover_pct"],
                neg["year_return_pct"],
                color="#E94F37",
                alpha=0.8,
                label="Negative Return Year",
            )

        for _, row in yearly_df.iterrows():
            ax.annotate(
                str(int(row["year"])),
                (row["avg_daily_turnover_pct"], row["year_return_pct"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Average Daily Turnover (%)")
        ax.set_ylabel("Annual Return (%)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_cost_drag_by_year(results: BacktestResults) -> pd.DataFrame:
    """
    Compute yearly trading-cost drag based on annual costs and average equity.
    """
    daily_df = pd.DataFrame(
        [
            {
                "date": r.date,
                "portfolio_value": r.portfolio_value,
                "costs": r.costs,
                "turnover": r.turnover,
            }
            for r in results.daily_results
        ]
    )
    if daily_df.empty:
        return pd.DataFrame()

    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["year"] = daily_df["date"].dt.year
    grouped = daily_df.groupby("year", as_index=False).agg(
        total_costs=("costs", "sum"),
        avg_equity=("portfolio_value", "mean"),
        rebalance_events=("turnover", lambda s: int((s > 0).sum())),
        avg_turnover=("turnover", "mean"),
    )
    grouped["cost_drag"] = np.where(
        grouped["avg_equity"] > 0,
        -(grouped["total_costs"] / grouped["avg_equity"]),
        np.nan,
    )
    grouped["cost_drag_pct"] = grouped["cost_drag"] * 100.0
    grouped["avg_turnover_pct"] = grouped["avg_turnover"] * 100.0
    return grouped


def plot_cost_drag_by_year(
    yearly_df: pd.DataFrame,
    figsize: tuple = (12, 6),
    title: str = "Cost Drag by Year",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot annual cost drag (%) and total costs ($) by year.
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    if yearly_df is None or yearly_df.empty:
        ax1.text(0.5, 0.5, "No yearly cost data", ha="center", va="center")
        ax1.set_axis_off()
    else:
        x = yearly_df["year"].astype(str)
        ax1.bar(x, yearly_df["cost_drag_pct"], color="#E94F37", alpha=0.75, label="Cost Drag (%)")
        ax1.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax1.set_ylabel("Cost Drag (%)")
        ax1.set_xlabel("Year")
        ax1.grid(True, axis="y", alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(x, yearly_df["total_costs"], color="#2E86AB", marker="o", linewidth=1.8, label="Total Costs ($)")
        ax2.set_ylabel("Total Costs ($)")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        ax1.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_rebalance_event_costs(results: BacktestResults) -> pd.DataFrame:
    """
    Compute cost statistics for rebalance events (turnover/cost days).
    """
    daily_df = pd.DataFrame(
        [
            {
                "Date": r.date,
                "PortfolioValue": r.portfolio_value,
                "Turnover": r.turnover,
                "CostDollars": r.costs,
            }
            for r in results.daily_results
        ]
    )
    if daily_df.empty:
        return pd.DataFrame()

    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    events = daily_df[(daily_df["Turnover"] > 0) | (daily_df["CostDollars"] > 0)].copy()
    if events.empty:
        return pd.DataFrame()

    events["TurnoverPct"] = events["Turnover"] * 100.0
    events["CostBps"] = np.where(
        events["PortfolioValue"] > 0,
        (events["CostDollars"] / events["PortfolioValue"]) * 10000.0,
        np.nan,
    )
    events["Year"] = events["Date"].dt.year
    return events.sort_values("Date").reset_index(drop=True)


def plot_rebalance_event_costs(
    events_df: pd.DataFrame,
    figsize: tuple = (14, 8),
    title: str = "Cost Drag by Rebalance Events",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot turnover-cost relationship and event cost time series.
    """
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.1, 1.0]},
    )
    if events_df is None or events_df.empty:
        ax1.text(0.5, 0.5, "No rebalance events available", ha="center", va="center")
        ax1.set_axis_off()
        ax2.set_axis_off()
    else:
        sc = ax1.scatter(
            events_df["TurnoverPct"],
            events_df["CostBps"],
            c=events_df["CostDollars"],
            cmap="viridis",
            alpha=0.8,
        )
        cbar = fig.colorbar(sc, ax=ax1)
        cbar.set_label("Cost ($)")
        ax1.set_xlabel("Rebalance Turnover (%)")
        ax1.set_ylabel("Trading Cost (bps of portfolio)")
        ax1.set_title("Rebalance Event Cost vs Turnover")
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            events_df["Date"],
            events_df["CostBps"],
            color="#E94F37",
            linewidth=1.2,
            label="Cost per event (bps)",
        )
        rolling = events_df.set_index("Date")["CostBps"].rolling(20, min_periods=1).median()
        ax2.plot(
            rolling.index,
            rolling.values,
            color="#2E86AB",
            linewidth=1.8,
            label="Rolling median (20 events)",
        )
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Cost (bps)")
        ax2.set_title(title)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def build_plotly_charts(
    results: BacktestResults,
    benchmark: Optional[pd.Series] = None,
) -> dict[str, object]:
    """
    Build interactive Plotly charts for W&B logging.

    Returns:
        Dict of chart_name -> Plotly Figure. Empty if Plotly is unavailable.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return {}

    charts: dict[str, object] = {}
    equity = results.get_equity_curve().dropna()
    returns = results.get_returns().dropna()
    if equity.empty:
        return charts

    # Equity curve vs benchmark (normalized).
    eq_norm = equity / equity.iloc[0] * 100.0
    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=eq_norm.index,
            y=eq_norm.values,
            mode="lines",
            name="Strategy",
            line=dict(width=2),
        )
    )
    if benchmark is not None and len(benchmark.dropna()) > 1:
        bench = benchmark.dropna()
        bench_norm = bench / bench.iloc[0] * 100.0
        fig_equity.add_trace(
            go.Scatter(
                x=bench_norm.index,
                y=bench_norm.values,
                mode="lines",
                name="Benchmark",
                line=dict(width=1.5),
            )
        )
    fig_equity.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Value (Starting = 100)",
        legend_title="Series",
    )
    charts["equity_curve"] = fig_equity

    # Drawdown chart.
    drawdown = compute_drawdown_series(equity) * 100.0
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#E94F37", width=1.5),
        )
    )
    fig_drawdown.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend_title="Series",
    )
    charts["drawdown"] = fig_drawdown

    # Combined equity + drawdown.
    fig_combined = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Portfolio Value", "Drawdown"),
        vertical_spacing=0.12,
    )
    fig_combined.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name="Portfolio Value",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )
    fig_combined.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color="#E94F37", width=1.5),
        ),
        row=2,
        col=1,
    )
    fig_combined.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig_combined.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig_combined.update_xaxes(title_text="Date", row=2, col=1)
    fig_combined.update_layout(title="Portfolio Performance", legend_title="Series")
    charts["equity_drawdown_combined"] = fig_combined

    # Rolling Sharpe.
    rolling_sharpe = get_rolling_sharpe(results, window=252).dropna()
    fig_sharpe = go.Figure()
    if not rolling_sharpe.empty:
        fig_sharpe.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name="Rolling Sharpe (252d)",
                line=dict(width=1.5),
            )
        )
    fig_sharpe.add_hline(y=0, line_dash="dash", line_color="red")
    fig_sharpe.add_hline(y=1, line_dash="dot", line_color="green")
    fig_sharpe.update_layout(
        title="Rolling Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        legend_title="Series",
    )
    charts["rolling_sharpe"] = fig_sharpe

    # Positions + exposure.
    positions = results.get_positions_count()
    exposure = pd.Series({r.date: r.gross_exposure * 100.0 for r in results.daily_results})
    fig_exposure = make_subplots(specs=[[{"secondary_y": True}]])
    fig_exposure.add_trace(
        go.Scatter(
            x=positions.index,
            y=positions.values,
            mode="lines",
            name="Positions",
            line=dict(width=1.5),
        ),
        secondary_y=False,
    )
    fig_exposure.add_trace(
        go.Scatter(
            x=exposure.index,
            y=exposure.values,
            mode="lines",
            name="Gross Exposure (%)",
            line=dict(width=1.5, color="#F18F01"),
        ),
        secondary_y=True,
    )
    fig_exposure.update_xaxes(title_text="Date")
    fig_exposure.update_yaxes(title_text="Number of Positions", secondary_y=False)
    fig_exposure.update_yaxes(title_text="Gross Exposure (%)", secondary_y=True)
    fig_exposure.update_layout(title="Positions and Exposure", legend_title="Series")
    charts["positions_exposure"] = fig_exposure

    # Turnover.
    turnover = results.get_turnover().dropna() * 100.0
    fig_turnover = go.Figure()
    if not turnover.empty:
        fig_turnover.add_trace(
            go.Bar(x=turnover.index, y=turnover.values, name="Daily Turnover (%)")
        )
        fig_turnover.add_hline(
            y=turnover.mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {turnover.mean():.2f}%",
        )
    fig_turnover.update_layout(
        title="Daily Turnover",
        xaxis_title="Date",
        yaxis_title="Turnover (%)",
        legend_title="Series",
    )
    charts["turnover"] = fig_turnover

    # Daily P&L.
    daily_pnl = equity.diff().dropna()
    colors = np.where(daily_pnl.values >= 0, "#2CA02C", "#D62728")
    fig_pnl = go.Figure()
    if not daily_pnl.empty:
        fig_pnl.add_trace(
            go.Bar(
                x=daily_pnl.index,
                y=daily_pnl.values,
                marker_color=colors,
                name="Daily P&L",
            )
        )
    fig_pnl.update_layout(
        title="Daily Profit and Loss",
        xaxis_title="Date",
        yaxis_title="Daily P&L ($)",
        legend_title="Series",
    )
    charts["daily_pnl"] = fig_pnl

    # Monthly returns heatmap.
    monthly = get_monthly_returns(results)
    if not monthly.empty:
        mdf = pd.DataFrame({"return": monthly})
        mdf["year"] = mdf.index.year
        mdf["month"] = mdf.index.month
        pivot = mdf.pivot(index="year", columns="month", values="return")
        fig_monthly = go.Figure(
            data=go.Heatmap(
                z=(pivot.values * 100.0),
                x=[str(c) for c in pivot.columns],
                y=[str(i) for i in pivot.index],
                colorscale="RdYlGn",
                colorbar=dict(title="Return (%)"),
            )
        )
        fig_monthly.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
        )
        charts["monthly_returns"] = fig_monthly

    # Returns distribution.
    fig_dist = go.Figure()
    if not returns.empty:
        fig_dist.add_trace(
            go.Histogram(
                x=returns.values * 100.0,
                nbinsx=50,
                name="Daily Returns",
                opacity=0.75,
            )
        )
        fig_dist.add_vline(
            x=float(returns.mean() * 100.0),
            line_dash="dash",
            line_color="#E94F37",
            annotation_text=f"Mean: {returns.mean()*100:.2f}%",
        )
        fig_dist.add_vline(
            x=float(returns.median() * 100.0),
            line_dash="dot",
            line_color="#6A994E",
            annotation_text=f"Median: {returns.median()*100:.2f}%",
        )
    fig_dist.update_layout(
        title="Distribution of Daily Returns",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        legend_title="Series",
        barmode="overlay",
    )
    charts["returns_distribution"] = fig_dist

    # Negative tail distribution.
    fig_tail = go.Figure()
    if not returns.empty:
        threshold = returns.quantile(0.05)
        tail = returns[returns <= threshold]
        if not tail.empty:
            fig_tail.add_trace(
                go.Histogram(
                    x=tail.values * 100.0,
                    nbinsx=max(10, min(40, len(tail))),
                    name="Tail Returns (<= 5th pct)",
                    opacity=0.8,
                    marker_color="#E94F37",
                )
            )
            fig_tail.add_vline(
                x=float(threshold * 100.0),
                line_dash="dash",
                line_color="#222222",
                annotation_text=f"Threshold: {threshold*100:.2f}%",
            )
            fig_tail.add_vline(
                x=float(tail.mean() * 100.0),
                line_dash="dot",
                line_color="#6A994E",
                annotation_text=f"Tail Mean: {tail.mean()*100:.2f}%",
            )
    fig_tail.update_layout(
        title="Negative Tail Distribution (<= 5th Percentile)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        legend_title="Series",
        barmode="overlay",
    )
    charts["negative_tail_distribution"] = fig_tail

    # Benchmark diagnostics (decomposition + rolling active), if benchmark exists.
    payload = _benchmark_diagnostics_payload(results, benchmark)
    if payload:
        labels = [
            "Active CAGR Gap",
            "Beta Gap",
            "Cost Drag",
            "Selection Residual",
        ]
        vals = [
            float(payload["active_cagr_gap"]) * 100.0,
            float(payload["beta_gap"]) * 100.0,
            float(payload["annual_cost_drag"]) * 100.0,
            float(payload["selection_residual"]) * 100.0,
        ]
        colors = ["#E94F37" if v < 0 else "#2E86AB" for v in vals]
        fig_diag = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=False,
            subplot_titles=(
                "Active CAGR Decomposition (Approx.)",
                "Rolling 1Y Active Return",
            ),
            vertical_spacing=0.18,
        )
        fig_diag.add_trace(
            go.Bar(
                x=labels,
                y=vals,
                marker_color=colors,
                name="Contribution (pp)",
            ),
            row=1,
            col=1,
        )
        fig_diag.add_hline(y=0.0, row=1, col=1, line_dash="dash", line_color="black")

        rolling_active = payload.get("rolling_active", pd.Series(dtype=float))
        if isinstance(rolling_active, pd.Series) and not rolling_active.empty:
            fig_diag.add_trace(
                go.Scatter(
                    x=rolling_active.index,
                    y=rolling_active.values * 100.0,
                    mode="lines",
                    name="Rolling Active Return",
                    line=dict(color="#6A994E", width=1.8),
                ),
                row=2,
                col=1,
            )
            fig_diag.add_hline(y=0.0, row=2, col=1, line_dash="dash", line_color="black")

        fig_diag.update_yaxes(title_text="Contribution (pp)", row=1, col=1)
        fig_diag.update_yaxes(title_text="Active Return (%)", row=2, col=1)
        fig_diag.update_xaxes(title_text="Date", row=2, col=1)
        fig_diag.update_layout(
            title="Benchmark Gap Diagnostics",
            legend_title="Series",
        )
        charts["benchmark_gap_diagnostics"] = fig_diag

    return charts


def plot_stock_with_trades(
    ticker: str,
    prices: pd.Series,
    trades: pd.DataFrame,
    figsize: tuple = (12, 6),
    max_markers_per_side: int = 400,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot individual stock price with buy/sell markers.
    
    Args:
        ticker: Stock ticker symbol
        prices: Price series for the stock
        trades: Trades DataFrame (filtered for this ticker)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    ax.plot(prices.index, prices.values, linewidth=1.5, color='#2E86AB', label=ticker)
    
    if not trades.empty:
        trades = trades.copy()
        trades['date'] = pd.to_datetime(trades['date'])
        
        buys = trades[trades['side'] == 'BUY']
        sells = trades[trades['side'] == 'SELL']

        if len(buys) > max_markers_per_side:
            buys = buys.sample(n=max_markers_per_side, random_state=42).sort_values('date')
        if len(sells) > max_markers_per_side:
            sells = sells.sample(n=max_markers_per_side, random_state=42).sort_values('date')

        buy_dates = pd.to_datetime(buys['date']).drop_duplicates()
        sell_dates = pd.to_datetime(sells['date']).drop_duplicates()
        buy_dates = buy_dates[buy_dates.isin(prices.index)]
        sell_dates = sell_dates[sell_dates.isin(prices.index)]

        if len(buy_dates) > 0:
            buy_prices = prices.reindex(buy_dates).dropna()
            ax.scatter(
                buy_prices.index,
                buy_prices.values,
                marker='^',
                color='green',
                s=40,
                zorder=5,
                edgecolors='darkgreen',
                linewidths=0.5,
            )
        if len(sell_dates) > 0:
            sell_prices = prices.reindex(sell_dates).dropna()
            ax.scatter(
                sell_prices.index,
                sell_prices.values,
                marker='v',
                color='red',
                s=40,
                zorder=5,
                edgecolors='darkred',
                linewidths=0.5,
            )
        
        ax.scatter([], [], marker='^', color='green', s=100, label=f'Buy ({len(buys)})')
        ax.scatter([], [], marker='v', color='red', s=100, label=f'Sell ({len(sells)})')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{ticker} - Price with Trades', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def generate_stock_charts(
    results: BacktestResults,
    price_data: pd.DataFrame,
    output_dir: Path,
    top_n: int = 10,
    mode: str = "top",
    max_charts: Optional[int] = None,
    chunk_size: int = 25,
    chunk_index: int = 0,
    max_markers_per_side: int = 400,
) -> list[Path]:
    """
    Generate individual stock charts.
    
    Args:
        results: Backtest results
        price_data: DataFrame with stock prices (columns = tickers)
        output_dir: Output directory
        top_n: Number of top stocks to chart when mode='top'
        mode: 'top' for most-traded stocks, 'all' for every stock in price_data,
            'none' to skip stock charts
        max_charts: Optional hard limit on number of charts to create
        chunk_size: Number of charts in each chunk
        chunk_index: 0-based chunk index to generate
        max_markers_per_side: Cap buy/sell markers per chart for speed
        
    Returns:
        List of paths to generated charts
    """
    trades = results.trades
    all_tickers = list(price_data.columns)
    if mode not in {"none", "top", "all"}:
        raise ValueError("mode must be 'none', 'top' or 'all'")

    if mode == "none":
        return []

    if mode == "all":
        tickers_to_plot = sorted(all_tickers)
    else:
        if trades.empty or 'ticker' not in trades.columns:
            return []
        trade_counts = trades.groupby('ticker').size().sort_values(ascending=False)
        tickers_to_plot = trade_counts.head(top_n).index.tolist()

    if chunk_size > 0:
        start = chunk_index * chunk_size
        end = start + chunk_size
        tickers_to_plot = tickers_to_plot[start:end]
    if max_charts is not None:
        tickers_to_plot = tickers_to_plot[:max_charts]
    
    stock_dir = output_dir / 'stocks'
    stock_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for ticker in tickers_to_plot:
        if ticker in price_data.columns:
            ticker_trades = (
                trades[trades['ticker'] == ticker]
                if (not trades.empty and 'ticker' in trades.columns)
                else pd.DataFrame()
            )
            prices = price_data[ticker].dropna()
            
            if prices.empty:
                continue

            save_path = stock_dir / f'{ticker}_timeseries.png'
            fig = plot_stock_with_trades(
                ticker,
                prices,
                ticker_trades,
                max_markers_per_side=max_markers_per_side,
                save_path=save_path,
            )
            paths.append(save_path)
            plt.close(fig)
    
    return paths


def _attach_realized_pnl_fifo(trades: pd.DataFrame) -> pd.DataFrame:
    """Attach realized PnL on SELL rows using FIFO lot matching."""
    if trades.empty:
        return trades.copy()

    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    if "realized_pnl" not in df.columns:
        df["realized_pnl"] = 0.0
    else:
        df["realized_pnl"] = df["realized_pnl"].fillna(0.0).astype(float)

    # If realized_pnl already has sell-side values, keep them.
    has_existing = ((df["side"] == "SELL") & (df["realized_pnl"] != 0.0)).any()
    if has_existing:
        return df

    for _, idx in df.groupby("ticker", sort=False).groups.items():
        inventory: list[list[float]] = []  # [buy_price, shares]
        for row_idx in idx:
            row = df.loc[row_idx]
            side = str(row["side"])
            qty = float(row["shares"])
            px = float(row["price"])

            if side == "BUY":
                inventory.append([px, qty])
                continue

            qty_to_sell = qty
            realized = 0.0
            while qty_to_sell > 1e-8 and inventory:
                buy_px, buy_qty = inventory[0]
                matched = min(qty_to_sell, buy_qty)
                realized += matched * (px - buy_px)
                qty_to_sell -= matched
                buy_qty -= matched

                if buy_qty <= 1e-8:
                    inventory.pop(0)
                else:
                    inventory[0][1] = buy_qty

            df.at[row_idx, "realized_pnl"] = realized

    return df


def _compute_roundtrip_trades_fifo(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Build FIFO-matched round-trip trade records.

    Each SELL is matched against prior BUY lots and expanded into one or more
    round-trip rows with realized PnL.
    """
    columns = [
        "Ticker",
        "OpenDate",
        "CloseDate",
        "Shares",
        "BuyPrice",
        "SellPrice",
        "RealizedPnL",
        "ReturnPct",
        "HoldingDays",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    df = trades.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    records: list[dict] = []

    for ticker, group in df.groupby("ticker", sort=False):
        inventory: list[list] = []  # [buy_price, buy_shares, buy_date]

        for _, row in group.iterrows():
            side = str(row["side"])
            qty = float(row["shares"])
            px = float(row["price"])
            dt = pd.Timestamp(row["date"])

            if side == "BUY":
                inventory.append([px, qty, dt])
                continue

            qty_to_sell = qty
            while qty_to_sell > 1e-8 and inventory:
                buy_px, buy_qty, buy_dt = inventory[0]
                matched = min(qty_to_sell, buy_qty)
                realized = matched * (px - buy_px)
                ret_pct = ((px / buy_px) - 1.0) * 100.0 if buy_px != 0 else np.nan

                records.append(
                    {
                        "Ticker": str(ticker),
                        "OpenDate": buy_dt,
                        "CloseDate": dt,
                        "Shares": float(matched),
                        "BuyPrice": float(buy_px),
                        "SellPrice": float(px),
                        "RealizedPnL": float(realized),
                        "ReturnPct": float(ret_pct) if not pd.isna(ret_pct) else np.nan,
                        "HoldingDays": int((dt - buy_dt).days),
                    }
                )

                qty_to_sell -= matched
                buy_qty -= matched
                if buy_qty <= 1e-8:
                    inventory.pop(0)
                else:
                    inventory[0][1] = buy_qty

    if not records:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(records)
    out["OpenDate"] = pd.to_datetime(out["OpenDate"])
    out["CloseDate"] = pd.to_datetime(out["CloseDate"])
    return out


def compute_worst_trades_table(
    results: BacktestResults,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return top-N worst FIFO round-trip trades by realized PnL.
    """
    roundtrips = _compute_roundtrip_trades_fifo(results.trades)
    if roundtrips.empty:
        return roundtrips

    table = roundtrips.sort_values("RealizedPnL", ascending=True).head(top_n).copy()
    table["OpenDate"] = table["OpenDate"].dt.strftime("%Y-%m-%d")
    table["CloseDate"] = table["CloseDate"].dt.strftime("%Y-%m-%d")
    return table.reset_index(drop=True)


def compute_ticker_health_table(
    results: BacktestResults,
    price_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-ticker portfolio-health metrics table.

    Metrics combine realized trade PnL and weight-based return contributions.
    """
    if price_data is None or price_data.empty:
        return pd.DataFrame()

    weights = results.get_weights_history()
    if weights.empty:
        return pd.DataFrame()

    common = [c for c in weights.columns if c in price_data.columns]
    if not common:
        return pd.DataFrame()

    stock_returns = price_data[common].pct_change().fillna(0.0)
    weights = weights[common].reindex(stock_returns.index).fillna(0.0)
    contrib = weights.shift(1).fillna(0.0) * stock_returns

    base = pd.DataFrame(index=common)
    base["TotalContribution"] = contrib.sum().astype(float)
    base["ContributionPct"] = base["TotalContribution"] * 100.0
    base["AvgWeight"] = weights.mean().astype(float)
    base["AvgWeightPct"] = base["AvgWeight"] * 100.0
    base["AvgAbsWeightPct"] = (weights.abs().mean() * 100.0).astype(float)
    base["HoldingDays"] = (weights.abs() > 1e-8).sum().astype(int)
    base["StockReturn"] = ((1.0 + stock_returns).prod() - 1.0).astype(float)
    base["StockReturnPct"] = base["StockReturn"] * 100.0
    base["AnnualVolatility"] = (stock_returns.std() * np.sqrt(252)).astype(float)
    base["AnnualVolatilityPct"] = base["AnnualVolatility"] * 100.0
    base["StockMaxDrawdown"] = ((price_data[common] / price_data[common].cummax()) - 1.0).min().astype(float)
    base["StockMaxDrawdownPct"] = base["StockMaxDrawdown"] * 100.0

    trades = _attach_realized_pnl_fifo(results.trades)
    if not trades.empty and "ticker" in trades.columns:
        sells = trades[trades["side"] == "SELL"].copy()
        grouped = sells.groupby("ticker")
        trade_metrics = pd.DataFrame(index=grouped.size().index)
        trade_metrics["RoundTrips"] = grouped.size().astype(int)
        trade_metrics["RealizedPnL"] = grouped["realized_pnl"].sum().astype(float)
        trade_metrics["AvgRealizedPnL"] = grouped["realized_pnl"].mean().astype(float)
        trade_metrics["WinRate"] = grouped["realized_pnl"].apply(lambda s: (s > 0).mean()).astype(float)

        slip = trades.groupby("ticker")["slippage_cost"].sum().rename("TotalSlippage")
        trade_metrics = trade_metrics.join(slip, how="outer")
        base = base.join(trade_metrics, how="outer")

    base = base.fillna(0.0)
    base.index.name = "Ticker"
    df = base.reset_index().sort_values("TotalContribution", ascending=False)
    return df


def compute_drawdown_detractor_metrics(
    results: BacktestResults,
    price_data: pd.DataFrame,
    lookback_days: int = 63,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Build detailed metrics table for drawdown detractor tickers.
    """
    equity = results.get_equity_curve().dropna()
    weights = results.get_weights_history()
    if len(equity) < 2 or price_data.empty or weights.empty:
        return pd.DataFrame()

    start_pos = max(0, len(equity) - lookback_days)
    window_start = equity.index[start_pos]
    window_end = equity.index[-1]

    common = [c for c in weights.columns if c in price_data.columns]
    if not common:
        return pd.DataFrame()

    stock_returns = price_data[common].pct_change().fillna(0.0)
    weights = weights[common].reindex(stock_returns.index).fillna(0.0)
    contrib = weights.shift(1).fillna(0.0) * stock_returns
    contrib_window = contrib.loc[window_start:window_end]
    if contrib_window.empty:
        return pd.DataFrame()

    detractors = contrib_window.sum().sort_values().head(top_n)
    rows: list[dict] = []
    trades = _attach_realized_pnl_fifo(results.trades)
    if not trades.empty:
        trades = trades.copy()
        trades["date"] = pd.to_datetime(trades["date"])
        trades = trades[(trades["date"] >= window_start) & (trades["date"] <= window_end)]

    for ticker, contribution in detractors.items():
        rets = stock_returns[ticker].loc[window_start:window_end]
        price_window = price_data[ticker].loc[window_start:window_end].dropna()
        w = weights[ticker].loc[window_start:window_end]

        max_dd = 0.0
        if len(price_window) > 1:
            max_dd = float(((price_window / price_window.cummax()) - 1.0).min())

        realized = 0.0
        round_trips = 0
        win_rate = 0.0
        if not trades.empty and ticker in trades["ticker"].values:
            sells = trades[(trades["ticker"] == ticker) & (trades["side"] == "SELL")]
            if not sells.empty:
                realized = float(sells["realized_pnl"].sum())
                round_trips = int(len(sells))
                win_rate = float((sells["realized_pnl"] > 0).mean())

        rows.append(
            {
                "Ticker": ticker,
                "Contribution": float(contribution),
                "ContributionPct": float(contribution * 100.0),
                "WindowReturn": float((1.0 + rets).prod() - 1.0),
                "WindowReturnPct": float(((1.0 + rets).prod() - 1.0) * 100.0),
                "WindowVolatility": float(rets.std() * np.sqrt(252)),
                "WindowVolatilityPct": float(rets.std() * np.sqrt(252) * 100.0),
                "WindowMaxDrawdown": max_dd,
                "WindowMaxDrawdownPct": float(max_dd * 100.0),
                "AvgWeightWindowPct": float(w.mean() * 100.0),
                "MaxWeightWindowPct": float(w.max() * 100.0),
                "RealizedPnLWindow": realized,
                "RoundTripsWindow": round_trips,
                "WinRateWindow": win_rate,
            }
        )

    return pd.DataFrame(rows).sort_values("Contribution", ascending=True)


def analyze_recent_drawdown(
    results: BacktestResults,
    price_data: pd.DataFrame,
    benchmark: Optional[pd.Series] = None,
    lookback_days: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze the recent drawdown window and estimate top ticker detractors.

    Uses prior-day portfolio weights to estimate return contributions.
    """
    equity = results.get_equity_curve().dropna()
    if len(equity) < 2:
        return pd.DataFrame(), pd.DataFrame()

    start_pos = max(0, len(equity) - lookback_days)
    window_start = equity.index[start_pos]
    window_end = equity.index[-1]
    window_equity = equity.loc[window_start:window_end]
    window_returns = results.get_returns().loc[window_start:window_end]

    portfolio_return = (
        window_equity.iloc[-1] / window_equity.iloc[0] - 1
        if len(window_equity) > 1 else 0.0
    )
    window_drawdown = compute_drawdown_series(window_equity)
    worst_day = window_returns.idxmin() if len(window_returns) else None
    worst_day_return = window_returns.min() if len(window_returns) else 0.0

    turnover = results.get_turnover().loc[window_start:window_end]
    total_costs = sum(
        r.costs for r in results.daily_results
        if window_start <= r.date <= window_end
    )

    benchmark_return = np.nan
    if benchmark is not None and len(benchmark) > 1:
        benchmark_window = benchmark.loc[window_start:window_end]
        if len(benchmark_window) > 1 and benchmark_window.iloc[0] != 0:
            benchmark_return = benchmark_window.iloc[-1] / benchmark_window.iloc[0] - 1

    summary = pd.DataFrame([
        ("Window Start", str(window_start.date())),
        ("Window End", str(window_end.date())),
        ("Portfolio Return", f"{portfolio_return:.2%}"),
        ("Benchmark Return", f"{benchmark_return:.2%}" if not pd.isna(benchmark_return) else "N/A"),
        ("Max Drawdown (Window)", f"{window_drawdown.min():.2%}"),
        ("Worst Day", str(worst_day.date()) if worst_day is not None else "N/A"),
        ("Worst Day Return", f"{worst_day_return:.2%}"),
        ("Avg Daily Turnover", f"{turnover.mean():.2%}" if len(turnover) else "N/A"),
        ("Total Trading Costs", f"${total_costs:,.0f}"),
    ], columns=["Metric", "Value"])

    detractor_metrics = compute_drawdown_detractor_metrics(
        results=results,
        price_data=price_data,
        lookback_days=lookback_days,
        top_n=15,
    )
    if detractor_metrics.empty:
        return summary, pd.DataFrame(columns=["Ticker", "Contribution", "ContributionPct"])

    detractors_df = detractor_metrics[["Ticker", "Contribution"]].copy()
    detractors_df["ContributionPct"] = detractors_df["Contribution"].map(lambda x: f"{x:.2%}")

    return summary, detractors_df


def plot_recent_drawdown_detractors(
    detractors: pd.DataFrame,
    figsize: tuple = (12, 6),
    title: str = "Recent Drawdown: Top Detractor Tickers",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plot estimated ticker-level return contribution detractors."""
    fig, ax = plt.subplots(figsize=figsize)

    if detractors.empty:
        ax.text(0.5, 0.5, "No detractor data available", ha='center', va='center')
        ax.set_axis_off()
    else:
        d = detractors.sort_values("Contribution", ascending=True)
        ax.barh(d["Ticker"], d["Contribution"] * 100, color='#E94F37', alpha=0.8)
        ax.set_xlabel("Contribution (pct points)")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_summary_table(metrics: PerformanceMetrics) -> pd.DataFrame:
    """
    Generate a summary table of metrics.
    
    Args:
        metrics: Performance metrics
        
    Returns:
        DataFrame with metrics summary
    """
    return pd.DataFrame(
        list(metrics.to_dict().items()),
        columns=['Metric', 'Value']
    )


def generate_full_report(
    results: BacktestResults,
    output_dir: Path,
    benchmark: Optional[pd.Series] = None,
    price_data: Optional[pd.DataFrame] = None,
    all_stock_charts: bool = False,
    stock_charts_mode: str = "top",
    stock_chart_top_n: int = 10,
    stock_chart_max: Optional[int] = None,
    stock_chart_chunk_size: int = 25,
    stock_chart_chunk_index: int = 0,
    stock_marker_limit: int = 400,
    include_trade_stats: bool = False,
) -> Path:
    """
    Generate a full HTML report with all charts.
    
    Args:
        results: Backtest results
        output_dir: Directory for output files
        benchmark: Optional benchmark series
        
    Returns:
        Path to generated report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Trade stats can be expensive on very large ledgers.
    trade_stats = compute_trade_stats(results.trades) if include_trade_stats else pd.DataFrame()
    
    # Backward compatibility: all_stock_charts flag overrides explicit mode.
    if all_stock_charts:
        stock_charts_mode = "all"

    # Generate core charts and close each figure immediately.
    fig = plot_equity_curve(
        results, benchmark,
        save_path=output_dir / 'equity_curve.png'
    )
    plt.close(fig)
    fig = plot_drawdown(
        results,
        save_path=output_dir / 'drawdown.png'
    )
    plt.close(fig)
    fig = plot_monthly_returns(
        results,
        save_path=output_dir / 'monthly_returns.png'
    )
    plt.close(fig)
    fig = plot_rolling_sharpe(
        results,
        save_path=output_dir / 'rolling_sharpe.png'
    )
    plt.close(fig)
    fig = plot_positions_and_exposure(
        results,
        save_path=output_dir / 'exposure.png'
    )
    plt.close(fig)
    fig = plot_turnover(
        results,
        save_path=output_dir / 'turnover.png'
    )
    plt.close(fig)
    
    # NEW: Combined equity + drawdown chart
    fig = plot_equity_and_drawdown(
        results, benchmark,
        save_path=output_dir / 'equity_drawdown_combined.png'
    )
    plt.close(fig)
    
    # NEW: Trading signals chart
    fig = plot_trading_signals(
        results,
        max_markers_per_side=max(200, stock_marker_limit * 4),
        save_path=output_dir / 'trading_signals.png'
    )
    plt.close(fig)
    
    # NEW: Daily P&L chart
    fig = plot_daily_pnl(
        results,
        save_path=output_dir / 'daily_pnl.png'
    )
    plt.close(fig)

    fig = plot_returns_distribution(
        results,
        save_path=output_dir / 'returns_distribution.png'
    )
    plt.close(fig)

    fig = plot_negative_tail_distribution(
        results,
        percentile=5.0,
        save_path=output_dir / 'negative_tail_distribution.png'
    )
    plt.close(fig)

    # Generate stock charts if price data is available
    stock_chart_paths = []
    drawdown_summary = pd.DataFrame()
    drawdown_detractors = pd.DataFrame()
    benchmark_diagnostics = pd.DataFrame()
    ticker_health_metrics = pd.DataFrame()
    drawdown_detractor_metrics = pd.DataFrame()
    turnover_perf_by_year = compute_turnover_performance_by_year(results)
    cost_drag_by_year = compute_cost_drag_by_year(results)
    rebalance_event_costs = compute_rebalance_event_costs(results)
    rebalance_event_costs_top = pd.DataFrame()
    worst_trades_top20 = compute_worst_trades_table(results, top_n=20)
    if not worst_trades_top20.empty:
        worst_trades_top20.to_csv(output_dir / "worst_trades_top20.csv", index=False)
    if not turnover_perf_by_year.empty:
        turnover_perf_by_year.to_csv(output_dir / "turnover_performance_by_year.csv", index=False)
    if not cost_drag_by_year.empty:
        cost_drag_by_year.to_csv(output_dir / "cost_drag_by_year.csv", index=False)
    if not rebalance_event_costs.empty:
        rebalance_event_costs.to_csv(output_dir / "rebalance_event_costs.csv", index=False)
        rebalance_event_costs_top = (
            rebalance_event_costs.sort_values("CostBps", ascending=False).head(50).copy()
        )
        rebalance_event_costs_top.to_csv(
            output_dir / "rebalance_event_costs_top50.csv",
            index=False,
        )

    fig = plot_turnover_vs_performance_by_year(
        turnover_perf_by_year,
        save_path=output_dir / "turnover_vs_performance_by_year.png",
    )
    plt.close(fig)
    fig = plot_cost_drag_by_year(
        cost_drag_by_year,
        save_path=output_dir / "cost_drag_by_year.png",
    )
    plt.close(fig)
    fig = plot_rebalance_event_costs(
        rebalance_event_costs,
        save_path=output_dir / "rebalance_event_costs.png",
    )
    plt.close(fig)
    if benchmark is not None:
        benchmark_diagnostics = compute_benchmark_diagnostics(results, benchmark)
        if not benchmark_diagnostics.empty:
            benchmark_diagnostics.to_csv(output_dir / "benchmark_diagnostics.csv", index=False)
        fig = plot_benchmark_gap_decomposition(
            results,
            benchmark,
            save_path=output_dir / "benchmark_gap_decomposition.png",
        )
        plt.close(fig)
        fig = plot_benchmark_capture_ratio(
            results,
            benchmark,
            save_path=output_dir / "benchmark_capture_ratio.png",
        )
        plt.close(fig)
    if price_data is not None:
        stock_chart_paths = generate_stock_charts(
            results,
            price_data,
            output_dir,
            top_n=stock_chart_top_n,
            mode=stock_charts_mode,
            max_charts=stock_chart_max,
            chunk_size=stock_chart_chunk_size,
            chunk_index=stock_chart_chunk_index,
            max_markers_per_side=stock_marker_limit,
        )
        drawdown_summary, drawdown_detractors = analyze_recent_drawdown(
            results,
            price_data,
            benchmark=benchmark,
            lookback_days=63,
        )
        fig = plot_recent_drawdown_detractors(
            drawdown_detractors,
            save_path=output_dir / "recent_drawdown_detractors.png",
        )
        plt.close(fig)
        ticker_health_metrics = compute_ticker_health_table(results, price_data)
        drawdown_detractor_metrics = compute_drawdown_detractor_metrics(
            results,
            price_data,
            lookback_days=63,
            top_n=20,
        )
        if not ticker_health_metrics.empty:
            ticker_health_metrics.to_csv(output_dir / "ticker_health_metrics.csv", index=False)
        if not drawdown_detractor_metrics.empty:
            drawdown_detractor_metrics.to_csv(
                output_dir / "drawdown_detractor_metrics.csv",
                index=False,
            )
    
    # Close all figures
    plt.close('all')
    
    # Generate HTML report
    html = _generate_html_report(
        metrics,
        output_dir,
        stock_chart_paths,
        trade_stats,
        drawdown_summary,
        drawdown_detractors,
        benchmark_diagnostics,
        ticker_health_metrics,
        drawdown_detractor_metrics,
        turnover_perf_by_year,
        cost_drag_by_year,
        rebalance_event_costs_top,
        worst_trades_top20,
        all_stock_charts=(stock_charts_mode == "all"),
    )
    
    report_path = output_dir / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path



def _generate_html_report(
    metrics: PerformanceMetrics,
    charts_dir: Path,
    stock_charts: list[Path] = [],
    trade_stats: Optional[pd.DataFrame] = None,
    drawdown_summary: Optional[pd.DataFrame] = None,
    drawdown_detractors: Optional[pd.DataFrame] = None,
    benchmark_diagnostics: Optional[pd.DataFrame] = None,
    ticker_health_metrics: Optional[pd.DataFrame] = None,
    drawdown_detractor_metrics: Optional[pd.DataFrame] = None,
    turnover_perf_by_year: Optional[pd.DataFrame] = None,
    cost_drag_by_year: Optional[pd.DataFrame] = None,
    rebalance_event_costs_top: Optional[pd.DataFrame] = None,
    worst_trades_top20: Optional[pd.DataFrame] = None,
    all_stock_charts: bool = False,
) -> str:
    """Generate HTML report content."""
    
    metrics_html = generate_summary_table(metrics).to_html(index=False)
    
    trade_stats_html = ""
    if trade_stats is not None and not trade_stats.empty:
        trade_stats_html = trade_stats.to_html(index=False, border=0)

    drawdown_summary_html = ""
    if drawdown_summary is not None and not drawdown_summary.empty:
        drawdown_summary_html = drawdown_summary.to_html(index=False, border=0)

    drawdown_detractors_html = ""
    if drawdown_detractors is not None and not drawdown_detractors.empty:
        drawdown_detractors_html = drawdown_detractors.to_html(index=False, border=0)

    benchmark_diag_html = ""
    if benchmark_diagnostics is not None and not benchmark_diagnostics.empty:
        benchmark_diag_html = benchmark_diagnostics.to_html(index=False, border=0)
    benchmark_section_html = ""
    if benchmark_diagnostics is not None and not benchmark_diagnostics.empty:
        benchmark_section_html = f"""
        <h2>Benchmark Gap Diagnostics</h2>
        <div style="max-height: 420px; overflow-y: auto;">
        {benchmark_diag_html}
        </div>
        <img src="benchmark_gap_decomposition.png" alt="Benchmark Gap Diagnostics">
        <h2>Benchmark Capture Ratios (Up/Down)</h2>
        <img src="benchmark_capture_ratio.png" alt="Benchmark Capture Ratios">
        """

    ticker_health_html = ""
    if ticker_health_metrics is not None and not ticker_health_metrics.empty:
        ticker_health_html = ticker_health_metrics.to_html(index=False, border=0)

    detractor_metrics_html = ""
    if drawdown_detractor_metrics is not None and not drawdown_detractor_metrics.empty:
        detractor_metrics_html = drawdown_detractor_metrics.to_html(index=False, border=0)

    turnover_perf_html = ""
    if turnover_perf_by_year is not None and not turnover_perf_by_year.empty:
        turnover_perf_html = turnover_perf_by_year.to_html(index=False, border=0)

    cost_drag_year_html = ""
    if cost_drag_by_year is not None and not cost_drag_by_year.empty:
        cost_drag_year_html = cost_drag_by_year.to_html(index=False, border=0)

    rebalance_event_costs_html = ""
    if rebalance_event_costs_top is not None and not rebalance_event_costs_top.empty:
        rebalance_event_costs_html = rebalance_event_costs_top.to_html(index=False, border=0)

    worst_trades_html = ""
    if worst_trades_top20 is not None and not worst_trades_top20.empty:
        worst_trades_html = worst_trades_top20.to_html(index=False, border=0)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate stock charts HTML
    stock_charts_html = ""
    if stock_charts:
        section_title = "All Stocks vs Time (with Trade Markers)" if all_stock_charts else "Top Traded Stocks"
        stock_charts_html = f"<h2>{section_title}</h2>"
        for path in stock_charts:
            # Path relative to output_dir
            rel_path = path.relative_to(charts_dir)
            stock_charts_html += f'<img src="{rel_path}" alt="{path.stem}"><br>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            img {{ max-width: 100%; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .timestamp {{ color: #999; font-size: 12px; }}
        </style>
    </head>
    <body>
        <h1>Backtest Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>
        
        <h2>Performance Summary</h2>
        {metrics_html}
        
        <h2>Ticker Trade Statistics</h2>
        <div style="max-height: 500px; overflow-y: auto;">
        {trade_stats_html}
        </div>

        <h2>Top 20 Worst Round-Trip Trades</h2>
        <div style="max-height: 500px; overflow-y: auto;">
        {worst_trades_html}
        </div>
        
        <h2>Combined Performance</h2>
        <img src="equity_drawdown_combined.png" alt="Combined Performance">

        <h2>Trading Signals</h2>
        <img src="trading_signals.png" alt="Trading Signals">

        <h2>Daily P&L</h2>
        <img src="daily_pnl.png" alt="Daily P&L">

        <h2>Distribution of Daily Returns</h2>
        <img src="returns_distribution.png" alt="Returns Distribution">

        <h2>Distribution of Negative Tails (&lt;= 5th Percentile)</h2>
        <img src="negative_tail_distribution.png" alt="Negative Tail Distribution">
        
        {benchmark_section_html}

        <h2>Recent Drawdown Diagnostics (Last ~3 Months)</h2>
        {drawdown_summary_html}
        <img src="recent_drawdown_detractors.png" alt="Recent Drawdown Detractors">
        <div style="max-height: 350px; overflow-y: auto;">
        {drawdown_detractors_html}
        </div>

        <h2>Equity Curve</h2>
        <img src="equity_curve.png" alt="Equity Curve">
        
        <h2>Drawdown</h2>
        <img src="drawdown.png" alt="Drawdown">
        
        <h2>Monthly Returns</h2>
        <img src="monthly_returns.png" alt="Monthly Returns">
        
        <h2>Rolling Sharpe Ratio</h2>
        <img src="rolling_sharpe.png" alt="Rolling Sharpe">
        
        <h2>Positions and Exposure</h2>
        <img src="exposure.png" alt="Exposure">
        
        <h2>Turnover</h2>
        <img src="turnover.png" alt="Turnover">

        <h2>Turnover vs Performance by Year</h2>
        <img src="turnover_vs_performance_by_year.png" alt="Turnover vs Performance by Year">
        <div style="max-height: 400px; overflow-y: auto;">
        {turnover_perf_html}
        </div>

        <h2>Cost Drag by Year</h2>
        <img src="cost_drag_by_year.png" alt="Cost Drag by Year">
        <div style="max-height: 400px; overflow-y: auto;">
        {cost_drag_year_html}
        </div>

        <h2>Cost Drag by Rebalance Events</h2>
        <img src="rebalance_event_costs.png" alt="Cost Drag by Rebalance Events">
        <div style="max-height: 400px; overflow-y: auto;">
        {rebalance_event_costs_html}
        </div>

        <h2>Ticker Health Metrics</h2>
        <div style="max-height: 500px; overflow-y: auto;">
        {ticker_health_html}
        </div>

        <h2>Drawdown Detractor Metrics (Detailed)</h2>
        <div style="max-height: 500px; overflow-y: auto;">
        {detractor_metrics_html}
        </div>

        {stock_charts_html}
    </body>
    </html>
    """
    
    return html
