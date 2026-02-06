"""
Metrics module for the equity strategy.

Computes performance metrics:
- CAGR, Sharpe, Sortino
- Max drawdown, volatility
- Turnover (daily/annual)
- Cost ratio
- Gate failure diagnostics
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .backtest import BacktestResults


@dataclass
class PerformanceMetrics:
    """
    Complete performance metrics from a backtest.
    
    Attributes:
        cagr: Compound annual growth rate
        sharpe: Sharpe ratio (annualized)
        sortino: Sortino ratio (annualized)
        max_drawdown: Maximum drawdown (as decimal, e.g., -0.20 = -20%)
        annual_volatility: Annualized volatility
        daily_turnover: Average daily turnover
        annual_turnover: Annualized turnover
        avg_positions: Average number of positions
        avg_holding_period: Approximate average holding period in days
        pct_days_invested: Percentage of days with positions
        total_costs: Total trading costs
        cost_ratio: Costs as percentage of gross PnL
        total_return: Total cumulative return
        num_trading_days: Number of trading days
    """
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    annual_volatility: float
    daily_turnover: float
    annual_turnover: float
    avg_positions: int
    avg_holding_period: float
    pct_days_invested: float
    total_costs: float
    cost_ratio: float
    total_return: float
    num_trading_days: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'CAGR': f"{self.cagr:.2%}",
            'Sharpe Ratio': f"{self.sharpe:.2f}",
            'Sortino Ratio': f"{self.sortino:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Annual Volatility': f"{self.annual_volatility:.2%}",
            'Daily Turnover': f"{self.daily_turnover:.2%}",
            'Annual Turnover': f"{self.annual_turnover:.1f}x",
            'Avg Positions': f"{self.avg_positions:.1f}",
            'Avg Holding Period': f"{self.avg_holding_period:.1f} days",
            '% Days Invested': f"{self.pct_days_invested:.1%}",
            'Total Costs': f"${self.total_costs:,.0f}",
            'Cost Ratio': f"{self.cost_ratio:.2%}",
            'Total Return': f"{self.total_return:.2%}",
            'Trading Days': f"{self.num_trading_days:,}",
        }
    
    def __repr__(self) -> str:
        lines = [f"{k}: {v}" for k, v in self.to_dict().items()]
        return "\n".join(lines)


def compute_cagr(equity_curve: pd.Series) -> float:
    """
    Compute compound annual growth rate.
    
    Args:
        equity_curve: Series of portfolio values over time
        
    Returns:
        CAGR as decimal (e.g., 0.15 = 15%)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    start_val = equity_curve.iloc[0]
    end_val = equity_curve.iloc[-1]
    
    if start_val <= 0:
        return 0.0
    
    # Calculate years
    start_date = equity_curve.index[0]
    end_date = equity_curve.index[-1]
    years = (end_date - start_date).days / 365.25
    
    if years <= 0:
        return 0.0
    
    cagr = (end_val / start_val) ** (1 / years) - 1
    return cagr


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate (default 0)
        annualization_factor: Trading days per year
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    mean_return = excess_returns.mean()
    std_return = returns.std()
    
    if std_return == 0 or pd.isna(std_return):
        return 0.0
    
    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return sharpe


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized Sortino ratio.
    
    Uses downside deviation (std of negative returns only).
    
    Args:
        returns: Daily returns
        risk_free_rate: Annual risk-free rate
        annualization_factor: Trading days per year
        
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / annualization_factor
    mean_return = excess_returns.mean()
    
    # Downside deviation
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if mean_return > 0 else 0.0
    
    downside_std = np.sqrt((negative_returns ** 2).mean())
    
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0
    
    sortino = (mean_return / downside_std) * np.sqrt(annualization_factor)
    return sortino


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute maximum drawdown.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Max drawdown as negative decimal (e.g., -0.20 = -20%)
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Compute running maximum
    running_max = equity_curve.cummax()
    
    # Drawdown at each point
    drawdowns = (equity_curve - running_max) / running_max
    
    return drawdowns.min()


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute drawdown series over time.
    
    Args:
        equity_curve: Series of portfolio values
        
    Returns:
        Series of drawdowns
    """
    running_max = equity_curve.cummax()
    return (equity_curve - running_max) / running_max


def compute_volatility(
    returns: pd.Series,
    annualization_factor: float = 252.0
) -> float:
    """
    Compute annualized volatility.
    
    Args:
        returns: Daily returns
        annualization_factor: Trading days per year
        
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(annualization_factor)


def compute_turnover_stats(
    turnover: pd.Series,
    annualization_factor: float = 252.0
) -> tuple[float, float]:
    """
    Compute turnover statistics.
    
    Args:
        turnover: Daily turnover series
        annualization_factor: Trading days per year
        
    Returns:
        Tuple of (daily_avg, annualized)
    """
    daily_avg = turnover.mean() if len(turnover) > 0 else 0.0
    annualized = daily_avg * annualization_factor
    return daily_avg, annualized


def compute_avg_holding_period(
    turnover: pd.Series
) -> float:
    """
    Estimate average holding period from turnover.
    
    Approximation: holding_period ≈ 1 / (2 * daily_turnover)
    
    Args:
        turnover: Daily turnover series
        
    Returns:
        Estimated average holding period in days
    """
    daily_turnover = turnover.mean() if len(turnover) > 0 else 0.0
    
    if daily_turnover <= 0:
        return float('inf')
    
    # Each day, turnover represents fraction of portfolio traded
    # If turnover = 0.05 (5%), full position replaced every 1/0.05 = 20 days
    # Divide by 2 because turnover counts both buys and sells
    return 1 / (2 * daily_turnover) if daily_turnover > 0 else float('inf')


def compute_cost_ratio(
    total_costs: float,
    gross_pnl: float
) -> float:
    """
    Compute trading costs as percentage of gross PnL.
    
    Args:
        total_costs: Total trading costs
        gross_pnl: Gross profit/loss
        
    Returns:
        Cost ratio
    """
    if gross_pnl <= 0:
        return float('inf') if total_costs > 0 else 0.0
    
    return total_costs / gross_pnl


def compute_metrics(results: BacktestResults) -> PerformanceMetrics:
    """
    Compute all performance metrics from backtest results.
    
    Args:
        results: BacktestResults from backtester
        
    Returns:
        PerformanceMetrics with all statistics
    """
    equity_curve = results.get_equity_curve()
    returns = results.get_returns()
    turnover = results.get_turnover()
    positions = results.get_positions_count()
    
    # Basic metrics
    cagr = compute_cagr(equity_curve)
    sharpe = compute_sharpe(returns)
    sortino = compute_sortino(returns)
    max_dd = compute_max_drawdown(equity_curve)
    vol = compute_volatility(returns)
    
    # Turnover
    daily_turnover, annual_turnover = compute_turnover_stats(turnover)
    
    # Positions
    avg_positions = positions.mean() if len(positions) > 0 else 0
    pct_invested = (positions > 0).mean() if len(positions) > 0 else 0
    
    # Holding period
    holding_period = compute_avg_holding_period(turnover)
    
    # Costs
    total_costs = sum(r.costs for r in results.daily_results)
    start_val = equity_curve.iloc[0] if len(equity_curve) > 0 else 0
    end_val = equity_curve.iloc[-1] if len(equity_curve) > 0 else 0
    gross_pnl = end_val - start_val
    cost_ratio = compute_cost_ratio(total_costs, gross_pnl)
    
    # Total return
    total_return = (end_val / start_val - 1) if start_val > 0 else 0
    
    return PerformanceMetrics(
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        annual_volatility=vol,
        daily_turnover=daily_turnover,
        annual_turnover=annual_turnover,
        avg_positions=int(avg_positions),
        avg_holding_period=holding_period,
        pct_days_invested=pct_invested,
        total_costs=total_costs,
        cost_ratio=cost_ratio if cost_ratio != float('inf') else 0.0,
        total_return=total_return,
        num_trading_days=len(equity_curve),
    )


def print_metrics_summary(metrics: PerformanceMetrics) -> None:
    """
    Print a formatted metrics summary.
    
    Args:
        metrics: Performance metrics
    """
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    for key, value in metrics.to_dict().items():
        print(f"{key:.<25} {value}")
    print("=" * 50)


def get_monthly_returns(results: BacktestResults) -> pd.Series:
    """
    Compute monthly returns.
    
    Args:
        results: Backtest results
        
    Returns:
        Series of monthly returns
    """
    equity_curve = results.get_equity_curve()
    
    # Resample to month-end (use 'ME' for month end)
    monthly = equity_curve.resample('ME').last()
    
    return monthly.pct_change().dropna()


def get_rolling_sharpe(
    results: BacktestResults,
    window: int = 252
) -> pd.Series:
    """
    Compute rolling Sharpe ratio.
    
    Args:
        results: Backtest results
        window: Rolling window in days
        
    Returns:
        Series of rolling Sharpe ratios
    """
    returns = results.get_returns()
    
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    
    return (rolling_mean / rolling_std) * np.sqrt(252)
