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
        
        # Get portfolio value at trade dates
        for _, trade in buys.iterrows():
            if trade['date'] in equity.index:
                val = equity.loc[trade['date']] / 1e6
                ax1.scatter(trade['date'], val, marker='^', color='green', s=20, alpha=0.5)
        
        for _, trade in sells.iterrows():
            if trade['date'] in equity.index:
                val = equity.loc[trade['date']] / 1e6
                ax1.scatter(trade['date'], val, marker='v', color='red', s=20, alpha=0.5)
        
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


def plot_stock_with_trades(
    ticker: str,
    prices: pd.Series,
    trades: pd.DataFrame,
    figsize: tuple = (12, 6),
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
        
        # Plot buy markers
        for _, trade in buys.iterrows():
            if trade['date'] in prices.index:
                price = prices.loc[trade['date']]
                ax.scatter(trade['date'], price, marker='^', color='green', 
                          s=100, zorder=5, edgecolors='darkgreen', linewidths=1)
        
        # Plot sell markers
        for _, trade in sells.iterrows():
            if trade['date'] in prices.index:
                price = prices.loc[trade['date']]
                ax.scatter(trade['date'], price, marker='v', color='red', 
                          s=100, zorder=5, edgecolors='darkred', linewidths=1)
        
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
) -> list[Path]:
    """
    Generate individual stock charts for most-traded stocks.
    
    Args:
        results: Backtest results
        price_data: DataFrame with stock prices (columns = tickers)
        output_dir: Output directory
        top_n: Number of top stocks to chart
        
    Returns:
        List of paths to generated charts
    """
    trades = results.trades
    if trades.empty or 'ticker' not in trades.columns:
        return []
    
    # Find most traded stocks
    trade_counts = trades.groupby('ticker').size().sort_values(ascending=False)
    top_tickers = trade_counts.head(top_n).index.tolist()
    
    stock_dir = output_dir / 'stocks'
    stock_dir.mkdir(parents=True, exist_ok=True)
    
    paths = []
    for ticker in top_tickers:
        if ticker in price_data.columns:
            ticker_trades = trades[trades['ticker'] == ticker]
            prices = price_data[ticker].dropna()
            
            save_path = stock_dir / f'{ticker}_trades.png'
            plot_stock_with_trades(ticker, prices, ticker_trades, save_path=save_path)
            paths.append(save_path)
            plt.close()
    
    return paths


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
    
    # Compute trade stats
    trade_stats = compute_trade_stats(results.trades)
    
    # Generate charts
    charts = {}
    charts['equity'] = plot_equity_curve(
        results, benchmark,
        save_path=output_dir / 'equity_curve.png'
    )
    charts['drawdown'] = plot_drawdown(
        results,
        save_path=output_dir / 'drawdown.png'
    )
    charts['monthly'] = plot_monthly_returns(
        results,
        save_path=output_dir / 'monthly_returns.png'
    )
    charts['sharpe'] = plot_rolling_sharpe(
        results,
        save_path=output_dir / 'rolling_sharpe.png'
    )
    charts['exposure'] = plot_positions_and_exposure(
        results,
        save_path=output_dir / 'exposure.png'
    )
    charts['turnover'] = plot_turnover(
        results,
        save_path=output_dir / 'turnover.png'
    )
    
    # NEW: Combined equity + drawdown chart
    charts['combined'] = plot_equity_and_drawdown(
        results, benchmark,
        save_path=output_dir / 'equity_drawdown_combined.png'
    )
    
    # NEW: Trading signals chart
    charts['signals'] = plot_trading_signals(
        results,
        save_path=output_dir / 'trading_signals.png'
    )
    
    # NEW: Daily P&L chart
    charts['pnl'] = plot_daily_pnl(
        results,
        save_path=output_dir / 'daily_pnl.png'
    )

    # Generate stock charts if price data is available
    stock_chart_paths = []
    if price_data is not None:
        stock_chart_paths = generate_stock_charts(
            results, price_data, output_dir, top_n=10
        )
    
    # Close all figures
    plt.close('all')
    
    # Generate HTML report
    html = _generate_html_report(metrics, output_dir, stock_chart_paths, trade_stats)
    
    report_path = output_dir / 'report.html'
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path



def _generate_html_report(
    metrics: PerformanceMetrics,
    charts_dir: Path,
    stock_charts: list[Path] = [],
    trade_stats: Optional[pd.DataFrame] = None,
) -> str:
    """Generate HTML report content."""
    
    metrics_html = generate_summary_table(metrics).to_html(index=False)
    
    trade_stats_html = ""
    if trade_stats is not None and not trade_stats.empty:
        trade_stats_html = trade_stats.to_html(index=False, border=0)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate stock charts HTML
    stock_charts_html = ""
    if stock_charts:
        stock_charts_html = "<h2>Top Traded Stocks</h2>"
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
        
        <h2>Combined Performance</h2>
        <img src="equity_drawdown_combined.png" alt="Combined Performance">

        <h2>Trading Signals</h2>
        <img src="trading_signals.png" alt="Trading Signals">

        <h2>Daily P&L</h2>
        <img src="daily_pnl.png" alt="Daily P&L">

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

        {stock_charts_html}
    </body>
    </html>
    """
    
    return html
