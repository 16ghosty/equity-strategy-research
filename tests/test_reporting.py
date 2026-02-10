"""Unit tests for reporting charts."""

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.backtest import BacktestResults, DailyResult
from strategy.config import StrategyConfig
from strategy.reporting import (
    compute_benchmark_diagnostics,
    compute_drawdown_detractor_metrics,
    compute_ticker_health_table,
    plot_benchmark_gap_decomposition,
    plot_negative_tail_distribution,
    plot_returns_distribution,
)


def _sample_results(tmp_path) -> BacktestResults:
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\n")
    config = StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        start_date="2024-01-01",
        end_date="2024-01-31",
        universe_size=1,
        top_k=1,
        log_level="WARNING",
    )

    dates = pd.bdate_range("2024-01-02", periods=8)
    portfolio_vals = [1000, 1020, 1005, 1010, 980, 995, 1002, 990]

    daily = []
    prev = None
    for dt, pv in zip(dates, portfolio_vals):
        ret = 0.0 if prev is None else pv / prev - 1.0
        daily.append(
            DailyResult(
                date=dt,
                portfolio_value=float(pv),
                daily_return=float(ret),
                positions=1,
                gross_exposure=1.0,
                turnover=0.1,
                costs=0.0,
                cash=0.0,
            )
        )
        prev = pv

    return BacktestResults(
        daily_results=daily,
        trades=pd.DataFrame(
            [
                {
                    "date": dates[0],
                    "ticker": "AAPL",
                    "side": "BUY",
                    "shares": 10.0,
                    "price": 100.0,
                    "notional": 1000.0,
                    "slippage_cost": 0.0,
                    "signal_date": dates[0],
                },
                {
                    "date": dates[-1],
                    "ticker": "AAPL",
                    "side": "SELL",
                    "shares": 10.0,
                    "price": 105.0,
                    "notional": 1050.0,
                    "slippage_cost": 0.0,
                    "signal_date": dates[-2],
                },
            ]
        ),
        config=config,
        gate_failures=pd.DataFrame(),
        weights_history=pd.DataFrame({"AAPL": [1.0] * len(dates)}, index=dates),
    )


def test_returns_distribution_labels(tmp_path):
    results = _sample_results(tmp_path)
    fig = plot_returns_distribution(results)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Daily Return (%)"
    assert ax.get_ylabel() == "Frequency"


def test_negative_tail_distribution_labels(tmp_path):
    results = _sample_results(tmp_path)
    fig = plot_negative_tail_distribution(results, percentile=5.0)
    ax = fig.axes[0]
    # Either chart labels exist or no-data placeholder disabled axis.
    if ax.axison:
        assert ax.get_xlabel() == "Daily Return (%)"
        assert ax.get_ylabel() == "Frequency"


def test_ticker_health_table(tmp_path):
    results = _sample_results(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {"AAPL": [100, 101, 99, 102, 100, 103, 104, 105]},
        index=dates,
    )
    table = compute_ticker_health_table(results, prices)
    assert not table.empty
    assert "Ticker" in table.columns
    assert "TotalContribution" in table.columns
    assert "RealizedPnL" in table.columns


def test_drawdown_detractor_metrics_table(tmp_path):
    results = _sample_results(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=8)
    prices = pd.DataFrame(
        {"AAPL": [100, 99, 98, 95, 96, 94, 93, 92]},
        index=dates,
    )
    table = compute_drawdown_detractor_metrics(results, prices, lookback_days=8, top_n=5)
    assert not table.empty
    assert "Ticker" in table.columns
    assert "Contribution" in table.columns
    assert "RealizedPnLWindow" in table.columns


def test_benchmark_diagnostics_table_and_chart(tmp_path):
    results = _sample_results(tmp_path)
    dates = pd.bdate_range("2024-01-02", periods=8)
    benchmark = pd.Series(
        [100, 101, 102, 101, 103, 104, 105, 106],
        index=dates,
        name="SPY",
    )

    table = compute_benchmark_diagnostics(results, benchmark)
    assert not table.empty
    assert "Metric" in table.columns
    assert "Value" in table.columns
    assert (table["Metric"] == "Beta to Benchmark").any()
    assert (table["Metric"] == "Active CAGR Gap").any()

    fig = plot_benchmark_gap_decomposition(results, benchmark)
    ax1 = fig.axes[0]
    assert ax1.get_ylabel() == "Contribution (percentage points)"
