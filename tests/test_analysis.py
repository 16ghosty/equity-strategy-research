"""
Unit tests for stress/sensitivity analysis helpers.
"""

import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.analysis import _apply_remove_top_winners, _roundtrip_pnl
from strategy.backtest import BacktestResults, DailyResult
from strategy.config import StrategyConfig


@pytest.fixture
def config(tmp_path):
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\n")
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        start_date="2024-01-01",
        end_date="2024-01-31",
        universe_size=1,
        top_k=1,
        log_level="WARNING",
    )


@pytest.fixture
def sample_results(config):
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    equity = [1000.0, 1100.0, 1080.0, 1030.0]
    daily_results = []
    prev = None
    for dt, pv in zip(dates, equity):
        ret = 0.0 if prev is None else pv / prev - 1.0
        daily_results.append(
            DailyResult(
                date=dt,
                portfolio_value=pv,
                daily_return=ret,
                positions=1,
                gross_exposure=1.0,
                turnover=0.1,
                costs=0.0,
                cash=0.0,
            )
        )
        prev = pv

    trades = pd.DataFrame(
        [
            {"date": dates[0], "ticker": "AAPL", "side": "BUY", "shares": 10.0, "price": 100.0, "notional": 1000.0, "slippage_cost": 0.0, "signal_date": dates[0]},
            {"date": dates[1], "ticker": "AAPL", "side": "SELL", "shares": 10.0, "price": 110.0, "notional": 1100.0, "slippage_cost": 0.0, "signal_date": dates[0]},
            {"date": dates[2], "ticker": "AAPL", "side": "BUY", "shares": 5.0, "price": 100.0, "notional": 500.0, "slippage_cost": 0.0, "signal_date": dates[2]},
            {"date": dates[3], "ticker": "AAPL", "side": "SELL", "shares": 5.0, "price": 90.0, "notional": 450.0, "slippage_cost": 0.0, "signal_date": dates[2]},
        ]
    )

    return BacktestResults(
        daily_results=daily_results,
        trades=trades,
        config=config,
        gate_failures=pd.DataFrame(),
        weights_history=pd.DataFrame({"AAPL": [1.0, 1.0, 1.0, 1.0]}, index=dates),
    )


def test_roundtrip_pnl_matches_fifo(sample_results):
    matched = _roundtrip_pnl(sample_results.trades)
    assert len(matched) == 2
    assert matched["pnl"].sum() == pytest.approx(50.0)
    assert matched["pnl"].max() == pytest.approx(100.0)


def test_remove_top_winners_reduces_equity(sample_results):
    adjusted_results, removed = _apply_remove_top_winners(sample_results, top_n=1)
    assert len(removed) == 1
    original = sample_results.get_equity_curve()
    adjusted = adjusted_results.get_equity_curve()
    assert adjusted.iloc[-1] == pytest.approx(original.iloc[-1] - removed["pnl"].iloc[0])
