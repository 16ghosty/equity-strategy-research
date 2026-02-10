"""Unit tests for tail-risk validation helpers."""

import pandas as pd
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.backtest import BacktestResults, DailyResult
from strategy.config import StrategyConfig
from strategy.validation import (
    _combined_stress_config,
    _slice_results,
    block_bootstrap_drawdown_distribution,
    compute_crisis_window_scorecards,
)


@pytest.fixture
def config(tmp_path):
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\n")
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        start_date="2024-01-01",
        end_date="2024-02-01",
        universe_size=1,
        top_k=1,
        log_level="WARNING",
    )


@pytest.fixture
def sample_results(config):
    dates = pd.bdate_range("2024-01-02", periods=15)
    returns = np.array([0.01, -0.02, 0.015, -0.03, 0.005, -0.01, 0.02, -0.015, 0.01, -0.005, 0.008, -0.012, 0.004, -0.02, 0.01])

    equity = [1000.0]
    for r in returns[1:]:
        equity.append(equity[-1] * (1.0 + r))

    daily_results = []
    prev = None
    for dt, pv in zip(dates, equity):
        ret = 0.0 if prev is None else pv / prev - 1.0
        daily_results.append(
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
        daily_results=daily_results,
        trades=pd.DataFrame(),
        config=config,
        gate_failures=pd.DataFrame(),
        weights_history=pd.DataFrame(index=dates),
    )


def test_slice_results_preserves_window(sample_results):
    sliced = _slice_results(sample_results, pd.Timestamp("2024-01-08"), pd.Timestamp("2024-01-19"))
    assert sliced is not None
    assert sliced.daily_results[0].date >= pd.Timestamp("2024-01-08")
    assert sliced.daily_results[-1].date <= pd.Timestamp("2024-01-19")


def test_crisis_scorecards_for_custom_window(sample_results):
    windows = (("custom", "2024-01-02", "2024-01-31"),)
    scorecards = compute_crisis_window_scorecards(sample_results, windows=windows)
    assert not scorecards.empty
    assert "window" in scorecards.columns
    assert scorecards.iloc[0]["window"] == "custom"


def test_block_bootstrap_drawdown_distribution(sample_results):
    returns = sample_results.get_returns()
    distribution = block_bootstrap_drawdown_distribution(
        returns,
        n_samples=100,
        block_size=5,
        seed=123,
    )
    assert len(distribution) == 100
    assert (distribution <= 0).all()


def test_combined_stress_config(config):
    stressed = _combined_stress_config(config)
    assert stressed.slippage_bps == pytest.approx(config.slippage_bps * 2.0)
    assert stressed.bad_fills_enabled is True
    assert stressed.bad_fills_multiplier >= 2.5
