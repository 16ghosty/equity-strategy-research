"""
Unit tests for the backtest module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.config import StrategyConfig
from strategy.backtest import (
    Backtester,
    BacktestState,
    BacktestResults,
    DailyResult,
)


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\nMSFT\nGOOGL\n")
    
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        start_date="2024-01-01",
        end_date="2024-06-30",
        universe_size=3,
        top_k=2,
        buffer=1,
        momentum_lookback=20,
        momentum_skip=2,
        lookback_dollar_vol=10,
        regime_ma_days=50,
        slippage_bps=10,
        initial_capital=1_000_000,
        log_level="WARNING",
    )


class TestBacktestState:
    """Tests for BacktestState."""
    
    def test_copy(self):
        """Test state copying."""
        state = BacktestState(
            date=pd.Timestamp("2024-01-15"),
            positions={'AAPL': 100},
            cash=50000,
            portfolio_value=100000,
            weights={'AAPL': 0.5},
        )
        
        copy = state.copy()
        
        # Modify original
        state.positions['MSFT'] = 50
        state.cash = 0
        
        # Copy should be unchanged
        assert 'MSFT' not in copy.positions
        assert copy.cash == 50000


class TestDailyResult:
    """Tests for DailyResult dataclass."""
    
    def test_daily_result(self):
        """Test daily result creation."""
        result = DailyResult(
            date=pd.Timestamp("2024-01-15"),
            portfolio_value=1_010_000,
            daily_return=0.01,
            positions=5,
            gross_exposure=0.95,
            turnover=0.1,
            costs=500,
            cash=50000,
        )
        
        assert result.daily_return == 0.01
        assert result.positions == 5


class TestBacktestResults:
    """Tests for BacktestResults."""
    
    @pytest.fixture
    def sample_results(self, config):
        """Create sample backtest results."""
        daily_results = [
            DailyResult(
                date=pd.Timestamp("2024-01-15"),
                portfolio_value=1_000_000,
                daily_return=0.0,
                positions=0,
                gross_exposure=0,
                turnover=0,
                costs=0,
                cash=1_000_000,
            ),
            DailyResult(
                date=pd.Timestamp("2024-01-16"),
                portfolio_value=1_010_000,
                daily_return=0.01,
                positions=3,
                gross_exposure=0.95,
                turnover=0.3,
                costs=500,
                cash=50_000,
            ),
            DailyResult(
                date=pd.Timestamp("2024-01-17"),
                portfolio_value=1_005_000,
                daily_return=-0.005,
                positions=3,
                gross_exposure=0.95,
                turnover=0.05,
                costs=100,
                cash=50_000,
            ),
        ]
        
        return BacktestResults(
            daily_results=daily_results,
            trades=pd.DataFrame(),
            config=config,
            gate_failures=pd.DataFrame(),
        )
    
    def test_get_equity_curve(self, sample_results):
        """Test equity curve extraction."""
        curve = sample_results.get_equity_curve()
        
        assert len(curve) == 3
        assert curve.iloc[0] == 1_000_000
        assert curve.iloc[1] == 1_010_000
    
    def test_get_returns(self, sample_results):
        """Test returns extraction."""
        returns = sample_results.get_returns()
        
        assert len(returns) == 3
        assert returns.iloc[1] == 0.01
        assert returns.iloc[2] == -0.005
    
    def test_get_turnover(self, sample_results):
        """Test turnover extraction."""
        turnover = sample_results.get_turnover()
        
        assert len(turnover) == 3
        assert turnover.iloc[1] == 0.3


class TestBacktester:
    """Tests for the Backtester class."""
    
    def test_initialization(self, config):
        """Test backtester initialization."""
        backtester = Backtester(config)
        
        assert backtester.config == config
        assert backtester.universe_selector is not None
        assert backtester.gate_evaluator is not None
        assert backtester.rank_computer is not None
        assert backtester.portfolio_constructor is not None
        assert backtester.execution_model is not None


# Note: Integration test with real data would require DataManager setup
# Those tests are marked as slow and can be run separately
