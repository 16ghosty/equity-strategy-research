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
from strategy.portfolio import PortfolioTarget


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

    def test_get_weights_history_default(self, sample_results):
        """Weights history should default to empty DataFrame."""
        weights = sample_results.get_weights_history()
        assert isinstance(weights, pd.DataFrame)
        assert weights.empty


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

    def test_should_rebalance_daily(self, config):
        """Daily rebalance should always signal."""
        config.rebalance_frequency = "daily"
        backtester = Backtester(config)
        assert backtester._should_rebalance(pd.Timestamp("2024-01-15"))
        assert backtester._should_rebalance(pd.Timestamp("2024-01-16"))

    def test_should_rebalance_weekly(self, config):
        """Weekly rebalance should only signal on configured weekday."""
        config.rebalance_frequency = "weekly"
        config.rebalance_weekday = 0  # Monday
        backtester = Backtester(config)
        assert backtester._should_rebalance(pd.Timestamp("2024-01-15"))  # Monday
        assert not backtester._should_rebalance(pd.Timestamp("2024-01-16"))  # Tuesday

    def test_should_rebalance_custom_weekdays(self, config):
        """Custom rebalance should signal only on configured weekday set."""
        config.rebalance_frequency = "custom"
        config.rebalance_weekdays = (0, 2, 4)  # Mon/Wed/Fri
        backtester = Backtester(config)
        assert backtester._should_rebalance(pd.Timestamp("2024-01-15"))  # Monday
        assert not backtester._should_rebalance(pd.Timestamp("2024-01-16"))  # Tuesday
        assert backtester._should_rebalance(pd.Timestamp("2024-01-17"))  # Wednesday

    def test_cash_sweep_applies_benchmark_return(self, config):
        """Positive cash should earn benchmark return when cash sweep is enabled."""
        config.cash_sweep_to_benchmark = True
        backtester = Backtester(config)
        benchmark_returns = pd.Series(
            [0.01],
            index=[pd.Timestamp("2024-01-16")],
            dtype=float,
        )

        swept_cash = backtester._apply_cash_sweep(
            cash=1_000_000.0,
            date=pd.Timestamp("2024-01-16"),
            sweep_returns=benchmark_returns,
            risk_off_regime=False,
        )

        assert swept_cash == pytest.approx(1_010_000.0)

    def test_cash_sweep_disabled_leaves_cash_unchanged(self, config):
        """Cash sweep off should keep cash unchanged."""
        config.cash_sweep_to_benchmark = False
        backtester = Backtester(config)
        benchmark_returns = pd.Series(
            [0.02],
            index=[pd.Timestamp("2024-01-16")],
            dtype=float,
        )

        swept_cash = backtester._apply_cash_sweep(
            cash=500_000.0,
            date=pd.Timestamp("2024-01-16"),
            sweep_returns=benchmark_returns,
            risk_off_regime=False,
        )

        assert swept_cash == pytest.approx(500_000.0)

    def test_cash_sweep_only_applies_to_positive_cash(self, config):
        """Negative cash should not be swept through benchmark return."""
        config.cash_sweep_to_benchmark = True
        backtester = Backtester(config)
        benchmark_returns = pd.Series(
            [0.03],
            index=[pd.Timestamp("2024-01-16")],
            dtype=float,
        )

        swept_cash = backtester._apply_cash_sweep(
            cash=-10_000.0,
            date=pd.Timestamp("2024-01-16"),
            sweep_returns=benchmark_returns,
            risk_off_regime=False,
        )

        assert swept_cash == pytest.approx(-10_000.0)

    def test_cash_sweep_risk_off_holds_cash(self, config):
        """Risk-off regime should keep cash flat when configured."""
        config.cash_sweep_to_benchmark = True
        config.cash_sweep_risk_off_to_cash = True
        backtester = Backtester(config)
        benchmark_returns = pd.Series(
            [0.05],
            index=[pd.Timestamp("2024-01-16")],
            dtype=float,
        )

        swept_cash = backtester._apply_cash_sweep(
            cash=100_000.0,
            date=pd.Timestamp("2024-01-16"),
            sweep_returns=benchmark_returns,
            risk_off_regime=True,
        )

        assert swept_cash == pytest.approx(100_000.0)

    def test_no_trade_band_keeps_small_weight_changes(self, config):
        """Small target deltas should be suppressed by no-trade band."""
        config.min_trade_weight_change = 0.01
        backtester = Backtester(config)
        target = PortfolioTarget(
            date=pd.Timestamp("2024-01-15"),
            weights={"AAPL": 0.105, "MSFT": 0.191},
            gross_exposure=0.296,
            num_positions=2,
        )
        current = {"AAPL": 0.10, "MSFT": 0.20}

        adjusted = backtester._apply_no_trade_band(current_weights=current, target=target)

        assert adjusted.weights["AAPL"] == pytest.approx(0.10)
        assert adjusted.weights["MSFT"] == pytest.approx(0.20)

    def test_beta_targeting_scales_stock_sleeve(self, config):
        """Beta targeting should reduce stock sleeve when beta target is lower."""
        config.beta_targeting_enabled = True
        config.beta_target_risk_on = 0.8
        config.beta_target_neutral = 0.8
        config.beta_target_risk_off = 0.8
        config.beta_target_hysteresis = 0.0
        config.beta_target_step_limit = 1.0
        backtester = Backtester(config)

        target = PortfolioTarget(
            date=pd.Timestamp("2024-01-15"),
            weights={"AAPL": 0.30, "MSFT": 0.30},
            gross_exposure=0.60,
            num_positions=2,
        )
        betas = pd.Series({"AAPL": 1.5, "MSFT": 1.5}, dtype=float)

        adjusted = backtester._apply_beta_targeting(
            target=target,
            betas=betas,
            benchmark_price=100.0,
            benchmark_ma=90.0,
            benchmark_vol=0.10,
        )

        assert sum(adjusted.weights.values()) < sum(target.weights.values())


# Note: Integration test with real data would require DataManager setup
# Those tests are marked as slow and can be run separately
