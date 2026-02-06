"""
Unit tests for the execution module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.config import StrategyConfig
from strategy.execution import (
    ExecutionModel,
    Trade,
    ExecutionResult,
    validate_execution_timing,
)


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\nMSFT\n")
    
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        slippage_bps=10,  # 10 bps
        use_atr_slippage=False,
        execution_delay=1,
        universe_size=10,
        top_k=5,
        log_level="WARNING",
    )


class TestExecutionModel:
    """Tests for the execution model."""
    
    def test_compute_slippage(self, config):
        """Test slippage calculation."""
        model = ExecutionModel(config)
        
        # 10 bps on $100 price = $0.10
        slippage = model.compute_slippage(price=100.0, side='BUY')
        assert abs(slippage - 0.10) < 0.001
    
    def test_buy_execution_price(self, config):
        """Test that buys have higher execution price."""
        model = ExecutionModel(config)
        
        open_price = 100.0
        exec_price, slippage = model.get_execution_price(open_price, 'BUY')
        
        assert exec_price > open_price
        assert slippage > 0
    
    def test_sell_execution_price(self, config):
        """Test that sells have lower execution price."""
        model = ExecutionModel(config)
        
        open_price = 100.0
        exec_price, slippage = model.get_execution_price(open_price, 'SELL')
        
        assert exec_price < open_price
        assert slippage > 0
    
    def test_execute_rebalance_buy(self, config):
        """Test executing a buy order."""
        model = ExecutionModel(config)
        
        signal_date = pd.Timestamp("2024-01-15")
        exec_date = pd.Timestamp("2024-01-16")
        
        new_positions, result = model.execute_rebalance(
            signal_date=signal_date,
            execution_date=exec_date,
            current_positions={},  # Empty portfolio
            target_weights={'AAPL': 0.5},  # 50% in AAPL
            portfolio_value=1_000_000,
            open_prices=pd.Series({'AAPL': 150.0}),
        )
        
        # Should have one trade
        assert result.num_trades == 1
        assert result.trades[0].ticker == 'AAPL'
        assert result.trades[0].side == 'BUY'
        assert result.total_cost > 0
        
        # Should have AAPL position
        assert 'AAPL' in new_positions
        assert new_positions['AAPL'] > 0
    
    def test_execute_rebalance_sell(self, config):
        """Test executing a sell order."""
        model = ExecutionModel(config)
        
        signal_date = pd.Timestamp("2024-01-15")
        exec_date = pd.Timestamp("2024-01-16")
        
        # Start with position, target zero
        new_positions, result = model.execute_rebalance(
            signal_date=signal_date,
            execution_date=exec_date,
            current_positions={'AAPL': 1000},  # 1000 shares
            target_weights={},  # Exit
            portfolio_value=150_000,  # $150k value
            open_prices=pd.Series({'AAPL': 150.0}),
        )
        
        assert result.num_trades == 1
        assert result.trades[0].side == 'SELL'
        assert 'AAPL' not in new_positions
    
    def test_trades_ledger(self, config):
        """Test that trades are recorded in ledger."""
        model = ExecutionModel(config)
        
        # Execute multiple rebalances
        for i in range(3):
            model.execute_rebalance(
                signal_date=pd.Timestamp(f"2024-01-{10+i}"),
                execution_date=pd.Timestamp(f"2024-01-{11+i}"),
                current_positions={},
                target_weights={'AAPL': 0.5},
                portfolio_value=1_000_000,
                open_prices=pd.Series({'AAPL': 150.0}),
            )
        
        ledger = model.get_trades_ledger()
        assert len(ledger) == 3
        assert 'ticker' in ledger.columns
        assert 'side' in ledger.columns
    
    def test_atr_slippage(self, config):
        """Test ATR-based slippage."""
        config.use_atr_slippage = True
        config.atr_slippage_mult = 0.1  # 10% of ATR
        model = ExecutionModel(config)
        
        # ATR of $2 with 10% mult = $0.20 slippage
        slippage = model.compute_slippage(price=100.0, side='BUY', atr=2.0)
        assert abs(slippage - 0.20) < 0.001


class TestTrade:
    """Tests for Trade dataclass."""
    
    def test_to_dict(self):
        """Test converting trade to dict."""
        trade = Trade(
            date=pd.Timestamp("2024-01-16"),
            ticker="AAPL",
            side="BUY",
            shares=100,
            price=150.10,
            notional=15010.0,
            slippage_cost=10.0,
            signal_date=pd.Timestamp("2024-01-15"),
        )
        
        d = trade.to_dict()
        assert d['ticker'] == 'AAPL'
        assert d['shares'] == 100
        assert d['signal_date'] == pd.Timestamp("2024-01-15")


class TestExecutionTiming:
    """Critical tests for execution timing (no look-ahead)."""
    
    def test_validate_timing_correct(self):
        """Test validation passes for correct timing."""
        trading_dates = pd.DatetimeIndex([
            "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"
        ])
        signal_dates = pd.DatetimeIndex(["2024-01-15", "2024-01-16"])
        execution_dates = pd.DatetimeIndex(["2024-01-16", "2024-01-17"])
        
        is_valid, errors = validate_execution_timing(
            signal_dates, execution_dates, trading_dates, delay=1
        )
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_timing_lookahead(self):
        """Test validation catches look-ahead bias."""
        trading_dates = pd.DatetimeIndex([
            "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"
        ])
        signal_dates = pd.DatetimeIndex(["2024-01-16"])
        execution_dates = pd.DatetimeIndex(["2024-01-16"])  # Same day = look-ahead!
        
        is_valid, errors = validate_execution_timing(
            signal_dates, execution_dates, trading_dates, delay=1
        )
        
        assert not is_valid
        assert len(errors) > 0
    
    def test_t1_execution_enforced(self, config):
        """Test that execution model enforces T+1."""
        model = ExecutionModel(config)
        
        signal_date = pd.Timestamp("2024-01-15")
        exec_date = pd.Timestamp("2024-01-16")  # T+1
        
        new_positions, result = model.execute_rebalance(
            signal_date=signal_date,
            execution_date=exec_date,
            current_positions={},
            target_weights={'AAPL': 0.5},
            portfolio_value=1_000_000,
            open_prices=pd.Series({'AAPL': 150.0}),
        )
        
        # Verify the trade has correct dates
        assert result.trades[0].signal_date == signal_date
        assert result.trades[0].date == exec_date
        assert (exec_date - signal_date).days >= 1
