"""
Unit tests for the metrics module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.metrics import (
    compute_cagr,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_drawdown_series,
    compute_volatility,
    compute_turnover_stats,
    compute_avg_holding_period,
    compute_cost_ratio,
    PerformanceMetrics,
)


class TestCAGR:
    """Tests for CAGR calculation."""
    
    def test_zero_cagr(self):
        """Test CAGR with no growth."""
        equity = pd.Series(
            [1000000] * 252,
            index=pd.date_range('2024-01-01', periods=252, freq='B')
        )
        cagr = compute_cagr(equity)
        assert abs(cagr) < 0.001
    
    def test_positive_cagr(self):
        """Test CAGR with growth."""
        # 100% return over 1 year
        equity = pd.Series(
            [1000000, 2000000],
            index=pd.DatetimeIndex(['2024-01-01', '2025-01-01'])
        )
        cagr = compute_cagr(equity)
        assert abs(cagr - 1.0) < 0.01  # ~100% CAGR
    
    def test_negative_cagr(self):
        """Test CAGR with loss."""
        # 50% loss over 1 year
        equity = pd.Series(
            [1000000, 500000],
            index=pd.DatetimeIndex(['2024-01-01', '2025-01-01'])
        )
        cagr = compute_cagr(equity)
        assert cagr < 0
        assert abs(cagr - (-0.5)) < 0.01


class TestSharpe:
    """Tests for Sharpe ratio calculation."""
    
    def test_zero_sharpe(self):
        """Test Sharpe with zero mean return but vol."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 252))
        sharpe = compute_sharpe(returns)
        # Should be close to zero with zero mean
        assert abs(sharpe) < 1.0
    
    def test_very_high_sharpe(self):
        """Test Sharpe with positive consistent returns."""
        # Constant positive returns with tiny noise
        returns = pd.Series([0.0005] * 252)  # Constant returns
        sharpe = compute_sharpe(returns)
        # With std near zero, Sharpe should be very high or handled gracefully
        # The actual value depends on numerical precision
        assert sharpe >= 0  # Either 0 (handled) or very high
    
    def test_sharpe_with_realistic_data(self):
        """Test Sharpe with realistic return distribution."""
        np.random.seed(42)
        # Daily return ~10 bps with 1% std
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = compute_sharpe(returns)
        # Expected Sharpe ≈ (0.001 / 0.01) * sqrt(252) ≈ 1.58
        assert 1.0 < sharpe < 2.5


class TestSortino:
    """Tests for Sortino ratio calculation."""
    
    def test_positive_sortino(self):
        """Test Sortino with mostly positive returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sortino = compute_sortino(returns)
        assert sortino > 0
    
    def test_sortino_vs_sharpe(self):
        """Test that Sortino >= Sharpe for positive returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = compute_sharpe(returns)
        sortino = compute_sortino(returns)
        # Sortino should be >= Sharpe when using only downside vol
        assert sortino >= sharpe * 0.9  # Allow some tolerance


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""
    
    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        equity = pd.Series(
            [1000000, 1010000, 1020000, 1030000],
            index=pd.date_range('2024-01-01', periods=4, freq='B')
        )
        dd = compute_max_drawdown(equity)
        assert dd == 0.0
    
    def test_known_drawdown(self):
        """Test with known drawdown."""
        # Start at 1M, peak at 1.2M, trough at 0.9M (25% DD), end at 1M
        equity = pd.Series(
            [1000000, 1200000, 900000, 1000000],
            index=pd.date_range('2024-01-01', periods=4, freq='B')
        )
        dd = compute_max_drawdown(equity)
        assert abs(dd - (-0.25)) < 0.001
    
    def test_drawdown_series(self):
        """Test drawdown series computation."""
        equity = pd.Series(
            [100, 110, 100, 90, 95],
            index=pd.date_range('2024-01-01', periods=5, freq='B')
        )
        dd_series = compute_drawdown_series(equity)
        
        assert len(dd_series) == 5
        assert dd_series.iloc[0] == 0.0  # First point = no drawdown
        assert dd_series.iloc[1] == 0.0  # New high = no drawdown
        assert dd_series.iloc[3] < dd_series.iloc[2]  # Deeper drawdown


class TestVolatility:
    """Tests for volatility calculation."""
    
    def test_zero_vol(self):
        """Test with constant returns."""
        returns = pd.Series([0.01] * 252)
        vol = compute_volatility(returns)
        assert vol == 0.0
    
    def test_known_vol(self):
        """Test with known volatility."""
        np.random.seed(42)
        daily_vol = 0.01  # 1% daily vol
        returns = pd.Series(np.random.normal(0, daily_vol, 10000))
        vol = compute_volatility(returns)
        # Annual vol should be ~15.9% (1% * sqrt(252))
        expected = daily_vol * np.sqrt(252)
        assert abs(vol - expected) < 0.01


class TestTurnover:
    """Tests for turnover statistics."""
    
    def test_turnover_stats(self):
        """Test turnover statistics calculation."""
        # 5% daily turnover
        turnover = pd.Series([0.05] * 100)
        daily, annual = compute_turnover_stats(turnover)
        
        assert abs(daily - 0.05) < 0.001
        assert abs(annual - 0.05 * 252) < 0.1
    
    def test_holding_period(self):
        """Test holding period estimation."""
        # 10% daily turnover
        turnover = pd.Series([0.10] * 100)
        holding = compute_avg_holding_period(turnover)
        
        # Holding period ≈ 1 / (2 * 0.10) = 5 days
        assert abs(holding - 5.0) < 0.1


class TestCostRatio:
    """Tests for cost ratio calculation."""
    
    def test_cost_ratio(self):
        """Test cost ratio calculation."""
        ratio = compute_cost_ratio(total_costs=1000, gross_pnl=100000)
        assert abs(ratio - 0.01) < 0.001  # 1%
    
    def test_cost_ratio_no_pnl(self):
        """Test cost ratio with zero PnL."""
        ratio = compute_cost_ratio(total_costs=1000, gross_pnl=0)
        assert ratio == float('inf')


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = PerformanceMetrics(
            cagr=0.15,
            sharpe=1.5,
            sortino=2.0,
            max_drawdown=-0.20,
            annual_volatility=0.15,
            daily_turnover=0.05,
            annual_turnover=12.6,
            avg_positions=10,
            avg_holding_period=5.0,
            pct_days_invested=0.95,
            total_costs=10000,
            cost_ratio=0.05,
            total_return=0.30,
            num_trading_days=252,
        )
        
        d = metrics.to_dict()
        assert 'CAGR' in d
        assert 'Sharpe Ratio' in d
        assert d['CAGR'] == "15.00%"
