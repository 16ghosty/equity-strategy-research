"""
Unit tests for the gates module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.gates import (
    GateResult,
    GateResults,
    liquidity_gate,
    volatility_gate,
    market_regime_gate,
    buffer_gate,
    GateEvaluator,
)


class TestLiquidityGate:
    """Tests for the liquidity gate."""
    
    def test_passes_when_above_threshold(self):
        """Test that gate passes when above threshold."""
        result = liquidity_gate(
            avg_dollar_volume=2_000_000,
            price=50.0,
            threshold=1_000_000,
            min_price=5.0
        )
        assert result.passed
        assert result.reason == "ok"
    
    def test_fails_when_below_threshold(self):
        """Test that gate fails when below threshold."""
        result = liquidity_gate(
            avg_dollar_volume=500_000,
            price=50.0,
            threshold=1_000_000,
            min_price=5.0
        )
        assert not result.passed
        assert result.reason == "low_dollar_volume"
    
    def test_fails_when_price_too_low(self):
        """Test that gate fails when price below minimum."""
        result = liquidity_gate(
            avg_dollar_volume=2_000_000,
            price=3.0,
            threshold=1_000_000,
            min_price=5.0
        )
        assert not result.passed
        assert "price_below" in result.reason
    
    def test_fails_when_no_data(self):
        """Test that gate fails gracefully with NaN data."""
        result = liquidity_gate(
            avg_dollar_volume=np.nan,
            price=50.0,
            threshold=1_000_000,
            min_price=5.0
        )
        assert not result.passed
        assert result.reason == "no_data"
    
    def test_exactly_at_threshold(self):
        """Test behavior exactly at threshold."""
        result = liquidity_gate(
            avg_dollar_volume=1_000_000,  # Exactly at threshold
            price=5.0,  # Exactly at min
            threshold=1_000_000,
            min_price=5.0
        )
        assert result.passed


class TestVolatilityGate:
    """Tests for the volatility gate."""
    
    def test_passes_below_cap(self):
        """Test that gate passes when vol below cap."""
        result = volatility_gate(
            volatility=0.30,
            vol_cap=0.60,
            use_sizing=False
        )
        assert result.passed
        assert result.reason == "ok"
    
    def test_fails_above_cap(self):
        """Test that gate fails when vol above cap."""
        result = volatility_gate(
            volatility=0.70,
            vol_cap=0.60,
            use_sizing=False
        )
        assert not result.passed
        assert "vol_exceeds" in result.reason
    
    def test_sizing_scale_calculation(self):
        """Test that sizing scale is calculated correctly."""
        result = volatility_gate(
            volatility=0.40,  # 40% vol
            vol_cap=0.60,
            use_sizing=True,
            target_vol=0.20  # 20% target
        )
        assert result.passed
        assert result.scale == 0.5  # 20% / 40% = 0.5
    
    def test_sizing_scale_capped_at_one(self):
        """Test that sizing scale is capped at 1.0."""
        result = volatility_gate(
            volatility=0.10,  # 10% vol (below target)
            vol_cap=0.60,
            use_sizing=True,
            target_vol=0.20  # 20% target
        )
        assert result.passed
        assert result.scale == 1.0  # Capped at 1.0
    
    def test_no_data(self):
        """Test handling of missing data."""
        result = volatility_gate(
            volatility=np.nan,
            vol_cap=0.60
        )
        assert not result.passed
        assert result.reason == "no_vol_data"


class TestMarketRegimeGate:
    """Tests for the market regime gate."""
    
    def test_risk_on_above_ma_low_vol(self):
        """Test risk-on when price above MA and low vol."""
        result = market_regime_gate(
            benchmark_price=450.0,
            benchmark_ma=400.0,
            benchmark_vol=0.15,
            vol_threshold=0.25
        )
        assert result.passed
        assert result.reason == "risk_on"
        assert result.scale == 1.0
    
    def test_risk_off_below_ma(self):
        """Test risk-off when price below MA."""
        result = market_regime_gate(
            benchmark_price=380.0,
            benchmark_ma=400.0,
            benchmark_vol=0.15,
            vol_threshold=0.25,
            reduce_exposure=0.5
        )
        assert result.passed  # Still passes but with reduced exposure
        assert "risk_off_trend" in result.reason
        assert result.scale == 0.5
    
    def test_risk_off_high_vol(self):
        """Test risk-off when vol above threshold."""
        result = market_regime_gate(
            benchmark_price=450.0,
            benchmark_ma=400.0,
            benchmark_vol=0.35,
            vol_threshold=0.25,
            reduce_exposure=0.5
        )
        assert result.passed
        assert "risk_off_vol" in result.reason
        assert result.scale == 0.5
    
    def test_risk_off_both_conditions(self):
        """Test risk-off when both conditions triggered."""
        result = market_regime_gate(
            benchmark_price=380.0,  # Below MA
            benchmark_ma=400.0,
            benchmark_vol=0.35,  # High vol
            vol_threshold=0.25,
            reduce_exposure=0.5
        )
        assert result.passed
        assert "risk_off_trend_and_vol" in result.reason
        assert result.scale == 0.5
    
    def test_missing_data_passes(self):
        """Test that missing data allows gate to pass."""
        result = market_regime_gate(
            benchmark_price=np.nan,
            benchmark_ma=400.0,
            benchmark_vol=0.15
        )
        assert result.passed
        assert "no_regime_data" in result.reason


class TestBufferGate:
    """Tests for the buffer/turnover gate."""
    
    def test_enter_when_top_k(self):
        """Test that new positions enter when rank <= top_k."""
        result = buffer_gate(
            current_rank=10,
            top_k=20,
            buffer=5,
            is_currently_held=False
        )
        assert result.passed
        assert result.reason == "enter_top_k"
    
    def test_no_enter_when_outside_top_k(self):
        """Test that new positions don't enter when rank > top_k."""
        result = buffer_gate(
            current_rank=25,
            top_k=20,
            buffer=5,
            is_currently_held=False
        )
        assert not result.passed
        assert "no_entry" in result.reason
    
    def test_hold_when_within_buffer(self):
        """Test that existing positions hold when within buffer."""
        result = buffer_gate(
            current_rank=23,  # Between top_k (20) and top_k + buffer (25)
            top_k=20,
            buffer=5,
            is_currently_held=True
        )
        assert result.passed
        assert result.reason == "hold_within_buffer"
    
    def test_exit_when_outside_buffer(self):
        """Test that existing positions exit when outside buffer."""
        result = buffer_gate(
            current_rank=30,  # Beyond top_k + buffer (25)
            top_k=20,
            buffer=5,
            is_currently_held=True
        )
        assert not result.passed
        assert "exit" in result.reason
    
    def test_exactly_at_exit_threshold(self):
        """Test behavior exactly at exit threshold."""
        result = buffer_gate(
            current_rank=25,  # Exactly at top_k + buffer
            top_k=20,
            buffer=5,
            is_currently_held=True
        )
        assert result.passed  # Should still hold at threshold
    
    def test_no_rank_fails(self):
        """Test that missing rank fails."""
        result = buffer_gate(
            current_rank=np.nan,
            top_k=20,
            buffer=5,
            is_currently_held=False
        )
        assert not result.passed


class TestGateResults:
    """Tests for aggregated gate results."""
    
    def test_final_passed_all_pass(self):
        """Test final_passed when all gates pass."""
        results = GateResults(
            ticker="AAPL",
            date=pd.Timestamp("2024-01-15"),
            results={
                "liquidity": GateResult(True, "ok"),
                "volatility": GateResult(True, "ok"),
            }
        )
        assert results.final_passed
    
    def test_final_passed_one_fails(self):
        """Test final_passed when one gate fails."""
        results = GateResults(
            ticker="AAPL",
            date=pd.Timestamp("2024-01-15"),
            results={
                "liquidity": GateResult(True, "ok"),
                "volatility": GateResult(False, "vol_exceeds_60%"),
            }
        )
        assert not results.final_passed
    
    def test_final_scale(self):
        """Test combined scale calculation."""
        results = GateResults(
            ticker="AAPL",
            date=pd.Timestamp("2024-01-15"),
            results={
                "volatility": GateResult(True, "ok", scale=0.5),
                "regime": GateResult(True, "risk_off", scale=0.5),
            }
        )
        assert results.final_scale == 0.25  # 0.5 * 0.5
    
    def test_fail_reasons(self):
        """Test failure reason collection."""
        results = GateResults(
            ticker="AAPL",
            date=pd.Timestamp("2024-01-15"),
            results={
                "liquidity": GateResult(False, "low_dollar_volume"),
                "volatility": GateResult(True, "ok"),
            }
        )
        assert "liquidity:low_dollar_volume" in results.fail_reasons


class TestGateEvaluator:
    """Tests for the gate evaluator framework."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a test evaluator."""
        return GateEvaluator(
            liquidity_threshold=1_000_000,
            min_price=5.0,
            vol_cap=0.60,
            use_vol_sizing=True,
            target_vol=0.20,
            top_k=20,
            buffer=5,
        )
    
    def test_evaluate_ticker(self, evaluator):
        """Test evaluating a single ticker."""
        result = evaluator.evaluate_ticker(
            ticker="AAPL",
            date=pd.Timestamp("2024-01-15"),
            avg_dollar_volume=5_000_000,
            price=150.0,
            volatility=0.25,
            rank=5,
            is_currently_held=False,
            benchmark_price=450.0,
            benchmark_ma=400.0,
            benchmark_vol=0.15,
        )
        
        assert result.ticker == "AAPL"
        assert "liquidity" in result.results
        assert "volatility" in result.results
        assert "regime" in result.results
        assert "buffer" in result.results
    
    def test_evaluate_universe(self, evaluator):
        """Test evaluating all tickers in universe."""
        universe = ["AAPL", "MSFT", "GOOGL"]
        
        results = evaluator.evaluate_universe(
            date=pd.Timestamp("2024-01-15"),
            universe=universe,
            avg_dollar_volumes=pd.Series({
                "AAPL": 5_000_000,
                "MSFT": 4_000_000,
                "GOOGL": 3_000_000,
            }),
            prices=pd.Series({
                "AAPL": 150.0,
                "MSFT": 350.0,
                "GOOGL": 140.0,
            }),
            volatilities=pd.Series({
                "AAPL": 0.25,
                "MSFT": 0.22,
                "GOOGL": 0.28,
            }),
            ranks=pd.Series({
                "AAPL": 1,
                "MSFT": 2,
                "GOOGL": 3,
            }),
            current_holdings=set(),
            benchmark_price=450.0,
            benchmark_ma=400.0,
            benchmark_vol=0.15,
        )
        
        assert len(results) == 3
        assert all(ticker in results for ticker in universe)
    
    def test_failure_tracking(self, evaluator):
        """Test that failures are tracked."""
        evaluator.evaluate_ticker(
            ticker="AAPL",
            date=pd.Timestamp("2024-01-15"),
            avg_dollar_volume=500_000,  # Below threshold
            price=150.0,
            volatility=0.25,
            rank=5,
            is_currently_held=False,
            benchmark_price=450.0,
            benchmark_ma=400.0,
            benchmark_vol=0.15,
        )
        
        summary = evaluator.get_failure_summary()
        assert not summary.empty
        assert "liquidity:low_dollar_volume" in summary.index
    
    def test_reset_failure_counts(self, evaluator):
        """Test resetting failure counts."""
        evaluator.failure_counts = {"test": 10}
        evaluator.reset_failure_counts()
        assert len(evaluator.failure_counts) == 0
