"""
Unit tests for the portfolio module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.config import StrategyConfig
from strategy.gates import GateResult, GateResults
from strategy.portfolio import (
    PortfolioConstructor,
    PortfolioTarget,
)


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\nMSFT\nGOOGL\n")
    
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        top_k=3,
        buffer=2,
        weight_scheme="equal",
        max_weight=0.25,
        max_gross_exposure=1.0,
        universe_size=10,
        log_level="WARNING",
    )


@pytest.fixture
def sample_ranks():
    """Create sample ranks."""
    return pd.Series({
        'AAPL': 1.0,
        'MSFT': 2.0,
        'GOOGL': 3.0,
        'AMZN': 4.0,
        'TSLA': 5.0,
    })


@pytest.fixture
def sample_gate_results():
    """Create sample gate results where all pass."""
    def make_results(ticker):
        return GateResults(
            ticker=ticker,
            date=pd.Timestamp("2024-01-15"),
            results={
                'liquidity': GateResult(True, "ok"),
                'volatility': GateResult(True, "ok", scale=1.0),
                'regime': GateResult(True, "risk_on", scale=1.0),
            }
        )
    
    return {
        'AAPL': make_results('AAPL'),
        'MSFT': make_results('MSFT'),
        'GOOGL': make_results('GOOGL'),
        'AMZN': make_results('AMZN'),
        'TSLA': make_results('TSLA'),
    }


class TestPortfolioConstructor:
    """Tests for portfolio construction."""
    
    def test_equal_weight_basic(self, config, sample_ranks, sample_gate_results):
        """Test basic equal weight portfolio."""
        constructor = PortfolioConstructor(config)
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
        )
        
        # Should hold top 3 (AAPL, MSFT, GOOGL)
        assert portfolio.num_positions == 3
        assert 'AAPL' in portfolio.weights
        assert 'MSFT' in portfolio.weights
        assert 'GOOGL' in portfolio.weights
        
        # Weights are capped at max_weight (0.25), so each gets 0.25
        # Total would be 0.75 (rescaled if needed)
        assert portfolio.weights['AAPL'] <= config.max_weight + 0.001
    
    def test_buffer_logic_hold(self, config, sample_ranks, sample_gate_results):
        """Test that existing holdings are kept within buffer."""
        constructor = PortfolioConstructor(config)
        
        # AMZN is rank 4, which is within buffer (top_k=3, buffer=2, so exit > 5)
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings={'AMZN'},  # Currently held
        )
        
        # AMZN should be kept even though rank > top_k
        assert 'AMZN' in portfolio.weights
    
    def test_buffer_logic_exit(self, config, sample_ranks, sample_gate_results):
        """Test that holdings exit when outside buffer."""
        constructor = PortfolioConstructor(config)
        
        # Modify ranks so TSLA is rank 6 (outside buffer)
        ranks = sample_ranks.copy()
        ranks['TSLA'] = 6.0
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=ranks,
            gate_results=sample_gate_results,
            current_holdings={'TSLA'},  # Currently held
        )
        
        # TSLA should exit (rank 6 > top_k + buffer = 5)
        assert 'TSLA' not in portfolio.weights
    
    def test_buffer_logic_no_entry(self, config, sample_ranks, sample_gate_results):
        """Test that new positions don't enter outside top_k."""
        constructor = PortfolioConstructor(config)
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),  # No current holdings
        )
        
        # AMZN (rank 4) should NOT enter (not in current holdings, rank > top_k)
        assert 'AMZN' not in portfolio.weights
    
    def test_max_weight_constraint(self, config, sample_ranks, sample_gate_results):
        """Test that max weight constraint is applied."""
        # Config has max_weight=0.25
        config.top_k = 2  # Fewer positions = higher weight per position
        constructor = PortfolioConstructor(config)
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
        )
        
        # Each weight should be at most 0.25
        for weight in portfolio.weights.values():
            assert weight <= config.max_weight + 0.001
    
    def test_gate_failure_excludes_ticker(self, config, sample_ranks, sample_gate_results):
        """Test that failed gates exclude tickers."""
        # Make AAPL fail liquidity gate
        sample_gate_results['AAPL'] = GateResults(
            ticker='AAPL',
            date=pd.Timestamp("2024-01-15"),
            results={
                'liquidity': GateResult(False, "low_dollar_volume"),
                'volatility': GateResult(True, "ok"),
            }
        )
        
        constructor = PortfolioConstructor(config)
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
        )
        
        # AAPL should be excluded despite being rank 1
        assert 'AAPL' not in portfolio.weights
    
    def test_volatility_sizing(self, config, sample_ranks, sample_gate_results):
        """Test that volatility sizing scales weights."""
        # Make AAPL have 0.5 scale from volatility
        sample_gate_results['AAPL'] = GateResults(
            ticker='AAPL',
            date=pd.Timestamp("2024-01-15"),
            results={
                'liquidity': GateResult(True, "ok"),
                'volatility': GateResult(True, "ok", scale=0.5),
                'regime': GateResult(True, "ok", scale=1.0),
            }
        )
        
        constructor = PortfolioConstructor(config)
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
        )
        
        # AAPL weight should be scaled down
        assert portfolio.weights['AAPL'] < portfolio.weights['MSFT']
    
    def test_regime_scale(self, config, sample_ranks, sample_gate_results):
        """Test that regime scale reduces all weights."""
        constructor = PortfolioConstructor(config)
        
        # Normal portfolio
        normal = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
            regime_scale=1.0
        )
        
        # Risk-off portfolio
        risk_off = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
            regime_scale=0.5
        )
        
        # Risk-off should have lower gross exposure
        assert risk_off.gross_exposure < normal.gross_exposure
    
    def test_inverse_vol_weights(self, config, sample_ranks, sample_gate_results):
        """Test inverse volatility weighting."""
        config.weight_scheme = "inverse_vol"
        constructor = PortfolioConstructor(config)
        
        volatilities = pd.Series({
            'AAPL': 0.20,  # Low vol = higher weight
            'MSFT': 0.40,  # High vol = lower weight
            'GOOGL': 0.30,
        })
        
        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
            volatilities=volatilities,
        )
        
        # AAPL should have higher weight than MSFT (lower vol)
        assert portfolio.weights['AAPL'] > portfolio.weights['MSFT']

    def test_sector_cap_constraint(self, config, sample_ranks, sample_gate_results):
        """Sector cap should limit aggregate exposure for crowded sectors."""
        config.top_k = 4
        config.max_weight = 1.0
        config.sector_cap_enabled = True
        config.sector_cap = 0.40
        constructor = PortfolioConstructor(config)

        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
            sector_map={
                "AAPL": "TECH",
                "MSFT": "TECH",
                "GOOGL": "TECH",
                "AMZN": "CONSUMER",
            },
        )

        tech_weight = sum(
            w for t, w in portfolio.weights.items()
            if t in {"AAPL", "MSFT", "GOOGL"}
        )
        assert tech_weight <= config.sector_cap + 1e-9

    def test_beta_cap_scales_exposure(self, config, sample_ranks, sample_gate_results):
        """Beta cap should de-risk high beta portfolios."""
        config.top_k = 3
        config.max_weight = 1.0
        config.beta_cap_enabled = True
        config.beta_cap = 0.70
        constructor = PortfolioConstructor(config)

        portfolio = constructor.construct_portfolio(
            date=pd.Timestamp("2024-01-15"),
            ranks=sample_ranks,
            gate_results=sample_gate_results,
            current_holdings=set(),
            betas=pd.Series({"AAPL": 2.0, "MSFT": 2.0, "GOOGL": 2.0}),
        )

        realized_beta = sum(portfolio.weights[t] * 2.0 for t in portfolio.weights)
        assert realized_beta <= config.beta_cap + 1e-6

    def test_drawdown_scaler_reduces_weights(self, config, sample_ranks, sample_gate_results):
        """Drawdown scaler should gradually reduce exposure in deeper drawdowns."""
        config.drawdown_scaler_enabled = True
        config.drawdown_scaler_start = -0.05
        config.drawdown_scaler_full = -0.20
        config.drawdown_scaler_min = 0.30
        constructor = PortfolioConstructor(config)

        full = constructor.compute_drawdown_scale(current_drawdown=-0.02)
        partial = constructor.compute_drawdown_scale(current_drawdown=-0.10)
        minimum = constructor.compute_drawdown_scale(current_drawdown=-0.30)

        assert full == pytest.approx(1.0)
        assert config.drawdown_scaler_min < partial < 1.0
        assert minimum == pytest.approx(config.drawdown_scaler_min)


class TestPortfolioTarget:
    """Tests for PortfolioTarget dataclass."""
    
    def test_get_weight(self):
        """Test getting weight for a ticker."""
        target = PortfolioTarget(
            date=pd.Timestamp("2024-01-15"),
            weights={'AAPL': 0.5, 'MSFT': 0.5},
            gross_exposure=1.0,
            num_positions=2,
        )
        
        assert target.get_weight('AAPL') == 0.5
        assert target.get_weight('GOOGL') == 0.0  # Not held


class TestTurnover:
    """Tests for turnover calculation."""
    
    def test_compute_turnover_no_change(self, config):
        """Test turnover when no changes."""
        constructor = PortfolioConstructor(config)
        
        old = {'AAPL': 0.5, 'MSFT': 0.5}
        new = {'AAPL': 0.5, 'MSFT': 0.5}
        
        turnover = constructor.compute_turnover(old, new)
        assert turnover == 0.0
    
    def test_compute_turnover_full_change(self, config):
        """Test turnover with complete portfolio change."""
        constructor = PortfolioConstructor(config)
        
        old = {'AAPL': 0.5, 'MSFT': 0.5}
        new = {'GOOGL': 0.5, 'AMZN': 0.5}
        
        turnover = constructor.compute_turnover(old, new)
        assert turnover == 1.0  # 100% turnover
    
    def test_compute_turnover_partial_change(self, config):
        """Test turnover with partial changes."""
        constructor = PortfolioConstructor(config)
        
        old = {'AAPL': 0.5, 'MSFT': 0.5}
        new = {'AAPL': 0.5, 'GOOGL': 0.5}  # MSFT out, GOOGL in
        
        turnover = constructor.compute_turnover(old, new)
        assert turnover == 0.5  # 50% turnover
