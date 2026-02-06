"""
Unit tests for the rank module.
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.config import StrategyConfig
from strategy.rank import (
    RankComputer,
    get_top_ranked_tickers,
    get_tickers_within_buffer,
)


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\nMSFT\nGOOGL\n")
    
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        momentum_lookback=20,  # Shorter for testing
        momentum_skip=2,
        top_k=10,
        universe_size=100,
        log_level="WARNING",
    )


@pytest.fixture
def sample_prices():
    """Create sample price data with known momentum."""
    dates = pd.date_range('2024-01-01', periods=50, freq='B')
    
    # Create prices with different momentum
    # AAPL: strong uptrend (+50% over 20 days)
    # MSFT: flat
    # GOOGL: downtrend (-20% over 20 days)
    # AMZN: moderate uptrend (+20% over 20 days)
    
    np.random.seed(42)
    
    aapl = 100 * np.exp(np.linspace(0, 0.5, 50))  # Strong up
    msft = 200 * np.ones(50)  # Flat
    googl = 150 * np.exp(np.linspace(0, -0.2, 50))  # Down
    amzn = 120 * np.exp(np.linspace(0, 0.2, 50))  # Moderate up
    
    return pd.DataFrame({
        'AAPL': aapl,
        'MSFT': msft,
        'GOOGL': googl,
        'AMZN': amzn,
    }, index=dates)


class TestRankComputer:
    """Tests for the RankComputer class."""
    
    def test_compute_momentum_scores(self, config, sample_prices):
        """Test momentum score calculation."""
        computer = RankComputer(config)
        scores = computer.compute_momentum_scores(sample_prices)
        
        assert scores.shape == sample_prices.shape
        
        # After enough history, AAPL should have highest momentum
        valid_idx = 30  # After lookback + skip
        assert scores.iloc[valid_idx]['AAPL'] > scores.iloc[valid_idx]['MSFT']
        assert scores.iloc[valid_idx]['AAPL'] > scores.iloc[valid_idx]['GOOGL']
    
    def test_compute_ranks(self, config, sample_prices):
        """Test rank calculation."""
        computer = RankComputer(config)
        scores = computer.compute_momentum_scores(sample_prices)
        ranks = computer.compute_ranks(scores)
        
        # After enough history, AAPL should have rank 1 (best momentum)
        valid_idx = 30
        assert ranks.iloc[valid_idx]['AAPL'] == 1.0
        # GOOGL should have worst rank (lowest momentum)
        assert ranks.iloc[valid_idx]['GOOGL'] == 4.0
    
    def test_ranks_only_for_universe(self, config, sample_prices):
        """Test that only universe members get ranks."""
        computer = RankComputer(config)
        scores = computer.compute_momentum_scores(sample_prices)
        
        # Create mask with only AAPL and MSFT in universe
        mask = pd.DataFrame(False, index=sample_prices.index, columns=sample_prices.columns)
        mask['AAPL'] = True
        mask['MSFT'] = True
        
        ranks = computer.compute_ranks(scores, universe_mask=mask)
        
        # Only AAPL and MSFT should have ranks
        valid_idx = 30
        assert not pd.isna(ranks.iloc[valid_idx]['AAPL'])
        assert not pd.isna(ranks.iloc[valid_idx]['MSFT'])
        assert pd.isna(ranks.iloc[valid_idx]['GOOGL'])
        assert pd.isna(ranks.iloc[valid_idx]['AMZN'])
    
    def test_no_lookahead(self, config, sample_prices):
        """Test that ranks at time t don't use future data."""
        computer = RankComputer(config)
        
        # Compute ranks using full data
        full_ranks = computer.compute_daily_ranks(sample_prices)
        
        # Compute ranks using only data up to day 30
        partial_prices = sample_prices.iloc[:31]
        partial_ranks = computer.compute_daily_ranks(partial_prices)
        
        # Ranks should be identical for days in both
        for dt in partial_ranks.index:
            if dt in full_ranks.index:
                pd.testing.assert_series_equal(
                    partial_ranks.loc[dt],
                    full_ranks.loc[dt],
                    check_names=False
                )


class TestGetTopRanked:
    """Tests for helper functions."""
    
    def test_get_top_ranked_tickers(self):
        """Test getting top K tickers."""
        ranks = pd.Series({
            'AAPL': 1.0,
            'MSFT': 2.0,
            'GOOGL': 3.0,
            'AMZN': 4.0,
        })
        
        top_2 = get_top_ranked_tickers(ranks, top_k=2)
        assert top_2 == ['AAPL', 'MSFT']
    
    def test_get_top_ranked_with_nan(self):
        """Test handling of NaN ranks."""
        ranks = pd.Series({
            'AAPL': 1.0,
            'MSFT': np.nan,
            'GOOGL': 2.0,
        })
        
        top_2 = get_top_ranked_tickers(ranks, top_k=2)
        assert len(top_2) == 2
        assert 'AAPL' in top_2
        assert 'GOOGL' in top_2
    
    def test_get_tickers_within_buffer(self):
        """Test getting tickers within buffer range."""
        ranks = pd.Series({
            'AAPL': 1.0,
            'MSFT': 5.0,  # Within buffer
            'GOOGL': 8.0,  # Outside buffer
            'AMZN': 3.0,  # In top_k
        })
        
        within = get_tickers_within_buffer(ranks, top_k=4, buffer=2)
        assert 'AAPL' in within
        assert 'AMZN' in within
        assert 'MSFT' in within  # 5 <= 4+2
        assert 'GOOGL' not in within  # 8 > 4+2
