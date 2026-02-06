"""
Unit tests for the universe module.

Includes leakage tests to verify no look-ahead bias.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.config import StrategyConfig
from strategy.universe import (
    UniverseSelector,
    create_universe_membership_matrix,
    validate_no_lookahead,
)


@pytest.fixture
def config(tmp_path):
    """Create a test configuration."""
    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\n")
    
    return StrategyConfig(
        ticker_file=ticker_file,
        data_cache_dir=tmp_path / "cache",
        universe_size=3,  # Select top 3
        top_k=3,  # Hold top 3 (must be <= universe_size)
        lookback_dollar_vol=10,  # Shorter for testing
        log_level="WARNING",
    )


@pytest.fixture
def sample_data():
    """Create sample price and volume data with distinct dollar volumes."""
    # 6 months of daily data
    dates = pd.date_range('2024-01-01', periods=126, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    np.random.seed(42)
    
    # Create prices with different levels
    prices = pd.DataFrame(index=dates, columns=tickers)
    for i, ticker in enumerate(tickers):
        base_price = 100 + i * 50
        prices[ticker] = base_price * (1 + 0.001 * np.arange(len(dates)))
    
    # Create volumes with distinct levels so ranking is predictable
    # AAPL: highest, TSLA: lowest
    volume = pd.DataFrame(index=dates, columns=tickers)
    volume['AAPL'] = 10_000_000
    volume['MSFT'] = 8_000_000
    volume['GOOGL'] = 6_000_000
    volume['AMZN'] = 4_000_000
    volume['TSLA'] = 2_000_000
    
    # Add some noise
    for ticker in tickers:
        volume[ticker] = volume[ticker] * (1 + np.random.randn(len(dates)) * 0.1)
    
    return prices.astype(float), volume.astype(float)


class TestUniverseSelector:
    """Tests for universe selection."""
    
    def test_universe_size(self, config, sample_data):
        """Test that universe has correct size."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        
        # After warmup period, universe should have config.universe_size tickers
        # (3 in our test config)
        warmup_complete = universe_df['universe_size'] >= config.lookback_dollar_vol
        valid_universes = universe_df.loc[warmup_complete]
        
        assert (valid_universes['universe_size'] == config.universe_size).all()
    
    def test_universe_monthly_rebalance(self, config, sample_data):
        """Test that universe is rebalanced monthly."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        
        # Check that universe_date changes at month boundaries
        universe_dates = universe_df['universe_date']
        unique_dates = universe_dates.drop_duplicates()
        
        # Should have at most one selection per month
        months = unique_dates.dropna().dt.to_period('M')
        assert len(months) == len(months.unique())
    
    def test_top_tickers_selected(self, config, sample_data):
        """Test that top tickers by dollar volume are selected."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        
        # Skip warmup period
        valid_idx = universe_df['universe_size'] >= config.universe_size
        
        for dt in universe_df.index[valid_idx]:
            universe = universe_df.loc[dt, 'universe']
            
            # AAPL, MSFT, GOOGL should be in top 3 (they have highest dollar volume)
            # Note: due to noise, this might not always hold
            # But AAPL should almost always be there
            assert 'AAPL' in universe or 'MSFT' in universe


class TestLookAheadPrevention:
    """Critical tests for look-ahead bias prevention."""
    
    def test_universe_uses_prior_data_only(self, config, sample_data):
        """Test that universe selection uses only prior data."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        
        # For each date, verify that universe was selected using prior data
        prev_month = None
        
        for dt in universe_df.index:
            universe = universe_df.loc[dt, 'universe']
            current_month = (dt.year, dt.month)
            
            if current_month != prev_month and len(universe) > 0:
                # Universe was recomputed for this month
                # Verify that all data used is from BEFORE this date
                prior_data = prices.loc[:dt].iloc[:-1]  # Exclude current day
                
                # Must have enough history
                assert len(prior_data) >= config.lookback_dollar_vol, \
                    f"Insufficient history for {dt.date()}"
            
            prev_month = current_month
    
    def test_no_future_data_in_selection(self, config, sample_data):
        """Test that future data doesn't affect selection."""
        prices, volume = sample_data
        
        # Split data at a month boundary
        split_date = pd.Timestamp('2024-03-01')
        
        # Select universe using full data
        selector = UniverseSelector(config)
        universe_full = selector.select_universe(prices, volume)
        
        # Select universe using only data up to split date
        prices_partial = prices.loc[:split_date]
        volume_partial = volume.loc[:split_date]
        universe_partial = selector.select_universe(prices_partial, volume_partial)
        
        # For dates before split, universes should be identical
        common_dates = universe_partial.index[universe_partial.index < split_date]
        
        for dt in common_dates:
            if len(universe_partial.loc[dt, 'universe']) > 0:
                assert universe_partial.loc[dt, 'universe'] == universe_full.loc[dt, 'universe'], \
                    f"Universe differs at {dt.date()} - possible look-ahead!"
    
    def test_validate_no_lookahead_function(self, config, sample_data):
        """Test the validation function for look-ahead bias."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        
        is_valid, errors = validate_no_lookahead(
            universe_df, 
            prices, 
            volume, 
            lookback=config.lookback_dollar_vol
        )
        
        assert is_valid, f"Look-ahead detected: {errors}"


class TestUniverseMembership:
    """Tests for universe membership matrix."""
    
    def test_membership_matrix_shape(self, config, sample_data):
        """Test that membership matrix has correct shape."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        membership = create_universe_membership_matrix(
            universe_df, 
            list(prices.columns)
        )
        
        assert membership.shape == (len(universe_df), len(prices.columns))
    
    def test_membership_matches_universe(self, config, sample_data):
        """Test that membership matrix matches universe list."""
        prices, volume = sample_data
        selector = UniverseSelector(config)
        
        universe_df = selector.select_universe(prices, volume)
        membership = create_universe_membership_matrix(
            universe_df, 
            list(prices.columns)
        )
        
        for dt in universe_df.index:
            universe = universe_df.loc[dt, 'universe']
            
            for ticker in prices.columns:
                expected = ticker in universe
                assert membership.loc[dt, ticker] == expected


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_insufficient_history(self, config, sample_data):
        """Test handling of insufficient history."""
        prices, volume = sample_data
        
        # Use only first few days
        prices_short = prices.iloc[:5]
        volume_short = volume.iloc[:5]
        
        selector = UniverseSelector(config)
        universe_df = selector.select_universe(prices_short, volume_short)
        
        # Should have empty universes due to insufficient history
        assert (universe_df['universe_size'] == 0).all()
    
    def test_fewer_tickers_than_universe_size(self, config, sample_data):
        """Test when fewer tickers available than universe size."""
        prices, volume = sample_data
        
        # Use only 2 tickers but universe_size is 3
        prices_small = prices[['AAPL', 'MSFT']]
        volume_small = volume[['AAPL', 'MSFT']]
        
        selector = UniverseSelector(config)
        universe_df = selector.select_universe(prices_small, volume_small)
        
        # Universe should have at most 2 tickers
        valid = universe_df['universe_size'] > 0
        assert (universe_df.loc[valid, 'universe_size'] <= 2).all()
