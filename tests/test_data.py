"""
Unit tests for the data module.

Tests data download, caching, and integrity.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy.config import StrategyConfig
from strategy.data import (
    DataManager,
    YFinanceProvider,
    load_tickers,
)


class TestLoadTickers:
    """Tests for ticker file loading."""
    
    def test_load_valid_tickers(self, tmp_path):
        """Test loading tickers from a valid file."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\nMSFT\nGOOGL\n")
        
        tickers = load_tickers(ticker_file)
        
        assert tickers == ["AAPL", "MSFT", "GOOGL"]
    
    def test_load_tickers_with_comments(self, tmp_path):
        """Test that comments are ignored."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL  # Apple Inc\n# This is a comment\nMSFT\n")
        
        tickers = load_tickers(ticker_file)
        
        assert tickers == ["AAPL", "MSFT"]
    
    def test_load_tickers_handles_whitespace(self, tmp_path):
        """Test that whitespace is stripped."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("  AAPL  \n\nMSFT\n  \n")
        
        tickers = load_tickers(ticker_file)
        
        assert tickers == ["AAPL", "MSFT"]
    
    def test_load_tickers_uppercase(self, tmp_path):
        """Test that tickers are uppercased."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("aapl\nMsft\ngoogl\n")
        
        tickers = load_tickers(ticker_file)
        
        assert tickers == ["AAPL", "MSFT", "GOOGL"]
    
    def test_load_tickers_removes_duplicates(self, tmp_path):
        """Test that duplicates are removed."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\nMSFT\nAAPL\n")
        
        tickers = load_tickers(ticker_file)
        
        assert tickers == ["AAPL", "MSFT"]
    
    def test_load_tickers_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        ticker_file = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            load_tickers(ticker_file)
    
    def test_load_tickers_empty_file(self, tmp_path):
        """Test that ValueError is raised for empty file."""
        ticker_file = tmp_path / "empty.txt"
        ticker_file.write_text("")
        
        with pytest.raises(ValueError, match="No tickers found"):
            load_tickers(ticker_file)


class TestYFinanceProvider:
    """Tests for YFinance data provider."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\nMSFT\n")
        
        return StrategyConfig(
            ticker_file=ticker_file,
            data_cache_dir=tmp_path / "cache",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            log_level="WARNING",
        )
    
    @pytest.fixture
    def provider(self, config):
        """Create a YFinance provider."""
        return YFinanceProvider(config)
    
    def test_cache_directory_created(self, provider):
        """Test that cache directory is created."""
        assert provider.cache_dir.exists()
    
    @pytest.mark.slow
    def test_download_single_ticker(self, provider, config):
        """Test downloading data for a single ticker."""
        # This test makes a real API call
        df = provider.get_benchmark("SPY", config.start_date, config.end_date)
        
        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        # First trading day may be after start_date due to holidays/weekends
        # Jan 1st is a holiday, so first trading day is Jan 2nd
        assert df.index.min().date() <= config.start_date + pd.Timedelta(days=5)
        assert df.index.max().date() >= config.end_date - pd.Timedelta(days=5)  # Allow for weekends
    
    @pytest.mark.slow
    def test_data_cached_after_download(self, provider, config):
        """Test that data is cached after download."""
        # Download data
        provider.get_benchmark("SPY", config.start_date, config.end_date)
        
        # Check cache file exists
        cache_path = provider._get_cache_path("SPY")
        assert cache_path.exists()
        
        # Verify cached data is valid
        cached_df = pd.read_parquet(cache_path)
        assert not cached_df.empty
    
    @pytest.mark.slow
    def test_cache_hit_on_second_call(self, provider, config):
        """Test that second call uses cache."""
        # First call (downloads)
        df1 = provider.get_benchmark("SPY", config.start_date, config.end_date)
        
        # Delete cache to verify it was created
        cache_path = provider._get_cache_path("SPY")
        assert cache_path.exists()
        
        # Second call (should use cache)
        df2 = provider.get_benchmark("SPY", config.start_date, config.end_date)
        
        # Check values are equal (ignoring potential datetime dtype differences)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


class TestDataManager:
    """Tests for the DataManager."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create a test configuration with small ticker list."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\nMSFT\n")
        
        return StrategyConfig(
            ticker_file=ticker_file,
            data_cache_dir=tmp_path / "cache",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            log_level="WARNING",
        )
    
    @pytest.mark.slow
    def test_load_data(self, config):
        """Test loading all data through DataManager."""
        manager = DataManager(config)
        manager.load_data()
        
        assert manager.ohlcv is not None
        assert manager.benchmark is not None
        assert len(manager.tickers) == 2
    
    @pytest.mark.slow
    def test_get_close_prices(self, config):
        """Test getting close prices."""
        manager = DataManager(config)
        manager.load_data()
        
        close_prices = manager.get_close_prices()
        
        assert isinstance(close_prices, pd.DataFrame)
        assert not close_prices.empty
        assert all(t in close_prices.columns for t in ["AAPL", "MSFT"])
    
    @pytest.mark.slow
    def test_get_volumes(self, config):
        """Test getting volumes."""
        manager = DataManager(config)
        manager.load_data()
        
        volumes = manager.get_volumes()
        
        assert isinstance(volumes, pd.DataFrame)
        assert not volumes.empty
        assert all(vol >= 0 for vol in volumes.values.flatten() if not np.isnan(vol))
    
    def test_data_not_loaded_error(self, config):
        """Test that accessing data before loading raises error."""
        manager = DataManager(config)
        
        with pytest.raises(ValueError, match="Data not loaded"):
            _ = manager.ohlcv


class TestDataIntegrity:
    """Tests for data integrity."""
    
    @pytest.fixture
    def loaded_manager(self, tmp_path):
        """Create a DataManager with loaded data."""
        ticker_file = tmp_path / "tickers.txt"
        ticker_file.write_text("AAPL\n")
        
        config = StrategyConfig(
            ticker_file=ticker_file,
            data_cache_dir=tmp_path / "cache",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            log_level="WARNING",
        )
        
        manager = DataManager(config)
        manager.load_data()
        return manager
    
    @pytest.mark.slow
    def test_no_future_dates(self, loaded_manager):
        """Test that data doesn't contain future dates."""
        from datetime import datetime
        
        max_date = loaded_manager.ohlcv.index.max().date()
        assert max_date <= datetime.now().date()
    
    @pytest.mark.slow
    def test_prices_positive(self, loaded_manager):
        """Test that prices are positive."""
        close_prices = loaded_manager.get_close_prices()
        
        # Drop NaNs and check all values are positive
        valid_prices = close_prices.dropna()
        assert (valid_prices > 0).all().all()
    
    @pytest.mark.slow
    def test_volumes_non_negative(self, loaded_manager):
        """Test that volumes are non-negative."""
        volumes = loaded_manager.get_volumes()
        
        # Drop NaNs and check all values are non-negative
        valid_volumes = volumes.dropna()
        assert (valid_volumes >= 0).all().all()
    
    @pytest.mark.slow
    def test_dates_are_trading_days(self, loaded_manager):
        """Test that all dates are weekdays (trading days)."""
        dates = loaded_manager.get_trading_dates()
        
        # Check no weekends
        weekdays = dates.dayofweek
        assert all(wd < 5 for wd in weekdays), "Data contains weekend dates"


# Mark slow tests for optional exclusion
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (require network)")
