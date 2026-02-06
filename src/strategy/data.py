"""
Data download and caching module.

Provides a DataProvider interface with yfinance implementation.
All downloaded data is cached locally as Parquet files for fast repeated runs.
"""

from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
import logging

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

from .config import StrategyConfig


class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_ohlcv(
        self, 
        tickers: list[str], 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for given tickers and date range.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with MultiIndex columns (ticker, field) where field is one of:
            Open, High, Low, Close, Volume, Adj Close
        """
        pass
    
    @abstractmethod
    def get_benchmark(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for benchmark ticker.
        
        Args:
            ticker: Benchmark ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        pass


class YFinanceProvider(DataProvider):
    """
    Data provider using yfinance for free OHLCV data.
    
    Features:
    - Downloads data from Yahoo Finance
    - Caches data to Parquet files
    - Handles missing data and ticker failures gracefully
    - Logs download progress and any issues
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the YFinance data provider.
        
        Args:
            config: Strategy configuration with cache directory and logging settings
        """
        self.config = config
        self.cache_dir = Path(config.data_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = config.get_logger("data")
    
    def _get_cache_path(self, ticker: str) -> Path:
        """Get the cache file path for a ticker."""
        # Sanitize ticker name for filesystem (handle tickers like BRK-B)
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_ticker}.parquet"
    
    def _load_from_cache(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache if available and covers the requested range.
        
        Returns:
            DataFrame if cache hit and covers range, None otherwise
        """
        cache_path = self._get_cache_path(ticker)
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            if df.empty:
                return None
            
            # Check if cached data covers the requested range
            cached_start = df.index.min().date()
            cached_end = df.index.max().date()
            
            if cached_start <= start_date and cached_end >= end_date:
                # Filter to requested range
                df = df.loc[str(start_date):str(end_date)]
                self.logger.debug(f"Cache hit for {ticker}: {len(df)} rows")
                return df
            else:
                self.logger.debug(
                    f"Cache miss for {ticker}: cached [{cached_start}, {cached_end}], "
                    f"requested [{start_date}, {end_date}]"
                )
                return None
        except Exception as e:
            self.logger.warning(f"Error reading cache for {ticker}: {e}")
            return None
    
    def _save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """Save data to cache."""
        if df.empty:
            return
        
        cache_path = self._get_cache_path(ticker)
        try:
            df.to_parquet(cache_path)
            self.logger.debug(f"Saved {ticker} to cache: {len(df)} rows")
        except Exception as e:
            self.logger.warning(f"Error saving cache for {ticker}: {e}")
    
    def _download_ticker(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a single ticker from yfinance.
        
        Returns:
            DataFrame with OHLCV data, or None if download failed
        """
        try:
            # Add buffer days to ensure we have enough data for lookbacks
            buffer_start = start_date - timedelta(days=365)
            
            # Download data
            df = yf.download(
                ticker,
                start=buffer_start.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),  # end is exclusive
                progress=False,
                auto_adjust=False,  # Keep Adj Close separate
            )
            
            if df.empty:
                self.logger.warning(f"No data returned for {ticker}")
                return None
            
            # Ensure index is DatetimeIndex
            df.index = pd.to_datetime(df.index)
            
            # Handle MultiIndex columns from yfinance (when downloading single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Validate required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for {ticker}: {missing_cols}")
                return None
            
            # Save full data to cache
            self._save_to_cache(ticker, df)
            
            # Return only the requested range
            df = df.loc[str(start_date):str(end_date)]
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error downloading {ticker}: {e}")
            return None
    
    def get_ohlcv(
        self, 
        tickers: list[str], 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for given tickers and date range.
        
        Uses cache when available, downloads missing data from yfinance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with MultiIndex columns (field, ticker) where field is one of:
            Open, High, Low, Close, Volume, Adj Close
        """
        self.logger.info(f"Fetching OHLCV for {len(tickers)} tickers from {start_date} to {end_date}")
        
        ticker_data = {}
        failed_tickers = []
        
        for ticker in tqdm(tickers, desc="Loading data"):
            # Try cache first
            df = self._load_from_cache(ticker, start_date, end_date)
            
            if df is None:
                # Download if not in cache
                df = self._download_ticker(ticker, start_date, end_date)
            
            if df is not None and not df.empty:
                ticker_data[ticker] = df
            else:
                failed_tickers.append(ticker)
        
        if failed_tickers:
            self.logger.warning(f"Failed to load {len(failed_tickers)} tickers: {failed_tickers[:10]}...")
        
        if not ticker_data:
            raise ValueError("No data loaded for any ticker")
        
        # Combine into single DataFrame with MultiIndex columns
        combined = pd.concat(ticker_data, axis=1)
        combined.columns = pd.MultiIndex.from_tuples(
            [(ticker, col) for ticker, df in ticker_data.items() for col in df.columns],
            names=['Ticker', 'Field']
        )
        
        # Reorganize to have Field as first level (more intuitive for accessing)
        # combined = combined.swaplevel(axis=1).sort_index(axis=1)
        
        self.logger.info(f"Loaded data for {len(ticker_data)} tickers, {len(combined)} trading days")
        
        return combined
    
    def get_benchmark(
        self, 
        ticker: str, 
        start_date: date, 
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for benchmark ticker.
        
        Args:
            ticker: Benchmark ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        """
        self.logger.info(f"Fetching benchmark {ticker} from {start_date} to {end_date}")
        
        # Try cache first
        df = self._load_from_cache(ticker, start_date, end_date)
        
        if df is None:
            # Download if not in cache
            df = self._download_ticker(ticker, start_date, end_date)
        
        if df is None or df.empty:
            raise ValueError(f"Failed to load benchmark data for {ticker}")
        
        return df


def load_tickers(ticker_file: Path) -> list[str]:
    """
    Load ticker symbols from a file.
    
    Args:
        ticker_file: Path to file with one ticker per line
        
    Returns:
        List of ticker symbols (uppercase, stripped, comments removed)
    """
    if not ticker_file.exists():
        raise FileNotFoundError(f"Ticker file not found: {ticker_file}")
    
    tickers = []
    with open(ticker_file, 'r') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if line:
                tickers.append(line.upper())
    
    if not tickers:
        raise ValueError(f"No tickers found in {ticker_file}")
    
    return list(dict.fromkeys(tickers))  # Remove duplicates, preserve order


class DataManager:
    """
    High-level data manager that coordinates data loading for the strategy.
    
    Provides a clean interface for accessing price data with proper alignment.
    """
    
    def __init__(self, config: StrategyConfig, provider: Optional[DataProvider] = None):
        """
        Initialize the data manager.
        
        Args:
            config: Strategy configuration
            provider: Data provider instance (defaults to YFinanceProvider)
        """
        self.config = config
        self.provider = provider or YFinanceProvider(config)
        self.logger = config.get_logger("data_manager")
        
        # Cached data
        self._ohlcv: Optional[pd.DataFrame] = None
        self._benchmark: Optional[pd.DataFrame] = None
        self._tickers: Optional[list[str]] = None
    
    def load_data(self) -> None:
        """Load all required data (tickers + benchmark)."""
        # Load ticker list
        self._tickers = load_tickers(self.config.ticker_file)
        self.logger.info(f"Loaded {len(self._tickers)} candidate tickers")
        
        # Load OHLCV data
        self._ohlcv = self.provider.get_ohlcv(
            self._tickers,
            self.config.start_date,
            self.config.end_date
        )
        
        # Load benchmark data
        self._benchmark = self.provider.get_benchmark(
            self.config.benchmark,
            self.config.start_date,
            self.config.end_date
        )
    
    @property
    def tickers(self) -> list[str]:
        """Get list of available tickers."""
        if self._tickers is None:
            self._tickers = load_tickers(self.config.ticker_file)
        return self._tickers
    
    @property
    def ohlcv(self) -> pd.DataFrame:
        """Get OHLCV data for all tickers."""
        if self._ohlcv is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._ohlcv
    
    @property
    def benchmark(self) -> pd.DataFrame:
        """Get benchmark OHLCV data."""
        if self._benchmark is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self._benchmark
    
    def get_close_prices(self) -> pd.DataFrame:
        """
        Get adjusted close prices for all tickers.
        
        Returns:
            DataFrame with tickers as columns, dates as index
        """
        prices = {}
        for ticker in self.ohlcv.columns.get_level_values('Ticker').unique():
            # Prefer Adj Close, fall back to Close
            if ('Adj Close' in self.ohlcv[(ticker,)].columns):
                prices[ticker] = self.ohlcv[(ticker, 'Adj Close')]
            else:
                prices[ticker] = self.ohlcv[(ticker, 'Close')]
        return pd.DataFrame(prices)
    
    def get_open_prices(self) -> pd.DataFrame:
        """
        Get open prices for all tickers.
        
        Returns:
            DataFrame with tickers as columns, dates as index
        """
        prices = {}
        for ticker in self.ohlcv.columns.get_level_values('Ticker').unique():
            prices[ticker] = self.ohlcv[(ticker, 'Open')]
        return pd.DataFrame(prices)
    
    def get_volumes(self) -> pd.DataFrame:
        """
        Get volume for all tickers.
        
        Returns:
            DataFrame with tickers as columns, dates as index
        """
        volumes = {}
        for ticker in self.ohlcv.columns.get_level_values('Ticker').unique():
            volumes[ticker] = self.ohlcv[(ticker, 'Volume')]
        return pd.DataFrame(volumes)
    
    def get_high_prices(self) -> pd.DataFrame:
        """Get high prices for all tickers."""
        prices = {}
        for ticker in self.ohlcv.columns.get_level_values('Ticker').unique():
            prices[ticker] = self.ohlcv[(ticker, 'High')]
        return pd.DataFrame(prices)
    
    def get_low_prices(self) -> pd.DataFrame:
        """Get low prices for all tickers."""
        prices = {}
        for ticker in self.ohlcv.columns.get_level_values('Ticker').unique():
            prices[ticker] = self.ohlcv[(ticker, 'Low')]
        return pd.DataFrame(prices)
    
    def get_trading_dates(self) -> pd.DatetimeIndex:
        """Get all trading dates in the data."""
        return self.ohlcv.index
    
    def get_benchmark_close(self) -> pd.Series:
        """Get benchmark adjusted close prices."""
        if 'Adj Close' in self.benchmark.columns:
            return self.benchmark['Adj Close']
        return self.benchmark['Close']
