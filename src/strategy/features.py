"""
Feature computation module.

Computes derived features from OHLCV data:
- Returns (daily, rolling)
- Volatility (realized)
- Momentum (with skip period)
- Dollar volume

All features are computed with strict look-ahead prevention.
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from prices.
    
    Args:
        prices: DataFrame with tickers as columns, dates as index
        
    Returns:
        DataFrame of daily returns (same shape as input)
    """
    return prices.pct_change()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from prices.
    
    Args:
        prices: DataFrame with tickers as columns, dates as index
        
    Returns:
        DataFrame of log returns (same shape as input)
    """
    return np.log(prices / prices.shift(1))


def compute_rolling_returns(
    prices: pd.DataFrame, 
    lookback: int,
    skip: int = 0
) -> pd.DataFrame:
    """
    Compute rolling returns over a lookback period, optionally skipping recent days.
    
    This is used for momentum calculations where we skip the most recent days
    to avoid short-term reversal effects.
    
    For day t, computes: price[t - skip] / price[t - skip - lookback] - 1
    
    Args:
        prices: DataFrame with tickers as columns, dates as index
        lookback: Number of days for return calculation
        skip: Number of recent days to skip (default 0)
        
    Returns:
        DataFrame of rolling returns
    """
    if skip > 0:
        # Shift prices to skip recent days, then compute return
        shifted_numerator = prices.shift(skip)
        shifted_denominator = prices.shift(skip + lookback)
    else:
        shifted_numerator = prices
        shifted_denominator = prices.shift(lookback)
    
    return shifted_numerator / shifted_denominator - 1


def compute_realized_volatility(
    returns: pd.DataFrame,
    lookback: int = 20,
    annualization_factor: float = 252.0
) -> pd.DataFrame:
    """
    Compute trailing realized volatility (annualized).
    
    For day t, uses returns from [t - lookback + 1, t] inclusive.
    Only uses data available at time t (no look-ahead).
    
    Args:
        returns: DataFrame of daily returns
        lookback: Rolling window size (default 20 days)
        annualization_factor: Trading days per year (default 252)
        
    Returns:
        DataFrame of annualized volatility
    """
    # Rolling standard deviation of returns
    rolling_std = returns.rolling(window=lookback, min_periods=lookback).std()
    
    # Annualize
    return rolling_std * np.sqrt(annualization_factor)


def compute_dollar_volume(
    close_prices: pd.DataFrame,
    volume: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute dollar volume (close price × volume).
    
    Args:
        close_prices: DataFrame of close prices
        volume: DataFrame of trading volume
        
    Returns:
        DataFrame of dollar volume
    """
    # Ensure alignment
    common_cols = close_prices.columns.intersection(volume.columns)
    common_idx = close_prices.index.intersection(volume.index)
    
    return close_prices.loc[common_idx, common_cols] * volume.loc[common_idx, common_cols]


def compute_average_dollar_volume(
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Compute trailing average dollar volume.
    
    For day t, computes mean of dollar volume over [t - lookback + 1, t].
    
    Args:
        close_prices: DataFrame of close prices
        volume: DataFrame of trading volume
        lookback: Rolling window size (default 20 days)
        
    Returns:
        DataFrame of average dollar volume
    """
    dollar_vol = compute_dollar_volume(close_prices, volume)
    return dollar_vol.rolling(window=lookback, min_periods=lookback).mean()


def compute_momentum_score(
    prices: pd.DataFrame,
    lookback: int = 60,
    skip: int = 5
) -> pd.DataFrame:
    """
    Compute momentum score for ranking.
    
    Baseline momentum: trailing return over lookback period, excluding
    the most recent skip days to avoid short-term reversal.
    
    For day t: (price[t - skip] / price[t - skip - lookback]) - 1
    
    Args:
        prices: DataFrame of prices (adjusted close recommended)
        lookback: Total lookback period for momentum (default 60 days)
        skip: Recent days to skip (default 5 days)
        
    Returns:
        DataFrame of momentum scores
    """
    return compute_rolling_returns(prices, lookback=lookback, skip=skip)


def compute_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    lookback: int = 14
) -> pd.DataFrame:
    """
    Compute Average True Range (ATR).
    
    True Range is the maximum of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|
    
    ATR is the rolling average of True Range.
    
    Args:
        high: DataFrame of high prices
        low: DataFrame of low prices
        close: DataFrame of close prices
        lookback: Rolling window for averaging (default 14)
        
    Returns:
        DataFrame of ATR values
    """
    prev_close = close.shift(1)
    
    # True Range components
    hl = high - low
    hc = (high - prev_close).abs()
    lc = (low - prev_close).abs()
    
    # True Range is the maximum
    true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1, level=0)
    
    # If we have MultiIndex, handle it properly
    if isinstance(hl.columns, pd.MultiIndex):
        true_range = hl.copy()
        for col in hl.columns:
            true_range[col] = pd.concat([hl[col], hc[col], lc[col]], axis=1).max(axis=1)
    else:
        true_range = pd.DataFrame({
            col: pd.concat([hl[col], hc[col], lc[col]], axis=1).max(axis=1)
            for col in hl.columns
        })
    
    # Average True Range
    return true_range.rolling(window=lookback, min_periods=lookback).mean()


class FeatureComputer:
    """
    High-level feature computation class.
    
    Computes and caches all features needed for the strategy.
    Ensures strict look-ahead prevention in all calculations.
    """
    
    def __init__(
        self,
        close_prices: pd.DataFrame,
        open_prices: Optional[pd.DataFrame] = None,
        high_prices: Optional[pd.DataFrame] = None,
        low_prices: Optional[pd.DataFrame] = None,
        volume: Optional[pd.DataFrame] = None
    ):
        """
        Initialize the feature computer.
        
        Args:
            close_prices: DataFrame of adjusted close prices
            open_prices: DataFrame of open prices (optional)
            high_prices: DataFrame of high prices (optional)
            low_prices: DataFrame of low prices (optional)
            volume: DataFrame of trading volume (optional)
        """
        self.close_prices = close_prices
        self.open_prices = open_prices
        self.high_prices = high_prices
        self.low_prices = low_prices
        self.volume = volume
        
        # Cached features
        self._returns: Optional[pd.DataFrame] = None
        self._volatility: Optional[pd.DataFrame] = None
        self._momentum: Optional[pd.DataFrame] = None
        self._avg_dollar_volume: Optional[pd.DataFrame] = None
        self._atr: Optional[pd.DataFrame] = None
    
    @property
    def returns(self) -> pd.DataFrame:
        """Get daily returns (cached)."""
        if self._returns is None:
            self._returns = compute_daily_returns(self.close_prices)
        return self._returns
    
    def get_volatility(self, lookback: int = 20) -> pd.DataFrame:
        """
        Get realized volatility.
        
        Args:
            lookback: Rolling window size
            
        Returns:
            DataFrame of annualized volatility
        """
        return compute_realized_volatility(self.returns, lookback=lookback)
    
    def get_momentum(self, lookback: int = 60, skip: int = 5) -> pd.DataFrame:
        """
        Get momentum score.
        
        Args:
            lookback: Total lookback period
            skip: Recent days to skip
            
        Returns:
            DataFrame of momentum scores
        """
        return compute_momentum_score(self.close_prices, lookback=lookback, skip=skip)
    
    def get_avg_dollar_volume(self, lookback: int = 20) -> pd.DataFrame:
        """
        Get average dollar volume.
        
        Args:
            lookback: Rolling window size
            
        Returns:
            DataFrame of average dollar volume
        """
        if self.volume is None:
            raise ValueError("Volume data required for dollar volume calculation")
        
        return compute_average_dollar_volume(
            self.close_prices, 
            self.volume, 
            lookback=lookback
        )
    
    def get_atr(self, lookback: int = 14) -> pd.DataFrame:
        """
        Get Average True Range.
        
        Args:
            lookback: Rolling window size
            
        Returns:
            DataFrame of ATR values
        """
        if self.high_prices is None or self.low_prices is None:
            raise ValueError("High and Low prices required for ATR calculation")
        
        return compute_atr(
            self.high_prices, 
            self.low_prices, 
            self.close_prices, 
            lookback=lookback
        )
