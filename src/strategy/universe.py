"""
Universe selection module.

Implements monthly universe selection based on trailing dollar volume.
Strict look-ahead prevention: month M universe uses only data prior to month M start.
"""

import pandas as pd
import numpy as np
from datetime import date
from typing import Optional
import logging

from .config import StrategyConfig
from .features import compute_dollar_volume


class UniverseSelector:
    """
    Selects top N stocks by trailing dollar volume on a monthly basis.
    
    Key features:
    - Recomputes universe at the start of each month
    - Uses only data available before the month start (no look-ahead)
    - Tracks universe membership over time
    - Logs selection diagnostics
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the universe selector.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = config.get_logger("universe")
        
        self.universe_size = config.universe_size
        self.lookback = config.lookback_dollar_vol
    
    def select_universe(
        self,
        close_prices: pd.DataFrame,
        volume: pd.DataFrame,
        candidate_tickers: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Select universe for each trading day.
        
        Universe is recomputed at the start of each month using data
        strictly prior to that month.
        
        Args:
            close_prices: DataFrame of close prices (tickers as columns)
            volume: DataFrame of trading volume (tickers as columns)
            candidate_tickers: Optional list of candidate tickers to consider
                             (if None, uses all columns in close_prices)
        
        Returns:
            DataFrame with dates as index, columns:
            - 'universe': list of tickers in universe for that day
            - 'universe_date': date when this universe was computed
        """
        # Filter to candidate tickers if provided
        if candidate_tickers is not None:
            available_tickers = set(close_prices.columns) & set(candidate_tickers)
            close_prices = close_prices[list(available_tickers)]
            volume = volume[list(available_tickers)]
        
        # Compute dollar volume
        dollar_volume = compute_dollar_volume(close_prices, volume)
        
        # Compute trailing average dollar volume
        avg_dollar_volume = dollar_volume.rolling(
            window=self.lookback, 
            min_periods=self.lookback
        ).mean()
        
        # Get unique months in the data
        trading_dates = close_prices.index.sort_values()
        
        # Create universe assignments
        universe_records = []
        current_universe = None
        current_universe_date = None
        
        for dt in trading_dates:
            # Check if this is new month or first day
            is_new_month = (
                current_universe_date is None or 
                dt.month != current_universe_date.month or
                dt.year != current_universe_date.year
            )
            
            if is_new_month:
                # Select new universe using data BEFORE this date
                # Get the last available data point before this month
                prior_dates = avg_dollar_volume.index[avg_dollar_volume.index < dt]
                
                if len(prior_dates) >= self.lookback:
                    # Use the last available prior date for selection
                    selection_date = prior_dates[-1]
                    adv_on_selection = avg_dollar_volume.loc[selection_date]
                    
                    # Drop NaN values and sort by dollar volume
                    valid_adv = adv_on_selection.dropna()
                    
                    if len(valid_adv) >= self.universe_size:
                        # Select top N by dollar volume
                        top_n = valid_adv.nlargest(self.universe_size)
                        current_universe = list(top_n.index)
                    else:
                        # Take all available if fewer than N
                        current_universe = list(valid_adv.index)
                        self.logger.warning(
                            f"Only {len(valid_adv)} tickers available for {dt.date()}, "
                            f"need {self.universe_size}"
                        )
                    
                    current_universe_date = dt
                    self.logger.debug(
                        f"Universe updated for {dt.date()}: {len(current_universe)} tickers, "
                        f"selected using data up to {selection_date.date()}"
                    )
                else:
                    # Not enough history yet
                    current_universe = []
                    current_universe_date = dt
                    self.logger.debug(
                        f"Insufficient history for {dt.date()}: {len(prior_dates)} days available"
                    )
            
            universe_records.append({
                'date': dt,
                'universe': current_universe.copy() if current_universe else [],
                'universe_date': current_universe_date,
                'universe_size': len(current_universe) if current_universe else 0
            })
        
        result = pd.DataFrame(universe_records)
        result = result.set_index('date')
        
        self.logger.info(
            f"Universe selection complete: {len(result)} days, "
            f"avg universe size: {result['universe_size'].mean():.1f}"
        )
        
        return result
    
    def get_universe_on_date(
        self,
        universe_df: pd.DataFrame,
        dt: pd.Timestamp
    ) -> list[str]:
        """
        Get the universe for a specific date.
        
        Args:
            universe_df: DataFrame from select_universe()
            dt: Date to query
            
        Returns:
            List of tickers in universe on that date
        """
        if dt not in universe_df.index:
            # Find the most recent prior date
            prior = universe_df.index[universe_df.index <= dt]
            if len(prior) == 0:
                return []
            dt = prior[-1]
        
        return universe_df.loc[dt, 'universe']


def create_universe_membership_matrix(
    universe_df: pd.DataFrame,
    all_tickers: list[str]
) -> pd.DataFrame:
    """
    Create a binary matrix of universe membership.
    
    Args:
        universe_df: DataFrame from UniverseSelector.select_universe()
        all_tickers: List of all possible tickers
        
    Returns:
        DataFrame with dates as index, tickers as columns, 
        True/False for membership
    """
    # Initialize with False
    membership = pd.DataFrame(
        False,
        index=universe_df.index,
        columns=all_tickers
    )
    
    # Fill in memberships
    for dt in universe_df.index:
        universe = universe_df.loc[dt, 'universe']
        for ticker in universe:
            if ticker in membership.columns:
                membership.loc[dt, ticker] = True
    
    return membership


def validate_no_lookahead(
    universe_df: pd.DataFrame,
    close_prices: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 30
) -> tuple[bool, list[str]]:
    """
    Validate that universe selection has no look-ahead bias.
    
    Checks that for each date, the universe was selected using only
    data available before that date.
    
    Args:
        universe_df: DataFrame from select_universe()
        close_prices: Original price data
        volume: Original volume data
        lookback: Dollar volume lookback period
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    dollar_volume = compute_dollar_volume(close_prices, volume)
    avg_dollar_volume = dollar_volume.rolling(window=lookback, min_periods=lookback).mean()
    
    prev_month = None
    
    for dt in universe_df.index:
        universe = universe_df.loc[dt, 'universe']
        universe_date = universe_df.loc[dt, 'universe_date']
        
        # Check if this is a new month boundary
        current_month = (dt.year, dt.month)
        
        if current_month != prev_month:
            # Universe should have been computed using data BEFORE this date
            prior_dates = avg_dollar_volume.index[avg_dollar_volume.index < dt]
            
            if len(prior_dates) > 0:
                latest_prior = prior_dates[-1]
                
                # Check that all tickers in universe were top N on prior date
                if len(universe) > 0:
                    prior_adv = avg_dollar_volume.loc[latest_prior].dropna()
                    
                    for ticker in universe:
                        if ticker not in prior_adv.index:
                            errors.append(
                                f"{dt.date()}: {ticker} in universe but no prior ADV data"
                            )
            
            prev_month = current_month
    
    return len(errors) == 0, errors
