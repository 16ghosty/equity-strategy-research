"""
Ranking module for the equity strategy.

Computes cross-sectional momentum scores and ranks tickers daily.
Uses trailing returns excluding recent days to avoid short-term reversal.
"""

import pandas as pd
import numpy as np
from typing import Optional

from .config import StrategyConfig
from .features import compute_momentum_score


class RankComputer:
    """
    Computes daily cross-sectional ranks for tickers.
    
    Ranking is based on momentum score (trailing return excluding recent days).
    Ranks are computed only among eligible tickers in the universe.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the rank computer.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.lookback = config.momentum_lookback
        self.skip = config.momentum_skip
        self.logger = config.get_logger("rank")
    
    def compute_momentum_scores(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute momentum score for all tickers.
        
        Score is the trailing return over lookback period, excluding
        the most recent skip days to avoid short-term reversal.
        
        Args:
            prices: DataFrame of adjusted close prices (tickers as columns)
            
        Returns:
            DataFrame of momentum scores (same shape as prices)
        """
        return compute_momentum_score(
            prices, 
            lookback=self.lookback, 
            skip=self.skip
        )
    
    def compute_ranks(
        self,
        momentum_scores: pd.DataFrame,
        universe_mask: Optional[pd.DataFrame] = None,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Compute cross-sectional ranks from momentum scores.
        
        Rank 1 = best (highest momentum by default).
        Only tickers in universe are ranked; others get NaN.
        
        Args:
            momentum_scores: DataFrame of momentum scores
            universe_mask: Optional boolean DataFrame (True = in universe)
            ascending: If True, lower scores get better ranks
            
        Returns:
            DataFrame of ranks (1 = best, NaN = not in universe)
        """
        scores = momentum_scores.copy()
        
        # Apply universe mask if provided
        if universe_mask is not None:
            # Set scores to NaN for tickers not in universe
            scores = scores.where(universe_mask, np.nan)
        
        # Compute cross-sectional ranks (along axis=1, i.e., across tickers)
        # rank() gives 1 = smallest, so for descending we use ascending=False
        ranks = scores.rank(axis=1, ascending=ascending, method='first')
        
        return ranks
    
    def compute_daily_ranks(
        self,
        prices: pd.DataFrame,
        universe_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute daily ranks for all tickers, optionally filtered by universe.
        
        Args:
            prices: DataFrame of adjusted close prices
            universe_df: Optional DataFrame from UniverseSelector with 'universe' column
            
        Returns:
            DataFrame of daily ranks (1 = best momentum, NaN = not ranked)
        """
        # Compute momentum scores
        momentum = self.compute_momentum_scores(prices)
        
        # Create universe mask if universe_df provided
        universe_mask = None
        if universe_df is not None:
            universe_mask = pd.DataFrame(
                False, 
                index=prices.index, 
                columns=prices.columns
            )
            for dt in universe_df.index:
                if dt in universe_mask.index:
                    universe = universe_df.loc[dt, 'universe']
                    for ticker in universe:
                        if ticker in universe_mask.columns:
                            universe_mask.loc[dt, ticker] = True
        
        # Compute ranks
        ranks = self.compute_ranks(momentum, universe_mask, ascending=False)
        
        self.logger.debug(
            f"Computed ranks for {len(prices)} days, "
            f"avg ranked tickers per day: {(~ranks.isna()).sum(axis=1).mean():.1f}"
        )
        
        return ranks


def get_top_ranked_tickers(
    ranks: pd.Series,
    top_k: int
) -> list[str]:
    """
    Get the top K ranked tickers for a given day.
    
    Args:
        ranks: Series of ranks for tickers on a specific day
        top_k: Number of top tickers to select
        
    Returns:
        List of ticker symbols for top K
    """
    valid_ranks = ranks.dropna()
    if len(valid_ranks) == 0:
        return []
    
    # Sort by rank (ascending, so rank 1 is first)
    sorted_ranks = valid_ranks.sort_values()
    return list(sorted_ranks.head(top_k).index)


def get_tickers_within_buffer(
    ranks: pd.Series,
    top_k: int,
    buffer: int
) -> list[str]:
    """
    Get tickers within the exit buffer (rank <= top_k + buffer).
    
    Args:
        ranks: Series of ranks for tickers
        top_k: Number of top positions
        buffer: Exit buffer
        
    Returns:
        List of tickers within buffer range
    """
    valid_ranks = ranks.dropna()
    threshold = top_k + buffer
    return list(valid_ranks[valid_ranks <= threshold].index)
