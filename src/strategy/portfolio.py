"""
Portfolio construction module.

Converts ranks and gate results into target portfolio weights.
Implements weight schemes, constraints, and buffer logic.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .config import StrategyConfig
from .gates import GateResults


@dataclass
class PortfolioTarget:
    """
    Target portfolio for a single day.
    
    Attributes:
        date: Target date
        weights: Dict mapping ticker to target weight
        gross_exposure: Total gross exposure (sum of absolute weights)
        num_positions: Number of positions
        turnover: Turnover from previous portfolio (if available)
    """
    date: pd.Timestamp
    weights: dict[str, float]
    gross_exposure: float
    num_positions: int
    turnover: Optional[float] = None
    
    def get_weight(self, ticker: str) -> float:
        """Get weight for a ticker (0 if not held)."""
        return self.weights.get(ticker, 0.0)


class PortfolioConstructor:
    """
    Constructs target portfolios from ranks and gate results.
    
    Features:
    - Equal-weight or inverse-vol weighting
    - Max single-name weight cap
    - Gross exposure limit
    - Buffer logic for entry/exit
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the portfolio constructor.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.top_k = config.top_k
        self.buffer = config.buffer
        self.weight_scheme = config.weight_scheme
        self.max_weight = config.max_weight
        self.max_gross_exposure = config.max_gross_exposure
        self.logger = config.get_logger("portfolio")
    
    def construct_portfolio(
        self,
        date: pd.Timestamp,
        ranks: pd.Series,
        gate_results: dict[str, GateResults],
        current_holdings: set[str],
        volatilities: Optional[pd.Series] = None,
        regime_scale: float = 1.0
    ) -> PortfolioTarget:
        """
        Construct target portfolio for a given day.
        
        Args:
            date: Target date
            ranks: Series of ranks by ticker
            gate_results: Dict of GateResults by ticker
            current_holdings: Set of currently held tickers
            volatilities: Optional volatilities for inverse-vol weighting
            regime_scale: Global scale from market regime gate
            
        Returns:
            PortfolioTarget with weights
        """
        # Step 1: Determine which tickers to hold based on buffer logic
        tickers_to_hold = self._apply_buffer_logic(
            ranks=ranks,
            gate_results=gate_results,
            current_holdings=current_holdings
        )
        
        if not tickers_to_hold:
            return PortfolioTarget(
                date=date,
                weights={},
                gross_exposure=0.0,
                num_positions=0
            )
        
        # Step 2: Compute raw weights
        if self.weight_scheme == "inverse_vol" and volatilities is not None:
            raw_weights = self._compute_inverse_vol_weights(
                tickers_to_hold, volatilities, gate_results
            )
        else:
            raw_weights = self._compute_equal_weights(
                tickers_to_hold, gate_results
            )
        
        # Step 3: Apply constraints
        constrained_weights = self._apply_constraints(raw_weights, regime_scale)
        
        # Step 4: Build result
        gross_exposure = sum(abs(w) for w in constrained_weights.values())
        
        return PortfolioTarget(
            date=date,
            weights=constrained_weights,
            gross_exposure=gross_exposure,
            num_positions=len(constrained_weights)
        )
    
    def _apply_buffer_logic(
        self,
        ranks: pd.Series,
        gate_results: dict[str, GateResults],
        current_holdings: set[str]
    ) -> list[str]:
        """
        Apply buffer logic to determine which tickers to hold.
        
        Entry: rank <= top_k AND passes all critical gates
        Exit: rank > top_k + buffer OR fails critical gates
        Hold: top_k < rank <= top_k + buffer AND passes critical gates
        
        Args:
            ranks: Series of ranks by ticker
            gate_results: Dict of GateResults by ticker
            current_holdings: Currently held tickers
            
        Returns:
            List of tickers to hold
        """
        tickers_to_hold = []
        exit_threshold = self.top_k + self.buffer
        
        for ticker, result in gate_results.items():
            rank = ranks.get(ticker, np.nan)
            
            # Check if passes critical gates (excluding buffer gate)
            critical_gates = ['liquidity', 'volatility']
            passes_critical = all(
                result.results.get(g, GateResults).passed 
                for g in critical_gates if g in result.results
            )
            
            if not passes_critical:
                continue
            
            if pd.isna(rank):
                continue
            
            is_currently_held = ticker in current_holdings
            
            if is_currently_held:
                # Exit only if rank exceeds buffer
                if rank <= exit_threshold:
                    tickers_to_hold.append(ticker)
            else:
                # Enter only if rank in top_k
                if rank <= self.top_k:
                    tickers_to_hold.append(ticker)
        
        # Sort by rank to ensure we take best if there are ties
        tickers_to_hold.sort(key=lambda t: ranks.get(t, float('inf')))
        
        # Limit to top_k if we have more (shouldn't happen but safety check)
        if len(tickers_to_hold) > self.top_k + self.buffer:
            tickers_to_hold = tickers_to_hold[:self.top_k + self.buffer]
        
        return tickers_to_hold
    
    def _compute_equal_weights(
        self,
        tickers: list[str],
        gate_results: dict[str, GateResults]
    ) -> dict[str, float]:
        """
        Compute equal weights, adjusted by gate scaling factors.
        
        Args:
            tickers: List of tickers to hold
            gate_results: Gate results with scaling factors
            
        Returns:
            Dict of raw weights
        """
        if not tickers:
            return {}
        
        # Start with equal weights
        base_weight = 1.0 / len(tickers)
        
        weights = {}
        for ticker in tickers:
            # Apply gate scales (e.g., volatility sizing)
            scale = gate_results[ticker].final_scale if ticker in gate_results else 1.0
            weights[ticker] = base_weight * scale
        
        return weights
    
    def _compute_inverse_vol_weights(
        self,
        tickers: list[str],
        volatilities: pd.Series,
        gate_results: dict[str, GateResults]
    ) -> dict[str, float]:
        """
        Compute inverse volatility weights.
        
        Weight proportional to 1/vol, normalized to sum to 1.
        
        Args:
            tickers: List of tickers to hold
            volatilities: Series of volatilities by ticker
            gate_results: Gate results with scaling factors
            
        Returns:
            Dict of raw weights
        """
        if not tickers:
            return {}
        
        # Compute inverse vol for each ticker
        inv_vols = {}
        for ticker in tickers:
            vol = volatilities.get(ticker, np.nan)
            if pd.isna(vol) or vol <= 0:
                inv_vols[ticker] = 1.0  # Default to equal weight contribution
            else:
                inv_vols[ticker] = 1.0 / vol
        
        # Normalize
        total_inv_vol = sum(inv_vols.values())
        if total_inv_vol == 0:
            return self._compute_equal_weights(tickers, gate_results)
        
        weights = {}
        for ticker in tickers:
            base_weight = inv_vols[ticker] / total_inv_vol
            # Apply gate scales
            scale = gate_results[ticker].final_scale if ticker in gate_results else 1.0
            weights[ticker] = base_weight * scale
        
        return weights
    
    def _apply_constraints(
        self,
        raw_weights: dict[str, float],
        regime_scale: float
    ) -> dict[str, float]:
        """
        Apply portfolio constraints.
        
        1. Apply regime scale
        2. Cap single-name weights
        3. Renormalize to respect gross exposure
        
        Args:
            raw_weights: Raw weights before constraints
            regime_scale: Global scale from market regime
            
        Returns:
            Constrained weights
        """
        if not raw_weights:
            return {}
        
        # Apply regime scale
        weights = {t: w * regime_scale for t, w in raw_weights.items()}
        
        # Cap individual weights
        capped = False
        for ticker in weights:
            if weights[ticker] > self.max_weight:
                weights[ticker] = self.max_weight
                capped = True
        
        # Renormalize if needed to respect max gross exposure
        gross = sum(abs(w) for w in weights.values())
        if gross > self.max_gross_exposure:
            scale = self.max_gross_exposure / gross
            weights = {t: w * scale for t, w in weights.items()}
        
        # Remove zero weights
        weights = {t: w for t, w in weights.items() if w > 1e-6}
        
        return weights
    
    def compute_turnover(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float]
    ) -> float:
        """
        Compute turnover between two portfolios.
        
        Turnover = sum of absolute weight changes / 2
        
        Args:
            old_weights: Previous weights
            new_weights: New weights
            
        Returns:
            Turnover as a fraction of portfolio
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        
        total_change = sum(
            abs(new_weights.get(t, 0) - old_weights.get(t, 0))
            for t in all_tickers
        )
        
        return total_change / 2
