"""
Gate functions for the equity strategy.

Each gate is a filter that determines eligibility for a ticker on a given day.
Gates return a GateResult with:
- passed: bool indicating if the gate passed
- reason: str explaining the result (e.g., 'ok', 'fail_liquidity', 'risk_off')
- scale: float for sizing gates (1.0 if not applicable)

All gates use only data available at decision time (no look-ahead).
"""

from dataclasses import dataclass
from typing import Optional, Callable
import pandas as pd
import numpy as np


@dataclass
class GateResult:
    """
    Result of a gate evaluation.
    
    Attributes:
        passed: Whether the gate passed
        reason: Explanation code (e.g., 'ok', 'fail_liquidity', 'fail_vol')
        scale: Scaling factor for position size (1.0 = full size, 0.5 = half, etc.)
    """
    passed: bool
    reason: str
    scale: float = 1.0
    
    def __repr__(self) -> str:
        return f"GateResult(passed={self.passed}, reason='{self.reason}', scale={self.scale:.2f})"


@dataclass
class GateResults:
    """
    Aggregated results from all gates for a ticker on a date.
    
    Attributes:
        ticker: The ticker symbol
        date: The evaluation date
        results: Dict mapping gate name to GateResult
        final_passed: Whether all gates passed
        final_scale: Combined scaling factor from all gates
        fail_reasons: List of reasons for any failed gates
    """
    ticker: str
    date: pd.Timestamp
    results: dict[str, GateResult]
    
    @property
    def final_passed(self) -> bool:
        """Check if all gates passed."""
        return all(r.passed for r in self.results.values())
    
    @property
    def final_scale(self) -> float:
        """Compute combined scale (product of all scales)."""
        scale = 1.0
        for r in self.results.values():
            scale *= r.scale
        return scale
    
    @property
    def fail_reasons(self) -> list[str]:
        """Get list of failure reasons."""
        return [f"{name}:{r.reason}" for name, r in self.results.items() if not r.passed]


# =============================================================================
# Individual Gate Functions
# =============================================================================

def liquidity_gate(
    avg_dollar_volume: float,
    price: float,
    threshold: float = 1_000_000,
    min_price: float = 5.0
) -> GateResult:
    """
    Liquidity gate based on average dollar volume and price.
    
    Args:
        avg_dollar_volume: Trailing average dollar volume
        price: Current price
        threshold: Minimum average dollar volume required
        min_price: Minimum price (default $5)
        
    Returns:
        GateResult with pass/fail and reason
    """
    if pd.isna(avg_dollar_volume):
        return GateResult(passed=False, reason="no_data")
    
    if pd.isna(price):
        return GateResult(passed=False, reason="no_price")
    
    if price < min_price:
        return GateResult(passed=False, reason=f"price_below_{min_price}")
    
    if avg_dollar_volume < threshold:
        return GateResult(passed=False, reason="low_dollar_volume")
    
    return GateResult(passed=True, reason="ok")


def volatility_gate(
    volatility: float,
    vol_cap: float = 0.60,
    use_sizing: bool = True,
    target_vol: float = 0.20
) -> GateResult:
    """
    Volatility gate that can either block or scale position size.
    
    Args:
        volatility: Annualized trailing volatility
        vol_cap: Maximum allowed volatility (block if exceeded)
        use_sizing: If True, return a sizing scale based on vol; if False, just block
        target_vol: Target volatility for sizing calculation
        
    Returns:
        GateResult with pass/fail, reason, and optional scale
    """
    if pd.isna(volatility):
        return GateResult(passed=False, reason="no_vol_data")
    
    if volatility > vol_cap:
        return GateResult(passed=False, reason=f"vol_exceeds_{vol_cap:.0%}")
    
    if use_sizing:
        # Inverse vol sizing: scale = target_vol / actual_vol, capped at 1.0
        scale = min(1.0, target_vol / volatility) if volatility > 0 else 1.0
        return GateResult(passed=True, reason="ok", scale=scale)
    
    return GateResult(passed=True, reason="ok")


def market_regime_gate(
    benchmark_price: float,
    benchmark_ma: float,
    benchmark_vol: float,
    vol_threshold: float = 0.25,
    reduce_exposure: float = 0.5
) -> GateResult:
    """
    Market regime gate based on benchmark trend and volatility.
    
    Risk-off if:
    - Benchmark is below its moving average, OR
    - Benchmark volatility exceeds threshold
    
    Args:
        benchmark_price: Current benchmark price
        benchmark_ma: Moving average of benchmark
        benchmark_vol: Annualized volatility of benchmark
        vol_threshold: Max benchmark vol before risk-off
        reduce_exposure: Scale factor when in risk-off mode
        
    Returns:
        GateResult with pass/fail, reason, and exposure scale
    """
    if pd.isna(benchmark_price) or pd.isna(benchmark_ma):
        return GateResult(passed=True, reason="no_regime_data", scale=1.0)
    
    is_below_ma = benchmark_price < benchmark_ma
    is_high_vol = benchmark_vol > vol_threshold if not pd.isna(benchmark_vol) else False
    
    if is_below_ma and is_high_vol:
        return GateResult(
            passed=True,  # Don't block, but reduce exposure
            reason="risk_off_trend_and_vol",
            scale=reduce_exposure
        )
    elif is_below_ma:
        return GateResult(
            passed=True,
            reason="risk_off_trend",
            scale=reduce_exposure
        )
    elif is_high_vol:
        return GateResult(
            passed=True,
            reason="risk_off_vol",
            scale=reduce_exposure
        )
    
    return GateResult(passed=True, reason="risk_on")


def buffer_gate(
    current_rank: int,
    top_k: int,
    buffer: int,
    is_currently_held: bool
) -> GateResult:
    """
    Turnover reduction gate using entry/exit buffers.
    
    - Entry: rank <= top_k to enter a new position
    - Exit: rank > top_k + buffer to exit an existing position
    
    Args:
        current_rank: Current rank of the ticker (1 = best)
        top_k: Number of top positions to hold
        buffer: Exit buffer (exit if rank > top_k + buffer)
        is_currently_held: Whether the position is currently held
        
    Returns:
        GateResult indicating whether to hold/enter the position
    """
    if pd.isna(current_rank):
        return GateResult(passed=False, reason="no_rank")
    
    rank = int(current_rank)
    exit_threshold = top_k + buffer
    
    if is_currently_held:
        # For existing positions: exit only if rank > exit_threshold
        if rank <= exit_threshold:
            return GateResult(passed=True, reason="hold_within_buffer")
        else:
            return GateResult(passed=False, reason=f"exit_rank_{rank}_exceeds_{exit_threshold}")
    else:
        # For new positions: enter only if rank <= top_k
        if rank <= top_k:
            return GateResult(passed=True, reason="enter_top_k")
        else:
            return GateResult(passed=False, reason=f"no_entry_rank_{rank}_exceeds_{top_k}")


# =============================================================================
# Gate Evaluation Framework
# =============================================================================

class GateEvaluator:
    """
    Evaluates all gates for tickers on each trading day.
    
    Coordinates gate evaluation and aggregates results.
    """
    
    def __init__(
        self,
        liquidity_threshold: float = 1_000_000,
        min_price: float = 5.0,
        vol_cap: float = 0.60,
        use_vol_sizing: bool = True,
        target_vol: float = 0.20,
        regime_ma_days: int = 200,
        regime_vol_threshold: float = 0.25,
        regime_reduce_exposure: float = 0.5,
        top_k: int = 20,
        buffer: int = 5,
    ):
        """
        Initialize the gate evaluator.
        
        Args:
            liquidity_threshold: Min avg dollar volume
            min_price: Min stock price
            vol_cap: Max stock volatility
            use_vol_sizing: Use inverse vol sizing
            target_vol: Target vol for sizing
            regime_ma_days: Days for benchmark MA
            regime_vol_threshold: Benchmark vol threshold
            regime_reduce_exposure: Scale when risk-off
            top_k: Number of positions to hold
            buffer: Exit buffer for ranks
        """
        self.liquidity_threshold = liquidity_threshold
        self.min_price = min_price
        self.vol_cap = vol_cap
        self.use_vol_sizing = use_vol_sizing
        self.target_vol = target_vol
        self.regime_ma_days = regime_ma_days
        self.regime_vol_threshold = regime_vol_threshold
        self.regime_reduce_exposure = regime_reduce_exposure
        self.top_k = top_k
        self.buffer = buffer
        
        # Gate failure tracking for diagnostics
        self.failure_counts: dict[str, int] = {}
    
    def evaluate_ticker(
        self,
        ticker: str,
        date: pd.Timestamp,
        avg_dollar_volume: float,
        price: float,
        volatility: float,
        rank: Optional[int],
        is_currently_held: bool,
        benchmark_price: float,
        benchmark_ma: float,
        benchmark_vol: float,
    ) -> GateResults:
        """
        Evaluate all gates for a single ticker on a given date.
        
        Args:
            ticker: Ticker symbol
            date: Evaluation date
            avg_dollar_volume: Trailing avg dollar volume
            price: Current price
            volatility: Trailing annualized volatility
            rank: Current rank (None if not ranked)
            is_currently_held: Whether position is currently held
            benchmark_price: Benchmark (e.g., SPY) price
            benchmark_ma: Benchmark moving average
            benchmark_vol: Benchmark volatility
            
        Returns:
            GateResults with all gate evaluations
        """
        results = {}
        
        # 1. Liquidity gate
        results['liquidity'] = liquidity_gate(
            avg_dollar_volume=avg_dollar_volume,
            price=price,
            threshold=self.liquidity_threshold,
            min_price=self.min_price
        )
        
        # 2. Volatility gate
        results['volatility'] = volatility_gate(
            volatility=volatility,
            vol_cap=self.vol_cap,
            use_sizing=self.use_vol_sizing,
            target_vol=self.target_vol
        )
        
        # 3. Market regime gate (global, same for all tickers)
        results['regime'] = market_regime_gate(
            benchmark_price=benchmark_price,
            benchmark_ma=benchmark_ma,
            benchmark_vol=benchmark_vol,
            vol_threshold=self.regime_vol_threshold,
            reduce_exposure=self.regime_reduce_exposure
        )
        
        # 4. Buffer gate (only if ranked)
        if rank is not None:
            results['buffer'] = buffer_gate(
                current_rank=rank,
                top_k=self.top_k,
                buffer=self.buffer,
                is_currently_held=is_currently_held
            )
        
        # Track failures for diagnostics
        gate_results = GateResults(ticker=ticker, date=date, results=results)
        if not gate_results.final_passed:
            for name, r in results.items():
                if not r.passed:
                    key = f"{name}:{r.reason}"
                    self.failure_counts[key] = self.failure_counts.get(key, 0) + 1
        
        return gate_results
    
    def evaluate_universe(
        self,
        date: pd.Timestamp,
        universe: list[str],
        avg_dollar_volumes: pd.Series,
        prices: pd.Series,
        volatilities: pd.Series,
        ranks: pd.Series,
        current_holdings: set[str],
        benchmark_price: float,
        benchmark_ma: float,
        benchmark_vol: float,
    ) -> dict[str, GateResults]:
        """
        Evaluate all gates for all tickers in universe on a given date.
        
        Args:
            date: Evaluation date
            universe: List of tickers in universe
            avg_dollar_volumes: Series of avg dollar volumes by ticker
            prices: Series of prices by ticker
            volatilities: Series of volatilities by ticker
            ranks: Series of ranks by ticker
            current_holdings: Set of currently held tickers
            benchmark_price: Benchmark price
            benchmark_ma: Benchmark MA
            benchmark_vol: Benchmark vol
            
        Returns:
            Dict mapping ticker to GateResults
        """
        results = {}
        
        for ticker in universe:
            adv = avg_dollar_volumes.get(ticker, np.nan)
            price = prices.get(ticker, np.nan)
            vol = volatilities.get(ticker, np.nan)
            rank = ranks.get(ticker, None)
            is_held = ticker in current_holdings
            
            results[ticker] = self.evaluate_ticker(
                ticker=ticker,
                date=date,
                avg_dollar_volume=adv,
                price=price,
                volatility=vol,
                rank=int(rank) if not pd.isna(rank) else None,
                is_currently_held=is_held,
                benchmark_price=benchmark_price,
                benchmark_ma=benchmark_ma,
                benchmark_vol=benchmark_vol,
            )
        
        return results
    
    def get_failure_summary(self) -> pd.DataFrame:
        """
        Get summary of gate failures.
        
        Returns:
            DataFrame with gate:reason as index and count as column
        """
        if not self.failure_counts:
            return pd.DataFrame(columns=['count'])
        
        df = pd.DataFrame.from_dict(
            self.failure_counts, 
            orient='index', 
            columns=['count']
        )
        df.index.name = 'gate:reason'
        return df.sort_values('count', ascending=False)
    
    def reset_failure_counts(self) -> None:
        """Reset failure tracking."""
        self.failure_counts = {}
