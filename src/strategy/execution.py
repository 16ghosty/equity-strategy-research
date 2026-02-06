"""
Execution module for the equity strategy.

Implements trade execution with:
- T+1 open fills (signals at day t close → execute at day t+1 open)
- Fixed or ATR-based slippage
- Trades ledger generation
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .config import StrategyConfig


@dataclass
class Trade:
    """
    A single trade execution.
    
    Attributes:
        date: Execution date
        ticker: Ticker symbol
        side: 'BUY' or 'SELL'
        shares: Number of shares
        price: Execution price (including slippage)
        notional: Trade value (shares × price)
        slippage_cost: Cost due to slippage
        signal_date: Date when signal was generated
    """
    date: pd.Timestamp
    ticker: str
    side: str  # 'BUY' or 'SELL'
    shares: float
    price: float
    notional: float
    slippage_cost: float
    signal_date: pd.Timestamp
    
    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            'date': self.date,
            'ticker': self.ticker,
            'side': self.side,
            'shares': self.shares,
            'price': self.price,
            'notional': self.notional,
            'slippage_cost': self.slippage_cost,
            'signal_date': self.signal_date,
        }


@dataclass
class ExecutionResult:
    """
    Result of executing trades for a day.
    
    Attributes:
        date: Execution date
        trades: List of executed trades
        total_cost: Total slippage cost
        gross_traded: Total notional traded
    """
    date: pd.Timestamp
    trades: list[Trade]
    total_cost: float
    gross_traded: float
    
    @property
    def num_trades(self) -> int:
        return len(self.trades)


class ExecutionModel:
    """
    Executes portfolio rebalances with realistic fills and costs.
    
    Features:
    - T+1 open fills (configurable delay)
    - Fixed slippage in basis points
    - Optional ATR-based slippage
    - Tracks all trades for analysis
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the execution model.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.slippage_bps = config.slippage_bps
        self.use_atr_slippage = config.use_atr_slippage
        self.atr_slippage_mult = config.atr_slippage_mult
        self.execution_delay = config.execution_delay
        self.logger = config.get_logger("execution")
        
        # Track all trades
        self.trades_ledger: list[Trade] = []
    
    def compute_slippage(
        self,
        price: float,
        side: str,
        atr: Optional[float] = None
    ) -> float:
        """
        Compute slippage cost for a trade.
        
        For buys, slippage increases execution price.
        For sells, slippage decreases execution price.
        
        Args:
            price: Base price (open price)
            side: 'BUY' or 'SELL'
            atr: Optional ATR for ATR-based slippage
            
        Returns:
            Slippage as absolute price adjustment
        """
        if self.use_atr_slippage and atr is not None and not pd.isna(atr):
            slippage = atr * self.atr_slippage_mult
        else:
            slippage = price * (self.slippage_bps / 10000)
        
        return slippage
    
    def get_execution_price(
        self,
        open_price: float,
        side: str,
        atr: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Get execution price including slippage.
        
        Args:
            open_price: Open price on execution day
            side: 'BUY' or 'SELL'
            atr: Optional ATR for slippage calculation
            
        Returns:
            Tuple of (execution_price, slippage_cost_per_share)
        """
        slippage = self.compute_slippage(open_price, side, atr)
        
        if side == 'BUY':
            execution_price = open_price + slippage
        else:
            execution_price = open_price - slippage
        
        return execution_price, slippage
    
    def execute_rebalance(
        self,
        signal_date: pd.Timestamp,
        execution_date: pd.Timestamp,
        current_positions: dict[str, float],  # ticker -> shares
        target_weights: dict[str, float],
        portfolio_value: float,
        open_prices: pd.Series,
        atrs: Optional[pd.Series] = None,
    ) -> tuple[dict[str, float], ExecutionResult]:
        """
        Execute a portfolio rebalance.
        
        Args:
            signal_date: Date when signals were generated
            execution_date: Date of execution (t+1)
            current_positions: Current positions in shares
            target_weights: Target portfolio weights
            portfolio_value: Current portfolio value for sizing
            open_prices: Open prices on execution date
            atrs: Optional ATRs for slippage
            
        Returns:
            Tuple of (new_positions, execution_result)
        """
        trades = []
        new_positions = current_positions.copy()
        
        # Calculate target shares for each ticker
        all_tickers = set(current_positions.keys()) | set(target_weights.keys())
        
        for ticker in all_tickers:
            current_shares = current_positions.get(ticker, 0)
            target_weight = target_weights.get(ticker, 0)
            
            # Get open price
            open_price = open_prices.get(ticker, np.nan)
            if pd.isna(open_price) or open_price <= 0:
                self.logger.warning(f"No valid open price for {ticker} on {execution_date}")
                continue
            
            # Calculate target shares
            target_value = portfolio_value * target_weight
            target_shares = target_value / open_price
            
            # Calculate trade
            shares_delta = target_shares - current_shares
            
            if abs(shares_delta) < 0.01:  # Skip tiny trades
                continue
            
            # Determine side
            side = 'BUY' if shares_delta > 0 else 'SELL'
            trade_shares = abs(shares_delta)
            
            # Get execution price with slippage
            atr = atrs.get(ticker) if atrs is not None else None
            exec_price, slippage_per_share = self.get_execution_price(
                open_price, side, atr
            )
            
            # Calculate trade value and slippage cost
            notional = trade_shares * exec_price
            slippage_cost = trade_shares * slippage_per_share
            
            # Create trade
            trade = Trade(
                date=execution_date,
                ticker=ticker,
                side=side,
                shares=trade_shares,
                price=exec_price,
                notional=notional,
                slippage_cost=slippage_cost,
                signal_date=signal_date,
            )
            trades.append(trade)
            self.trades_ledger.append(trade)
            
            # Update positions
            new_positions[ticker] = target_shares
        
        # Clean up zero positions
        new_positions = {t: s for t, s in new_positions.items() if abs(s) > 0.01}
        
        # Create execution result
        total_cost = sum(t.slippage_cost for t in trades)
        gross_traded = sum(t.notional for t in trades)
        
        result = ExecutionResult(
            date=execution_date,
            trades=trades,
            total_cost=total_cost,
            gross_traded=gross_traded,
        )
        
        return new_positions, result
    
    def get_trades_ledger(self) -> pd.DataFrame:
        """
        Get all trades as a DataFrame.
        
        Returns:
            DataFrame with all executed trades
        """
        if not self.trades_ledger:
            return pd.DataFrame(columns=[
                'date', 'ticker', 'side', 'shares', 'price', 
                'notional', 'slippage_cost', 'signal_date'
            ])
        
        return pd.DataFrame([t.to_dict() for t in self.trades_ledger])
    
    def reset_ledger(self) -> None:
        """Clear the trades ledger."""
        self.trades_ledger = []
    
    def get_total_costs(self) -> float:
        """Get total slippage costs incurred."""
        return sum(t.slippage_cost for t in self.trades_ledger)
    
    def get_total_traded(self) -> float:
        """Get total notional traded."""
        return sum(t.notional for t in self.trades_ledger)


def validate_execution_timing(
    signal_dates: pd.DatetimeIndex,
    execution_dates: pd.DatetimeIndex,
    trading_dates: pd.DatetimeIndex,
    delay: int = 1
) -> tuple[bool, list[str]]:
    """
    Validate that execution timing is correct (no look-ahead).
    
    Args:
        signal_dates: Dates when signals were generated
        execution_dates: Dates when trades were executed
        trading_dates: All available trading dates
        delay: Expected delay between signal and execution
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    trading_dates_list = list(trading_dates)
    
    for signal_date, exec_date in zip(signal_dates, execution_dates):
        # Find position of signal date in trading dates
        if signal_date not in trading_dates_list:
            errors.append(f"Signal date {signal_date} not in trading dates")
            continue
        
        signal_idx = trading_dates_list.index(signal_date)
        expected_exec_idx = signal_idx + delay
        
        if expected_exec_idx >= len(trading_dates_list):
            continue  # End of data
        
        expected_exec_date = trading_dates_list[expected_exec_idx]
        
        if exec_date < expected_exec_date:
            errors.append(
                f"Execution on {exec_date} is before expected {expected_exec_date} "
                f"(signal on {signal_date})"
            )
    
    return len(errors) == 0, errors
