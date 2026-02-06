"""
Central configuration dataclass for the equity strategy.

All tunable parameters are defined here for reproducibility and easy experimentation.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal
import logging


@dataclass
class StrategyConfig:
    """
    Central configuration for the daily gated ranked-allocation strategy.
    
    Attributes:
        # Data settings
        ticker_file: Path to file containing candidate tickers (one per line)
        benchmark: Benchmark ticker for regime gate (default: SPY)
        start_date: Backtest start date
        end_date: Backtest end date
        data_cache_dir: Directory for cached OHLCV data
        
        # Universe settings
        universe_size: Number of stocks to select (top N by dollar volume)
        lookback_dollar_vol: Days for trailing dollar volume calculation
        universe_rebalance_freq: How often to recompute universe ('monthly')
        
        # Gate thresholds
        liquidity_threshold: Minimum average dollar volume (20-day)
        min_price: Minimum stock price filter
        vol_cap: Maximum annualized volatility (as decimal, e.g., 0.6 = 60%)
        use_vol_sizing: If True, scale positions by inverse vol instead of blocking
        regime_ma_days: Moving average days for regime gate
        regime_vol_threshold: Regime gate vol threshold (annualized)
        
        # Ranking settings
        momentum_lookback: Days for momentum calculation
        momentum_skip: Recent days to skip (avoid short-term reversal)
        
        # Portfolio settings
        top_k: Number of positions to hold
        buffer: Exit buffer (exit if rank > top_k + buffer)
        weight_scheme: 'equal' or 'inverse_vol'
        max_weight: Maximum single-name weight
        max_gross_exposure: Maximum gross exposure (1.0 for long-only)
        
        # Execution settings
        slippage_bps: Fixed slippage in basis points per trade
        use_atr_slippage: If True, use ATR-based slippage model
        atr_slippage_mult: Multiplier for ATR slippage (fraction of ATR)
        execution_delay: Days between signal and execution (1 = t+1 open)
        
        # Backtest settings
        initial_capital: Starting portfolio value
        random_seed: Seed for reproducibility
        
        # Logging
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    # Data settings
    ticker_file: Path = field(default_factory=lambda: Path("data/tickers.txt"))
    benchmark: str = "SPY"
    start_date: date = field(default_factory=lambda: date(2019, 1, 1))
    end_date: date = field(default_factory=lambda: date(2024, 12, 31))
    data_cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    
    # Universe settings
    universe_size: int = 100
    lookback_dollar_vol: int = 30
    universe_rebalance_freq: Literal["monthly"] = "monthly"
    
    # Gate thresholds
    liquidity_threshold: float = 1_000_000  # $1M avg daily dollar volume
    min_price: float = 5.0
    vol_cap: float = 0.60  # 60% annualized vol
    use_vol_sizing: bool = True  # Scale by inverse vol instead of blocking
    regime_ma_days: int = 200
    regime_vol_threshold: float = 0.25  # 25% annualized vol threshold for regime
    
    # Ranking settings
    momentum_lookback: int = 60
    momentum_skip: int = 5
    
    # Portfolio settings
    top_k: int = 20
    buffer: int = 5
    weight_scheme: Literal["equal", "inverse_vol"] = "equal"
    max_weight: float = 0.10  # 10% max single name
    max_gross_exposure: float = 1.0
    
    # Execution settings
    slippage_bps: float = 10.0
    use_atr_slippage: bool = False
    atr_slippage_mult: float = 0.1  # 10% of daily ATR
    execution_delay: int = 1  # t+1 execution
    
    # Backtest settings
    initial_capital: float = 1_000_000.0
    random_seed: int = 42
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string paths to Path objects
        if isinstance(self.ticker_file, str):
            self.ticker_file = Path(self.ticker_file)
        if isinstance(self.data_cache_dir, str):
            self.data_cache_dir = Path(self.data_cache_dir)
        
        # Convert string dates to date objects
        if isinstance(self.start_date, str):
            self.start_date = date.fromisoformat(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = date.fromisoformat(self.end_date)
        
        # Validate constraints
        assert self.universe_size > 0, "universe_size must be positive"
        assert self.top_k > 0, "top_k must be positive"
        assert self.top_k <= self.universe_size, "top_k must be <= universe_size"
        assert self.buffer >= 0, "buffer must be non-negative"
        assert 0 < self.max_weight <= 1, "max_weight must be in (0, 1]"
        assert 0 < self.max_gross_exposure <= 1, "max_gross_exposure must be in (0, 1]"
        assert self.slippage_bps >= 0, "slippage_bps must be non-negative"
        assert self.start_date < self.end_date, "start_date must be before end_date"
        assert self.execution_delay >= 1, "execution_delay must be at least 1 (t+1)"
    
    def get_logger(self, name: str) -> logging.Logger:
        """Create a logger with the configured level."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(getattr(logging, self.log_level.upper()))
        return logger
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "ticker_file": str(self.ticker_file),
            "benchmark": self.benchmark,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "data_cache_dir": str(self.data_cache_dir),
            "universe_size": self.universe_size,
            "lookback_dollar_vol": self.lookback_dollar_vol,
            "universe_rebalance_freq": self.universe_rebalance_freq,
            "liquidity_threshold": self.liquidity_threshold,
            "min_price": self.min_price,
            "vol_cap": self.vol_cap,
            "use_vol_sizing": self.use_vol_sizing,
            "regime_ma_days": self.regime_ma_days,
            "regime_vol_threshold": self.regime_vol_threshold,
            "momentum_lookback": self.momentum_lookback,
            "momentum_skip": self.momentum_skip,
            "top_k": self.top_k,
            "buffer": self.buffer,
            "weight_scheme": self.weight_scheme,
            "max_weight": self.max_weight,
            "max_gross_exposure": self.max_gross_exposure,
            "slippage_bps": self.slippage_bps,
            "use_atr_slippage": self.use_atr_slippage,
            "atr_slippage_mult": self.atr_slippage_mult,
            "execution_delay": self.execution_delay,
            "initial_capital": self.initial_capital,
            "random_seed": self.random_seed,
            "log_level": self.log_level,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        """Create config from dictionary."""
        return cls(**d)


# Default configuration instance
DEFAULT_CONFIG = StrategyConfig()
