"""
Central configuration dataclass for the equity strategy.

All tunable parameters are defined here for reproducibility and easy experimentation.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Optional
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
        sector_cap_enabled: If True, cap aggregate exposure by sector
        sector_cap: Maximum weight allowed per sector
        sector_map_file: Optional CSV/JSON mapping file (ticker->sector)
        beta_cap_enabled: If True, cap portfolio beta exposure
        beta_cap: Maximum allowed portfolio beta (long-only weighted beta)
        beta_lookback_days: Lookback window for rolling beta estimation
        drawdown_scaler_enabled: If True, scale exposure down during deep drawdowns
        drawdown_scaler_start: Drawdown level where de-risking starts (negative)
        drawdown_scaler_full: Drawdown level where minimum exposure is reached (negative)
        drawdown_scaler_min: Minimum exposure scale in deep drawdown
        
        # Execution settings
        slippage_bps: Fixed slippage in basis points per trade
        use_atr_slippage: If True, use ATR-based slippage model
        atr_slippage_mult: Multiplier for ATR slippage (fraction of ATR)
        execution_delay: Days between signal and execution (1 = t+1 open)
        rebalance_frequency: Rebalance cadence ('daily', 'weekly', or 'custom')
        rebalance_weekday: Weekday for weekly rebalance (0=Mon ... 4=Fri)
        rebalance_weekdays: Weekdays for custom rebalancing (e.g., Mon/Wed/Fri = [0,2,4])
        evaluate_exits_daily: If True, evaluate exits on non-rebalance days
        entries_on_rebalance_only: If True, block new entries on non-rebalance days
        bad_fills_enabled: If True, increase slippage on high-volatility days
        bad_fills_vol_threshold: Annualized benchmark vol threshold for bad fills
        bad_fills_multiplier: Slippage multiplier applied on bad-fill days
        randomize_ranks: If True, randomize cross-sectional ranks (edge sanity test)
        randomize_ranks_seed: Seed for rank randomization
        cash_sweep_to_benchmark: If True, apply sweep-asset return to positive idle cash
        cash_sweep_asset: Sweep asset for positive idle cash ('benchmark' or 'tbill')
        cash_sweep_tbill_ticker: T-bill proxy ticker used when cash_sweep_asset='tbill'
        cash_sweep_risk_off_to_cash: If True, keep idle cash uninvested during risk-off regime
        
        min_trade_weight_change: Minimum absolute weight delta required to trade
            (no-trade band / churn filter)
        beta_targeting_enabled: If True, scale stock sleeve toward target beta regime
        beta_target_risk_on: Target portfolio beta in risk-on regime
        beta_target_neutral: Target portfolio beta in neutral regime
        beta_target_risk_off: Target portfolio beta in risk-off regime
        beta_target_hysteresis: Minimum beta error before adjusting sleeve
        beta_target_step_limit: Max stock-sleeve exposure change per day
        beta_target_stock_min: Minimum stock sleeve exposure allowed
        beta_target_stock_max: Maximum stock sleeve exposure allowed

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
    momentum_lookback: int = 120
    momentum_skip: int = 5
    
    # Portfolio settings
    top_k: int = 10
    buffer: int = 25
    weight_scheme: Literal["equal", "inverse_vol"] = "equal"
    max_weight: float = 0.10  # 10% max single name
    max_gross_exposure: float = 1.0
    sector_cap_enabled: bool = False
    sector_cap: float = 0.35
    sector_map_file: Optional[Path] = None
    beta_cap_enabled: bool = False
    beta_cap: float = 1.20
    beta_lookback_days: int = 63
    drawdown_scaler_enabled: bool = False
    drawdown_scaler_start: float = -0.08
    drawdown_scaler_full: float = -0.20
    drawdown_scaler_min: float = 0.35
    
    # Execution settings
    slippage_bps: float = 10.0
    use_atr_slippage: bool = False
    atr_slippage_mult: float = 0.1  # 10% of daily ATR
    execution_delay: int = 1  # t+1 execution
    rebalance_frequency: Literal["daily", "weekly", "custom"] = "daily"
    rebalance_weekday: int = 0  # Monday
    rebalance_weekdays: tuple[int, ...] = (0, 2, 4)
    evaluate_exits_daily: bool = False
    entries_on_rebalance_only: bool = False
    bad_fills_enabled: bool = False
    bad_fills_vol_threshold: float = 0.35
    bad_fills_multiplier: float = 2.0
    randomize_ranks: bool = False
    randomize_ranks_seed: int = 42
    cash_sweep_to_benchmark: bool = True
    cash_sweep_asset: Literal["benchmark", "tbill"] = "benchmark"
    cash_sweep_tbill_ticker: str = "BIL"
    cash_sweep_risk_off_to_cash: bool = True
    min_trade_weight_change: float = 0.0
    beta_targeting_enabled: bool = False
    beta_target_risk_on: float = 0.90
    beta_target_neutral: float = 0.70
    beta_target_risk_off: float = 0.25
    beta_target_hysteresis: float = 0.05
    beta_target_step_limit: float = 0.05
    beta_target_stock_min: float = 0.0
    beta_target_stock_max: float = 1.0
    
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
        if isinstance(self.sector_map_file, str):
            self.sector_map_file = Path(self.sector_map_file)
        if isinstance(self.rebalance_weekdays, list):
            self.rebalance_weekdays = tuple(int(x) for x in self.rebalance_weekdays)
        
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
        assert 0 < self.sector_cap <= 1, "sector_cap must be in (0, 1]"
        assert self.beta_cap > 0, "beta_cap must be positive"
        assert self.beta_lookback_days >= 20, "beta_lookback_days should be at least 20"
        assert self.drawdown_scaler_full < 0, "drawdown_scaler_full must be negative"
        assert self.drawdown_scaler_start < 0, "drawdown_scaler_start must be negative"
        assert self.drawdown_scaler_full < self.drawdown_scaler_start, \
            "drawdown_scaler_full must be lower than drawdown_scaler_start"
        assert 0 < self.drawdown_scaler_min <= 1, "drawdown_scaler_min must be in (0, 1]"
        assert self.slippage_bps >= 0, "slippage_bps must be non-negative"
        assert self.start_date < self.end_date, "start_date must be before end_date"
        assert self.execution_delay >= 1, "execution_delay must be at least 1 (t+1)"
        assert self.rebalance_frequency in {"daily", "weekly", "custom"}, \
            "rebalance_frequency must be 'daily', 'weekly', or 'custom'"
        assert 0 <= self.rebalance_weekday <= 4, "rebalance_weekday must be in [0, 4]"
        assert len(self.rebalance_weekdays) > 0, "rebalance_weekdays must not be empty"
        assert all(0 <= d <= 4 for d in self.rebalance_weekdays), \
            "rebalance_weekdays entries must be in [0, 4]"
        assert self.bad_fills_vol_threshold >= 0, "bad_fills_vol_threshold must be non-negative"
        assert self.bad_fills_multiplier >= 1, "bad_fills_multiplier must be >= 1"
        assert self.cash_sweep_asset in {"benchmark", "tbill"}, \
            "cash_sweep_asset must be 'benchmark' or 'tbill'"
        assert self.min_trade_weight_change >= 0, \
            "min_trade_weight_change must be non-negative"
        assert 0 <= self.beta_target_risk_on <= 1.5, "beta_target_risk_on must be in [0, 1.5]"
        assert 0 <= self.beta_target_neutral <= 1.5, "beta_target_neutral must be in [0, 1.5]"
        assert 0 <= self.beta_target_risk_off <= 1.5, "beta_target_risk_off must be in [0, 1.5]"
        assert self.beta_target_hysteresis >= 0, "beta_target_hysteresis must be non-negative"
        assert self.beta_target_step_limit >= 0, "beta_target_step_limit must be non-negative"
        assert 0 <= self.beta_target_stock_min <= 1, \
            "beta_target_stock_min must be in [0, 1]"
        assert 0 <= self.beta_target_stock_max <= 1, \
            "beta_target_stock_max must be in [0, 1]"
        assert self.beta_target_stock_min <= self.beta_target_stock_max, \
            "beta_target_stock_min must be <= beta_target_stock_max"
    
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
            "sector_cap_enabled": self.sector_cap_enabled,
            "sector_cap": self.sector_cap,
            "sector_map_file": str(self.sector_map_file) if self.sector_map_file else None,
            "beta_cap_enabled": self.beta_cap_enabled,
            "beta_cap": self.beta_cap,
            "beta_lookback_days": self.beta_lookback_days,
            "drawdown_scaler_enabled": self.drawdown_scaler_enabled,
            "drawdown_scaler_start": self.drawdown_scaler_start,
            "drawdown_scaler_full": self.drawdown_scaler_full,
            "drawdown_scaler_min": self.drawdown_scaler_min,
            "slippage_bps": self.slippage_bps,
            "use_atr_slippage": self.use_atr_slippage,
            "atr_slippage_mult": self.atr_slippage_mult,
            "execution_delay": self.execution_delay,
            "rebalance_frequency": self.rebalance_frequency,
            "rebalance_weekday": self.rebalance_weekday,
            "rebalance_weekdays": list(self.rebalance_weekdays),
            "evaluate_exits_daily": self.evaluate_exits_daily,
            "entries_on_rebalance_only": self.entries_on_rebalance_only,
            "bad_fills_enabled": self.bad_fills_enabled,
            "bad_fills_vol_threshold": self.bad_fills_vol_threshold,
            "bad_fills_multiplier": self.bad_fills_multiplier,
            "randomize_ranks": self.randomize_ranks,
            "randomize_ranks_seed": self.randomize_ranks_seed,
            "cash_sweep_to_benchmark": self.cash_sweep_to_benchmark,
            "cash_sweep_asset": self.cash_sweep_asset,
            "cash_sweep_tbill_ticker": self.cash_sweep_tbill_ticker,
            "cash_sweep_risk_off_to_cash": self.cash_sweep_risk_off_to_cash,
            "min_trade_weight_change": self.min_trade_weight_change,
            "beta_targeting_enabled": self.beta_targeting_enabled,
            "beta_target_risk_on": self.beta_target_risk_on,
            "beta_target_neutral": self.beta_target_neutral,
            "beta_target_risk_off": self.beta_target_risk_off,
            "beta_target_hysteresis": self.beta_target_hysteresis,
            "beta_target_step_limit": self.beta_target_step_limit,
            "beta_target_stock_min": self.beta_target_stock_min,
            "beta_target_stock_max": self.beta_target_stock_max,
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
