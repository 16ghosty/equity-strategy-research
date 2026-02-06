"""
Equity Strategy Package

A reproducible Python research repository for a daily gated,
ranked-allocation equity strategy.
"""

from .config import StrategyConfig
from .data import DataManager, YFinanceProvider, load_tickers
from .features import FeatureComputer
from .universe import UniverseSelector
from .gates import GateEvaluator, GateResult, GateResults
from .rank import RankComputer
from .portfolio import PortfolioConstructor, PortfolioTarget
from .execution import ExecutionModel
from .backtest import Backtester, BacktestResults
from .metrics import compute_metrics, PerformanceMetrics
from .reporting import generate_full_report

__version__ = "0.1.0"

__all__ = [
    # Config
    "StrategyConfig",
    # Data
    "DataManager",
    "YFinanceProvider",
    "load_tickers",
    # Features
    "FeatureComputer",
    # Universe
    "UniverseSelector",
    # Gates
    "GateEvaluator",
    "GateResult",
    "GateResults",
    # Ranking
    "RankComputer",
    # Portfolio
    "PortfolioConstructor",
    "PortfolioTarget",
    # Execution
    "ExecutionModel",
    # Backtest
    "Backtester",
    "BacktestResults",
    # Metrics
    "compute_metrics",
    "PerformanceMetrics",
    # Reporting
    "generate_full_report",
]
