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
from .analysis import run_stress_tests, run_parameter_sensitivity
from .validation import run_validation_suite

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
    # Analysis
    "run_stress_tests",
    "run_parameter_sensitivity",
    "run_validation_suite",
]
