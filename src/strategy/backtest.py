"""
Backtest engine for the equity strategy.

Orchestrates the daily simulation loop:
1. Universe selection
2. Feature computation
3. Gate evaluation
4. Ranking
5. Portfolio construction
6. Execution with T+1 fills
7. Performance tracking
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import date

from .config import StrategyConfig
from .data import DataManager
from .features import FeatureComputer
from .universe import UniverseSelector
from .gates import GateEvaluator
from .rank import RankComputer
from .portfolio import PortfolioConstructor, PortfolioTarget
from .execution import ExecutionModel, ExecutionResult


@dataclass
class BacktestState:
    """
    State of the backtest at any point in time.
    
    Attributes:
        date: Current date
        positions: Dict of ticker -> shares
        cash: Available cash
        portfolio_value: Total portfolio value
        weights: Current portfolio weights
    """
    date: pd.Timestamp
    positions: dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
    portfolio_value: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    
    def copy(self) -> 'BacktestState':
        return BacktestState(
            date=self.date,
            positions=self.positions.copy(),
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            weights=self.weights.copy(),
        )


@dataclass
class DailyResult:
    """
    Result for a single day of backtesting.
    
    Attributes:
        date: The date
        portfolio_value: End-of-day portfolio value
        daily_return: Daily return
        positions: Number of positions
        gross_exposure: Total exposure
        turnover: Turnover from previous day
        costs: Trading costs incurred
        cash: Cash balance
    """
    date: pd.Timestamp
    portfolio_value: float
    daily_return: float
    positions: int
    gross_exposure: float
    turnover: float
    costs: float
    cash: float


@dataclass
class BacktestResults:
    """
    Complete results from a backtest run.
    
    Attributes:
        daily_results: List of daily results
        trades: DataFrame of all trades
        config: Configuration used
        gate_failures: Gate failure summary
    """
    daily_results: list[DailyResult]
    trades: pd.DataFrame
    config: StrategyConfig
    gate_failures: pd.DataFrame
    weights_history: Optional[pd.DataFrame] = None
    
    def get_equity_curve(self) -> pd.Series:
        """Get portfolio value over time."""
        return pd.Series(
            {r.date: r.portfolio_value for r in self.daily_results}
        )
    
    def get_returns(self) -> pd.Series:
        """Get daily returns."""
        return pd.Series(
            {r.date: r.daily_return for r in self.daily_results}
        )
    
    def get_turnover(self) -> pd.Series:
        """Get daily turnover."""
        return pd.Series(
            {r.date: r.turnover for r in self.daily_results}
        )
    
    def get_positions_count(self) -> pd.Series:
        """Get number of positions over time."""
        return pd.Series(
            {r.date: r.positions for r in self.daily_results}
        )

    def get_weights_history(self) -> pd.DataFrame:
        """Get historical portfolio weights (dates x tickers)."""
        if self.weights_history is None:
            return pd.DataFrame()
        return self.weights_history.copy()


class Backtester:
    """
    Main backtest engine.
    
    Runs the daily simulation loop and tracks all results.
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the backtester.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.logger = config.get_logger("backtest")
        
        # Initialize components
        self.universe_selector = UniverseSelector(config)
        self.gate_evaluator = GateEvaluator(
            liquidity_threshold=config.liquidity_threshold,
            min_price=config.min_price,
            vol_cap=config.vol_cap,
            use_vol_sizing=config.use_vol_sizing,
            regime_ma_days=config.regime_ma_days,
            regime_vol_threshold=config.regime_vol_threshold,
            top_k=config.top_k,
            buffer=config.buffer,
        )
        self.rank_computer = RankComputer(config)
        self.portfolio_constructor = PortfolioConstructor(config)
        self.execution_model = ExecutionModel(config)
        self.rebalance_frequency = config.rebalance_frequency
        self.rebalance_weekday = config.rebalance_weekday

    def _should_rebalance(self, dt: pd.Timestamp) -> bool:
        """Return True if strategy should generate a new signal on this date."""
        if self.rebalance_frequency == "daily":
            return True
        return dt.weekday() == self.rebalance_weekday

    def _randomize_ranks(self, ranks: pd.DataFrame) -> pd.DataFrame:
        """
        Randomize rank ordering for each day while preserving eligible tickers.

        Used for edge robustness checks. A valid strategy should lose alpha under
        this perturbation.
        """
        randomized = ranks.copy() * np.nan
        rng = np.random.default_rng(self.config.randomize_ranks_seed)

        for dt in ranks.index:
            valid = ranks.loc[dt].dropna().index.to_numpy()
            if len(valid) == 0:
                continue
            shuffled = rng.permutation(valid)
            randomized.loc[dt, shuffled] = np.arange(1, len(valid) + 1, dtype=float)

        return randomized

    def _apply_cash_sweep(
        self,
        cash: float,
        date: pd.Timestamp,
        sweep_returns: pd.Series,
        risk_off_regime: bool,
    ) -> float:
        """
        Apply sweep-asset return to positive idle cash.

        During risk-off regime periods, cash can optionally remain as cash.
        """
        if not self.config.cash_sweep_to_benchmark:
            return cash
        if cash <= 0:
            return cash
        if self.config.cash_sweep_risk_off_to_cash and risk_off_regime:
            return cash
        if date not in sweep_returns.index:
            return cash

        ret = sweep_returns.loc[date]
        if pd.isna(ret):
            return cash
        return cash * (1.0 + float(ret))

    def _is_risk_off_regime(
        self,
        benchmark_price: float,
        benchmark_ma: float,
        benchmark_vol: float,
    ) -> bool:
        """Mirror market regime gate logic for cash sweep behavior."""
        if pd.isna(benchmark_price) or pd.isna(benchmark_ma):
            return False
        is_below_ma = benchmark_price < benchmark_ma
        is_high_vol = (
            benchmark_vol > self.config.regime_vol_threshold
            if not pd.isna(benchmark_vol)
            else False
        )
        return bool(is_below_ma or is_high_vol)
    
    def run(self, data_manager: DataManager) -> BacktestResults:
        """
        Run the backtest.
        
        Args:
            data_manager: DataManager with loaded data
            
        Returns:
            BacktestResults with full simulation results
        """
        self.logger.info(
            f"Starting backtest from {self.config.start_date} to {self.config.end_date}"
        )
        
        # Get data
        close_prices = data_manager.get_close_prices()
        open_prices = data_manager.get_open_prices()
        volumes = data_manager.get_volumes()
        high_prices = data_manager.get_high_prices()
        low_prices = data_manager.get_low_prices()
        benchmark_close = data_manager.get_benchmark_close()
        
        # Initialize feature computer
        feature_computer = FeatureComputer(
            close_prices=close_prices,
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            volume=volumes,
        )
        
        # Compute features
        self.logger.info("Computing features...")
        volatilities = feature_computer.get_volatility(lookback=20)
        avg_dollar_volumes = feature_computer.get_avg_dollar_volume(lookback=20)
        
        # Compute benchmark features
        benchmark_returns = benchmark_close.pct_change()
        benchmark_vol = benchmark_returns.rolling(20).std() * np.sqrt(252)
        benchmark_ma = benchmark_close.rolling(self.config.regime_ma_days).mean()
        stock_returns = close_prices.pct_change()
        sweep_price = benchmark_close
        if self.config.cash_sweep_asset == "tbill":
            try:
                sweep_price = data_manager.get_asset_close(self.config.cash_sweep_tbill_ticker)
            except Exception as e:
                self.logger.warning(
                    "Failed to load cash sweep T-bill proxy %s (%s). Falling back to benchmark.",
                    self.config.cash_sweep_tbill_ticker,
                    e,
                )
                sweep_price = benchmark_close
        sweep_returns = sweep_price.reindex(close_prices.index).ffill().pct_change()
        rolling_betas = None
        if self.config.beta_cap_enabled:
            cov_to_benchmark = stock_returns.rolling(
                self.config.beta_lookback_days
            ).cov(benchmark_returns)
            benchmark_var = benchmark_returns.rolling(self.config.beta_lookback_days).var()
            rolling_betas = cov_to_benchmark.div(benchmark_var, axis=0)
            rolling_betas = rolling_betas.replace([np.inf, -np.inf], np.nan)
        
        # Select universe
        self.logger.info("Selecting universe...")
        universe_df = self.universe_selector.select_universe(
            close_prices, volumes, list(close_prices.columns)
        )
        
        # Compute ranks
        self.logger.info("Computing ranks...")
        ranks = self.rank_computer.compute_daily_ranks(close_prices, universe_df)
        if self.config.randomize_ranks:
            self.logger.warning("Randomizing ranks for robustness stress test.")
            ranks = self._randomize_ranks(ranks)
        
        # Initialize state
        state = BacktestState(
            date=pd.Timestamp(self.config.start_date),
            positions={},
            cash=self.config.initial_capital,
            portfolio_value=self.config.initial_capital,
            weights={},
        )
        
        daily_results = []
        weight_records: list[pd.Series] = []
        trading_dates = close_prices.index.sort_values()
        peak_portfolio_value = self.config.initial_capital
        sector_map = data_manager.get_sector_map()
        
        # Filter to backtest period and ensure enough warmup
        warmup_days = max(
            self.config.momentum_lookback + self.config.momentum_skip,
            self.config.regime_ma_days,
            self.config.lookback_dollar_vol,
        ) + 10  # Extra buffer
        
        start_idx = warmup_days
        if self.config.start_date:
            start_date_ts = pd.Timestamp(self.config.start_date)
            start_idx = max(
                start_idx,
                (trading_dates >= start_date_ts).argmax()
            )
        
        end_idx = len(trading_dates)
        if self.config.end_date:
            end_date_ts = pd.Timestamp(self.config.end_date)
            end_mask = trading_dates <= end_date_ts
            if end_mask.any():
                end_idx = end_mask.sum()
        
        backtest_dates = trading_dates[start_idx:end_idx]
        
        self.logger.info(f"Backtesting {len(backtest_dates)} days...")
        
        pending_target: Optional[PortfolioTarget] = None
        prev_portfolio_value = self.config.initial_capital
        
        for i, dt in enumerate(backtest_dates):
            bm_price = benchmark_close.loc[dt] if dt in benchmark_close.index else np.nan
            bm_ma = benchmark_ma.loc[dt] if dt in benchmark_ma.index else np.nan
            bm_vol = benchmark_vol.loc[dt] if dt in benchmark_vol.index else np.nan

            if i > 0:
                risk_off_regime = self._is_risk_off_regime(
                    benchmark_price=bm_price,
                    benchmark_ma=bm_ma,
                    benchmark_vol=bm_vol,
                )
                state.cash = self._apply_cash_sweep(
                    cash=state.cash,
                    date=dt,
                    sweep_returns=sweep_returns,
                    risk_off_regime=risk_off_regime,
                )

            # Update portfolio value using close prices
            position_value = sum(
                shares * close_prices.loc[dt, ticker]
                for ticker, shares in state.positions.items()
                if ticker in close_prices.columns and not pd.isna(close_prices.loc[dt, ticker])
            )
            state.portfolio_value = position_value + state.cash
            state.date = dt
            peak_portfolio_value = max(peak_portfolio_value, state.portfolio_value)
            current_drawdown = (
                state.portfolio_value / peak_portfolio_value - 1.0
                if peak_portfolio_value > 0 else 0.0
            )
            
            # Calculate daily return
            daily_return = (state.portfolio_value / prev_portfolio_value - 1) if prev_portfolio_value > 0 else 0
            
            # Execute pending trades from yesterday's signal (T+1)
            costs = 0.0
            turnover = 0.0
            
            if pending_target is not None:
                slippage_multiplier = 1.0
                if (
                    self.config.bad_fills_enabled
                    and dt in benchmark_vol.index
                    and not pd.isna(benchmark_vol.loc[dt])
                    and benchmark_vol.loc[dt] >= self.config.bad_fills_vol_threshold
                ):
                    slippage_multiplier = self.config.bad_fills_multiplier

                # Execute at today's open
                new_positions, exec_result = self.execution_model.execute_rebalance(
                    signal_date=pending_target.date,
                    execution_date=dt,
                    current_positions=state.positions,
                    target_weights=pending_target.weights,
                    portfolio_value=state.portfolio_value,
                    open_prices=open_prices.loc[dt] if dt in open_prices.index else pd.Series(),
                    slippage_multiplier=slippage_multiplier,
                )
                
                state.positions = new_positions
                costs = exec_result.total_cost
                # Trade principal must move through cash; slippage is already
                # embedded in execution prices and therefore in net cash flow.
                state.cash += exec_result.net_cash_flow
                
                # Compute turnover
                turnover = self.portfolio_constructor.compute_turnover(
                    state.weights, pending_target.weights
                )
                state.weights = pending_target.weights
            
            # Generate signal for tomorrow (use today's close)
            universe = universe_df.loc[dt, 'universe'] if dt in universe_df.index else []
            
            if len(universe) > 0 and self._should_rebalance(dt):
                # Get features for today
                prices_today = close_prices.loc[dt]
                vols_today = volatilities.loc[dt] if dt in volatilities.index else pd.Series()
                advs_today = avg_dollar_volumes.loc[dt] if dt in avg_dollar_volumes.index else pd.Series()
                ranks_today = ranks.loc[dt] if dt in ranks.index else pd.Series()
                betas_today = (
                    rolling_betas.loc[dt]
                    if rolling_betas is not None and dt in rolling_betas.index
                    else pd.Series(dtype=float)
                )
                
                # Evaluate gates
                gate_results = self.gate_evaluator.evaluate_universe(
                    date=dt,
                    universe=universe,
                    avg_dollar_volumes=advs_today,
                    prices=prices_today,
                    volatilities=vols_today,
                    ranks=ranks_today,
                    current_holdings=set(state.positions.keys()),
                    benchmark_price=bm_price,
                    benchmark_ma=bm_ma,
                    benchmark_vol=bm_vol,
                )
                
                # Get regime scale
                regime_scale = 1.0
                for gr in gate_results.values():
                    if 'regime' in gr.results:
                        regime_scale = gr.results['regime'].scale
                        break
                
                # Construct target portfolio
                pending_target = self.portfolio_constructor.construct_portfolio(
                    date=dt,
                    ranks=ranks_today,
                    gate_results=gate_results,
                    current_holdings=set(state.positions.keys()),
                    volatilities=vols_today,
                    regime_scale=regime_scale,
                    betas=betas_today,
                    sector_map=sector_map,
                    current_drawdown=current_drawdown,
                )
            else:
                pending_target = None

            weight_records.append(
                pd.Series(state.weights, name=dt, dtype=float)
            )
            
            # Record daily result
            gross_exposure = sum(abs(w) for w in state.weights.values())
            
            daily_results.append(DailyResult(
                date=dt,
                portfolio_value=state.portfolio_value,
                daily_return=daily_return,
                positions=len(state.positions),
                gross_exposure=gross_exposure,
                turnover=turnover,
                costs=costs,
                cash=state.cash,
            ))
            
            prev_portfolio_value = state.portfolio_value
        
        self.logger.info(f"Backtest complete. Final portfolio value: ${state.portfolio_value:,.2f}")

        weights_history = pd.DataFrame(weight_records).fillna(0.0)
        if not weights_history.empty:
            weights_history.index = pd.to_datetime(weights_history.index)
        
        return BacktestResults(
            daily_results=daily_results,
            trades=self.execution_model.get_trades_ledger(),
            config=self.config,
            gate_failures=self.gate_evaluator.get_failure_summary(),
            weights_history=weights_history,
        )
