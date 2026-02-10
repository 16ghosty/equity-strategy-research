#!/usr/bin/env python3
"""
Stress-test and parameter-sensitivity runner for the equity strategy.

Examples:
    python -m strategy.analysis --mode stress
    python -m strategy.analysis --mode stress --stress-scenario double_slippage
    python -m strategy.analysis --mode sensitivity
    python -m strategy.analysis --mode all --all-stock-charts
"""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .backtest import BacktestResults, Backtester, DailyResult
from .config import StrategyConfig
from .data import DataManager, YFinanceProvider
from .metrics import PerformanceMetrics, compute_metrics
from .reporting import generate_full_report
from .wandb_utils import (
    WandbSettings,
    finish_run,
    init_wandb_run,
    is_wandb_available,
    log_artifact_dir,
    log_dataframe_table,
    log_inline_images,
    log_metrics,
)


STRESS_SCENARIOS = (
    "baseline",
    "double_slippage",
    "bad_fills",
    "weekly_rebalance",
    "remove_top10_best_trades",
    "randomize_ranks",
)

SENSITIVITY_GRID = {
    "top_k": [10, 20, 30],
    "momentum_lookback": [20, 60, 120],
    "buffer": [5, 10, 15],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stress tests and parameter sensitivity analysis"
    )
    parser.add_argument("--start", "-s", type=str, default="2020-01-01")
    parser.add_argument("--end", "-e", type=str, default=None)
    parser.add_argument("--tickers", "-t", type=str, default="data/tickers.txt")
    parser.add_argument("--output", "-o", type=str, default="reports/output/analysis")
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--mode", choices=["stress", "sensitivity", "all"], default="all")
    parser.add_argument(
        "--stress-scenario",
        choices=STRESS_SCENARIOS,
        default="baseline",
        help="Run one stress scenario when mode=stress",
    )
    parser.add_argument(
        "--all-stress-scenarios",
        action="store_true",
        help="Run all stress scenarios when mode=stress",
    )
    parser.add_argument(
        "--all-stock-charts",
        action="store_true",
        help="Alias for --stock-charts-mode all (kept for compatibility)",
    )
    parser.add_argument(
        "--stock-charts-mode",
        choices=["none", "top", "all"],
        default="none",
        help="Stock chart mode for generated reports",
    )
    parser.add_argument(
        "--stock-charts-all-runs",
        action="store_true",
        help="Generate stock charts for every scenario run (can be slow)",
    )
    parser.add_argument(
        "--stock-chart-top-n",
        type=int,
        default=10,
        help="Number of stock charts when stock-charts-mode=top",
    )
    parser.add_argument(
        "--stock-chart-max",
        type=int,
        default=None,
        help="Optional hard cap on generated stock charts",
    )
    parser.add_argument(
        "--stock-chart-chunk-size",
        type=int,
        default=25,
        help="Number of stock charts per chunk",
    )
    parser.add_argument(
        "--stock-chart-chunk-index",
        type=int,
        default=0,
        help="0-based chunk index for stock charts",
    )
    parser.add_argument(
        "--stock-marker-limit",
        type=int,
        default=400,
        help="Maximum buy/sell markers per side on each stock chart",
    )
    parser.add_argument(
        "--include-trade-stats",
        action="store_true",
        help="Include detailed trade stats table in each report (can be slow)",
    )
    parser.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=True,
        help="Enable Weights & Biases logging (default: enabled)",
    )
    parser.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="equity-strategy",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team (optional)",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
        help="Weights & Biases mode",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="analysis",
        help="Weights & Biases group name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases summary run name",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        default="analysis,stress,sensitivity",
        help="Comma-separated Weights & Biases tags",
    )
    parser.add_argument(
        "--wandb-log-artifacts",
        dest="wandb_log_artifacts",
        action="store_true",
        default=True,
        help="Upload run output directories as Weights & Biases artifacts (default: enabled)",
    )
    parser.add_argument(
        "--no-wandb-log-artifacts",
        dest="wandb_log_artifacts",
        action="store_false",
        help="Disable Weights & Biases artifact uploads",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> StrategyConfig:
    if args.config:
        with open(args.config) as f:
            return StrategyConfig.from_dict(json.load(f))

    return StrategyConfig(
        ticker_file=Path(args.tickers),
        start_date=args.start,
        end_date=args.end if args.end else date.today().isoformat(),
        initial_capital=args.capital,
        log_level="DEBUG" if args.verbose else "INFO",
    )


def _override_config(base: StrategyConfig, **overrides) -> StrategyConfig:
    d = base.to_dict()
    d.update(overrides)
    return StrategyConfig.from_dict(d)


def _metrics_row(name: str, metrics: PerformanceMetrics) -> dict:
    return {
        "scenario": name,
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "max_drawdown": metrics.max_drawdown,
        "annual_volatility": metrics.annual_volatility,
        "daily_turnover": metrics.daily_turnover,
        "annual_turnover": metrics.annual_turnover,
        "avg_positions": metrics.avg_positions,
        "avg_holding_period": metrics.avg_holding_period,
        "pct_days_invested": metrics.pct_days_invested,
        "total_costs": metrics.total_costs,
        "cost_ratio": metrics.cost_ratio,
        "total_return": metrics.total_return,
        "num_trading_days": metrics.num_trading_days,
        "var_95": metrics.var_95,
        "var_99": metrics.var_99,
        "es_95": metrics.es_95,
        "es_99": metrics.es_99,
        "cdar_95": metrics.cdar_95,
        "skewness": metrics.skewness,
        "kurtosis": metrics.kurtosis,
        "worst_1d_return": metrics.worst_1d_return,
        "worst_5d_return": metrics.worst_5d_return,
        "tail_ratio": metrics.tail_ratio,
    }


def _numeric_metrics(metrics: PerformanceMetrics) -> dict[str, float]:
    """Numeric metrics payload for wandb logging."""
    return {
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "max_drawdown": metrics.max_drawdown,
        "annual_volatility": metrics.annual_volatility,
        "daily_turnover": metrics.daily_turnover,
        "annual_turnover": metrics.annual_turnover,
        "avg_positions": float(metrics.avg_positions),
        "avg_holding_period": metrics.avg_holding_period,
        "pct_days_invested": metrics.pct_days_invested,
        "total_costs": metrics.total_costs,
        "cost_ratio": metrics.cost_ratio,
        "total_return": metrics.total_return,
        "num_trading_days": float(metrics.num_trading_days),
        "var_95": metrics.var_95,
        "var_99": metrics.var_99,
        "es_95": metrics.es_95,
        "es_99": metrics.es_99,
        "cdar_95": metrics.cdar_95,
        "skewness": metrics.skewness,
        "kurtosis": metrics.kurtosis,
        "worst_1d_return": metrics.worst_1d_return,
        "worst_5d_return": metrics.worst_5d_return,
        "tail_ratio": metrics.tail_ratio,
    }


def _save_run_artifacts(
    name: str,
    results: BacktestResults,
    metrics: PerformanceMetrics,
    run_dir: Path,
    benchmark: pd.Series,
    prices: pd.DataFrame,
    stock_charts_mode: str = "none",
    stock_chart_top_n: int = 10,
    stock_chart_max: Optional[int] = None,
    stock_chart_chunk_size: int = 25,
    stock_chart_chunk_index: int = 0,
    stock_marker_limit: int = 400,
    include_trade_stats: bool = False,
    wandb_settings: Optional[WandbSettings] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[list[str]] = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    with open(run_dir / "config.json", "w") as f:
        json.dump(results.config.to_dict(), f, indent=2)

    results.trades.to_csv(run_dir / "trades.csv", index=False)
    if results.weights_history is not None and not results.weights_history.empty:
        results.weights_history.to_csv(run_dir / "weights_history.csv")

    generate_full_report(
        results=results,
        output_dir=run_dir,
        benchmark=benchmark,
        price_data=prices,
        stock_charts_mode=stock_charts_mode,
        stock_chart_top_n=stock_chart_top_n,
        stock_chart_max=stock_chart_max,
        stock_chart_chunk_size=stock_chart_chunk_size,
        stock_chart_chunk_index=stock_chart_chunk_index,
        stock_marker_limit=stock_marker_limit,
        include_trade_stats=include_trade_stats,
    )

    if wandb_settings is not None and wandb_settings.enabled:
        run = init_wandb_run(
            settings=wandb_settings,
            config=results.config.to_dict(),
            run_name=name,
            group=wandb_group,
            tags=wandb_tags,
        )
        log_metrics(run, _numeric_metrics(metrics), prefix="metrics")
        equity = results.get_equity_curve()
        returns = results.get_returns()
        if len(equity) > 0:
            equity_df = equity.rename("equity").reset_index()
            equity_df.columns = ["date", "equity"]
            log_dataframe_table(run, "equity_curve", equity_df)
        if len(returns) > 0:
            returns_df = returns.rename("returns").reset_index()
            returns_df.columns = ["date", "returns"]
            log_dataframe_table(run, "daily_returns", returns_df)
        if not results.trades.empty:
            log_dataframe_table(run, "trades_head", results.trades.head(2000))
        log_inline_images(
            run,
            directory=run_dir,
            key_prefix="charts",
            patterns=("*.png",),
            recursive=True,
        )
        log_artifact_dir(
            run,
            settings=wandb_settings,
            directory=run_dir,
            artifact_name=f"{name}-artifacts",
            artifact_type="analysis-run-output",
        )
        finish_run(run, settings=wandb_settings)


def _run_backtest(
    config: StrategyConfig,
    data_manager: DataManager,
) -> tuple[BacktestResults, PerformanceMetrics]:
    backtester = Backtester(config)
    results = backtester.run(data_manager)
    metrics = compute_metrics(results)
    return results, metrics


def _roundtrip_pnl(trades: pd.DataFrame) -> pd.DataFrame:
    """FIFO match BUY/SELL lots into realized round-trip records."""
    if trades.empty:
        return pd.DataFrame(columns=["ticker", "open_date", "close_date", "shares", "pnl"])

    records = []
    for ticker, group in trades.groupby("ticker"):
        g = group.sort_values("date")
        inventory: list[list] = []  # [buy_price, shares, buy_date]
        for _, row in g.iterrows():
            side = row["side"]
            px = float(row["price"])
            qty = float(row["shares"])
            dt = pd.Timestamp(row["date"])

            if side == "BUY":
                inventory.append([px, qty, dt])
                continue

            qty_to_sell = qty
            while qty_to_sell > 1e-8 and inventory:
                buy_px, buy_qty, buy_dt = inventory[0]
                matched = min(qty_to_sell, buy_qty)
                pnl = matched * (px - buy_px)
                records.append({
                    "ticker": ticker,
                    "open_date": buy_dt,
                    "close_date": dt,
                    "shares": matched,
                    "pnl": pnl,
                })
                qty_to_sell -= matched
                buy_qty -= matched
                if buy_qty <= 1e-8:
                    inventory.pop(0)
                else:
                    inventory[0][1] = buy_qty

    return pd.DataFrame(records)


def _clone_results_with_equity(
    base_results: BacktestResults,
    new_equity: pd.Series,
) -> BacktestResults:
    """Create a BacktestResults copy using a modified equity curve."""
    daily_by_date = {r.date: r for r in base_results.daily_results}
    dates = list(new_equity.index)
    cloned: list[DailyResult] = []
    prev = None
    for dt in dates:
        base = daily_by_date[dt]
        pv = float(new_equity.loc[dt])
        ret = 0.0 if prev is None or prev == 0 else (pv / prev - 1.0)
        cloned.append(
            DailyResult(
                date=dt,
                portfolio_value=pv,
                daily_return=ret,
                positions=base.positions,
                gross_exposure=base.gross_exposure,
                turnover=base.turnover,
                costs=base.costs,
                cash=base.cash,
            )
        )
        prev = pv

    return BacktestResults(
        daily_results=cloned,
        trades=base_results.trades.copy(),
        config=base_results.config,
        gate_failures=base_results.gate_failures.copy(),
        weights_history=(
            base_results.weights_history.copy()
            if base_results.weights_history is not None else None
        ),
    )


def _apply_remove_top_winners(
    baseline_results: BacktestResults,
    top_n: int = 10,
) -> tuple[BacktestResults, pd.DataFrame]:
    """Stress scenario: remove the realized PnL from the top-N winning round trips."""
    matched = _roundtrip_pnl(baseline_results.trades)
    if matched.empty:
        return baseline_results, matched

    winners = matched[matched["pnl"] > 0].sort_values("pnl", ascending=False).head(top_n).copy()
    if winners.empty:
        return baseline_results, winners

    adjusted_equity = baseline_results.get_equity_curve().copy()
    for _, row in winners.iterrows():
        close_date = pd.Timestamp(row["close_date"])
        pnl = float(row["pnl"])
        adjusted_equity.loc[adjusted_equity.index >= close_date] -= pnl

    adjusted_equity = adjusted_equity.clip(lower=1.0)
    adjusted_results = _clone_results_with_equity(baseline_results, adjusted_equity)
    return adjusted_results, winners


def _stress_scenario_config(base: StrategyConfig, scenario: str) -> Optional[StrategyConfig]:
    if scenario == "baseline":
        return base
    if scenario == "double_slippage":
        return _override_config(base, slippage_bps=base.slippage_bps * 2.0)
    if scenario == "bad_fills":
        return _override_config(
            base,
            bad_fills_enabled=True,
            bad_fills_vol_threshold=max(0.30, base.regime_vol_threshold),
            bad_fills_multiplier=2.5,
        )
    if scenario == "weekly_rebalance":
        return _override_config(base, rebalance_frequency="weekly", rebalance_weekday=0)
    if scenario == "randomize_ranks":
        return _override_config(
            base,
            randomize_ranks=True,
            randomize_ranks_seed=base.random_seed,
        )
    if scenario == "remove_top10_best_trades":
        return None
    raise ValueError(f"Unknown stress scenario: {scenario}")


def _plot_stress_summary(stress_df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_palette("husl")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    ordered = stress_df.copy()

    axes[0].bar(ordered["scenario"], ordered["cagr"] * 100, alpha=0.8, color="#2E86AB")
    axes[0].set_ylabel("CAGR (%)")
    axes[0].set_title("Stress Test Comparison")

    axes[1].bar(ordered["scenario"], ordered["sharpe"], alpha=0.8, color="#6A994E")
    axes[1].set_ylabel("Sharpe")

    axes[2].bar(ordered["scenario"], ordered["max_drawdown"] * 100, alpha=0.8, color="#E94F37")
    axes[2].set_ylabel("Max Drawdown (%)")
    axes[2].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "stress_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_stress_tests(
    base_config: StrategyConfig,
    data_manager: DataManager,
    output_dir: Path,
    benchmark: pd.Series,
    prices: pd.DataFrame,
    scenario: Optional[str] = None,
    all_scenarios: bool = True,
    stock_charts_mode: str = "none",
    stock_charts_all_runs: bool = False,
    stock_chart_top_n: int = 10,
    stock_chart_max: Optional[int] = None,
    stock_chart_chunk_size: int = 25,
    stock_chart_chunk_index: int = 0,
    stock_marker_limit: int = 400,
    include_trade_stats: bool = False,
    wandb_settings: Optional[WandbSettings] = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_names = STRESS_SCENARIOS if all_scenarios else (scenario or "baseline",)

    rows = []
    baseline_results = None
    baseline_metrics = None

    if "baseline" in scenario_names or "remove_top10_best_trades" in scenario_names:
        baseline_results, baseline_metrics = _run_backtest(base_config, data_manager)
        baseline_dir = output_dir / "baseline"
        _save_run_artifacts(
            "baseline",
            baseline_results,
            baseline_metrics,
            baseline_dir,
            benchmark,
            prices,
            stock_charts_mode=stock_charts_mode,
            stock_chart_top_n=stock_chart_top_n,
            stock_chart_max=stock_chart_max,
            stock_chart_chunk_size=stock_chart_chunk_size,
            stock_chart_chunk_index=stock_chart_chunk_index,
            stock_marker_limit=stock_marker_limit,
            include_trade_stats=include_trade_stats,
            wandb_settings=wandb_settings,
            wandb_group="stress",
            wandb_tags=["stress", "baseline"],
        )
        rows.append(_metrics_row("baseline", baseline_metrics))

    for name in scenario_names:
        if name == "baseline":
            continue

        if name == "remove_top10_best_trades":
            if baseline_results is None:
                baseline_results, _ = _run_backtest(base_config, data_manager)
            stressed_results, removed = _apply_remove_top_winners(baseline_results, top_n=10)
            stressed_metrics = compute_metrics(stressed_results)
            run_dir = output_dir / name
            _save_run_artifacts(
                name,
                stressed_results,
                stressed_metrics,
                run_dir,
                benchmark,
                prices,
                stock_charts_mode=(stock_charts_mode if stock_charts_all_runs else "none"),
                stock_chart_top_n=stock_chart_top_n,
                stock_chart_max=stock_chart_max,
                stock_chart_chunk_size=stock_chart_chunk_size,
                stock_chart_chunk_index=stock_chart_chunk_index,
                stock_marker_limit=stock_marker_limit,
                include_trade_stats=include_trade_stats,
                wandb_settings=wandb_settings,
                wandb_group="stress",
                wandb_tags=["stress", name],
            )
            removed.to_csv(run_dir / "removed_top10_winners.csv", index=False)
            rows.append(_metrics_row(name, stressed_metrics))
            continue

        cfg = _stress_scenario_config(base_config, name)
        if cfg is None:
            continue
        results, metrics = _run_backtest(cfg, data_manager)
        _save_run_artifacts(
            name,
            results,
            metrics,
            output_dir / name,
            benchmark,
            prices,
            stock_charts_mode=(stock_charts_mode if stock_charts_all_runs else "none"),
            stock_chart_top_n=stock_chart_top_n,
            stock_chart_max=stock_chart_max,
            stock_chart_chunk_size=stock_chart_chunk_size,
            stock_chart_chunk_index=stock_chart_chunk_index,
            stock_marker_limit=stock_marker_limit,
            include_trade_stats=include_trade_stats,
            wandb_settings=wandb_settings,
            wandb_group="stress",
            wandb_tags=["stress", name],
        )
        rows.append(_metrics_row(name, metrics))

    stress_df = pd.DataFrame(rows)
    stress_df.to_csv(output_dir / "stress_test_results.csv", index=False)
    with open(output_dir / "stress_test_results.json", "w") as f:
        json.dump(stress_df.to_dict(orient="records"), f, indent=2)

    if not stress_df.empty:
        _plot_stress_summary(stress_df, output_dir)
    return stress_df


def _plot_parameter_sensitivity(
    sensitivity_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if sensitivity_df.empty:
        return

    for param, group in sensitivity_df.groupby("parameter"):
        g = group.sort_values("value")
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        axes[0].plot(g["value"], g["cagr"] * 100, marker="o", linewidth=2, color="#2E86AB")
        axes[0].set_ylabel("CAGR (%)")
        axes[0].set_title(f"Sensitivity: {param}")

        axes[1].plot(g["value"], g["sharpe"], marker="o", linewidth=2, color="#6A994E")
        axes[1].set_ylabel("Sharpe")

        axes[2].plot(g["value"], g["max_drawdown"] * 100, marker="o", linewidth=2, color="#E94F37")
        axes[2].set_ylabel("Max Drawdown (%)")
        axes[2].set_xlabel(param)

        for ax in axes:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"sensitivity_{param}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_parameter_sensitivity(
    base_config: StrategyConfig,
    data_manager: DataManager,
    output_dir: Path,
    benchmark: pd.Series,
    prices: pd.DataFrame,
    stock_charts_mode: str = "none",
    stock_charts_all_runs: bool = False,
    stock_chart_top_n: int = 10,
    stock_chart_max: Optional[int] = None,
    stock_chart_chunk_size: int = 25,
    stock_chart_chunk_index: int = 0,
    stock_marker_limit: int = 400,
    include_trade_stats: bool = False,
    wandb_settings: Optional[WandbSettings] = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for parameter, values in SENSITIVITY_GRID.items():
        for value in values:
            cfg = _override_config(base_config, **{parameter: value})
            results, metrics = _run_backtest(cfg, data_manager)
            run_name = f"{parameter}_{value}"
            _save_run_artifacts(
                run_name,
                results,
                metrics,
                output_dir / run_name,
                benchmark,
                prices,
                stock_charts_mode=(stock_charts_mode if stock_charts_all_runs else "none"),
                stock_chart_top_n=stock_chart_top_n,
                stock_chart_max=stock_chart_max,
                stock_chart_chunk_size=stock_chart_chunk_size,
                stock_chart_chunk_index=stock_chart_chunk_index,
                stock_marker_limit=stock_marker_limit,
                include_trade_stats=include_trade_stats,
                wandb_settings=wandb_settings,
                wandb_group="sensitivity",
                wandb_tags=["sensitivity", parameter],
            )
            row = _metrics_row(run_name, metrics)
            row["parameter"] = parameter
            row["value"] = value
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "parameter_sensitivity_results.csv", index=False)
    with open(output_dir / "parameter_sensitivity_results.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    _plot_parameter_sensitivity(df, output_dir)
    return df


def _render_analysis_html(
    output_dir: Path,
    stress_df: Optional[pd.DataFrame] = None,
    sensitivity_df: Optional[pd.DataFrame] = None,
) -> Path:
    stress_html = ""
    sensitivity_html = ""
    stress_img = ""
    sensitivity_imgs = ""

    if stress_df is not None and not stress_df.empty:
        stress_html = stress_df.to_html(index=False)
        if (output_dir / "stress" / "stress_summary.png").exists():
            stress_img = '<img src="stress/stress_summary.png" alt="Stress Summary">'

    if sensitivity_df is not None and not sensitivity_df.empty:
        sensitivity_html = sensitivity_df.to_html(index=False)
        for parameter in SENSITIVITY_GRID:
            p = output_dir / "sensitivity" / f"sensitivity_{parameter}.png"
            if p.exists():
                sensitivity_imgs += (
                    f'<h3>{parameter}</h3>'
                    f'<img src="sensitivity/sensitivity_{parameter}.png" alt="{parameter}">'
                )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Strategy Analysis Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 16px 0; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        img {{ max-width: 100%; margin: 16px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
      </style>
    </head>
    <body>
      <h1>Stress + Sensitivity Analysis</h1>
      <h2>Stress Tests</h2>
      {stress_img}
      {stress_html}
      <h2>Parameter Sensitivity</h2>
      {sensitivity_imgs}
      {sensitivity_html}
    </body>
    </html>
    """
    path = output_dir / "analysis_report.html"
    with open(path, "w") as f:
        f.write(html)
    return path


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    logger = config.get_logger("analysis")
    stock_charts_mode = "all" if args.all_stock_charts else args.stock_charts_mode
    wandb_settings = WandbSettings(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group=args.wandb_group,
        run_name=args.wandb_run_name,
        tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()],
        log_artifacts=args.wandb_log_artifacts,
    )
    if args.wandb and not is_wandb_available():
        logger.warning(
            "W&B was enabled but wandb is not importable. "
            "Install tracking extras or disable --wandb."
        )

    logger.info("Loading data for analysis runs...")
    data_manager = DataManager(config, YFinanceProvider(config))
    data_manager.load_data()

    benchmark = data_manager.get_benchmark_close()
    prices = data_manager.get_close_prices()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    stress_df = None
    sensitivity_df = None

    if args.mode in {"stress", "all"}:
        logger.info("Running stress tests...")
        stress_df = run_stress_tests(
            base_config=config,
            data_manager=data_manager,
            output_dir=out_root / "stress",
            benchmark=benchmark,
            prices=prices,
            scenario=args.stress_scenario,
            all_scenarios=args.all_stress_scenarios or args.mode == "all",
            stock_charts_mode=stock_charts_mode,
            stock_charts_all_runs=args.stock_charts_all_runs,
            stock_chart_top_n=args.stock_chart_top_n,
            stock_chart_max=args.stock_chart_max,
            stock_chart_chunk_size=args.stock_chart_chunk_size,
            stock_chart_chunk_index=args.stock_chart_chunk_index,
            stock_marker_limit=args.stock_marker_limit,
            include_trade_stats=args.include_trade_stats,
            wandb_settings=wandb_settings,
        )

    if args.mode in {"sensitivity", "all"}:
        logger.info("Running parameter sensitivity...")
        sensitivity_df = run_parameter_sensitivity(
            base_config=config,
            data_manager=data_manager,
            output_dir=out_root / "sensitivity",
            benchmark=benchmark,
            prices=prices,
            stock_charts_mode=stock_charts_mode,
            stock_charts_all_runs=args.stock_charts_all_runs,
            stock_chart_top_n=args.stock_chart_top_n,
            stock_chart_max=args.stock_chart_max,
            stock_chart_chunk_size=args.stock_chart_chunk_size,
            stock_chart_chunk_index=args.stock_chart_chunk_index,
            stock_marker_limit=args.stock_marker_limit,
            include_trade_stats=args.include_trade_stats,
            wandb_settings=wandb_settings,
        )

    report_path = _render_analysis_html(
        output_dir=out_root,
        stress_df=stress_df,
        sensitivity_df=sensitivity_df,
    )
    logger.info(f"Analysis report saved to: {report_path}")

    if wandb_settings.enabled:
        summary_run = init_wandb_run(
            settings=wandb_settings,
            config=config.to_dict(),
            run_name=args.wandb_run_name or "analysis-summary",
            group=args.wandb_group or "analysis",
            tags=[*(wandb_settings.tags or []), "summary"],
        )
        if stress_df is not None and not stress_df.empty:
            log_dataframe_table(summary_run, "stress_results", stress_df)
        if sensitivity_df is not None and not sensitivity_df.empty:
            log_dataframe_table(summary_run, "sensitivity_results", sensitivity_df)
        log_inline_images(
            summary_run,
            directory=out_root / "stress",
            key_prefix="summary_charts/stress",
            patterns=("stress_summary.png",),
            recursive=False,
        )
        log_inline_images(
            summary_run,
            directory=out_root / "sensitivity",
            key_prefix="summary_charts/sensitivity",
            patterns=("sensitivity_*.png",),
            recursive=False,
        )
        log_metrics(summary_run, {"report_path": str(report_path)}, prefix="artifacts")
        log_artifact_dir(
            summary_run,
            settings=wandb_settings,
            directory=out_root,
            artifact_name=f"analysis-summary-{out_root.name}",
            artifact_type="analysis-output",
        )
        finish_run(summary_run, settings=wandb_settings)

    # Final sync sweep for any run fragments not captured by per-run finishes.
    finish_run(None, settings=wandb_settings)


if __name__ == "__main__":
    main()
