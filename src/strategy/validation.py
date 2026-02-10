#!/usr/bin/env python3
"""
Validation suite for tail-risk diagnostics.

Implements:
- Crisis window scorecards
- Block-bootstrap drawdown distribution
- Combined cost + volatility stress
- Clearly labeled training vs validation output bundles
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import BacktestResults, Backtester, DailyResult
from .config import StrategyConfig
from .data import DataManager, YFinanceProvider
from .metrics import compute_metrics
from .reporting import generate_full_report
from .wandb_utils import (
    WandbSettings,
    finish_run,
    init_wandb_run,
    log_artifact_dir,
    log_dataframe_table,
    log_inline_images,
    log_metrics,
)


CRISIS_WINDOWS = (
    ("gfc", "2008-09-01", "2009-06-30"),
    ("covid_crash", "2020-02-19", "2020-06-30"),
    ("inflation_shock", "2022-01-03", "2022-10-31"),
    ("recent_2024", "2024-01-01", "2024-12-31"),
)


@dataclass
class PhaseRun:
    label: str
    config: StrategyConfig
    results: BacktestResults
    metrics: object
    phase_dir: Path


def _build_config(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        ticker_file=Path(args.tickers),
        start_date=args.training_start,
        end_date=args.validation_end if args.validation_end else date.today().isoformat(),
        initial_capital=args.capital,
        log_level="DEBUG" if args.verbose else "INFO",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tail-risk validation suite")
    parser.add_argument("--tickers", "-t", type=str, default="data/tickers.txt")
    parser.add_argument("--output", "-o", type=str, default="reports/output/validation")
    parser.add_argument("--capital", type=float, default=1_000_000)

    # Defaults split train/validation to avoid leakage.
    parser.add_argument("--training-start", type=str, default="2008-01-01")
    parser.add_argument("--training-end", type=str, default="2023-12-31")
    parser.add_argument("--validation-start", type=str, default="2024-01-01")
    parser.add_argument("--validation-end", type=str, default=None)

    parser.add_argument("--stock-charts-mode", choices=["none", "top", "all"], default="none")
    parser.add_argument("--stock-chart-top-n", type=int, default=10)
    parser.add_argument("--stock-chart-max", type=int, default=None)
    parser.add_argument("--stock-chart-chunk-size", type=int, default=25)
    parser.add_argument("--stock-chart-chunk-index", type=int, default=0)
    parser.add_argument("--stock-marker-limit", type=int, default=400)
    parser.add_argument("--include-trade-stats", action="store_true")

    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--bootstrap-block-size", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=42)

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
    parser.add_argument("--wandb-project", type=str, default="equity-strategy")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-group", type=str, default="validation")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="validation,tail-risk")
    parser.add_argument(
        "--wandb-log-artifacts",
        dest="wandb_log_artifacts",
        action="store_true",
        default=True,
        help="Upload output bundles as W&B artifacts (default: enabled)",
    )
    parser.add_argument(
        "--no-wandb-log-artifacts",
        dest="wandb_log_artifacts",
        action="store_false",
        help="Disable W&B artifact uploads",
    )

    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def _override_config(base: StrategyConfig, **overrides) -> StrategyConfig:
    d = base.to_dict()
    d.update(overrides)
    return StrategyConfig.from_dict(d)


def _run_backtest(config: StrategyConfig) -> tuple[BacktestResults, object, DataManager]:
    data_manager = DataManager(config, YFinanceProvider(config))
    data_manager.load_data()
    results = Backtester(config).run(data_manager)
    metrics = compute_metrics(results)
    return results, metrics, data_manager


def _slice_results(results: BacktestResults, start: pd.Timestamp, end: pd.Timestamp) -> Optional[BacktestResults]:
    """Slice results to date window and recompute daily returns consistently."""
    subset = [r for r in results.daily_results if start <= r.date <= end]
    if len(subset) < 2:
        return None

    rebuilt: list[DailyResult] = []
    prev = None
    for row in subset:
        ret = 0.0 if prev is None or prev == 0 else row.portfolio_value / prev - 1.0
        rebuilt.append(
            DailyResult(
                date=row.date,
                portfolio_value=row.portfolio_value,
                daily_return=ret,
                positions=row.positions,
                gross_exposure=row.gross_exposure,
                turnover=row.turnover,
                costs=row.costs,
                cash=row.cash,
            )
        )
        prev = row.portfolio_value

    trades = results.trades.copy()
    if not trades.empty and "date" in trades.columns:
        trade_dates = pd.to_datetime(trades["date"])
        trades = trades[(trade_dates >= start) & (trade_dates <= end)].copy()

    weights = results.get_weights_history()
    if not weights.empty:
        weights = weights.loc[(weights.index >= start) & (weights.index <= end)].copy()

    return BacktestResults(
        daily_results=rebuilt,
        trades=trades,
        config=results.config,
        gate_failures=results.gate_failures.copy(),
        weights_history=weights,
    )


def compute_crisis_window_scorecards(
    results: BacktestResults,
    windows: tuple[tuple[str, str, str], ...] = CRISIS_WINDOWS,
) -> pd.DataFrame:
    """Compute per-crisis scorecards over predefined market stress windows."""
    rows: list[dict] = []
    for name, start, end in windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        sliced = _slice_results(results, start_ts, end_ts)
        if sliced is None:
            continue

        metrics = compute_metrics(sliced)
        returns = sliced.get_returns().dropna()
        rows.append(
            {
                "window": name,
                "start": start,
                "end": end,
                "trading_days": int(len(returns)),
                "total_return": metrics.total_return,
                "max_drawdown": metrics.max_drawdown,
                "annual_volatility": metrics.annual_volatility,
                "worst_1d_return": metrics.worst_1d_return,
                "var_95": metrics.var_95,
                "es_95": metrics.es_95,
                "sharpe": metrics.sharpe,
            }
        )

    return pd.DataFrame(rows)


def block_bootstrap_drawdown_distribution(
    returns: pd.Series,
    n_samples: int = 500,
    block_size: int = 20,
    seed: int = 42,
) -> pd.Series:
    """Simulate max drawdown distribution via block-bootstrap return paths."""
    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float)

    arr = clean.to_numpy()
    n = len(arr)
    block_size = max(2, min(block_size, n))
    rng = np.random.default_rng(seed)

    max_dds = []
    max_start = max(1, n - block_size + 1)
    for _ in range(n_samples):
        synthetic: list[float] = []
        while len(synthetic) < n:
            start = int(rng.integers(0, max_start))
            synthetic.extend(arr[start:start + block_size].tolist())
        sample = np.asarray(synthetic[:n], dtype=float)

        equity = np.cumprod(1.0 + sample)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dds.append(float(drawdown.min()))

    return pd.Series(max_dds, name="max_drawdown")


def plot_bootstrap_drawdown_distribution(
    distribution: pd.Series,
    save_path: Path,
    title: str = "Block Bootstrap Max Drawdown Distribution",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if distribution.empty:
        ax.text(0.5, 0.5, "No bootstrap data", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.hist(distribution * 100, bins=40, alpha=0.75, color="#2E86AB", label="Bootstrap samples")
        p5 = distribution.quantile(0.05) * 100
        p50 = distribution.quantile(0.50) * 100
        p95 = distribution.quantile(0.95) * 100
        ax.axvline(p5, color="#E94F37", linestyle="--", label=f"5th pct: {p5:.2f}%")
        ax.axvline(p50, color="#6A994E", linestyle="-.", label=f"Median: {p50:.2f}%")
        ax.axvline(p95, color="#6A994E", linestyle=":", label=f"95th pct: {p95:.2f}%")
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _combined_stress_config(base_config: StrategyConfig) -> StrategyConfig:
    """Combined tail stress: trading costs worsen exactly when vol is elevated."""
    return _override_config(
        base_config,
        slippage_bps=base_config.slippage_bps * 2.0,
        bad_fills_enabled=True,
        bad_fills_vol_threshold=max(0.30, base_config.regime_vol_threshold),
        bad_fills_multiplier=max(2.5, base_config.bad_fills_multiplier),
    )


def run_combined_cost_vol_stress(
    base_config: StrategyConfig,
    data_manager: DataManager,
) -> tuple[BacktestResults, object]:
    stress_cfg = _combined_stress_config(base_config)
    stressed_results = Backtester(stress_cfg).run(data_manager)
    stressed_metrics = compute_metrics(stressed_results)
    return stressed_results, stressed_metrics


def _numeric_metrics(metrics: object) -> dict[str, float]:
    return {
        "cagr": metrics.cagr,
        "sharpe": metrics.sharpe,
        "sortino": metrics.sortino,
        "max_drawdown": metrics.max_drawdown,
        "annual_volatility": metrics.annual_volatility,
        "total_return": metrics.total_return,
        "var_95": metrics.var_95,
        "var_99": metrics.var_99,
        "es_95": metrics.es_95,
        "es_99": metrics.es_99,
        "cdar_95": metrics.cdar_95,
        "tail_ratio": metrics.tail_ratio,
    }


def _save_phase_outputs(
    phase: PhaseRun,
    data_manager: DataManager,
    benchmark: pd.Series,
    prices: pd.DataFrame,
    stock_charts_mode: str,
    stock_chart_top_n: int,
    stock_chart_max: Optional[int],
    stock_chart_chunk_size: int,
    stock_chart_chunk_index: int,
    stock_marker_limit: int,
    include_trade_stats: bool,
    bootstrap_samples: int,
    bootstrap_block_size: int,
    bootstrap_seed: int,
    wandb_settings: Optional[WandbSettings],
    wandb_tags: list[str],
) -> dict[str, object]:
    phase.phase_dir.mkdir(parents=True, exist_ok=True)

    # Main backtest report bundle for this phase.
    report_path = generate_full_report(
        results=phase.results,
        output_dir=phase.phase_dir,
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
    with open(phase.phase_dir / "metrics.json", "w") as f:
        json.dump(phase.metrics.to_dict(), f, indent=2)
    with open(phase.phase_dir / "config.json", "w") as f:
        json.dump(phase.config.to_dict(), f, indent=2)
    phase.results.trades.to_csv(phase.phase_dir / "trades.csv", index=False)

    # Crisis scorecards.
    crisis_df = compute_crisis_window_scorecards(phase.results)
    crisis_df.to_csv(phase.phase_dir / "crisis_window_scorecards.csv", index=False)

    # Block-bootstrap max drawdown distribution.
    returns = phase.results.get_returns().dropna()
    bootstrap = block_bootstrap_drawdown_distribution(
        returns,
        n_samples=bootstrap_samples,
        block_size=bootstrap_block_size,
        seed=bootstrap_seed,
    )
    bootstrap.to_frame(name="max_drawdown").to_csv(
        phase.phase_dir / "bootstrap_drawdown_distribution.csv", index=False
    )
    plot_bootstrap_drawdown_distribution(
        bootstrap,
        save_path=phase.phase_dir / "bootstrap_drawdown_distribution.png",
        title=f"{phase.label.title()} - Block Bootstrap Max Drawdown Distribution",
    )

    # Combined cost + vol stress on same data window.
    _, stress_metrics = run_combined_cost_vol_stress(phase.config, data_manager)
    with open(phase.phase_dir / "combined_cost_vol_stress_metrics.json", "w") as f:
        json.dump(stress_metrics.to_dict(), f, indent=2)

    stress_summary = pd.DataFrame(
        [
            {"run": "baseline", **_numeric_metrics(phase.metrics)},
            {"run": "combined_cost_vol_stress", **_numeric_metrics(stress_metrics)},
        ]
    )
    stress_summary.to_csv(phase.phase_dir / "combined_cost_vol_stress_comparison.csv", index=False)

    if wandb_settings is not None and wandb_settings.enabled:
        run = init_wandb_run(
            settings=wandb_settings,
            config=phase.config.to_dict(),
            run_name=f"{phase.label}-validation",
            group=wandb_settings.group,
            tags=[*wandb_tags, phase.label],
        )
        log_metrics(run, _numeric_metrics(phase.metrics), prefix=f"{phase.label}/baseline")
        log_metrics(run, _numeric_metrics(stress_metrics), prefix=f"{phase.label}/combined_stress")
        if not crisis_df.empty:
            log_dataframe_table(run, f"{phase.label}_crisis_scorecards", crisis_df)
        log_dataframe_table(run, f"{phase.label}_stress_comparison", stress_summary)
        log_inline_images(
            run,
            directory=phase.phase_dir,
            key_prefix=f"{phase.label}/charts",
            patterns=("*.png",),
            recursive=True,
        )
        log_artifact_dir(
            run,
            settings=wandb_settings,
            directory=phase.phase_dir,
            artifact_name=f"{phase.label}-validation-{phase.phase_dir.name}",
            artifact_type="validation-output",
        )
        finish_run(run, settings=wandb_settings)

    return {
        "phase": phase.label,
        "report_path": str(report_path),
        "metrics": _numeric_metrics(phase.metrics),
        "stress_metrics": _numeric_metrics(stress_metrics),
        "crisis_windows": len(crisis_df),
        "bootstrap_samples": len(bootstrap),
    }


def _render_suite_report(output_dir: Path, summaries: list[dict[str, object]]) -> Path:
    rows = []
    for s in summaries:
        row = {
            "Phase": s["phase"],
            "Report": s["report_path"],
            "CAGR": f"{s['metrics']['cagr']:.2%}",
            "Max Drawdown": f"{s['metrics']['max_drawdown']:.2%}",
            "ES(95%)": f"{s['metrics']['es_95']:.2%}",
            "Tail Ratio": f"{s['metrics']['tail_ratio']:.3f}",
            "Stress Max Drawdown": f"{s['stress_metrics']['max_drawdown']:.2%}",
            "Crisis Windows Covered": s["crisis_windows"],
            "Bootstrap Samples": s["bootstrap_samples"],
        }
        rows.append(row)

    table = pd.DataFrame(rows).to_html(index=False)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Training vs Validation Tail-Risk Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 32px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
      </style>
    </head>
    <body>
      <h1>Tail-Risk Validation Suite</h1>
      <p>Outputs are split and labeled by phase to avoid confusion.</p>
      <h2>Phase Summary</h2>
      {table}
    </body>
    </html>
    """
    report_path = output_dir / "validation_suite_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    return report_path


def run_validation_suite(
    base_config: StrategyConfig,
    output_dir: Path,
    training_start: str,
    training_end: str,
    validation_start: str,
    validation_end: Optional[str] = None,
    stock_charts_mode: str = "none",
    stock_chart_top_n: int = 10,
    stock_chart_max: Optional[int] = None,
    stock_chart_chunk_size: int = 25,
    stock_chart_chunk_index: int = 0,
    stock_marker_limit: int = 400,
    include_trade_stats: bool = False,
    bootstrap_samples: int = 500,
    bootstrap_block_size: int = 20,
    bootstrap_seed: int = 42,
    wandb_settings: Optional[WandbSettings] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = _override_config(base_config, start_date=training_start, end_date=training_end)
    validation_cfg = _override_config(
        base_config,
        start_date=validation_start,
        end_date=validation_end if validation_end else date.today().isoformat(),
    )

    phase_defs = [
        ("training", training_cfg),
        ("validation", validation_cfg),
    ]

    summaries: list[dict[str, object]] = []
    for label, cfg in phase_defs:
        results, metrics, data_manager = _run_backtest(cfg)
        phase_dir = output_dir / f"{label}_{cfg.start_date.isoformat()}_{cfg.end_date.isoformat()}"

        phase = PhaseRun(
            label=label,
            config=cfg,
            results=results,
            metrics=metrics,
            phase_dir=phase_dir,
        )
        summary = _save_phase_outputs(
            phase=phase,
            data_manager=data_manager,
            benchmark=data_manager.get_benchmark_close(),
            prices=data_manager.get_close_prices(),
            stock_charts_mode=stock_charts_mode,
            stock_chart_top_n=stock_chart_top_n,
            stock_chart_max=stock_chart_max,
            stock_chart_chunk_size=stock_chart_chunk_size,
            stock_chart_chunk_index=stock_chart_chunk_index,
            stock_marker_limit=stock_marker_limit,
            include_trade_stats=include_trade_stats,
            bootstrap_samples=bootstrap_samples,
            bootstrap_block_size=bootstrap_block_size,
            bootstrap_seed=bootstrap_seed,
            wandb_settings=wandb_settings,
            wandb_tags=["validation", "tail-risk"],
        )
        summaries.append(summary)

    with open(output_dir / "validation_suite_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    return _render_suite_report(output_dir, summaries)


def main() -> None:
    args = parse_args()
    config = _build_config(args)
    logger = config.get_logger("validation")

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

    report_path = run_validation_suite(
        base_config=config,
        output_dir=Path(args.output),
        training_start=args.training_start,
        training_end=args.training_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        stock_charts_mode=args.stock_charts_mode,
        stock_chart_top_n=args.stock_chart_top_n,
        stock_chart_max=args.stock_chart_max,
        stock_chart_chunk_size=args.stock_chart_chunk_size,
        stock_chart_chunk_index=args.stock_chart_chunk_index,
        stock_marker_limit=args.stock_marker_limit,
        include_trade_stats=args.include_trade_stats,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_size=args.bootstrap_block_size,
        bootstrap_seed=args.bootstrap_seed,
        wandb_settings=wandb_settings,
    )
    logger.info(f"Validation suite report saved to: {report_path}")
    # Final sync sweep for any run fragments not captured by per-run finishes.
    finish_run(None, settings=wandb_settings)


if __name__ == "__main__":
    main()
