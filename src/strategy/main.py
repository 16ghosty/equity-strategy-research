#!/usr/bin/env python3
"""
Main entry point for running the equity strategy backtest.

Usage:
    python -m strategy.main --start 2020-01-01 --end 2024-12-31
    python -m strategy.main --config custom_config.json
"""

import argparse
import json
from pathlib import Path
from datetime import date

import pandas as pd

from .config import StrategyConfig
from .data import DataManager, YFinanceProvider, load_tickers
from .backtest import Backtester
from .metrics import compute_metrics, print_metrics_summary
from .reporting import (
    build_plotly_charts,
    compute_benchmark_diagnostics,
    compute_drawdown_detractor_metrics,
    compute_ticker_health_table,
    generate_full_report,
)
from .wandb_utils import (
    WandbSettings,
    finish_run,
    init_wandb_run,
    log_artifact_dir,
    log_dataframe_table,
    log_inline_images,
    log_metrics,
    log_plotly_figures,
    set_summary_metrics,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run momentum equity strategy backtest"
    )
    
    parser.add_argument(
        '--start', '-s',
        type=str,
        default='2020-01-01',
        help='Backtest start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end', '-e',
        type=str,
        default=None,
        help='Backtest end date (YYYY-MM-DD), defaults to today'
    )
    
    parser.add_argument(
        '--tickers', '-t',
        type=str,
        default='data/tickers.txt',
        help='Path to ticker file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='reports/output',
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to JSON config file'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=1_000_000,
        help='Initial capital'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Number of positions to hold'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating HTML report'
    )

    parser.add_argument(
        '--cash-sweep-to-benchmark',
        dest='cash_sweep_to_benchmark',
        action='store_true',
        default=True,
        help='Invest positive idle cash in benchmark return stream (default: enabled)'
    )
    parser.add_argument(
        '--no-cash-sweep-to-benchmark',
        dest='cash_sweep_to_benchmark',
        action='store_false',
        help='Disable benchmark cash sweep for idle cash'
    )
    parser.add_argument(
        '--cash-sweep-asset',
        choices=['benchmark', 'tbill'],
        default='benchmark',
        help='Asset used for idle-cash sweep when enabled'
    )
    parser.add_argument(
        '--cash-sweep-tbill-ticker',
        type=str,
        default='BIL',
        help='T-bill proxy ticker used when --cash-sweep-asset tbill'
    )
    parser.add_argument(
        '--cash-sweep-risk-off-to-cash',
        dest='cash_sweep_risk_off_to_cash',
        action='store_true',
        default=True,
        help='Keep idle cash uninvested during tail-risk regime (default: enabled)'
    )
    parser.add_argument(
        '--no-cash-sweep-risk-off-to-cash',
        dest='cash_sweep_risk_off_to_cash',
        action='store_false',
        help='Do not switch idle cash to pure cash during tail-risk regime'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--all-stock-charts',
        action='store_true',
        help='Generate per-stock time-series charts for all stocks in the report'
    )

    parser.add_argument(
        '--stock-charts-mode',
        choices=['none', 'top', 'all'],
        default='top',
        help='Stock chart generation mode for report'
    )

    parser.add_argument(
        '--stock-chart-top-n',
        type=int,
        default=10,
        help='Number of stock charts when using --stock-charts-mode top'
    )

    parser.add_argument(
        '--stock-chart-max',
        type=int,
        default=None,
        help='Optional hard cap on number of stock charts generated'
    )

    parser.add_argument(
        '--stock-chart-chunk-size',
        type=int,
        default=25,
        help='Number of stock charts per chunk'
    )

    parser.add_argument(
        '--stock-chart-chunk-index',
        type=int,
        default=0,
        help='0-based chunk index for stock chart generation'
    )

    parser.add_argument(
        '--stock-marker-limit',
        type=int,
        default=400,
        help='Maximum buy/sell markers per side on each stock chart'
    )

    parser.add_argument(
        '--include-trade-stats',
        action='store_true',
        help='Include detailed trade statistics table in HTML report (can be slow)'
    )

    parser.add_argument(
        '--wandb',
        dest='wandb',
        action='store_true',
        default=True,
        help='Enable Weights & Biases logging (default: enabled)'
    )
    parser.add_argument(
        '--no-wandb',
        dest='wandb',
        action='store_false',
        help='Disable Weights & Biases logging'
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='equity-strategy',
        help='Weights & Biases project name'
    )

    parser.add_argument(
        '--wandb-entity',
        type=str,
        default=None,
        help='Weights & Biases entity/team (optional)'
    )

    parser.add_argument(
        '--wandb-mode',
        choices=['online', 'offline', 'disabled'],
        default='online',
        help='Weights & Biases mode'
    )

    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Weights & Biases run name (optional)'
    )

    parser.add_argument(
        '--wandb-group',
        type=str,
        default='main-backtest',
        help='Weights & Biases run group'
    )

    parser.add_argument(
        '--wandb-tags',
        type=str,
        default='main,backtest',
        help='Comma-separated Weights & Biases tags'
    )

    parser.add_argument(
        '--wandb-log-artifacts',
        dest='wandb_log_artifacts',
        action='store_true',
        default=True,
        help='Upload output directory as a Weights & Biases artifact (default: enabled)'
    )
    parser.add_argument(
        '--no-wandb-log-artifacts',
        dest='wandb_log_artifacts',
        action='store_false',
        help='Disable Weights & Biases artifact uploads'
    )

    parser.add_argument(
        '--wandb-log-plotly',
        action='store_true',
        help='Log interactive Plotly charts to Weights & Biases'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Build config
    if args.config:
        # Load from JSON file
        with open(args.config) as f:
            config_dict = json.load(f)
        config = StrategyConfig.from_dict(config_dict)
    else:
        # Build from arguments
        end_date = args.end if args.end else date.today().isoformat()
        
        config = StrategyConfig(
            ticker_file=Path(args.tickers),
            start_date=args.start,
            end_date=end_date,
            initial_capital=args.capital,
            top_k=args.top_k,
            cash_sweep_to_benchmark=args.cash_sweep_to_benchmark,
            cash_sweep_asset=args.cash_sweep_asset,
            cash_sweep_tbill_ticker=args.cash_sweep_tbill_ticker,
            cash_sweep_risk_off_to_cash=args.cash_sweep_risk_off_to_cash,
            log_level='DEBUG' if args.verbose else 'INFO',
        )
    
    logger = config.get_logger("main")

    wandb_settings = WandbSettings(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        group=args.wandb_group,
        run_name=args.wandb_run_name,
        tags=[t.strip() for t in args.wandb_tags.split(',') if t.strip()],
        log_artifacts=args.wandb_log_artifacts,
    )
    run = init_wandb_run(
        settings=wandb_settings,
        config=config.to_dict(),
    )
    if args.wandb and run is None:
        logger.warning(
            "W&B was enabled but no run could be initialized. "
            "Install wandb and/or verify settings."
        )
    logger.info("=" * 60)
    logger.info("MOMENTUM EQUITY STRATEGY BACKTEST")
    logger.info("=" * 60)
    
    # Load tickers
    tickers = load_tickers(config.ticker_file)
    logger.info(f"Loaded {len(tickers)} tickers from {config.ticker_file}")
    
    # Initialize data provider
    provider = YFinanceProvider(config)
    
    # Initialize data manager and load data
    logger.info("Loading data...")
    data_manager = DataManager(config, provider)
    data_manager.load_data()
    
    logger.info(
        f"Loaded {len(data_manager.get_trading_dates())} trading days "
        f"for {len(data_manager.tickers)} tickers"
    )
    
    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(config)
    results = backtester.run(data_manager)
    benchmark = data_manager.get_benchmark_close()
    prices = data_manager.get_close_prices()
    
    # Compute metrics
    metrics = compute_metrics(results)
    print_metrics_summary(metrics)

    numeric_metrics = {
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
    log_metrics(run, numeric_metrics, prefix="main")
    set_summary_metrics(run, numeric_metrics, prefix="main")
    metrics_table = pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in metrics.to_dict().items()]
    )
    log_dataframe_table(run, "main_metrics_table", metrics_table)
    
    # Generate report
    report_path = None
    if not args.no_report:
        logger.info("Generating report...")
        output_dir = Path(args.output)
        
        report_path = generate_full_report(
            results=results,
            output_dir=output_dir,
            benchmark=benchmark,
            price_data=prices,
            all_stock_charts=args.all_stock_charts,
            stock_charts_mode=args.stock_charts_mode,
            stock_chart_top_n=args.stock_chart_top_n,
            stock_chart_max=args.stock_chart_max,
            stock_chart_chunk_size=args.stock_chart_chunk_size,
            stock_chart_chunk_index=args.stock_chart_chunk_index,
            stock_marker_limit=args.stock_marker_limit,
            include_trade_stats=args.include_trade_stats,
        )
        
        logger.info(f"Report saved to: {report_path}")
    
    # Save trades ledger
    trades_path = Path(args.output) / 'trades.csv'
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    results.trades.to_csv(trades_path, index=False)
    logger.info(f"Trades saved to: {trades_path}")
    
    # Save metrics
    metrics_path = Path(args.output) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Save ticker health and drawdown detractor metrics tables
    ticker_health = compute_ticker_health_table(results, prices)
    benchmark_diagnostics = compute_benchmark_diagnostics(results, benchmark)
    drawdown_detractor_metrics = compute_drawdown_detractor_metrics(
        results,
        prices,
        lookback_days=63,
        top_n=20,
    )

    ticker_health_path = Path(args.output) / "ticker_health_metrics.csv"
    if not ticker_health.empty:
        ticker_health.to_csv(ticker_health_path, index=False)
        logger.info(f"Ticker health metrics saved to: {ticker_health_path}")

    detractor_metrics_path = Path(args.output) / "drawdown_detractor_metrics.csv"
    if not drawdown_detractor_metrics.empty:
        drawdown_detractor_metrics.to_csv(detractor_metrics_path, index=False)
        logger.info(f"Drawdown detractor metrics saved to: {detractor_metrics_path}")

    benchmark_diag_path = Path(args.output) / "benchmark_diagnostics.csv"
    if not benchmark_diagnostics.empty:
        benchmark_diagnostics.to_csv(benchmark_diag_path, index=False)
        logger.info(f"Benchmark diagnostics saved to: {benchmark_diag_path}")
    
    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)

    if run is not None:
        output_dir = Path(args.output)
        if args.wandb_log_plotly:
            plotly_charts = build_plotly_charts(results=results, benchmark=benchmark)
            logged = log_plotly_figures(
                run,
                plotly_charts,
                key_prefix="interactive",
            )
            log_metrics(
                run,
                {"plotly_charts_logged": float(logged)},
                prefix="artifacts",
            )
        log_inline_images(
            run,
            directory=output_dir,
            key_prefix="charts",
            patterns=("*.png",),
            recursive=True,
        )
        if not ticker_health.empty:
            log_dataframe_table(run, "ticker_health_metrics", ticker_health)
        if not drawdown_detractor_metrics.empty:
            log_dataframe_table(
                run,
                "drawdown_detractor_metrics",
                drawdown_detractor_metrics,
            )
        if not benchmark_diagnostics.empty:
            log_dataframe_table(run, "benchmark_diagnostics", benchmark_diagnostics)
        log_metrics(
            run,
            {
                "report_path": str(report_path) if report_path is not None else "",
                "output_dir": str(output_dir),
                "ticker_health_metrics_path": str(ticker_health_path) if ticker_health_path.exists() else "",
                "drawdown_detractor_metrics_path": str(detractor_metrics_path) if detractor_metrics_path.exists() else "",
                "benchmark_diagnostics_path": str(benchmark_diag_path) if benchmark_diag_path.exists() else "",
            },
            prefix="artifacts",
        )
        log_artifact_dir(
            run,
            settings=wandb_settings,
            directory=output_dir,
            artifact_name=f"main-backtest-{output_dir.name}",
            artifact_type="backtest-output",
        )
    finish_run(run, settings=wandb_settings)
    # Final sync sweep for any run fragments not captured by run finish.
    finish_run(None, settings=wandb_settings)
    
    return results, metrics


if __name__ == '__main__':
    main()
