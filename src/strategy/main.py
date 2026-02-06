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

from .config import StrategyConfig
from .data import DataManager, YFinanceProvider, load_tickers
from .backtest import Backtester
from .metrics import compute_metrics, print_metrics_summary
from .reporting import generate_full_report


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
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
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
            log_level='DEBUG' if args.verbose else 'INFO',
        )
    
    logger = config.get_logger("main")
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
    
    # Compute metrics
    metrics = compute_metrics(results)
    print_metrics_summary(metrics)
    
    # Generate report
    if not args.no_report:
        logger.info("Generating report...")
        output_dir = Path(args.output)
        
        # Get benchmark for comparison
        benchmark = data_manager.get_benchmark_close()
        
        # Get price data for stock charts
        prices = data_manager.get_close_prices()
        
        report_path = generate_full_report(
            results=results,
            output_dir=output_dir,
            benchmark=benchmark,
            price_data=prices,
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
    
    logger.info("=" * 60)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 60)
    
    return results, metrics


if __name__ == '__main__':
    main()
