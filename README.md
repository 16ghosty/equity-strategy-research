# Equity Strategy Research Repository

A reproducible Python research repository for a **daily gated, ranked-allocation equity strategy**.

## Overview

This repository implements a momentum-based equity strategy with:

- **Dynamic Universe Selection**: Monthly rebalancing of top 100 stocks by trailing dollar volume
- **Multiple Filter Gates**: Liquidity, volatility, and market regime filters
- **Momentum Ranking**: Cross-sectional momentum scoring with skip period
- **Portfolio Construction**: Equal-weight or inverse-vol weighting with constraints
- **Buffer Logic**: Hysteresis for entry/exit to reduce turnover
- **T+1 Execution**: Realistic fills at next-day open with slippage costs

## Key Features

### Look-Ahead Prevention
- All signals use only data available at decision time
- Universe selection uses data strictly before the selection month
- Execution occurs at T+1 open (signals at day T close)
- Comprehensive unit tests validate no data leakage

### Reproducibility
- Deterministic random seeds
- Data cached to Parquet files
- All parameters in a single configuration dataclass
- Full test coverage with 131 unit tests

### Modularity
- Clean separation of concerns across modules
- Easy to modify individual components
- Comprehensive type hints and docstrings

## Project Structure

```
equity_strategy/
├── pyproject.toml          # Project config and dependencies
├── data/
│   ├── tickers.txt         # List of tickers to trade
│   └── cache/              # Cached OHLCV data (Parquet)
├── src/
│   └── strategy/
│       ├── __init__.py     # Package exports
│       ├── config.py       # Central configuration dataclass
│       ├── data.py         # Data download and caching
│       ├── features.py     # Feature computation (returns, vol, etc.)
│       ├── universe.py     # Universe selection logic
│       ├── gates.py        # Filter gates (liquidity, vol, regime)
│       ├── rank.py         # Momentum ranking
│       ├── portfolio.py    # Portfolio construction
│       ├── execution.py    # Trade execution with costs
│       ├── backtest.py     # Main backtest engine
│       ├── metrics.py      # Performance metrics
│       ├── reporting.py    # Charts and HTML reports
│       ├── analysis.py     # Stress + sensitivity runner
│       ├── validation.py   # Training/validation tail-risk suite
│       └── main.py         # CLI runner
├── tests/                  # Unit tests
└── reports/
    └── output/             # Generated reports
```

## Installation

```bash
# Clone and navigate to project
cd equity_strategy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Quick Start

### Running a Backtest

```bash
# Run with default settings
python -m strategy.main

# Custom date range and output
python -m strategy.main --start 2020-01-01 --end 2024-12-31 --output reports/my_run

# More options
python -m strategy.main --help
```

### Using in Python

```python
from strategy import (
    StrategyConfig,
    DataManager,
    YFinanceProvider,
    Backtester,
    compute_metrics,
    generate_full_report,
    load_tickers,
)

# 1. Configure the strategy
config = StrategyConfig(
    ticker_file="data/tickers.txt",
    start_date="2020-01-01",
    end_date="2024-12-31",
    top_k=20,           # Number of positions
    buffer=5,           # Exit buffer
    momentum_lookback=60,
    momentum_skip=5,
    slippage_bps=10,
)

# 2. Load data
tickers = load_tickers(config.ticker_file)
provider = YFinanceProvider(cache_dir=config.data_cache_dir)
data_manager = DataManager(
    tickers=tickers,
    provider=provider,
    start_date=config.start_date,
    end_date=config.end_date,
)
data_manager.load_all()

# 3. Run backtest
backtester = Backtester(config)
results = backtester.run(data_manager)

# 4. Analyze results
metrics = compute_metrics(results)
print(metrics)

# 5. Generate report
generate_full_report(results, output_dir="reports/output")
```

## Configuration Options

All parameters are centralized in `StrategyConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `universe_size` | 100 | Number of stocks in tradeable universe |
| `top_k` | 20 | Number of positions to hold |
| `buffer` | 5 | Exit buffer (exit if rank > top_k + buffer) |
| `momentum_lookback` | 60 | Days for momentum calculation |
| `momentum_skip` | 5 | Recent days to skip (avoid reversal) |
| `liquidity_threshold` | 1,000,000 | Min average dollar volume |
| `min_price` | 5.0 | Min stock price |
| `vol_cap` | 0.60 | Max annualized volatility |
| `weight_scheme` | "equal" | "equal" or "inverse_vol" |
| `max_weight` | 0.10 | Max single position weight |
| `slippage_bps` | 10 | Slippage in basis points |
| `execution_delay` | 1 | Days delay for execution (T+1) |

## Baseline Configuration

Going forward, the baseline strategy configuration is:

| Parameter | Baseline Value |
|-----------|----------------|
| `top_k` | 10 |
| `buffer` | 10 |
| `momentum_lookback` | 120 |
| `momentum_skip` | 5 |
| `liquidity_threshold` | 1,000,000 |
| `min_price` | 5.0 |
| `vol_cap` | 0.60 |
| `weight_scheme` | `"equal"` |
| `max_weight` | 0.10 |

Reference baseline config file:
- `configs/baseline_strategy.json`

Run command:

```bash
python -m strategy.main \
  --config configs/baseline_strategy.json \
  --output reports/output/baseline_run \
  --wandb --wandb-mode online --wandb-log-artifacts
```

## Strategy Logic

### 1. Universe Selection (Monthly)
- Select top N stocks by trailing 30-day average dollar volume
- Rebalance on first trading day of each month
- Use only data from prior month (no look-ahead)

### 2. Filter Gates (Daily)
- **Liquidity Gate**: Reject if avg dollar volume < threshold or price < $5
- **Volatility Gate**: Reject if volatility > cap; optionally size inversely to vol
- **Market Regime Gate**: Reduce exposure when benchmark is below MA or vol is high
- **Buffer Gate**: Use hysteresis for entry/exit to reduce turnover

### 3. Ranking (Daily)
- Compute momentum score: trailing return over `lookback` days, excluding recent `skip` days
- Rank cross-sectionally (rank 1 = highest momentum)

### 4. Portfolio Construction
- Entry: rank ≤ top_k AND passes all gates
- Exit: rank > top_k + buffer OR fails critical gates
- Apply weight scheme (equal or inverse-vol)
- Apply max weight and gross exposure constraints

### 5. Execution
- Generate signals at day T close
- Execute at day T+1 open with slippage

## Performance Metrics

The `metrics.py` module computes:
- **CAGR**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside-adjusted return
- **Max Drawdown**: Worst peak-to-trough decline
- **Annual Volatility**: Annualized return standard deviation
- **Turnover**: Daily and annualized portfolio turnover
- **Cost Ratio**: Trading costs as % of gross PnL
- **VaR / ES**: VaR(95/99) and Expected Shortfall(95/99)
- **CDaR(95%)**: Conditional drawdown-at-risk
- **Tail shape**: Skew, kurtosis, worst 1-day/5-day, tail ratio

## Reports

Generated HTML reports include:
- Equity curve (vs benchmark)
- Drawdown chart
- Monthly returns heatmap
- Rolling Sharpe ratio
- Positions and exposure over time
- Daily turnover analysis
- Daily return distribution
- Negative-tail distribution (<= 5th percentile)

## Weights & Biases (Optional)

W&B integration is available for both:
- `strategy.main` (primary backtest runs)
- `strategy.analysis` (stress tests and parameter sensitivity runs)

Setup:

```bash
# Optional dependency
pip install -e ".[tracking]"

# Set key in environment (do not hardcode in code or config files)
export WANDB_API_KEY="your_api_key_here"
```

Example usage:

```bash
# Main backtest tracking
python -m strategy.main --wandb --wandb-log-artifacts

# Stress + sensitivity tracking
python -m strategy.analysis --mode all --wandb --wandb-log-artifacts

# Training vs validation suite (labeled outputs)
python -m strategy.validation \
  --training-start 2008-01-01 --training-end 2023-12-31 \
  --validation-start 2024-01-01 --validation-end 2026-02-09 \
  --wandb --wandb-log-artifacts
```

Notes:
- Keep API keys in environment variables only.
- Default mode is online. Use `--wandb-mode offline` for deferred syncing.
- Offline runs can be synced later with `wandb sync`.

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only (skip network calls)
pytest tests/ -v -m "not slow"

# Run with coverage
pytest tests/ --cov=src/strategy --cov-report=html
```

## Validation Checklist

- [x] Universe selection uses only prior-month data
- [x] Signals at day T use only data available at day T close
- [x] Execution at day T+1 open (not same day)
- [x] No look-ahead in feature calculations
- [x] All tests pass (131 unit tests)
- [x] Reproducible results with fixed random seed

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Author

Gautam Marathe
