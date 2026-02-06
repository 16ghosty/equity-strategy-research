# Quant Research Review: Current Equity Strategy vs Antifragile Asset Allocation Model

## Scope
This document compares:
- The implementation in this repository (`README.md` + `src/strategy/*`)
- The white paper: `00K_Antifragile-Asset-Allocation-Model_GioeleGiordano_1st-Place.pdf`

The review is written from a quant research engineering perspective: implementation realism, reproducibility, risk design, and portability to production research workflows.

## Executive Summary
The two systems are materially different in objective and architecture:
- The white paper proposes a **monthly, ETF-based, dual-sleeve allocation model** explicitly designed for antifragility via a dedicated black-swan hedging sleeve.
- This repository implements a **daily, single-stock momentum selection model** with strong engineering quality (no-lookahead controls, T+1 execution, slippage, tests), but **without a dedicated convex/tail-risk sleeve**.

Conclusion:
- Your current codebase is stronger as a research platform and execution simulator.
- The white paper is stronger in explicit tail-event portfolio design.
- The highest-value integration is to add a **dynamic hedge sleeve allocator** on top of the existing engine.

## What The White Paper Implements (Observed)
From sections III-VI (pages 6-16), the paper describes:
- A **Sector Rotation Model** on 11 S&P sector ETFs (monthly rebalance).
- Ranking factors:
  - Absolute Momentum (4-month ROC)
  - Volatility model (edited GARCH-style)
  - Average relative correlations
  - ATR trend/breakout system
- Portfolio rule: pick top 5 ETFs, equal-weight.
- A **Black Swan Hedging Model** on 7 ETFs across asset classes:
  - Take top 3 candidates
  - Include only if absolute momentum is positive
  - Unused weight goes to cash proxy (1-3Y Treasury ETF)
- **Antifragile merge logic**:
  - Sector sleeve drives growth allocation
  - Missing/negative sector momentum is replaced by hedging sleeve
  - If all 5 sector picks are negative momentum, hedging sleeve can be 100%
- Data/implementation notes (page 14):
  - Monthly and daily data over ~Aug 2003-Feb 2019
  - Gross results shown with **no transaction costs**
  - Implementation spread across RStudio, Metastock, and Excel

## What This Repository Implements (Observed)
Core behavior in `src/strategy/`:
- **Universe**: dynamic top-N by trailing dollar volume (monthly universe refresh).
- **Signal frequency**: daily ranking and gating.
- **Ranking**: cross-sectional momentum (lookback/skip).
- **Gates**:
  - Liquidity and min-price
  - Volatility cap (+ optional inverse-vol scaling)
  - Market regime exposure scaling (benchmark MA + vol condition)
  - Buffer hysteresis for entry/exit turnover control
- **Portfolio construction**:
  - `top_k` + buffer logic
  - Equal or inverse-vol weights
  - Max weight and gross exposure limits
- **Execution model**:
  - Signal on close, execute T+1 open
  - Slippage model (bps or ATR-based)
  - Trade ledger
- **Research engineering quality**:
  - Central config object
  - Modular components
  - Test suite with 131 tests
  - Explicit no-lookahead checks in multiple modules

## Implementation Differences That Matter
| Dimension | White Paper Model | Current Repository |
|---|---|---|
| Asset universe | 11 sector ETFs + 7 hedge ETFs | Dynamic stock universe (top dollar volume names) |
| Rebalance cadence | Monthly allocation | Daily signal loop, monthly universe refresh |
| Alpha model | Multi-factor ETF ranking (M/V/C/T) | Single-core factor: cross-sectional momentum |
| Tail risk design | Dedicated black-swan sleeve + cash substitution | No dedicated hedge sleeve; only regime scaling and filters |
| Portfolio topology | Two-sleeve allocator (growth + hedge) | Single long-only sleeve with gating |
| Cost realism | Backtest results shown gross of costs | T+1 execution and slippage modeled |
| Reproducibility | Manual multi-tool workflow (R/Metastock/Excel) | Python package, modular code, tests |
| Data handling | Historical interpolation noted | Provider abstraction + cached OHLCV pipeline |

## Pros And Cons

### White Paper Approach
Pros:
- Explicitly targets crash behavior, not just average risk-adjusted return.
- Uses cross-asset hedge sleeve and cash fallback; this is structurally robust to deep equity selloffs.
- Monthly cadence reduces turnover and implementation complexity.

Cons:
- Lower implementation reproducibility (multi-tool/manual stack).
- Limited transparency on factor engineering specifics (some details in tables/formulas are not fully operationalized in code form).
- Reported results are gross of transaction costs.
- Static ETF basket design may miss broader cross-sectional equity alpha.

### Current Repository Approach
Pros:
- Strong engineering hygiene: modular architecture, deterministic configuration, unit tests.
- Better execution realism (T+1 open fills, slippage, turnover/cost tracking).
- Good no-lookahead discipline across universe/ranking/execution.
- Flexible enough to support rapid hypothesis testing.

Cons:
- Not antifragile by construction; there is no explicit convex crisis sleeve.
- Alpha breadth is narrow (mostly momentum + eligibility gating).
- Regime gate is coarse relative to a full risk-state allocation model.
- Concentration and correlation clustering risk remain in stressed regimes.

## High-Value Features To Integrate Into Existing Model

### 1) Add a dedicated hedge sleeve (highest impact)
- Implement a second portfolio sleeve with defensive/tail ETFs.
- Add allocator logic:
  - Growth sleeve weight from current model
  - Hedge sleeve receives residual when growth momentum breadth degrades
  - Cash fallback when hedge sleeve absolute momentum is weak

### 2) Add correlation-aware ranking term
- Introduce rolling average correlation penalty/score into ranking.
- Prefer low-correlation candidates when momentum is similar.

### 3) Add trend/breakout factor as rank component
- Convert ATR breakout into a formal feature (`trend_state`, `breakout_strength`).
- Blend with momentum using explicit factor weights and z-score normalization.

### 4) Add absolute momentum gate
- Require positive absolute momentum for entry (not just relative rank).
- Route non-qualifying capital to cash or hedge sleeve.

### 5) Add sleeve-level risk budgeting
- Target sleeve volatility and max drawdown constraints.
- Add dynamic gross exposure and sleeve caps by regime.

### 6) Expand evaluation for crisis robustness
- Add regime-conditional metrics:
  - Performance in benchmark drawdown windows
  - Tail quantile returns
  - Convexity proxy during volatility spikes
- Run robustness tests on 2008, 2020, 2022-type windows.

## Practical Integration Order
1. Implement two-sleeve portfolio object and allocator (growth + hedge + cash).
2. Add absolute momentum and correlation terms to factor stack.
3. Introduce monthly overlay layer while preserving daily execution realism.
4. Add crisis-specific analytics and walk-forward validation.

## Final Research View
If the objective is to remain a high-quality research/backtest engine, your current implementation is already strong.  
If the objective is true antifragility, the missing piece is structural: **explicit capital transfer from risk-on sleeve to a validated defensive sleeve under stress signals**, not just tighter gates on the same long equity book.
