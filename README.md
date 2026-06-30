# Heston Model Volatility Surface Arbitrage — Research Report

## Summary
This report is a comprehensive technical investigation into building a **Heston Model-based volatility surface arbitrage system**. The concept involves: (1) downloading options chain data, (2) calibrating the Heston stochastic volatility model to the observed implied volatility surface, (3) identifying strikes where the market's implied volatility deviates from the Heston-implied "fair" volatility, and (4) executing delta-hedged volatility strategies (straddles/strangles) to capture mispricings. The project is a quantitative finance application that bridges statistical modeling, numerical optimization, and derivatives trading strategy execution.

## What This System Would Do

### Core Idea
The Heston model extends Black-Scholes by treating volatility as a **stochastic process** rather than a constant. This gives it the ability to naturally generate the volatility smile/skew that real options markets exhibit. If you calibrate Heston parameters to the options chain, you get a "fair value" surface. Strikes where the market IV diverges significantly from this surface represent potential arbitrage opportunities — the market is mispricing the option's volatility.

### Expected Workflow
1. **Data Ingestion** – Fetch options chain for a liquid underlying (e.g., SPY, AAPL) via `yfinance` or broker API
2. **Surface Construction** – Extract implied volatilities across all strikes × expirations into a 2D surface
3. **Heston Calibration** – Use numerical optimization to find Heston parameters that minimize the difference between market IVs and Heston-computed IVs
4. **Mispricing Detection** – Compute z-scores or deviation metrics per strike; flag outliers
5. **Strategy Generation** – For mispriced strikes, generate straddle/strangle trades with delta-neutral weighting
6. **P&L Attribution** – Decompose daily P&L into volatility capture, delta drift, theta decay, and gamma scalping components

---

## Mathematical Foundation

### The Heston Model (1993)
The model assumes two correlated stochastic differential equations:

**Asset price process (with dividends):**
```
dS_t = μ S_t dt + √v_t S_t dW_t^S
```

**Variance process (CIR-style mean-reverting):**
```
dv_t = κ(θ - v_t)dt + ξ√v_t dW_t^v
```

**Correlation structure:**
```
dW_t^S · dW_t^v = ρ dt
```

### Parameters to Calibrate

| Parameter | Symbol | Meaning | Typical Range |
|-----------|--------|---------|---------------|
| Mean reversion speed | κ | How fast variance reverts to long-term mean | 1–5 |
| Long-term variance | θ | The level v_t tends toward (≈ VIX²) | 0.01–0.10 |
| Volatility of volatility | ξ | "Vol of vol" — how jumpy variance is | 0.1–0.8 |
| Correlation | ρ | Leverage effect: negative for equities | -0.9 to 0.0 |
| Initial variance | v₀ | Current instantaneous variance | 0.01–0.10 |

### The Calibration Problem
Given a set of market option prices {C_mkt(Tᵢ, Kⱼ)}, find the parameter vector Ψ = (κ, θ, ξ, ρ, v₀) that minimizes:

```
min_Ψ Σᵢ Σⱼ wᵢⱼ · (IV_mkt(Tᵢ, Kⱼ) - IV_Heston(Tᵢ, Kⱼ; Ψ))²
```

Where `IV_Heston` is computed by:
1. Computing Heston characteristic function (closed-form, via `py_vollib` or manual implementation)
2. Inverting via Carr-Madan FFT or Lewis approach to get option prices
3. Inverting the Black-Scholes formula to convert price back to implied volatility

**This is the hardest part.** The characteristic function inversion is numerically sensitive. Key challenges:
- Branch cuts in the complex log function (the "Little Heston Trap" by Albrecher et al.)
- Choosing integration limits and quadrature scheme
- Avoiding the Heston "stiffness" during optimization

---

## Architecture Recommendation

### System Modules

```
heston-arbitrage/
├── data/
│   ├── downloader.py          — Fetches options chains via yfinance / broker API
│   ├── cleaner.py             — Filters illiquid strikes, handles dividends, splices chains
│   └── surface.py             — Builds 2D IV surface (strike × expiry) with interpolation
├── models/
│   ├── heston.py              — Heston characteristic function, price → IV conversion
│   ├── calibrator.py          — scipy.optimize wrapper: loss function, constraints, multi-start
│   └── black_scholes.py       — BS pricing/IV functions (wrapper around py_vollib)
├── detection/
│   ├── mispricing.py          — Deviation scoring: z-score, residual analysis, regime filtering
│   └── signals.py             — Signal generation: which strikes/expiries to trade
├── strategy/
│   ├── straddle.py            — Straddle construction, Greeks computation, delta hedging logic
│   ├── strangle.py            — Strangle (OTM put + OTM call) variant
│   └── portfolio.py           — Multi-leg position tracking, margin estimation
├── backtest/
│   ├── engine.py              — Event-driven backtesting loop with daily rebalance
│   ├── pnl.py                 — Attribution: vol capture vs delta drift vs gamma vs theta
│   └── metrics.py             — Sharpe, Sortino, max drawdown, hit rate, etc.
├── visualization/
│   ├── surface_plot.py        — 3D IV surface before/after calibration
│   └── pnl_decomposition.py   — Stacked area charts for P&L attribution
├── config.py                  — Ticker, optimization bounds, trade parameters
├── main.py                    — Orchestration: download → calibrate → detect → backtest
└── requirements.txt
```

### Data Flow (Detailed)

```
Step 1: yfinance.Ticker("SPY").option_chain(date)
         → DataFrame of calls/puts: strike, lastPrice, bid, ask, impliedVol, delta, gamma, theta, vega
         
Step 2: For each expiry, build a curve: IV(strike, T)
         → Surface class handles expiry interpolation (linear in total variance)
         
Step 3: Calibrator.solve(market_surface, initial_guess, bounds)
         → Multi-start optimization (e.g., 20 random starts + L-BFGS-B or differential_evolution)
         → Returns optimal (κ, θ, ξ, ρ, v₀)
         
Step 4: For each (strike, expiry) compute:
         IV_heston = heston_price_to_iv(Ψ, S, r, q, T, K, flag)
         deviation = market_iv - heston_iv
         z_score = deviation / std(deviation_across_surface)
         
Step 5: Filter cells where |z_score| > threshold (e.g., 1.5 or 2.0)
         → Generate straddle/strangle at that strike & expiry
         → Compute hedge ratios: delta-neutral position
         
Step 6: Daily:
         - Rebalance delta hedge (short underlying against long options position)
         - Track realized vs. implied variance
         - Attribute P&L
```

---

## Key Design Decisions & Non-Obvious Challenges

### 1. Calibration Is Fragile — Multi-Start Is Mandatory
The Heston calibration objective function is **non-convex with many local minima**. A single `scipy.optimize.minimize` call will almost certainly find a local (bad) minimum. Use `differential_evolution` (global optimizer from scipy) or a multi-start approach with 20–50 random initial parameter vectors.

### 2. The "Little Heston Trap"
When implementing the characteristic function, the complex logarithm's branch cut causes discontinuities. Use the **Albrecher et al. (2007)** formulation that re-expresses the characteristic function to avoid this. If using `py_vollib`, this is already handled — but if implementing manually, it's a critical pitfall.

### 3. Option Chain Data Quality
`yfinance` options data has known issues:
- **Stale quotes** near closing auctions
- **Illiquid strikes** with zero volume but wide bid-ask spreads
- **Dividend adjustments** that affect forward price
- **Missing expiries** on low-volume days
- **Handling**: filter only strikes with volume > 0 and bid > 0; use mid-price, not last price

### 4. Parallel Structure Problem
When the Heston correlation `ρ` approaches ±1, or when `ξ` (vol of vol) is large relative to κ, the characteristic function inversion becomes numerically unstable. Solution: enforce constraints during optimization:
```
ρ ∈ [-0.98, 0.98]
ξ² ≤ 2κθ  (Feller condition — bounds v_t > 0)
```

### 5. Attribution Is Not Trivial
Decomposing P&L into vol vs. delta vs. gamma requires daily re-pricing of the option position. The "volatility capture" component is the change in option value due to the gap between realized volatility and implied volatility at entry. This is computed via:
```
P&L_vol = θ_adjusted * (σ_implied² - σ_realized²) * dt
```
This approximation works only for small dt; for daily rebalancing it's adequate.

### 6. Transaction Costs Kill Small Mispricings
The strategy relies on buying options (long straddle/strangle) — options are subject to wide bid-ask spreads, especially OTM. Mispricings need to exceed the round-trip transaction cost by a significant margin (2×–3×) to be viable.

### 7. Regime Dependency
Heston parameter estimates drift over time. Calibration should be re-run at minimum weekly — daily is better for live trading. A rolling calibration window of 60–90 trading days for SPY options is typical.

---

## Tools & Dependencies

| Tool | Purpose | Notes |
|------|---------|-------|
| `py_vollib` | Black-Scholes IV computation | Wraps `lets_be_rational` for accurate BS IV |
| `scipy.optimize` | Heston calibration | `differential_evolution` for global, `minimize` (L-BFGS-B) for local refinement |
| `numpy` | Array operations | Complex number handling for characteristic function |
| `pandas` | DataFrame for options chains, surface construction | Panel data operations |
| `yfinance` | Options chain download | Free but rate-limited; higher-quality paid: Polygon, Tradier, IBKR |
| `matplotlib` / `plotly` | Surface plots, P&L charts | 3D surface + stack plots |
| `numba` (optional) | JIT-compile characteristic function for speed | ~20× speedup on calibration loop |

---

## Suggested Implementation Plan (Development Order)

1. **Data layer first** — Write `downloader.py` and `cleaner.py`. Get clean options chains into a DataFrame. Print and validate. This is the base everything depends on.

2. **Heston pricing** — Implement `heston_price_to_iv()` using py_vollib's `black_scholes.implied_volatility` and Heston price via `py_vollib.heston` or a custom `numba`-accelerated implementation. Test against known values.

3. **Calibration** — Write `calibrator.py`. Use `scipy.optimize.differential_evolution` with bounds. Plot the fitted surface vs. market surface. If RMS error is consistently < 2 vol points (0.02 IV), calibration is working.

4. **Mispricing detection** — Simple z-score thresholding first. Then add filtering for bid-ask spreads, open interest, volume.

5. **Strategy simulation** — Paper-trade with daily rebalancing. Compute P&L attribution using recorded daily option prices and underlying price moves.

6. **Backtest engine** — Full historical backtest over 2+ years. Account for slippage (half spread), commissions, and margin.

---

## What Would Surprise a Developer New to This

- **Heston prices require complex numbers and Fourier inversion.** The pricing formula involves integrating the characteristic function from 0 to ∞. A naive numerical integration is slow; FFT-based methods (Carr-Madan 1999) are faster but require careful setup.
- **The calibration takes 30 seconds to 2 minutes** for a single surface with ~100 options. That makes real-time calibration impossible — you calibrate once per day.
- **The Feller condition (2κθ > ξ²) is almost never satisfied** by real market data. The process can hit zero variance. Various "absorption" or "reflection" fixes exist in literature but Heston pricing formulas via characteristic function handle this in closed form regardless.
- **Dividends and interest rates matter** — the forward price must be correct. For SPY, the dividend yield is ~1.3% and changes the forward by 0.1–0.3% per month. Ignoring this shifts the entire IV surface.
- **The implied volatility surface from yfinance is already interpolated** by the exchange's closing auction process. You're fitting a model to a surface that's already been smoothed by market makers.

---

## Verification Metrics

A successful implementation should demonstrate:
1. **Calibration fit**: RMS IV error < 0.03 (3 vol points) on SPY surface
2. **Parameter stability**: Week-over-week parameter estimates that drift smoothly (not jump around)
3. **Detectable patterns**: Mispricings tend to cluster around events (earnings, FOMC) — verify this
4. **Non-negative P&L (before costs)**: The strategy should capture a small positive edge over 100+ trades; zero or negative P&L indicates the calibration is overfit
5. **Negative P&L after costs suggests**: The strategy isn't viable with retail-trading execution — you'd need a direct market access feed
