# Portfolio Optimizer

A Markowitz mean-variance portfolio optimization and risk evaluation toolkit, built in Python. Given any list of ticker symbols, it computes optimal portfolio weights, traces the efficient frontier, and produces a full risk report — including tail risk (VaR/CVaR), drawdown, and distribution-shape diagnostics.

## What it does

1. **Data pipeline** — downloads historical prices for any list of tickers (via `yfinance`), computes daily returns, and builds annualized expected return (μ) and covariance (Σ) estimates.
2. **Optimization** — solves for:
   - the **Global Minimum Variance (GMV)** portfolio
   - the **Maximum Sharpe Ratio** portfolio
   - the full **efficient frontier** (minimum-variance weights across a range of target returns)

   using constrained numerical optimization (`scipy.optimize.minimize`, SLSQP), with long-only bounds by default.
3. **Risk evaluation** — for any chosen weight vector, reports:
   - Parametric and historical **Value at Risk (VaR)** and **Conditional VaR (CVaR)**
   - **Maximum drawdown**
   - **Sortino** and **Calmar** ratios
   - **Skewness** and **excess kurtosis** (to flag where the normal-distribution assumption behind parametric VaR is likely understating real risk)
   - A **diversification ratio** (how much real diversification benefit the portfolio is capturing)
4. **Stress testing** — applies real historical crisis-period returns (e.g. the 2020 COVID crash, the 2022 rate-hike selloff) to the current portfolio weights.
5. **Asset selection** — a greedy **forward stepwise selection** routine that builds a portfolio from a larger candidate universe by repeatedly adding whichever asset improves the max-Sharpe ratio the most, stopping once no candidate helps further.
6. **Comparing portfolios** — `run_portfolio_analysis(..., compare_to=<previous result>)` prints a side-by-side comparison against one earlier portfolio in the same call. For comparing three or more, `run_portfolio_analysis(..., save_as="label")` saves each portfolio into a shared registry as you go, and `compare_all_portfolios()` prints all of them together in one table.

## Quick start

```bash
pip install -r requirements.txt
```

```python
from portfolio_optimizer import run_portfolio_analysis

results = run_portfolio_analysis(["VXUS", "BND", "GLD", "VNQ"])
```

This prints the GMV and Max-Sharpe portfolio weights, plots the efficient frontier, runs the full risk evaluation on both portfolios, and stress-tests the Max-Sharpe portfolio against historical crisis windows.

### Selecting assets from a larger universe

```python
from portfolio_optimizer import forward_stepwise_selection, run_portfolio_analysis

candidates = ["VXUS", "BND", "GLD", "VNQ", "TLT", "DBC", "VWO", "SHY"]
selected_tickers, history = forward_stepwise_selection(candidates, max_assets=5)

results = run_portfolio_analysis(selected_tickers)
```

See `notebooks/demo.ipynb` for a full worked example with output.

## Methodology notes

- **Expected returns** default to naive historical means, which is a known limitation — see "Limitations" below.
- **Covariance** is the sample covariance of daily returns, annualized by ×252.
- All optimization is **long-only** by default (`allow_short=False`); pass `allow_short=True` to permit negative weights.
- Parametric VaR/CVaR assume normally distributed returns; historical VaR/CVaR make no distributional assumption. The two are reported side by side deliberately — a large gap between them is itself informative (it means the normal-distribution assumption is materially understating tail risk for that asset mix).

## Interpreting the risk metrics

`evaluate_portfolio_risk()` and `compare_portfolios()`/`compare_all_portfolios()` print the numeric results only — the guidance below explains what typical values mean.

**VaR vs. CVaR (tail risk)**
CVaR should always be ≥ VaR (it's the average loss *beyond* the VaR threshold, so it can't be smaller). If historical VaR is notably larger than parametric VaR, real returns have fatter tails than the normal distribution assumes — the parametric number is understating risk for that asset mix.

**Max drawdown (path risk)**
-10% to -20%: a routine correction. -20% to -40%: a typical bear-market-level drawdown. Beyond -40%: a severe, crisis-level decline (e.g. 2008 GFC, 2000 dot-com).

**Sortino and Calmar ratios (risk-adjusted return)**
Same rough scale as Sharpe: below 0 means losing money relative to the risk-free rate; 0–1 is mediocre; 1–2 is good; above 2 is very good (be skeptical of anything above 3 in a backtest — check for overfitting/data snooping before trusting it). Sortino being higher than Sharpe is expected and healthy, since it excludes upside volatility from the penalty. Calmar reflects return per unit of *worst-case* pain rather than average volatility.

**Skewness and excess kurtosis (distribution shape)**
Skew = 0 is symmetric; negative skew (common in equities) means large losses are more extreme/frequent than large gains — worse than a normal distribution suggests. Excess kurtosis = 0 matches a normal distribution; kurtosis > 0 ("fat tails") means extreme moves in either direction happen more often than a normal distribution predicts — a common, well-documented feature of real asset returns, and the reason historical VaR/CVaR often exceed the parametric estimates above.

**Diversification ratio**
Ratio = 1 means no diversification benefit (as if correlations were effectively 1 and you'd just held one asset scaled up). A ratio well above 1 (roughly 1.3–1.5+) means the portfolio's actual volatility is meaningfully lower than a naive blend of the individual asset volatilities would suggest — genuine diversification is being captured. Higher is generally better, holding return constant.

**Forward stepwise selection — improvement per step and `min_improvement`**
Diminishing improvement per step is expected — the first few assets usually add the most value (the biggest diversification gains), with later additions offering less. The `min_improvement` parameter (default `0.005`) enforces a real stopping point: without it, the algorithm would keep adding assets even when the true benefit is negligible, since in a long-only optimizer adding an asset can never mathematically *reduce* the achievable Sharpe ratio (the optimizer can always assign it zero weight).Raise `min_improvement` for a stricter, more parsimonious selection; lower it to allow smaller marginal gains through.

## Limitations

- **Historical mean returns are a noisy estimator of true expected return.** This is a known, fundamental issue — feeding raw historical means into a mean-variance optimizer is prone to producing unstable, overconcentrated weights, since the optimizer effectively bets hardest on whichever asset got lucky in the sample period. More robust alternatives used in practice include CAPM/factor-implied returns, Black-Litterman and shrinkage estimators.
- **Survivorship bias** — data sourced via `yfinance`/Yahoo Finance only reflects currently-listed tickers, which can overstate historical performance versus a point-in-time universe that included since-delisted assets.
- **Square-root-of-time scaling** (used for annualizing volatility and historical VaR) assumes i.i.d. returns, which real markets only approximately satisfy — volatility clustering (well captured by GARCH-type models) means realized risk is time-varying in ways this static annualization doesn't reflect.
- **Forward stepwise selection is a greedy heuristic**, not an exhaustive search — it isn't guaranteed to find the globally optimal subset of assets, though it's computationally tractable where exhaustive search over large candidate universes is not.
- **The portfolio comparison registry (`save_as`, `compare_all_portfolios`) does not persist between sessions.** It's an in-memory Python dictionary that exists only for the lifetime of the running kernel/process — restarting the kernel, closing the notebook, or restarting your machine clears it. Portfolios saved via `save_as` must be re-run in the current session before `compare_all_portfolios()` has anything to compare.

## Requirements

See `requirements.txt`. Requires internet access to fetch price data via `yfinance`.

## License

MIT
