"""
portfolio_optimizer.py
=======================
A Markowitz mean-variance portfolio optimization and evaluation toolkit.

Given a list of ticker symbols, this module:
  - downloads historical price data and computes annualized return/risk
    statistics (Stage 1-2)
  - solves for the Global Minimum Variance and Maximum Sharpe Ratio
    portfolios, and traces the efficient frontier, via constrained
    numerical optimization (Stage 3)
  - evaluates a chosen portfolio's risk profile: VaR/CVaR (parametric
    and historical), max drawdown, Sortino/Calmar ratios, distribution
    shape (skew/kurtosis), and a diversification ratio, plus historical
    crisis stress tests (Stage 4)
  - runs the full pipeline end-to-end for any ticker list (Stage 5)
  - performs greedy forward asset selection from a larger candidate
    universe (Stage 6)

Quick start:
    from portfolio_optimizer import run_portfolio_analysis

    results = run_portfolio_analysis(["VXUS", "BND", "GLD", "VNQ"])

See README.md for methodology notes and interpretation guidance.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ============================================================
# STAGE 1 — DATA (generalized to N tickers)
# ============================================================

def load_portfolio_data(tickers, start="2010-01-01"):
    """
    Downloads adjusted close prices for any list of tickers, computes
    daily returns, and returns:
      - returns_df: DataFrame of daily returns (columns = tickers)
      - mu: array of annualized expected returns (naive historical mean)
      - Sigma: annualized covariance matrix (as a plain numpy array)
      - annual_vols: array of each asset's own annualized volatility
    This replaces the separate eq_data/bond_data blocks — same logic,
    just looped over an arbitrary list instead of hardcoded twice.
    """
    # Download all tickers in ONE call — yfinance handles multiple
    # tickers natively and aligns them on the same date index for you,
    # which avoids the manual pd.concat step you had to do for 2 assets.
    raw = yf.download(tickers, start=start, auto_adjust=True)["Close"]

    # If only one ticker is passed, yfinance returns a Series, not a
    # DataFrame — normalize that edge case.
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=tickers[0])

    returns_df = raw.pct_change().dropna()

    # Same cleaning logic as before — drop any ticker with too much
    # missing data, then re-drop NaN rows so everything aligns exactly.
    missing_pct = returns_df.isna().mean()
    good_tickers = missing_pct[missing_pct < 0.05].index.tolist()
    if len(good_tickers) < len(tickers):
        dropped = set(tickers) - set(good_tickers)
        print(f"Dropped tickers with excessive missing data: {dropped}")
    returns_df = returns_df[good_tickers].dropna()

    mu = returns_df.mean().values * 252
    annual_vols = returns_df.std().values * np.sqrt(252)
    Sigma = (returns_df.cov() * 252).values

    return returns_df, mu, Sigma, annual_vols, good_tickers


# ============================================================
# STAGE 2 — PORTFOLIO METRICS (generalized — works for any n)
# ============================================================

def portfolio_variance(w, Sigma):
    return w @ Sigma @ w

def portfolio_stats(w, mu, Sigma, risk_free_rate):
    ret = w @ mu
    vol = np.sqrt(portfolio_variance(w, Sigma))
    sharpe = (ret - risk_free_rate) / vol
    return ret, vol, sharpe


# ============================================================
# STAGE 3 — OPTIMIZERS (already general — reused as-is from before)
# ============================================================

def find_min_variance_portfolio(mu, Sigma, target_return, allow_short=False):
    n = len(mu)
    initial_guess = np.repeat(1 / n, n)
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu - target_return},
    ]
    bounds = [(0, 1)] * n if not allow_short else None
    result = minimize(portfolio_variance, initial_guess, args=(Sigma,),
                       method="SLSQP", constraints=constraints, bounds=bounds)
    if not result.success:
        print("Warning: optimizer did not converge —", result.message)
    return result.x

def find_gmv_portfolio(Sigma, allow_short=False):
    n = Sigma.shape[0]
    initial_guess = np.repeat(1 / n, n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n if not allow_short else None
    result = minimize(portfolio_variance, initial_guess, args=(Sigma,),
                       method="SLSQP", constraints=constraints, bounds=bounds)
    return result.x

def negative_sharpe(w, mu, Sigma, risk_free_rate):
    ret, vol, sharpe = portfolio_stats(w, mu, Sigma, risk_free_rate)
    return -sharpe

def find_max_sharpe_portfolio(mu, Sigma, risk_free_rate, allow_short=False):
    n = len(mu)
    initial_guess = np.repeat(1 / n, n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n if not allow_short else None
    result = minimize(negative_sharpe, initial_guess, args=(mu, Sigma, risk_free_rate),
                       method="SLSQP", constraints=constraints, bounds=bounds)
    if not result.success:
        print("Warning: optimizer did not converge —", result.message)
    return result.x


# ============================================================
# STAGE 4 — POST-SELECTION RISK EVALUATION
# ============================================================
# These metrics assess how a chosen portfolio would actually behave —
# tail risk, path risk (drawdown), distribution shape, and how much
# real diversification benefit it's capturing. Reference thresholds are
# rules of thumb from common practice, not hard rules — always interpret
# relative to the specific asset class and time period.

def evaluate_portfolio_risk(w, returns_df, mu, Sigma, annual_vols,
                              risk_free_rate, confidence=0.95, freq=252,
                              portfolio_size=1_000_000, label="Portfolio"):
    """
    Computes a full risk evaluation for a given weight vector, using the
    SAME historical returns_df that mu/Sigma were built from.
    """
    tickers = list(returns_df.columns)
    port_daily_returns = returns_df.values @ w   # daily portfolio return series

    ret = w @ mu
    vol = np.sqrt(w @ Sigma @ w)
    sharpe = (ret - risk_free_rate) / vol

    z = stats.norm.ppf(confidence)  # e.g. 1.645 for 95%

    # --- Parametric (normal-distribution) VaR and CVaR ---
    var_parametric = portfolio_size * (z * vol - ret)
    # Parametric CVaR uses the normal distribution's tail expectation formula
    cvar_parametric = portfolio_size * (
        (stats.norm.pdf(z) / (1 - confidence)) * vol - ret
    )

    # --- Historical (empirical) VaR and CVaR ---
    # Annualize by scaling the empirical daily distribution's tail, using
    # the same sqrt(time) logic as elsewhere, applied to a 1-year horizon
    annual_port_returns = (1 + port_daily_returns) ** freq - 1  # rough annualized path, illustrative
    daily_var_pct = -np.percentile(port_daily_returns, (1 - confidence) * 100)
    var_historical = portfolio_size * daily_var_pct * np.sqrt(freq)  # scaled to 1yr, approximate

    tail_losses = port_daily_returns[port_daily_returns <= -daily_var_pct]
    cvar_historical = portfolio_size * (-tail_losses.mean()) * np.sqrt(freq) if len(tail_losses) > 0 else np.nan

    # --- Max drawdown (from the actual historical daily path) ---
    cumulative = (1 + port_daily_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # --- Sortino ratio (downside-only volatility) ---
    downside_returns = port_daily_returns[port_daily_returns < 0]
    downside_vol_annual = downside_returns.std() * np.sqrt(freq)
    sortino = (ret - risk_free_rate) / downside_vol_annual if downside_vol_annual > 0 else np.nan

    # --- Calmar ratio (return / max drawdown, not volatility) ---
    calmar = ret / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # --- Skewness and kurtosis (distribution shape) ---
    skew = stats.skew(port_daily_returns)
    kurt = stats.kurtosis(port_daily_returns)  # excess kurtosis (normal = 0)

    # --- Diversification ratio ---
    # (weighted avg of individual vols) / (actual portfolio vol)
    weighted_avg_vol = w @ annual_vols
    diversification_ratio = weighted_avg_vol / vol

    # ---------------- PRINT RESULTS ----------------
    # Interpretation guidance for each of these metrics (what typical
    # ranges mean, how to read VaR vs CVaR, etc.) is in README.md under
    # "Interpreting the risk metrics" — kept out of this printed output
    # since evaluate_portfolio_risk runs multiple times per session and
    # repeating multi-paragraph explanations each time gets unwieldy fast.
    print(f"\n{'=' * 55}")
    print(f"RISK EVALUATION — {label}")
    print(f"{'=' * 55}")

    print(f"\nExpected return: {ret:.2%} | Volatility: {vol:.2%} | Sharpe: {sharpe:.2f}")

    print(f"\n--- Tail risk (95% confidence, ${portfolio_size:,.0f} portfolio) ---")
    print(f"Parametric VaR:   ${var_parametric:,.0f}  ({var_parametric/portfolio_size:.2%})")
    print(f"Parametric CVaR:  ${cvar_parametric:,.0f}  ({cvar_parametric/portfolio_size:.2%})")
    print(f"Historical VaR:   ${var_historical:,.0f}  ({var_historical/portfolio_size:.2%})")
    print(f"Historical CVaR:  ${cvar_historical:,.0f}  ({cvar_historical/portfolio_size:.2%})")

    print(f"\n--- Path risk ---")
    print(f"Max drawdown: {max_drawdown:.2%}")

    print(f"\n--- Risk-adjusted return ratios ---")
    print(f"Sortino ratio: {sortino:.2f}   Calmar ratio: {calmar:.2f}")

    print(f"\n--- Distribution shape ---")
    print(f"Skewness: {skew:.2f}   Excess kurtosis: {kurt:.2f}")

    print(f"\n--- Diversification ---")
    print(f"Diversification ratio: {diversification_ratio:.2f}")

    print(f"\n(See README.md \"Interpreting the risk metrics\" for guidance on "
          f"what these values typically mean.)")

    return {
        "return": ret, "volatility": vol, "sharpe": sharpe,
        "var_parametric": var_parametric, "cvar_parametric": cvar_parametric,
        "var_historical": var_historical, "cvar_historical": cvar_historical,
        "max_drawdown": max_drawdown, "sortino": sortino, "calmar": calmar,
        "skew": skew, "kurtosis": kurt,
        "diversification_ratio": diversification_ratio,
    }


def stress_test_portfolio(w, returns_df, scenarios, portfolio_size=1_000_000, label="Portfolio"):
    """
    Applies historical crisis-period returns to the CURRENT weights, to
    see what the portfolio would have lost if that scenario repeated.

    scenarios: dict of {scenario_name: (start_date, end_date)}
    """
    print(f"\n{'=' * 55}")
    print(f"STRESS TEST — {label}")
    print(f"{'=' * 55}")

    for name, (start, end) in scenarios.items():
        window = returns_df.loc[start:end]
        if window.empty:
            print(f"{name}: no data available in this window for these assets")
            continue
        # Compound the daily returns over the window, then apply weights
        cumulative_asset_returns = (1 + window).prod() - 1
        scenario_portfolio_return = cumulative_asset_returns.values @ w
        dollar_impact = portfolio_size * scenario_portfolio_return
        print(f"{name} ({start} to {end}): portfolio return = "
              f"{scenario_portfolio_return:.2%}  (${dollar_impact:,.0f} on "
              f"${portfolio_size:,.0f})")


# ============================================================
# USAGE — call this function with any ticker list you want.
# Everything above (data loading, optimizers) only needs to be run
# ONCE per session; this function is what you call repeatedly with
# different arguments.
# ============================================================

def run_portfolio_analysis(tickers, risk_free_rate=0.0469, start="2010-01-01",
                             compare_to=None, compare_label="This portfolio",
                             baseline_label="Baseline", save_as="auto",
                             plot_frontier=False, evaluate_gmv=False):
    """
    Runs the full pipeline for any ticker list: load data, compute GMV
    and max-Sharpe portfolios.

    compare_to: optionally pass a previous results dict to print a
        side-by-side comparison against this one, at the end of this call.

    save_as: label to save this result into the module-level portfolio
        registry (see compare_all_portfolios()). Defaults to "auto", which
        generates a label from the ticker list itself (e.g. "VXUS+BND"),
        so every call is saved automatically and nothing gets silently
        lost between calls — unlike reusing the same Python variable name
        for different calls, which DOES overwrite the earlier one (that's
        ordinary Python variable assignment, not something this function
        controls). Pass an explicit string to use a custom label instead,
        or save_as=None to skip saving entirely.

    plot_frontier: if True, plots the efficient frontier. Defaults to
        False — the frontier plot mostly re-illustrates the GMV/Max-Sharpe
        points you already get as numbers, so it's off by default to keep
        output focused on the tables and risk evaluation. Set True for a
        one-off visual, e.g. when first exploring a new asset universe.

    evaluate_gmv: if True, runs the full risk evaluation (VaR/CVaR,
        drawdown, Sortino/Calmar, skew/kurtosis, diversification ratio)
        for the GMV portfolio too, not just Max-Sharpe. Defaults to False,
        since GMV's weights/return/volatility are still always computed
        and shown in the summary table regardless — this only controls
        whether its FULL risk breakdown is also printed. Worth turning on
        occasionally as a sanity check: GMV minimizes variance specifically,
        not tail risk directly, so it's possible (if unusual) for GMV to
        have worse VaR/CVaR than Max-Sharpe despite its lower volatility.

    Example:
        run_portfolio_analysis(["VXUS", "BND", "GLD", "VNQ"])              # auto-saved as "VXUS+BND+GLD+VNQ"
        run_portfolio_analysis(["VXUS", "BND", "GLD", "VNQ", "TLT"], save_as="Manual 5-asset")
        compare_all_portfolios()   # compares everything saved so far
    """
    returns_df, mu_naive, Sigma, annual_vols, tickers_used = load_portfolio_data(
        tickers, start=start)

    # --- Asset-level stats table: tickers as columns, metrics as rows,
    # formatted as percentages to 2 decimal places. Built with already-
    # formatted strings (rather than DataFrame.applymap/.map) so this
    # works across pandas versions without depending on either method.
    asset_table = pd.DataFrame(
        {t: [f"{mu_naive[i]:.2%}", f"{annual_vols[i]:.2%}"] for i, t in enumerate(tickers_used)},
        index=["Expected Return", "Volatility"],
    )
    print("Asset-level statistics (annualized):")
    print(asset_table.to_string())

    w_gmv = find_gmv_portfolio(Sigma)
    ret_gmv, vol_gmv, sharpe_gmv = portfolio_stats(w_gmv, mu_naive, Sigma, risk_free_rate)

    w_max_sharpe = find_max_sharpe_portfolio(mu_naive, Sigma, risk_free_rate)
    ret_ms, vol_ms, sharpe_ms = portfolio_stats(w_max_sharpe, mu_naive, Sigma, risk_free_rate)

    # --- Weights table: tickers as columns, one row per portfolio,
    # formatted as percentages to 2 decimal places ---
    weights_table = pd.DataFrame(
        {t: [f"{w_gmv[i]:.2%}", f"{w_max_sharpe[i]:.2%}"] for i, t in enumerate(tickers_used)},
        index=["GMV Weight", "Max-Sharpe Weight"],
    )
    print("\nPortfolio weights:")
    print(weights_table.to_string())

    # --- Summary stats table: one row per portfolio ---
    summary_table = pd.DataFrame(
        {
            "Expected Return": [f"{ret_gmv:.2%}", f"{ret_ms:.2%}"],
            "Volatility": [f"{vol_gmv:.2%}", f"{vol_ms:.2%}"],
            "Sharpe": [f"{sharpe_gmv:.2f}", f"{sharpe_ms:.2f}"],
        },
        index=["GMV", "Max-Sharpe"],
    )
    print("\nPortfolio summary:")
    print(summary_table.to_string())

    if plot_frontier:
        # Efficient frontier — genuinely curved with 3+ assets, since Sigma
        # actually gets to influence the shape (unlike the 2-asset straight line)
        target_returns = np.linspace(mu_naive.min(), mu_naive.max(), 40)
        frontier_vols = []
        for r in target_returns:
            w = find_min_variance_portfolio(mu_naive, Sigma, r)
            frontier_vols.append(np.sqrt(portfolio_variance(w, Sigma)))

        plt.figure(figsize=(8, 5))
        plt.plot(frontier_vols, target_returns, marker="o", markersize=3, label="Efficient frontier")
        plt.scatter(annual_vols, mu_naive, color="red", zorder=5)
        for i, name in enumerate(tickers_used):
            plt.annotate(name, (annual_vols[i], mu_naive[i]))
        plt.scatter([vol_gmv], [ret_gmv], color="green", marker="*", s=150, label="GMV portfolio", zorder=6)
        plt.scatter([vol_ms], [ret_ms], color="purple", marker="*", s=150, label="Max-Sharpe portfolio", zorder=6)
        plt.xlabel("Volatility (annualized)")
        plt.ylabel("Expected return (annualized)")
        plt.title(f"Efficient Frontier — {', '.join(tickers_used)}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"efficient_frontier_{'_'.join(tickers_used)}.png", dpi=150)
        plt.show()

    # --- Risk evaluation ---
    # GMV's detailed risk evaluation is skipped by default (evaluate_gmv=False)
    # since it's rarely acted on once Max-Sharpe is available — GMV's weights,
    # return, and volatility are still computed above regardless. Pass
    # evaluate_gmv=True to include it, e.g. as a sanity check that GMV
    # isn't hiding worse tail risk despite its lower volatility.
    eval_gmv = None
    if evaluate_gmv:
        eval_gmv = evaluate_portfolio_risk(
            w_gmv, returns_df, mu_naive, Sigma, annual_vols, risk_free_rate,
            label="GMV portfolio")

    eval_max_sharpe = evaluate_portfolio_risk(
        w_max_sharpe, returns_df, mu_naive, Sigma, annual_vols, risk_free_rate,
        label="Max-Sharpe portfolio")

    # --- Historical stress tests, applied to the max-Sharpe weights ---
    # Adjust/add scenarios as needed — dates only matter for periods where
    # your chosen tickers actually have data.
    crisis_scenarios = {
        "2020 COVID crash": ("2020-02-19", "2020-03-23"),
        "2022 rate-hike selloff": ("2022-01-01", "2022-10-12"),
    }
    stress_test_portfolio(w_max_sharpe, returns_df, crisis_scenarios,
                           label="Max-Sharpe portfolio")

    current_results = {
        "returns_df": returns_df, "mu": mu_naive, "Sigma": Sigma,
        "annual_vols": annual_vols, "tickers": tickers_used,
        "w_gmv": w_gmv, "w_max_sharpe": w_max_sharpe,
        "gmv_stats": (ret_gmv, vol_gmv, sharpe_gmv),
        "max_sharpe_stats": (ret_ms, vol_ms, sharpe_ms),
        "eval_gmv": eval_gmv, "eval_max_sharpe": eval_max_sharpe,
    }

    # If a previous portfolio was passed in, print the comparison here —
    # this is what lets you get a comparison from a single function call,
    # as long as you already have the earlier result to hand it.
    if compare_to is not None:
        compare_portfolios({
            baseline_label: compare_to,
            compare_label: current_results,
        })

    if save_as == "auto":
        save_as = "+".join(tickers_used)
    if save_as is not None:
        register_portfolio(save_as, current_results)

    return current_results


# --- Example calls — run any of these, as many times as you like ---
# results = run_portfolio_analysis(["VXUS", "BND", "GLD", "VNQ"])
# results = run_portfolio_analysis(["AAPL", "MSFT", "TLT", "GLD"], risk_free_rate=0.045)


def forward_stepwise_selection(candidate_tickers, risk_free_rate=0.0469,
                                 start="2010-01-01", max_assets=None,
                                 min_improvement=0.005):
    """
    Greedily builds a portfolio by adding one asset at a time — whichever
    candidate improves the max-Sharpe portfolio's Sharpe ratio the most —
    until no further candidate improves it by at least min_improvement
    (or max_assets is reached).

    This is a long-only-compatible alternative to L1-based sparsity:
    instead of starting with everything and shrinking some weights to
    zero, it starts with nothing and adds assets one at a time, which
    stays entirely within simple (0, 1) weight bounds throughout.

    min_improvement: the minimum Sharpe ratio increase required to keep
        adding assets (default 0.005). This matters more than it might
        look: adding an asset to a long-only max-Sharpe portfolio can
        mathematically never REDUCE the achievable Sharpe ratio, since
        the optimizer can always set a new asset's weight to exactly 0
        if it doesn't help. Combined with ordinary numerical noise from
        the optimizer (it rarely lands on the exact same floating-point
        value twice), a naive "stop only when improvement is <= 0" rule
        essentially never triggers — some candidate will almost always
        look infinitesimally better, even when the true improvement is
        economically meaningless. min_improvement enforces a genuine,
        meaningful threshold instead of relying on exact-zero comparison.

    Returns the final selected ticker list and its risk/return stats.
    """
    returns_df, mu_full, Sigma_full, annual_vols_full, all_tickers = \
        load_portfolio_data(candidate_tickers, start=start)

    selected = []
    remaining = list(all_tickers)
    best_sharpe_so_far = -np.inf
    history = []

    max_assets = max_assets or len(all_tickers)

    print(f"{'=' * 60}")
    print("FORWARD STEPWISE SELECTION")
    print(f"(minimum Sharpe improvement to keep adding: {min_improvement})")
    print(f"{'=' * 60}")

    while remaining and len(selected) < max_assets:
        best_candidate = None
        best_candidate_sharpe = best_sharpe_so_far

        for candidate in remaining:
            trial_assets = selected + [candidate]
            idx = [all_tickers.index(t) for t in trial_assets]
            mu_trial = mu_full[idx]
            Sigma_trial = Sigma_full[np.ix_(idx, idx)]

            w_trial = find_max_sharpe_portfolio(mu_trial, Sigma_trial, risk_free_rate)
            _, _, sharpe_trial = portfolio_stats(w_trial, mu_trial, Sigma_trial, risk_free_rate)

            if sharpe_trial > best_candidate_sharpe:
                best_candidate_sharpe = sharpe_trial
                best_candidate = candidate

        # FIX: require a MEANINGFUL improvement, not just any improvement
        # at all — see min_improvement docstring above for why the naive
        # "> 0" check essentially never stops the loop in practice.
        if best_candidate is None or (best_candidate_sharpe - best_sharpe_so_far) < min_improvement:
            if best_candidate is not None:
                print(f"\nBest remaining candidate ({best_candidate}) only improves "
                      f"Sharpe by {best_candidate_sharpe - best_sharpe_so_far:.4f}, "
                      f"below the {min_improvement} threshold — stopping.")
            else:
                print(f"\nNo remaining candidate improves Sharpe further "
                      f"(currently {best_sharpe_so_far:.3f}) — stopping.")
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        improvement = best_candidate_sharpe - best_sharpe_so_far
        best_sharpe_so_far = best_candidate_sharpe
        history.append((best_candidate, best_sharpe_so_far))

        print(f"Step {len(selected)}: added {best_candidate:<8} "
              f"-> Sharpe = {best_sharpe_so_far:.3f} "
              f"(+{improvement:.4f})")

    print(f"\nFinal selected assets: {selected}")
    print(f"Final Sharpe ratio: {best_sharpe_so_far:.3f}")
    print("(See README.md \"Interpreting the risk metrics\" for guidance on "
          "reading the per-step improvement.)")

    return selected, history


# ============================================================
# STAGE 7 — COMPARING MULTIPLE PORTFOLIOS
# ============================================================

# A simple module-level store so portfolios can be saved by label as you
# go, then compared all at once — rather than manually tracking a growing
# set of variables (results_1, results_2, results_3, ...) yourself.
#
# Worth knowing since it's slightly different from everything else in this
# file: this IS mutable state that persists across function calls within
# the same Python/notebook session — not a pure function. It resets if you
# restart the kernel, and re-running a cell that calls run_portfolio_analysis
# with the same save_as label will silently overwrite the earlier entry
# (a warning is printed when that happens).
_portfolio_registry = {}


def register_portfolio(label, results, overwrite=True):
    """
    Saves a results dict (from run_portfolio_analysis) under a label, for
    later comparison via compare_all_portfolios(). Called automatically
    if you pass save_as=<label> to run_portfolio_analysis — you generally
    won't need to call this directly.
    """
    if label in _portfolio_registry and not overwrite:
        print(f"'{label}' already saved — pass overwrite=True to replace it. Skipping.")
        return
    if label in _portfolio_registry:
        print(f"Note: overwriting previously saved portfolio '{label}'.")
    _portfolio_registry[label] = results


def list_saved_portfolios():
    """Prints the labels currently saved in the registry."""
    if not _portfolio_registry:
        print("No portfolios saved yet.")
        return
    print("Saved portfolios:", list(_portfolio_registry.keys()))


def clear_portfolio_registry():
    """Clears all saved portfolios — useful when starting a fresh comparison."""
    _portfolio_registry.clear()
    print("Portfolio registry cleared.")


def compare_all_portfolios(stat_key="max_sharpe_stats"):
    """
    Compares every portfolio currently saved in the registry, in the
    order they were added. Requires at least 2 saved portfolios.
    """
    if len(_portfolio_registry) < 2:
        print(f"Need at least 2 saved portfolios to compare — "
              f"currently have {len(_portfolio_registry)}. "
              f"Use save_as=<label> in run_portfolio_analysis() to add more.")
        return
    compare_portfolios(_portfolio_registry, stat_key=stat_key)



def compare_portfolios(portfolios, stat_key="max_sharpe_stats"):
    """
    Prints a side-by-side comparison table for any number of portfolios,
    each produced by run_portfolio_analysis(). Includes return/vol/Sharpe
    plus the full risk evaluation (VaR, CVaR, drawdown, Sortino, Calmar,
    skew, kurtosis, diversification ratio) for whichever portfolio type
    stat_key selects.

    portfolios: dict mapping a label (str) to a results dict returned by
                run_portfolio_analysis(), e.g.:
                    {
                        "Manual (4 assets)": results,
                        "Stepwise (5 assets)": selected_results,
                    }
    stat_key: which portfolio to compare — "max_sharpe_stats" (default,
              uses the eval_max_sharpe risk evaluation) or "gmv_stats"
              (uses eval_gmv).

    See README.md "Interpreting the risk metrics" for guidance on reading
    these values.

    Example:
        compare_portfolios({
            "Manual (4 assets)": results,
            "Stepwise (5 assets)": selected_results,
        })
    """
    labels = list(portfolios.keys())
    eval_key = "eval_gmv" if stat_key == "gmv_stats" else "eval_max_sharpe"

    # If comparing GMV portfolios but evaluate_gmv=False was used when they
    # were built, eval_gmv will be None for one or more of them — fall back
    # to return/vol/Sharpe only rather than erroring.
    eval_available = all(portfolios[l].get(eval_key) is not None for l in labels)
    if not eval_available:
        print(f"Note: full risk evaluation not available for one or more "
              f"portfolios under '{eval_key}' — showing return/volatility/"
              f"Sharpe only. Pass evaluate_gmv=True to run_portfolio_analysis() "
              f"to include it for GMV comparisons.\n")

    # Row spec: (display name, source, key/index, format)
    # source is "stats" for the (return, vol, sharpe) tuple, "eval" for
    # the evaluate_portfolio_risk() dict, or "meta" for anything else.
    rows = [
        ("Expected Return", "stats", 0, "{:.2%}"),
        ("Volatility", "stats", 1, "{:.2%}"),
        ("Sharpe", "stats", 2, "{:.2f}"),
    ]
    if eval_available:
        rows += [
            ("Parametric VaR", "eval", "var_parametric", "${:,.0f}"),
            ("Parametric CVaR", "eval", "cvar_parametric", "${:,.0f}"),
            ("Historical VaR", "eval", "var_historical", "${:,.0f}"),
            ("Historical CVaR", "eval", "cvar_historical", "${:,.0f}"),
            ("Max Drawdown", "eval", "max_drawdown", "{:.2%}"),
            ("Sortino Ratio", "eval", "sortino", "{:.2f}"),
            ("Calmar Ratio", "eval", "calmar", "{:.2f}"),
            ("Skewness", "eval", "skew", "{:.2f}"),
            ("Excess Kurtosis", "eval", "kurtosis", "{:.2f}"),
            ("Diversification Ratio", "eval", "diversification_ratio", "{:.2f}"),
        ]
    rows.append(("# Assets", "meta", "n_assets", "{:.0f}"))

    table = {}
    for label in labels:
        col = []
        for _, source, key, fmt in rows:
            if source == "stats":
                value = portfolios[label][stat_key][key]
            elif source == "eval":
                value = portfolios[label][eval_key][key]
            else:  # meta
                value = len(portfolios[label]["tickers"])
            col.append(fmt.format(value))
        table[label] = col

    df = pd.DataFrame(table, index=[r[0] for r in rows])
    print(f"Comparison ({'Max-Sharpe' if stat_key == 'max_sharpe_stats' else 'GMV'} portfolios):")
    print(df.to_string())

    best_label = max(labels, key=lambda l: portfolios[l][stat_key][2])
    print(f"\nHighest Sharpe: {best_label} "
          f"({portfolios[best_label][stat_key][2]:.3f})")
