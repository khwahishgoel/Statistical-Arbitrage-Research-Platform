"""
Statistical Arbitrage Research Platform
Phase 8a: Hidden Markov Model — Regime Detection (Python)

Fits a 2-state HMM on market returns to detect:
  - State 0: mean-reverting regime  → stat arb signals ON
  - State 1: trending regime        → stat arb signals OFF

Then re-runs the backtest with the regime filter applied
and compares performance vs the unfiltered strategy.
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

load_dotenv(Path(__file__).parent / ".env")

from db import load_prices, trades_col, get_db

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# HMM parameters
N_STATES      = 2      # mean-reverting vs trending
N_ITER        = 200    # EM algorithm iterations
RANDOM_STATE  = 42

# Signal thresholds (same as phase 5)
ENTRY_Z  = 2.0
EXIT_Z   = 0.5
STOP_Z   = 3.5
POSITION_SIZE = 10_000
COST_BPS = 10  # total cost per leg


# ── Load data ─────────────────────────────────────────────────────────────────
def load_market_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Build features for the HMM from SPY-proxy (market returns).
    We use the average return and volatility of all stocks as a
    market proxy since we may not have SPY in our universe.
    Features:
      - mean_return: average log return across universe
      - volatility:  rolling 10-day std of mean returns
      - abs_return:  absolute mean return (regime signal)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()

    mean_ret  = log_ret.mean(axis=1)
    vol       = mean_ret.rolling(10).std()
    abs_ret   = mean_ret.abs()

    features = pd.DataFrame({
        "mean_return": mean_ret,
        "volatility":  vol,
        "abs_return":  abs_ret,
    }).dropna()

    return features


def load_all_trades() -> pd.DataFrame:
    docs = list(trades_col().find({}, {"_id": 0}))
    df   = pd.DataFrame(docs)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    return df.sort_values("entry_date").reset_index(drop=True)


# ── Fit HMM ───────────────────────────────────────────────────────────────────
def fit_hmm(features: pd.DataFrame) -> tuple:
    """
    Fit a Gaussian HMM on market features.
    Returns the model and the predicted state sequence.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=N_ITER,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    model.fit(X)
    states = model.predict(X)

    return model, states, scaler, features.index


def identify_regimes(model, states: np.ndarray,
                     features: pd.DataFrame) -> tuple:
    """
    Identify which state corresponds to mean-reverting vs trending.
    Mean-reverting regime = lower volatility + smaller absolute returns.
    """
    state_stats = {}
    for s in range(N_STATES):
        mask = states == s
        state_stats[s] = {
            "mean_vol":    features["volatility"].values[mask].mean(),
            "mean_absret": features["abs_return"].values[mask].mean(),
            "count":       mask.sum(),
            "pct":         mask.mean() * 100,
        }

    # Mean-reverting state = lower volatility
    mr_state   = min(state_stats, key=lambda s: state_stats[s]["mean_vol"])
    tend_state = 1 - mr_state

    print(f"  State {mr_state}: mean-reverting "
          f"(vol={state_stats[mr_state]['mean_vol']:.4f}, "
          f"{state_stats[mr_state]['pct']:.1f}% of days)")
    print(f"  State {tend_state}: trending "
          f"(vol={state_stats[tend_state]['mean_vol']:.4f}, "
          f"{state_stats[tend_state]['pct']:.1f}% of days)")

    return mr_state, tend_state, state_stats


# ── Apply regime filter to trades ─────────────────────────────────────────────
def apply_regime_filter(trades: pd.DataFrame,
                        regime_series: pd.Series,
                        mr_state: int) -> pd.DataFrame:
    """
    Keep a trade only if the market was in mean-reverting regime
    on the entry date. Discard trades entered during trending regime.
    """
    filtered = []
    for _, row in trades.iterrows():
        entry = row["entry_date"].normalize()
        # Find closest regime date
        if entry in regime_series.index:
            regime = regime_series[entry]
        else:
            # Use nearest available date
            idx = regime_series.index.get_indexer([entry], method="nearest")
            regime = regime_series.iloc[idx[0]]

        if regime == mr_state:
            filtered.append(row)

    df = pd.DataFrame(filtered).reset_index(drop=True)
    print(f"  Trades kept:    {len(df)} / {len(trades)} "
          f"({len(df)/len(trades)*100:.1f}%)")
    print(f"  Trades filtered: {len(trades) - len(df)}")
    return df


# ── Recompute metrics ─────────────────────────────────────────────────────────
def compute_metrics(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {}

    start = trades["exit_date"].min().normalize()
    end   = trades["exit_date"].max().normalize()
    dates = pd.date_range(start, end, freq="B")
    daily = pd.Series(0.0, index=dates)
    for _, row in trades.iterrows():
        d = row["exit_date"].normalize()
        if d in daily.index:
            daily[d] += row["net_pnl"]

    cum      = daily.cumsum()
    roll_max = cum.cummax()
    drawdown = cum - roll_max

    active       = daily[daily != 0]
    total_pnl    = cum.iloc[-1]
    capital      = 60_000
    n_years      = (end - start).days / 365.25
    ann_return   = (total_pnl / capital) / n_years * 100
    sharpe       = ((active.mean() - 0.05/252) / active.std()
                    * np.sqrt(252)) if active.std() > 0 else 0
    max_dd       = drawdown.min()
    max_dd_pct   = (max_dd / capital) * 100
    calmar       = ann_return / abs(max_dd_pct) if max_dd_pct != 0 else 0
    wins         = trades[trades["net_pnl"] > 0]
    losses       = trades[trades["net_pnl"] <= 0]
    win_rate     = len(wins) / len(trades) * 100
    pf           = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())
                    if len(losses) > 0 else float("inf"))
    stop_rate    = (trades["exit_type"] == "stop_loss").mean() * 100

    return {
        "label":            label,
        "total_pnl":        round(total_pnl, 2),
        "annualised_return": round(ann_return, 2),
        "sharpe_ratio":     round(sharpe, 2),
        "calmar_ratio":     round(calmar, 2),
        "max_drawdown":     round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_trades":     len(trades),
        "win_rate":         round(win_rate, 2),
        "profit_factor":    round(pf, 2),
        "stop_loss_rate":   round(stop_rate, 1),
        "cum_series":       cum,
        "drawdown_series":  drawdown,
    }


# ── Comparison chart ──────────────────────────────────────────────────────────
def plot_comparison(baseline: dict, filtered: dict,
                    regime_series: pd.Series, mr_state: int) -> None:
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[2.5, 1.5, 1],
                            hspace=0.5, wspace=0.35)

    # ── Panel 1: Equity curve comparison ──
    ax1 = fig.add_subplot(gs[0, :])
    base_cum = baseline["cum_series"]
    filt_cum = filtered["cum_series"]

    ax1.plot(base_cum.index, base_cum.values,
             color="#B4B2A9", lw=1.2, ls="--", label="Baseline (no filter)")
    ax1.plot(filt_cum.index, filt_cum.values,
             color="#185FA5", lw=1.8, label="HMM-filtered")
    ax1.fill_between(filt_cum.index, filt_cum.values, 0,
                     where=(filt_cum.values >= 0),
                     color="#EAF3DE", alpha=0.3)
    ax1.fill_between(filt_cum.index, filt_cum.values, 0,
                     where=(filt_cum.values < 0),
                     color="#FCEBEB", alpha=0.3)
    ax1.axhline(0, color="#888780", lw=0.5, ls="--")

    # Shade trending regime periods
    trending_days = regime_series[regime_series != mr_state]
    if len(trending_days) > 0:
        in_trend = False
        t_start  = None
        prev_date = None
        for date, state in regime_series.items():
            if state != mr_state and not in_trend:
                in_trend = True
                t_start  = date
            elif state == mr_state and in_trend:
                ax1.axvspan(t_start, date,
                            alpha=0.08, color="#E24B4A", lw=0)
                in_trend = False
            prev_date = date
        if in_trend:
            ax1.axvspan(t_start, prev_date,
                        alpha=0.08, color="#E24B4A", lw=0)

    ax1.set_title("HMM Regime Filter: Baseline vs Filtered Strategy",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative PnL ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(lw=0.3, color="#E8E6DF")
    # Small note about shading
    ax1.text(0.99, 0.04, "pink shading = trending regime (signals off)",
             transform=ax1.transAxes, ha="right", fontsize=8,
             color="#A32D2D", style="italic")

    # ── Panel 2: Regime state over time ──
    ax2 = fig.add_subplot(gs[1, :])
    colors_map = {mr_state: "#3B6D11", 1 - mr_state: "#E24B4A"}
    regime_colors = [colors_map[s] for s in regime_series.values]
    ax2.scatter(regime_series.index, regime_series.values,
                c=regime_colors, s=3, linewidths=0, alpha=0.7)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([
        f"State 0 ({'MR' if mr_state==0 else 'Trend'})",
        f"State 1 ({'MR' if mr_state==1 else 'Trend'})"
    ], fontsize=9)
    ax2.set_title("Market Regime (HMM State Sequence)",
                  fontsize=10, pad=6)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(axis="x", lw=0.3, color="#E8E6DF")

    # ── Panel 3: Metrics comparison table ──
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    metrics_rows = [
        ["Total PnL",
         f"${baseline['total_pnl']:,.0f}",
         f"${filtered['total_pnl']:,.0f}"],
        ["Ann. Return",
         f"{baseline['annualised_return']:.1f}%",
         f"{filtered['annualised_return']:.1f}%"],
        ["Sharpe Ratio",
         f"{baseline['sharpe_ratio']:.2f}",
         f"{filtered['sharpe_ratio']:.2f}"],
        ["Calmar Ratio",
         f"{baseline['calmar_ratio']:.2f}",
         f"{filtered['calmar_ratio']:.2f}"],
        ["Max Drawdown",
         f"${baseline['max_drawdown']:,.0f} ({baseline['max_drawdown_pct']:.1f}%)",
         f"${filtered['max_drawdown']:,.0f} ({filtered['max_drawdown_pct']:.1f}%)"],
        ["Win Rate",
         f"{baseline['win_rate']:.1f}%",
         f"{filtered['win_rate']:.1f}%"],
        ["Profit Factor",
         f"{baseline['profit_factor']:.2f}",
         f"{filtered['profit_factor']:.2f}"],
        ["Total Trades",
         str(baseline['total_trades']),
         str(filtered['total_trades'])],
        ["Stop-loss Rate",
         f"{baseline['stop_loss_rate']:.1f}%",
         f"{filtered['stop_loss_rate']:.1f}%"],
    ]

    table = ax3.table(
        cellText=metrics_rows,
        colLabels=["Metric", "Baseline", "HMM-filtered"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E8E6DF")
        if r == 0:
            cell.set_facecolor("#E6F1FB")
            cell.set_text_props(fontweight="bold", color="#0C447C")
        elif r % 2 == 0:
            cell.set_facecolor("#F8F7F4")
        else:
            cell.set_facecolor("white")
        # Highlight improvements in filtered column
        if r > 0 and c == 2:
            metric_name = metrics_rows[r-1][0]
            if metric_name in ["Sharpe Ratio", "Win Rate",
                               "Profit Factor", "Ann. Return",
                               "Calmar Ratio", "Total PnL"]:
                try:
                    base_val = float(metrics_rows[r-1][1].replace(
                        "$","").replace(",","").replace("%",""))
                    filt_val = float(metrics_rows[r-1][2].replace(
                        "$","").replace(",","").replace("%",""))
                    if filt_val > base_val:
                        cell.set_facecolor("#EAF3DE")
                        cell.set_text_props(color="#27500A")
                    elif filt_val < base_val:
                        cell.set_facecolor("#FCEBEB")
                        cell.set_text_props(color="#A32D2D")
                except:
                    pass

    path = OUTPUT_DIR / "hmm_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Phase 8a: HMM Regime Detection ──\n")

    # Load data
    prices = load_prices()
    trades = load_all_trades()

    # Build market features
    print("Building market features...")
    features = load_market_features(prices)
    print(f"  {len(features)} days of features\n")

    # Fit HMM
    print("Fitting HMM...")
    model, states, scaler, dates = fit_hmm(features)
    print(f"  Converged: {model.monitor_.converged}")
    print(f"  Log-likelihood: {model.monitor_.history[-1]:.2f}\n")

    # Identify regimes
    print("Identifying regimes...")
    mr_state, trend_state, state_stats = identify_regimes(
        model, states, features)

    # Build regime time series
    regime_series = pd.Series(states, index=dates)

    # Baseline metrics (no filter)
    print("\nBaseline performance (no filter):")
    baseline = compute_metrics(trades, "Baseline")
    print(f"  Sharpe: {baseline['sharpe_ratio']:.2f}  |  "
          f"PnL: ${baseline['total_pnl']:,.0f}  |  "
          f"Trades: {baseline['total_trades']}")

    # Apply regime filter
    print("\nApplying HMM regime filter...")
    filtered_trades = apply_regime_filter(trades, regime_series, mr_state)

    # Filtered metrics
    print("\nFiltered performance (HMM-filtered):")
    filtered = compute_metrics(filtered_trades, "HMM-filtered")
    print(f"  Sharpe: {filtered['sharpe_ratio']:.2f}  |  "
          f"PnL: ${filtered['total_pnl']:,.0f}  |  "
          f"Trades: {filtered['total_trades']}")

    # Plot comparison
    print("\nGenerating comparison chart...")
    plot_comparison(baseline, filtered, regime_series, mr_state)

    # Print improvement summary
    print("\n── Regime filter impact ──")
    sharpe_delta = filtered["sharpe_ratio"] - baseline["sharpe_ratio"]
    dd_delta     = filtered["max_drawdown"] - baseline["max_drawdown"]
    pnl_delta    = filtered["total_pnl"]   - baseline["total_pnl"]

    print(f"  Sharpe:       {baseline['sharpe_ratio']:.2f} → "
          f"{filtered['sharpe_ratio']:.2f} "
          f"({'↑' if sharpe_delta > 0 else '↓'}{abs(sharpe_delta):.2f})")
    print(f"  Max drawdown: ${baseline['max_drawdown']:,.0f} → "
          f"${filtered['max_drawdown']:,.0f} "
          f"({'↓' if dd_delta > 0 else '↑'}${abs(dd_delta):,.0f})")
    print(f"  Total PnL:    ${baseline['total_pnl']:,.0f} → "
          f"${filtered['total_pnl']:,.0f} "
          f"({'↑' if pnl_delta > 0 else '↓'}${abs(pnl_delta):,.0f})")
    print(f"  Win rate:     {baseline['win_rate']:.1f}% → "
          f"{filtered['win_rate']:.1f}%")

    # Save to MongoDB
    col = get_db()["hmm_results"]
    col.replace_one(
        {"report_type": "hmm_comparison"},
        {
            "report_type":    "hmm_comparison",
            "baseline":       {k: v for k, v in baseline.items()
                               if k != "cum_series"
                               and k != "drawdown_series"},
            "filtered":       {k: v for k, v in filtered.items()
                               if k != "cum_series"
                               and k != "drawdown_series"},
            "mr_state":       int(mr_state),
            "trend_state":    int(trend_state),
            "state_stats":    {str(k): {sk: int(sv) if isinstance(sv, np.integer) 
                             else float(sv) if isinstance(sv, np.floating)
                             else sv 
                             for sk, sv in v.items()} 
                   for k, v in state_stats.items()},
            "generated_at":   datetime.now(timezone.utc),
        },
        upsert=True,
    )
    print("\nSaved to MongoDB → hmm_results collection")
    print("\nPhase 8a complete.")