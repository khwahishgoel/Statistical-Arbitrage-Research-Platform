"""
Statistical Arbitrage Research Platform
Phase 6: Backtesting Engine (Python)

Reads the trade log from MongoDB (phase 5 output),
builds a full portfolio equity curve, and computes
professional-grade performance metrics.

Produces:
  - output/equity_curve.png     
  - output/backtest_report.csv  
"""

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from db import trades_col, get_db

OUTPUT_DIR  = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RISK_FREE   = 0.05   # annualised risk-free rate (current T-bill ~5%)
POSITION_SIZE = 10_000


# ── Load trades from MongoDB ──────────────────────────────────────────────────
def load_trades() -> pd.DataFrame:
    docs = list(trades_col().find({}, {"_id": 0}))
    df   = pd.DataFrame(docs)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date").reset_index(drop=True)
    print(f"Loaded {len(df)} trades from MongoDB\n")
    return df


# ── Build daily equity curve ──────────────────────────────────────────────────
def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a daily PnL series from the trade log.
    PnL is attributed on the exit date of each trade.
    """
    start = trades["exit_date"].min().normalize()
    end   = trades["exit_date"].max().normalize()
    dates = pd.date_range(start, end, freq="B")  # business days only

    daily_pnl = pd.Series(0.0, index=dates)
    for _, row in trades.iterrows():
        exit_day = row["exit_date"].normalize()
        if exit_day in daily_pnl.index:
            daily_pnl[exit_day] += row["net_pnl"]

    cum_pnl    = daily_pnl.cumsum()
    roll_max   = cum_pnl.cummax()
    drawdown   = cum_pnl - roll_max

    return pd.DataFrame({
        "daily_pnl":  daily_pnl,
        "cum_pnl":    cum_pnl,
        "drawdown":   drawdown,
    })


# Performance metrics 
def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    daily_pnl   = equity["daily_pnl"]
    cum_pnl     = equity["cum_pnl"]
    drawdown    = equity["drawdown"]

    # Filter to days with trades only for Sharpe (avoid diluting with flat days)
    active_days = daily_pnl[daily_pnl != 0]
    n_days      = len(active_days)

    total_pnl      = cum_pnl.iloc[-1]
    avg_daily_pnl  = active_days.mean()
    std_daily_pnl  = active_days.std()

    # Annualised Sharpe (using active trading days)
    sharpe = ((avg_daily_pnl - RISK_FREE / 252) /
              std_daily_pnl * np.sqrt(252)) if std_daily_pnl > 0 else 0

    # Annualised return (assume $30k deployed: 3 pairs × $10k per leg)
    capital        = 3 * POSITION_SIZE * 2   # long + short legs
    n_years        = (equity.index[-1] - equity.index[0]).days / 365.25
    ann_return     = (total_pnl / capital) / n_years * 100

    # Drawdown metrics
    max_dd         = drawdown.min()
    max_dd_pct     = (max_dd / capital) * 100

    # Calmar ratio = annualised return / max drawdown %
    calmar         = ann_return / abs(max_dd_pct) if max_dd_pct != 0 else 0

    # Win/loss stats
    wins           = trades[trades["net_pnl"] > 0]
    losses         = trades[trades["net_pnl"] <= 0]
    win_rate       = len(wins) / len(trades) * 100
    avg_win        = wins["net_pnl"].mean()   if len(wins)   > 0 else 0
    avg_loss       = losses["net_pnl"].mean() if len(losses) > 0 else 0
    profit_factor  = (wins["net_pnl"].sum() /
                      abs(losses["net_pnl"].sum())) if len(losses) > 0 else float("inf")

    # Average holding period
    avg_hold = trades["holding_days"].mean()

    # Stop-loss rate
    stop_rate = (trades["exit_type"] == "stop_loss").mean() * 100

    return {
        "total_pnl":        round(total_pnl, 2),
        "annualised_return": round(ann_return, 2),
        "sharpe_ratio":     round(sharpe, 2),
        "calmar_ratio":     round(calmar, 2),
        "max_drawdown":     round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_trades":     len(trades),
        "win_rate":         round(win_rate, 2),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(profit_factor, 2),
        "avg_hold_days":    round(avg_hold, 1),
        "stop_loss_rate":   round(stop_rate, 1),
        "capital_deployed": capital,
        "n_years":          round(n_years, 2),
    }


# Main chart — equity curve  
def plot_equity_curve(equity: pd.DataFrame, trades: pd.DataFrame,
                      metrics: dict) -> None:
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=[3, 1.2, 1.2],
                            hspace=0.45, wspace=0.3)

    # Panel 1: Cumulative PnL 
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(equity.index, equity["cum_pnl"], 0,
                     where=(equity["cum_pnl"] >= 0),
                     color="#EAF3DE", alpha=0.6)
    ax1.fill_between(equity.index, equity["cum_pnl"], 0,
                     where=(equity["cum_pnl"] < 0),
                     color="#FCEBEB", alpha=0.6)
    ax1.plot(equity.index, equity["cum_pnl"],
             color="#185FA5", lw=1.5, zorder=3)
    ax1.axhline(0, color="#888780", lw=0.6, ls="--")

    # Mark individual trade exits
    for _, row in trades.iterrows():
        exit_day = row["exit_date"].normalize()
        if exit_day in equity.index:
            y = equity.loc[exit_day, "cum_pnl"]
            color = "#3B6D11" if row["net_pnl"] > 0 else "#E24B4A"
            ax1.scatter(exit_day, y, color=color, s=18, zorder=4,
                        alpha=0.7, linewidths=0)

    ax1.set_title("Portfolio Equity Curve — Statistical Arbitrage Strategy",
                  fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative PnL ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.grid(axis="y", lw=0.4, color="#E8E6DF")
    ax1.grid(axis="x", lw=0.2, color="#E8E6DF")

    # Annotation: final PnL
    ax1.annotate(
        f"  Final PnL: ${metrics['total_pnl']:,.0f}",
        xy=(equity.index[-1], equity["cum_pnl"].iloc[-1]),
        fontsize=9, color="#185FA5", fontweight="bold",
        ha="right",
    )

    # ── Panel 2: Drawdown ──
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(equity.index, equity["drawdown"], 0,
                     color="#FCEBEB", alpha=0.8)
    ax2.plot(equity.index, equity["drawdown"],
             color="#E24B4A", lw=0.8)
    ax2.axhline(0, color="#888780", lw=0.4)
    ax2.set_ylabel("Drawdown ($)", fontsize=10)
    ax2.set_title("Drawdown", fontsize=10, pad=6)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(axis="y", lw=0.4, color="#E8E6DF")

    # ── Panel 3: Per-pair PnL bars ──
    ax3 = fig.add_subplot(gs[2, 0])
    pair_pnl = trades.groupby("pair")["net_pnl"].sum().sort_values()
    colors   = ["#E24B4A" if v < 0 else "#3B6D11" for v in pair_pnl]
    bars = ax3.barh(pair_pnl.index, pair_pnl.values, color=colors,
                    height=0.5, edgecolor="none")
    ax3.axvline(0, color="#888780", lw=0.6)
    ax3.set_title("PnL by pair", fontsize=10, pad=6)
    ax3.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.grid(axis="x", lw=0.4, color="#E8E6DF")
    for bar, val in zip(bars, pair_pnl.values):
        ax3.text(val + (200 if val >= 0 else -200),
                 bar.get_y() + bar.get_height() / 2,
                 f"${val:,.0f}",
                 va="center", ha="left" if val >= 0 else "right",
                 fontsize=8,
                 color="#3B6D11" if val >= 0 else "#E24B4A")

    # ── Panel 4: Metrics table ──
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    rows = [
        ["Total PnL",          f"${metrics['total_pnl']:,.0f}"],
        ["Ann. Return",        f"{metrics['annualised_return']:.1f}%"],
        ["Sharpe Ratio",       f"{metrics['sharpe_ratio']:.2f}"],
        ["Calmar Ratio",       f"{metrics['calmar_ratio']:.2f}"],
        ["Max Drawdown",       f"${metrics['max_drawdown']:,.0f}  ({metrics['max_drawdown_pct']:.1f}%)"],
        ["Win Rate",           f"{metrics['win_rate']:.1f}%"],
        ["Profit Factor",      f"{metrics['profit_factor']:.2f}"],
        ["Total Trades",       str(metrics['total_trades'])],
        ["Avg Hold",           f"{metrics['avg_hold_days']:.1f} days"],
        ["Stop-loss Rate",     f"{metrics['stop_loss_rate']:.1f}%"],
    ]
    table = ax4.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
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

    plt.suptitle(
        f"Backtest: {metrics['n_years']:.1f} years  |  "
        f"Capital: ${metrics['capital_deployed']:,}  |  "
        f"3 pairs  |  $10k per leg",
        fontsize=10, color="#5F5E5A", y=0.98,
    )

    path = OUTPUT_DIR / "equity_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved → {path}")


# ── Per-pair equity curves ────────────────────────────────────────────────────
def plot_pair_breakdown(trades: pd.DataFrame,
                        equity_full: pd.DataFrame) -> None:
    pairs = trades["pair"].unique()
    fig, axes = plt.subplots(len(pairs), 1,
                             figsize=(12, 4 * len(pairs)),
                             facecolor="white")
    if len(pairs) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pairs):
        pair_trades = trades[trades["pair"] == pair].copy()
        start = pair_trades["exit_date"].min().normalize()
        end   = pair_trades["exit_date"].max().normalize()
        dates = pd.date_range(start, end, freq="B")
        daily = pd.Series(0.0, index=dates)
        for _, row in pair_trades.iterrows():
            d = row["exit_date"].normalize()
            if d in daily.index:
                daily[d] += row["net_pnl"]
        cum = daily.cumsum()

        color = "#185FA5" if cum.iloc[-1] > 0 else "#E24B4A"
        ax.fill_between(cum.index, cum.values, 0,
                        color=("#EAF3DE" if cum.iloc[-1] > 0 else "#FCEBEB"),
                        alpha=0.5)
        ax.plot(cum.index, cum.values, color=color, lw=1.2)
        ax.axhline(0, color="#888780", lw=0.5, ls="--")
        ax.set_title(
            f"{pair}  |  {len(pair_trades)} trades  |  "
            f"Win rate: {(pair_trades['net_pnl']>0).mean()*100:.1f}%  |  "
            f"PnL: ${cum.iloc[-1]:,.0f}",
            fontsize=11, fontweight="bold",
        )
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(lw=0.3, color="#E8E6DF")

    plt.tight_layout()
    path = OUTPUT_DIR / "pair_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Phase 6: Backtesting Engine ──\n")

    trades = load_trades()
    equity = build_equity_curve(trades)
    metrics = compute_metrics(trades, equity)

    print("── Performance metrics ──")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")

    print("\nGenerating charts...")
    plot_equity_curve(equity, trades, metrics)
    plot_pair_breakdown(trades, equity)

    # Save report
    report_df = pd.DataFrame([metrics])
    report_df["generated_at"] = datetime.now(timezone.utc).isoformat()
    csv_path  = OUTPUT_DIR / "backtest_report.csv"
    report_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # Save to MongoDB
    col = get_db()["backtest_reports"]
    col.replace_one(
        {"report_type": "full_portfolio"},
        {**metrics, "report_type": "full_portfolio",
         "generated_at": datetime.now(timezone.utc)},
        upsert=True,
    )
    print("Saved to MongoDB → backtest_reports collection")

    print(f"""
── Summary ──
  Total PnL:       ${metrics['total_pnl']:,.0f}
  Annualised return: {metrics['annualised_return']:.1f}%
  Sharpe ratio:    {metrics['sharpe_ratio']:.2f}
  Win rate:        {metrics['win_rate']:.1f}%
  Max drawdown:    ${metrics['max_drawdown']:,.0f}

Phase 6 complete.
""")