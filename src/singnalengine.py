"""
Statistical Arbitrage Research Platform
Phase 5: Signal Engine (Python)

Reads spread model parameters from MongoDB (phase 4 output),
recomputes z-scores on the full price history, generates a
structured trade log, and writes signals + trades to MongoDB.
"""

from pathlib import Path
from datetime import datetime, timezone
import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import UpdateOne

load_dotenv(Path(__file__).parent / ".env")

from db import (
    load_prices, get_db, signals_col, trades_col, cointegration_col
)
ROLL_WINDOW    = 60     # days for rolling mean/sd
ENTRY_Z        = 2.0    # enter when |z| exceeds this
EXIT_Z         = 0.5    # exit when |z| falls below this
STOP_Z         = 3.5    # stop-loss when |z| exceeds this
POSITION_SIZE  = 10_000 # $ per leg (each trade = $10k long + $10k short)
COST_BPS       = 7      # transaction cost per leg in basis points
SLIPPAGE_BPS   = 3      # slippage per leg in basis points
TOTAL_COST     = (COST_BPS + SLIPPAGE_BPS) / 10_000  # fraction per leg

def load_tradeable_pairs() -> pd.DataFrame:
    col  = cointegration_col()
    docs = list(col.find({"tradeable": True}, {"_id": 0}))
    df   = pd.DataFrame(docs)
    print(f"Loaded {len(df)} tradeable pairs from MongoDB\n")
    return df

def compute_spread(prices: pd.DataFrame, a: str, b: str,
                   beta: float) -> pd.Series:
    spread = prices[a] - beta * prices[b]
    spread.name = f"{a}/{b}"
    return spread


def compute_zscore(spread: pd.Series,
                   window: int = ROLL_WINDOW) -> pd.DataFrame:
    roll_mean = spread.rolling(window).mean()
    roll_sd   = spread.rolling(window).std()
    zscore    = (spread - roll_mean) / roll_sd
    return pd.DataFrame({
        "spread":    spread,
        "roll_mean": roll_mean,
        "roll_sd":   roll_sd,
        "zscore":    zscore,
    })

def generate_trade_log(zdf: pd.DataFrame, pair: str,
                       a: str, b: str, beta: float) -> pd.DataFrame:
    """
    Walks through z-score history and generates one row per trade event.
    Each round trip = one entry row + one exit row.
    """
    trades   = []
    in_trade = False
    entry    = {}

    for date, row in zdf.iterrows():
        z = row["zscore"]
        if pd.isna(z):
            continue

        if not in_trade:
            if z > ENTRY_Z:
                entry = {
                    "pair":        pair,
                    "ticker_a":    a,
                    "ticker_b":    b,
                    "direction":   "short_spread",
                    "entry_date":  date,
                    "entry_z":     round(z, 4),
                    "entry_spread": round(row["spread"], 4),
                    "beta":        round(beta, 6),
                }
                in_trade = True

            elif z < -ENTRY_Z:
                entry = {
                    "pair":        pair,
                    "ticker_a":    a,
                    "ticker_b":    b,
                    "direction":   "long_spread",
                    "entry_date":  date,
                    "entry_z":     round(z, 4),
                    "entry_spread": round(row["spread"], 4),
                    "beta":        round(beta, 6),
                }
                in_trade = True

        else:
            exit_triggered = (
                abs(z) < EXIT_Z or   
                abs(z) > STOP_Z     
            )
            if exit_triggered:
                exit_type = "stop_loss" if abs(z) > STOP_Z else "reversion"

                spread_change = row["spread"] - entry["entry_spread"]
                if entry["direction"] == "short_spread":
                    raw_pnl = -spread_change * (POSITION_SIZE / abs(entry["entry_spread"]) if entry["entry_spread"] != 0 else 1)
                else:
                    raw_pnl =  spread_change * (POSITION_SIZE / abs(entry["entry_spread"]) if entry["entry_spread"] != 0 else 1)

                cost = 4 * POSITION_SIZE * TOTAL_COST
                net_pnl = raw_pnl - cost

                holding_days = (date - entry["entry_date"]).days

                trade = {
                    **entry,
                    "exit_date":    date,
                    "exit_z":       round(z, 4),
                    "exit_spread":  round(row["spread"], 4),
                    "exit_type":    exit_type,
                    "holding_days": holding_days,
                    "raw_pnl":      round(raw_pnl, 2),
                    "cost":         round(cost, 2),
                    "net_pnl":      round(net_pnl, 2),
                    "logged_at":    datetime.now(timezone.utc),
                }
                trades.append(trade)
                in_trade = False
                entry    = {}

    return pd.DataFrame(trades)

def save_trades(trades_df: pd.DataFrame) -> None:
    if trades_df.empty:
        print("  No trades to save.")
        return
    col = trades_col()
    col.create_index([("pair", 1), ("entry_date", 1)], unique=True)

    ops = []
    for _, row in trades_df.iterrows():
        doc = row.to_dict()
        for k in ("entry_date", "exit_date"):
            if isinstance(doc[k], pd.Timestamp):
                doc[k] = doc[k].to_pydatetime()
        ops.append(UpdateOne(
            {"pair": doc["pair"], "entry_date": doc["entry_date"]},
            {"$set": doc},
            upsert=True,
        ))
    if ops:
        result = col.bulk_write(ops, ordered=False)
        print(f"  Trades upserted: {result.upserted_count} new, "
              f"{result.modified_count} updated")


def save_current_signals(signals: list[dict]) -> None:
    """Save the current z-score snapshot for each pair."""
    col = signals_col()
    col.create_index("pair", unique=True)
    ops = [
        UpdateOne(
            {"pair": s["pair"]},
            {"$set": {**s, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        for s in signals
    ]
    if ops:
        col.bulk_write(ops, ordered=False)

def print_performance(trades_df: pd.DataFrame, pair: str) -> dict:
    df = trades_df[trades_df["pair"] == pair].copy()
    if df.empty:
        return {}

    total_trades  = len(df)
    wins          = (df["net_pnl"] > 0).sum()
    win_rate      = wins / total_trades * 100
    total_pnl     = df["net_pnl"].sum()
    avg_pnl       = df["net_pnl"].mean()
    avg_hold      = df["holding_days"].mean()
    pnl_std       = df["net_pnl"].std()
    sharpe        = (avg_pnl / pnl_std * np.sqrt(252 / avg_hold)
                     if pnl_std > 0 else 0)

    cum_pnl = df["net_pnl"].cumsum()
    roll_max = cum_pnl.cummax()
    drawdown = (cum_pnl - roll_max)
    max_dd   = drawdown.min()

    print(f"  trades:      {total_trades}")
    print(f"  win rate:    {win_rate:.1f}%")
    print(f"  total PnL:   ${total_pnl:,.2f}")
    print(f"  avg PnL:     ${avg_pnl:,.2f} per trade")
    print(f"  avg hold:    {avg_hold:.1f} days")
    print(f"  sharpe:      {sharpe:.2f}")
    print(f"  max drawdown: ${max_dd:,.2f}")

    return {
        "pair":          pair,
        "total_trades":  int(total_trades),
        "win_rate":      round(win_rate, 2),
        "total_pnl":     round(total_pnl, 2),
        "avg_pnl":       round(avg_pnl, 2),
        "avg_hold_days": round(avg_hold, 1),
        "sharpe":        round(sharpe, 2),
        "max_drawdown":  round(max_dd, 2),
    }

if __name__ == "__main__":
    print("── Phase 5: Signal Engine ──\n")

    prices = load_prices()
    log_returns = np.log(prices / prices.shift(1)).dropna()
    pairs  = load_tradeable_pairs()

    all_trades    = []
    current_sigs  = []
    perf_summary  = []

    for _, pair_row in pairs.iterrows():
        pair = pair_row["pair"]
        a    = pair_row["ticker_a"]
        b    = pair_row["ticker_b"]
        beta = float(pair_row["beta"])

        print(f"── Processing: {pair} ──")
        spread = compute_spread(prices, a, b, beta)
        zdf    = compute_zscore(spread)

        trades_df = generate_trade_log(zdf, pair, a, b, beta)
        all_trades.append(trades_df)

        print(f"  Generated {len(trades_df)} completed trades")
        perf = print_performance(trades_df, pair)
        if perf:
            perf_summary.append(perf)

        current_z      = float(zdf["zscore"].dropna().iloc[-1])
        current_spread = float(zdf["spread"].dropna().iloc[-1])
        if current_z > ENTRY_Z:
            live_signal = "short_spread"
        elif current_z < -ENTRY_Z:
            live_signal = "long_spread"
        else:
            live_signal = "no_signal"

        current_sigs.append({
            "pair":           pair,
            "ticker_a":       a,
            "ticker_b":       b,
            "current_zscore": round(current_z, 4),
            "current_spread": round(current_spread, 4),
            "signal":         live_signal,
            "entry_threshold": ENTRY_Z,
            "exit_threshold":  EXIT_Z,
        })
        print(f"  Current z: {current_z:.4f} → {live_signal}\n")

    all_trades_df = pd.concat(all_trades, ignore_index=True)
    print("── Portfolio summary ──")
    print(f"  Total trades across all pairs: {len(all_trades_df)}")
    print(f"  Total net PnL: ${all_trades_df['net_pnl'].sum():,.2f}")
    print(f"  Overall win rate: {(all_trades_df['net_pnl'] > 0).mean()*100:.1f}%")
    print("\nSaving to MongoDB...")
    save_trades(all_trades_df)
    save_current_signals(current_sigs)

    
    perf_df  = pd.DataFrame(perf_summary)
    csv_path = Path(__file__).parent.parent / "output/signal_engine_results.csv"
    perf_df.to_csv(csv_path, index=False)
    print(f"Saved performance summary → {csv_path}")

    print("\n── Current live signals ──")
    for s in current_sigs:
        z   = s["current_zscore"]
        sig = s["signal"]
        bar = "█" * min(int(abs(z) * 5), 20)
        print(f"  {s['pair']:10s}  z={z:+.4f}  {bar}  {sig}")

    print("\nPhase 5 complete. Run phase6_backtest.py next.")