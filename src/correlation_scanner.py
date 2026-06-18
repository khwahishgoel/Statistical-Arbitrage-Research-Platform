import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pymongo import UpdateOne

load_dotenv(Path(__file__).parent / ".env")

from db import load_prices, pairs_col
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

ROLLING_DAYS = 60
CORR_THRESH  = 0.80
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

#correlation
def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def rolling_correlation(returns: pd.DataFrame, a: str, b: str,
                         window: int = ROLLING_DAYS) -> pd.Series:
    return returns[a].rolling(window).corr(returns[b])

#scanningpairs
def scan_pairs(returns: pd.DataFrame, threshold: float = CORR_THRESH) -> pd.DataFrame:
    results = []
    for a, b in itertools.combinations(returns.columns, 2):
        full_corr   = returns[a].corr(returns[b])
        recent_corr = returns[a].tail(ROLLING_DAYS).corr(returns[b].tail(ROLLING_DAYS))
        if full_corr >= threshold:
            results.append({
                "pair":        f"{a}/{b}",
                "ticker_a":    a,
                "ticker_b":    b,
                "full_corr":   round(float(full_corr), 4),
                "recent_corr": round(float(recent_corr), 4),
                "stable":      bool(abs(full_corr - recent_corr) < 0.10),
                "scanned_at":  datetime.utcnow(),
            })
    if not results:
        return pd.DataFrame(columns=["pair", "ticker_a", "ticker_b", "full_corr", "recent_corr", "stable", "scanned_at"])
    return pd.DataFrame(results).sort_values("full_corr", ascending=False).reset_index(drop=True)

#savepairstodb
def save_pairs(pairs_df: pd.DataFrame) -> None:
    col = pairs_col()
    col.create_index("pair", unique=True)
    ops = [
        UpdateOne(
            {"pair": row["pair"]},
            {"$set": row.to_dict()},
            upsert=True,
        )
        for _, row in pairs_df.iterrows()
    ]
    if ops:
        result = col.bulk_write(ops, ordered=False)
        print(f"  Pairs upserted: {result.upserted_count} new, {result.modified_count} updated")


def load_pairs(min_corr: float = CORR_THRESH) -> pd.DataFrame:
    docs = list(pairs_col().find(
        {"full_corr": {"$gte": min_corr}},
        {"_id": 0}
    ))
    return pd.DataFrame(docs).sort_values("full_corr", ascending=False)

def plot_heatmap(corr_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, ax=ax,
    )
    ax.set_title("Pairwise Correlation Matrix — Log Returns", fontsize=13, pad=12)
    plt.tight_layout()
    path = OUTPUT_DIR / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


def plot_rolling_correlation(returns: pd.DataFrame, a: str, b: str) -> None:
    roll = rolling_correlation(returns, a, b)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(roll.index, roll.values, color="#185FA5", lw=1.5)
    ax.axhline(CORR_THRESH, color="#E24B4A", ls="--", lw=1, label=f"threshold {CORR_THRESH}")
    ax.axhline(0, color="#888780", ls=":", lw=0.8)
    ax.fill_between(roll.index, roll.values, CORR_THRESH,
                    where=(roll.values >= CORR_THRESH), alpha=0.12, color="#185FA5")
    ax.set_title(f"Rolling {ROLLING_DAYS}-Day Correlation: {a} vs {b}", fontsize=12)
    ax.set_ylabel("Pearson ρ")
    ax.legend(fontsize=9)
    ax.grid(axis="y", lw=0.4, color="#D3D1C7")
    plt.tight_layout()
    path = OUTPUT_DIR / f"rolling_corr_{a}_{b}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


if __name__ == "__main__":
    prices  = load_prices()
    returns = compute_log_returns(prices)
    print(f"Loaded: {returns.shape[0]} days × {returns.shape[1]} tickers\n")

    if returns.empty:
        print("No price data loaded — check that the DB is populated (run the ingestion script first).")
        raise SystemExit(1)

    corr_matrix = compute_correlation_matrix(returns)
    pairs_df    = scan_pairs(returns)

    print("── Top correlated pairs ──")
    print(pairs_df.head(15).to_string(index=False))

    print("\nSaving pairs to MongoDB...")
    save_pairs(pairs_df)

    plot_heatmap(corr_matrix)
    if not pairs_df.empty:
        top = pairs_df.iloc[0]
        plot_rolling_correlation(returns, top["ticker_a"], top["ticker_b"])

    print(f"\nPhase 2 complete — {len(pairs_df)} candidate pairs stored.")
    print("Run phase3_cointegration.py next.")