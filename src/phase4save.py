"""
Phase 4: Save spread model results (CSV → MongoDB)
Run this after phase4_spread_modeling.R finishes.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from db import get_db
from pymongo import UpdateOne

CSV_PATH = Path(__file__).parent.parent / "output/spread_model_results.csv"


def save_spread_models(df: pd.DataFrame) -> None:
    col = get_db()["spread_models"]
    col.create_index("pair", unique=True)

    ops = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        doc["saved_at"] = datetime.now(timezone.utc)
        # Convert any NaN to None for MongoDB
        doc = {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
               for k, v in doc.items()}
        ops.append(UpdateOne(
            {"pair": doc["pair"]},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        result = col.bulk_write(ops, ordered=False)
        print(f"  Spread models upserted: {result.upserted_count} new, "
              f"{result.modified_count} updated")


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run phase4_spread_modeling.R first.")
        exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} pairs from {CSV_PATH}\n")

    print("── Spread model summary ──")
    print(df[["pair", "current_zscore", "n_long_signals",
              "n_short_signals", "halflife_mean"]].to_string(index=False))

    print("\nSaving to MongoDB...")
    save_spread_models(df)
    print("\nPhase 4 complete.")
    print("Run phase5_signals.py next to generate live trade signals.")