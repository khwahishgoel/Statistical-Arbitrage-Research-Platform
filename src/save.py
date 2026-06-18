"""
Phase 3: Save cointegration results (CSV → MongoDB)
Run this after phase3_cointegration.R finishes.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from db import cointegration_col
from pymongo import UpdateOne

CSV_PATH = Path(__file__).parent.parent / "output/cointegration_results.csv"


def save_cointegration(df: pd.DataFrame) -> None:
    col = cointegration_col()
    col.create_index("pair", unique=True)

    ops = []
    for _, row in df.iterrows():
        doc = row.to_dict()
        # Convert numpy bools/floats to native Python types for MongoDB
        doc["cointegrated"] = bool(doc["cointegrated"])
        doc["tradeable"]    = bool(doc["tradeable"])
        doc["saved_at"]     = datetime.now(timezone.utc)
        ops.append(UpdateOne(
            {"pair": doc["pair"]},
            {"$set": doc},
            upsert=True,
        ))

    if ops:
        result = col.bulk_write(ops, ordered=False)
        print(f"  Cointegration upserted: {result.upserted_count} new, "
              f"{result.modified_count} updated")


if __name__ == "__main__":
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found.")
        print("Run phase3_cointegration.R first.")
        exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} pairs from {CSV_PATH}\n")

    tradeable = df[df["tradeable"] == True]
    print(f"── Tradeable pairs (cointegrated + half-life in range) ──")
    print(tradeable[["pair", "eg_pvalue", "halflife_days", "beta"]].to_string(index=False))

    print("\nSaving to MongoDB...")
    save_cointegration(df)
    print("\nPhase 3 complete.")
    print(f"{len(tradeable)} tradeable pairs ready for phase 4 spread modeling.")