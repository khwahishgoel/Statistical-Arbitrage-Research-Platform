from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.collection import Collection
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import certifi

load_dotenv(Path(__file__).parent / ".env")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "stat_arb"

_client = MongoClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
)

def get_db():
    global _client
    if _client is None:
        _client = MongoClient(
            MONGO_URI,
            tlsCAFile="/opt/homebrew/etc/openssl@3/cert.pem",
        )
    return _client[DB_NAME]


def prices_col() -> Collection:
    return get_db()["prices"]


def universe_col() -> Collection:
    return get_db()["universe"]


def pairs_col() -> Collection:
    return get_db()["candidate_pairs"]


def cointegration_col() -> Collection:
    return get_db()["cointegration"]


def signals_col() -> Collection:
    return get_db()["signals"]


def trades_col() -> Collection:
    return get_db()["trades"]


def save_prices(prices: pd.DataFrame) -> None:
    """
    Upsert prices into MongoDB.
    Schema: one document per (date, ticker).
      { date: datetime, ticker: str, close: float }
    Uses bulk upsert — safe to call multiple times.
    """
    col = prices_col()
    col.create_index([("date", ASCENDING), ("ticker", ASCENDING)], unique=True)

    ops = []
    for date, row in prices.iterrows():
        for ticker, close in row.items():
            if pd.isna(close):
                continue
            ops.append(UpdateOne(
                {"date": date.to_pydatetime(), "ticker": ticker},
                {"$set": {"close": float(close)}},
                upsert=True,
            ))
    if ops:
        result = col.bulk_write(ops, ordered=False)
        print(f"  Prices upserted: {result.upserted_count} new, "
              f"{result.modified_count} updated")

def load_prices(tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Load prices from MongoDB → wide DataFrame (date index, ticker columns).
    Pass tickers=None to load the full universe.
    """
    col   = prices_col()
    query = {"ticker": {"$in": tickers}} if tickers else {}
    docs  = list(col.find(query, {"_id": 0, "date": 1, "ticker": 1, "close": 1}))

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="ticker", values="close").sort_index()
    pivot.index.name = "Date"
    return pivot


def last_price_date() -> pd.Timestamp | None:
    """Return the most recent date stored, or None if collection is empty."""
    col = prices_col()
    doc = col.find_one({}, sort=[("date", -1)], projection={"date": 1})
    return pd.Timestamp(doc["date"]) if doc else None

def save_universe(tickers: list[str], universe_map: dict) -> None:
    col = universe_col()
    col.create_index("ticker", unique=True)
    ops = []
    for ticker in tickers:
        sector = next((s for s, ts in universe_map.items() if ticker in ts), "unknown")
        ops.append(UpdateOne(
            {"ticker": ticker},
            {"$set": {"sector": sector, "last_updated": datetime.utcnow()}},
            upsert=True,
        ))
    if ops:
        col.bulk_write(ops, ordered=False)


def load_universe() -> pd.DataFrame:
    docs = list(universe_col().find({}, {"_id": 0}))
    return pd.DataFrame(docs)