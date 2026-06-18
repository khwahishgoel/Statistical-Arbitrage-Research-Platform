import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from db import save_prices, save_universe, load_prices, last_price_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UNIVERSE = {
    "tech":     ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL"],
    "payments": ["V", "MA", "PYPL", "AXP"],
    "consumer": ["KO", "PEP", "MCD", "YUM"],
    "energy":   ["XOM", "CVX", "COP", "SLB"],
    "banks":    ["JPM", "BAC", "GS", "MS"],
    "semis":    ["INTC", "QCOM", "TXN", "AMAT"],
    "etfs": ["XLE", "XLF", "XLK", "XLV"],   
    "utilities": ["NEE", "DUK", "SO", "AEP"], 
    "railroads": ["UNP", "CSX"]
}              

ALL_TICKERS = [t for sector in UNIVERSE.values() for t in sector]
START_DATE  = (datetime.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
END_DATE    = datetime.today().strftime("%Y-%m-%d")

def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices and return a clean wide DataFrame."""
    log.info(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else \
             raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.ffill()
    prices = prices.dropna(thresh=int(len(prices) * 0.95), axis=1)
    log.info(f"Clean universe: {list(prices.columns)} ({len(prices)} trading days)")
    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """log(P_t / P_{t-1}) — used in correlation and cointegration phases."""
    return np.log(prices / prices.shift(1)).dropna()

if __name__ == "__main__":
    last = last_price_date()

    if last is not None:
        log.info(f"MongoDB has data through {last.date()} — incremental update.")
        start = last.strftime("%Y-%m-%d")
    else:
        log.info("No existing data — full download.")
        start = START_DATE

    prices = download_prices(ALL_TICKERS, start=start, end=END_DATE)

    log.info("Saving prices to MongoDB...")
    save_prices(prices)
    save_universe(list(prices.columns), UNIVERSE)
    prices_full = load_prices()
    returns = compute_log_returns(prices_full)

    print("\n── Price data (last 5 rows) ──")
    print(prices_full.tail())
    print(f"\n── Shape: {prices_full.shape[0]} days × {prices_full.shape[1]} tickers ──")
    print("\n── Log returns (last 5 rows) ──")
    print(returns.tail())
    print("\nPhase 1 complete. Run phase2_correlation_scanner.py next.")