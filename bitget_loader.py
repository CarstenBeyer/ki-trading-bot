# bitget_loader.py (patch)
import ccxt
import pandas as pd
import os
from typing import Literal, Optional
from dotenv import load_dotenv

Timeframe = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w","1M"]
load_dotenv(dotenv_path="config/secrets.env")

def get_exchange() -> ccxt.bitget:
    cfg = { "enableRateLimit": True }
    api_key = os.getenv("API_KEY"); api_secret = os.getenv("API_SECRET"); api_passphrase = os.getenv("API_PASSPHRASE")
    if api_key and api_secret and api_passphrase:
        cfg.update({ "apiKey": api_key, "secret": api_secret, "password": api_passphrase })
    ex = ccxt.bitget(cfg)
    ex.load_markets()
    return ex

# --- NEW: helper to convert timeframe to milliseconds
def _tf_ms(tf: str) -> int:
    m = {"1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000,"30m":1_800_000,
         "1h":3_600_000,"2h":7_200_000,"4h":14_400_000,"6h":21_600_000,"12h":43_200_000,
         "1d":86_400_000,"1w":604_800_000,"1M":2_592_000_000}
    return m[tf]

def fetch_ohlcv(symbol: str = "BTC/USDT",
                timeframe: Timeframe = "1h",
                limit: int = 500,
                since_ms: Optional[int] = None,
                ensure_latest: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV from Bitget (Spot). Returns a DataFrame indexed by UTC timestamps.
    - If ensure_latest=True (default), we page so that the window ends at 'now' and
      includes the *most recent* candles (no stale last date).
    """
    ex = get_exchange()

    # If caller provided since_ms, respect it verbatim
    if not ensure_latest or since_ms is not None:
        raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since_ms)
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts").sort_index()

    # Otherwise: compute a 'since' so the result ends at now, and paginate to fill
    tfms = _tf_ms(timeframe)
    now_ms = ex.milliseconds()
    # Pull a slightly larger window to be safe, then trim
    window = int(limit * 1.3) + 10
    since_ms = now_ms - window * tfms

    out = []
    next_since = since_ms
    max_batch = 1000  # ccxt will cap anyway; Bitget typically ≤ 1000
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=min(max_batch, window))
        if not batch:
            break
        out.extend(batch)
        # advance by one candle to avoid duplication
        next_since = batch[-1][0] + tfms
        # stop when we've reached (or passed) 'now'
        if next_since >= now_ms + tfms:
            break

    if not out:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    df = pd.DataFrame(out, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    # Trim to the **last `limit` rows** so we end at (near) now
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df
