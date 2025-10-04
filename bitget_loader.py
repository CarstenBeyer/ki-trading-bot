# bitget_loader.py
import os
from typing import Literal, Optional, List, Tuple
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv

Timeframe = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w","1M"]
load_dotenv(dotenv_path="config/secrets.env")


def get_exchange() -> ccxt.bitget:
    cfg = {"enableRateLimit": True}
    k, s, p = os.getenv("API_KEY"), os.getenv("API_SECRET"), os.getenv("API_PASSPHRASE")
    if k and s and p:
        cfg.update({"apiKey": k, "secret": s, "password": p})
    ex = ccxt.bitget(cfg)
    ex.load_markets()
    return ex


def _tf_ms(tf: str) -> int:
    return {
        "1m": 60_000,  "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000, "12h": 43_200_000,
        "1d": 86_400_000, "1w": 604_800_000, "1M": 2_592_000_000,
    }[tf]


def _to_df(raw) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    if df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def _gap_blocks(idx: pd.DatetimeIndex, tf_ms: int, tol: float = 1.5) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if len(idx) < 2:
        return []
    diffs = idx.to_series().diff().dt.total_seconds().mul(1000)
    big = diffs > (tol * tf_ms)
    if not big.any():
        return []
    gaps = []
    missing_starts = big.index[big]
    for start in missing_starts:
        prev = idx[idx.get_loc(start) - 1]
        # expected next time after prev:
        gaps.append((prev + pd.Timedelta(milliseconds=tf_ms), start - pd.Timedelta(milliseconds=tf_ms)))
    return gaps


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: Timeframe = "1h",
    limit: int = 500,
    since_ms: Optional[int] = None,
    *,
    ensure_latest: bool = True,
    fill_gaps: bool = True,
    break_lines_at_gaps: bool = True,
) -> pd.DataFrame:
    """
    Robust OHLCV:
      - Backward pagination anchored to 'now' (or 'since_ms' if given).
      - Optionally re-fetches missing blocks detected between candles.
      - Optionally inserts np.nan at gap boundaries to prevent 'ramps' in plots.
    """
    ex = get_exchange()
    tfms = _tf_ms(timeframe)
    # conservative per-call cap (Bitget daily ~300, intraday ~1000)
    per_call_cap = 300 if timeframe in ("1d", "1w", "1M") else 1000

    if since_ms is None and ensure_latest:
        until = ex.milliseconds() + tfms  # a bit into the next bar
    else:
        # anchor at the end of the first window
        until = (since_ms + limit * tfms) if since_ms is not None else None

    out = []
    needed = int(limit)
    loops = 0
    max_loops = 100

    while needed > 0 and loops < max_loops:
        loops += 1
        step = min(per_call_cap, needed)
        if until is None:
            # first shot without anchor
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=step)
        else:
            since = until - step * tfms
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=step,
                                   params={"until": until, "endTime": until})
        if not batch:
            break
        out.extend(batch)
        needed -= len(batch)
        first_ts = batch[0][0]
        until = first_ts  # step further back

        if first_ts <= 0:
            break

        # safety: if we already have more than needed, stop
        if len(out) >= limit + per_call_cap:
            break

    df = _to_df(out)
    if df.empty:
        return df

    # keep the most recent `limit`
    if len(df) > limit:
        df = df.iloc[-limit:]

    # fill obvious missing blocks by re-fetching within each gap
    if fill_gaps:
        gaps = _gap_blocks(df.index, tfms, tol=1.5)
        for gstart, gend in gaps:
            # Fetch a little wider than the gap so server rounding doesn't miss it
            wider_since = int((gstart - pd.Timedelta(milliseconds=5*tfms)).timestamp() * 1000)
            wider_until = int((gend   + pd.Timedelta(milliseconds=5*tfms)).timestamp() * 1000)
            try:
                extra = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=wider_since, limit=per_call_cap,
                                       params={"until": wider_until, "endTime": wider_until})
                dfe = _to_df(extra)
                if not dfe.empty:
                    df = pd.concat([df, dfe]).drop_duplicates().sort_index()
            except Exception:
                pass
        if len(df) > limit:
            df = df.iloc[-limit:]

    # break chart lines at remaining gaps (use np.nan, not pd.NA)
    if break_lines_at_gaps:
        gaps = _gap_blocks(df.index, tfms, tol=1.5)
        if gaps:
            # mark the first candle after each gap as NaN → Matplotlib breaks the line
            diffs = df.index.to_series().diff().dt.total_seconds().mul(1000)
            big = diffs > (1.5 * tfms)
            gap_idx = big.index[big]        # these are the first bars AFTER gaps
            df.loc[gap_idx, ["open","high","low","close","volume"]] = np.nan

    return df
