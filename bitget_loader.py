# bitget_loader.py
import ccxt
import pandas as pd
import os
from typing import Literal, Optional
from dotenv import load_dotenv

Timeframe = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w","1M"]

# Lade deine Secrets aus config/secrets.env
load_dotenv(dotenv_path="config/secrets.env")

def get_exchange() -> ccxt.bitget:
    """
    Erstellt einen Bitget-Client. Ohne Keys -> nur öffentliche Daten (OHLCV, Ticker).
    Mit Keys -> später Orders möglich.
    """
    cfg = {
        "enableRateLimit": True,  # ccxt kümmert sich um Sleeps zwischen Requests
    }

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    api_passphrase = os.getenv("API_PASSPHRASE")

    print("Using Bitget exchange with API key:", api_key is not None)

    if api_key and api_secret and api_passphrase:
        cfg.update({
            "apiKey": api_key,
            "secret": api_secret,
            "password": api_passphrase
        })

    ex = ccxt.bitget(cfg)
    ex.load_markets()
    return ex


def _timeframe_ms(ex: ccxt.Exchange, timeframe: str) -> int:
    # ccxt: parse_timeframe returns seconds (float); we convert to ms
    return int(ex.parse_timeframe(timeframe) * 1000)


def _as_df(candles: list) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["ts","open","high","low","close","volume"])
    if df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"], index=pd.DatetimeIndex([], name="ts"))
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    # sicherstellen: float dtype
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: Timeframe = "1h",
    limit: int = 500,
    since_ms: Optional[int] = None,
    until_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lädt OHLCV von Bitget (Spot) mit automatischer Pagination.

    Args:
        symbol: z.B. "ETH/USDT"
        timeframe: z.B. "1m", "1h", "1d"
        limit: gewünschte Anzahl Bars (kann > 1000 sein; Pagination wird intern gehandhabt)
        since_ms: Startzeit (ms seit Epoch). Wenn None, wird aus limit/timeframe rückwärts geschätzt.
        until_ms: Endzeit (ms seit Epoch). Optional – begrenzt die oberen Zeitstempel.

    Returns:
        DataFrame mit Index=UTC-Zeitstempel und Spalten: open, high, low, close, volume
    """
    ex = get_exchange()
    step_ms = _timeframe_ms(ex, timeframe)

    # Wenn since_ms nicht gesetzt ist, schätze Startpunkt so, dass wir genügend Historie bekommen.
    # Wir gehen konservativ etwas weiter zurück (10% Puffer).
    if since_ms is None:
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        back_ms = int(step_ms * limit * 1.1)
        since_ms = max(0, (until_ms or now_ms) - back_ms)

    all_rows = []
    fetched = 0
    next_since = since_ms

    # Bitget/ccxt liefern meist max. 1000 Bars/Request
    MAX_PER_REQ = 1000

    while fetched < limit:
        per_request = min(MAX_PER_REQ, limit - fetched)
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=per_request)
        if not candles:
            break

        # Wenn ein until_ms gesetzt ist, vor dem Zusammenfügen filtern
        if until_ms is not None:
            candles = [c for c in candles if c[0] <= until_ms]
            if not candles:
                break

        all_rows.extend(candles)
        fetched = len(all_rows)

        # Nächster Startzeitpunkt: letzter Zeitstempel + eine Zeiteinheit
        last_ts = candles[-1][0]
        next_since = last_ts + step_ms

        # Wenn weniger als angefordert zurück kam, ist vermutlich Schluss (kein älteres Material)
        if len(candles) < per_request:
            break

    df = _as_df(all_rows)

    # Falls wir mehr als 'limit' Bars haben (wegen großzügiger since-Schätzung), die letzten 'limit' behalten
    if len(df) > limit:
        df = df.iloc[-limit:]

    return df


def fetch_ohlcv_days(
    symbol: str = "BTC/USDT",
    timeframe: Timeframe = "1m",
    days: int = 30,
    until_ms: Optional[int] = None,
) -> pd.DataFrame:
    """
    Komfort-Funktion: hole ca. 'days' Tage Historie für Timeframe.
    Achtung: 24/7-Markt – wir rechnen 24h pro Tag.

    Beispiel:
        df = fetch_ohlcv_days("ETH/USDT", "1m", days=30)
    """
    ex = get_exchange()
    step_ms = _timeframe_ms(ex, timeframe)
    bars_needed = int(days * 24 * 60 * 60 * 1000 / step_ms)  # Tage -> benötigte Bars
    return fetch_ohlcv(symbol, timeframe, limit=bars_needed, since_ms=None, until_ms=until_ms)


def fetch_ticker(symbol: str = "BTC/USDT") -> dict:
    """Einfacher Ticker (Last, Bid/Ask, etc.)."""
    ex = get_exchange()
    return ex.fetch_ticker(symbol)


def plot_candles(df: pd.DataFrame, title: str = "Candles"):
    import mplfinance as mpf
    data = df[["open","high","low","close","volume"]].copy()
    data.index.name = "Date"
    mpf.plot(data, type="candle", volume=True, title=title, style="classic")


if __name__ == "__main__":
    # Beispiele:
    # 1) ~30.000 1m-Bars (ca. 3 Wochen) für ETH/USDT
    df = fetch_ohlcv("ETH/USDT", "1m", limit=30000)
    print(df.tail())
    print(len(df), "rows")

    # 2) Alternativ: 30 Tage 1m mit convenience helper
    # df2 = fetch_ohlcv_days("ETH/USDT", "1m", days=30)
    # print(df2.tail())
    # print(len(df2), "rows")

    # 3) Plot (optional)
    # plot_candles(df, "ETH/USDT — 1m (sample)")
