import ccxt
import pandas as pd
from typing import Literal, Optional

Timeframe = Literal["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d","1w","1M"]

def get_exchange(api_key: str | None = None,
                 api_secret: str | None = None,
                 passphrase: str | None = None) -> ccxt.bitget:
    """
    Erstellt einen Bitget-Client. Ohne Keys -> nur öffentliche Daten (OHLCV, Ticker).
    Mit Keys -> später Orders möglich.
    """
    cfg = {
        "enableRateLimit": True,
    }
    if api_key and api_secret and passphrase:
        cfg.update({"apiKey": api_key, "secret": api_secret, "password": passphrase})
    ex = ccxt.bitget(cfg)
    ex.load_markets()
    return ex

def fetch_ohlcv(symbol: str = "BTC/USDT",
                timeframe: Timeframe = "1h",
                limit: int = 500,
                since_ms: Optional[int] = None) -> pd.DataFrame:
    """
    Lädt OHLCV von Bitget (Spot). Gibt DataFrame mit UTC-Index zurück.
    """
    ex = get_exchange()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since_ms)
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")

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
    # Beispiel: letzte 100 1d-Kerzen für BTC/USDT
    df = fetch_ohlcv("BTC/USDT", "1d", 100)
    print(df)
    plot_candles(df)
    
    #print(fetch_ticker("BTC/USDT"))
