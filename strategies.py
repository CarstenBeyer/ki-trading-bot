import pandas as pd

def sma_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Gibt 1 (long) oder 0 (flat) je Bar zurück.
    Handeln tun wir am NÄCHSTEN Bar -> daher im Backtest .shift(1).
    """
    ma_fast = df["close"].rolling(fast).mean()
    ma_slow = df["close"].rolling(slow).mean()
    sig = (ma_fast > ma_slow).astype(int)
    return sig