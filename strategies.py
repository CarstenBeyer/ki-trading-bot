import pandas as pd
import numpy as np

def sma_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Gibt 1 (long) oder 0 (flat) je Bar zurück.
    Handeln tun wir am NÄCHSTEN Bar -> daher im Backtest .shift(1).
    """
    ma_fast = df["close"].rolling(fast).mean()
    ma_slow = df["close"].rolling(slow).mean()
    sig = (ma_fast > ma_slow).astype(int)
    return sig

def donchian_breakout_signal(df: pd.DataFrame, entry_n: int = 20, exit_n: int = 10) -> pd.Series:
    """
    Long/Flat: 1 wenn im Trade, 0 sonst.
    Entry: Close > rolling High(entry_n)
    Exit : Close < rolling Low(exit_n)
    State-basiert, damit Entry/Exit-Regeln sauber gelten.
    """
    close = df["close"]
    # Höchst-/Tiefststände OHNE aktuellen Bar (konservativ)
    # -> damit das Signal nicht durch den aktuellen Bar 'gelookaheadet' ist.
    hh = close.shift(1).rolling(entry_n).max()
    ll = close.shift(1).rolling(exit_n).min()

    up_break   = close > hh
    down_break = close < ll

    pos = np.zeros(len(df), dtype=int)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0 and up_break.iloc[i]:
            in_pos = 1
        elif in_pos == 1 and down_break.iloc[i]:
            in_pos = 0
        pos[i] = in_pos

    sig = pd.Series(pos, index=df.index, name="donchian_long")
    return sig

# strategies.py (ergänzen)
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # EWM als Standard-RSI-Glättung
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def rsi_meanrev_signal(df: pd.DataFrame, n: int = 14, buy_thr: float = 30, exit_thr: float = 55) -> pd.Series:
    """
    Long/Flat: 1 wenn im Trade, 0 sonst.
    Entry: RSI < buy_thr
    Exit : RSI > exit_thr
    """
    r = _rsi(df["close"], n=n)
    buy  = r < buy_thr
    sell = r > exit_thr

    pos = []
    in_pos = 0
    for b, s in zip(buy, sell):
        if in_pos == 0 and b:
            in_pos = 1
        elif in_pos == 1 and s:
            in_pos = 0
        pos.append(in_pos)
    sig = pd.Series(pos, index=df.index, name="rsi_meanrev_long")
    return sig