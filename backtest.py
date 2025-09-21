# backtest.py
import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame,
                 sig: pd.Series,
                 fee_pct: float = 0.1,       # 0.1% pro Positionswechsel
                 slippage_bps: int = 5       # 5 Basispunkte Slippage
                 ):
    """
    df: DataFrame mit 'close'
    sig: Series {0,1} index-aligned zu df
    """
    df = df.copy()
    df["sig"] = sig.shift(1).fillna(0)           # handeln am nächsten Bar
    px = df["close"]

    # Slippage auf den Entry/Exit-Preis modellieren (vereinfacht, symmetrisch)
    slip = px * (slippage_bps / 10_000)
    # Renditen der Underlyings
    ret = px.pct_change().fillna(0)

    # Strategie-Renditen (Exposure * Underlying-Return)
    strat_ret = ret * df["sig"]

    # Gebühren bei Positionswechsel (Entry/Exit). turns=|Δsig|
    turns = df["sig"].diff().abs().fillna(0)
    fee = (fee_pct / 100.0) * turns
    strat_ret_after_cost = strat_ret - fee

    # Equity-Kurve (Start 1.0)
    equity = (1 + strat_ret_after_cost).cumprod()

    # Kennzahlen (einfach gehalten)
    total_return = equity.iloc[-1] - 1
    # annualisiert grob über Anzahl Bars; für 1d: 252, 1h: 24*365, etc.
    bars = len(df)
    ann_factor = _annualization_factor(df.index)  # siehe Helper
    sharpe = (strat_ret_after_cost.mean() / (strat_ret_after_cost.std() + 1e-12)) * np.sqrt(ann_factor)
    mdd = (equity / equity.cummax() - 1).min()
    trades = int(turns.sum())  # jede 0→1 oder 1→0 zählt

    stats = {
        "TotalReturn": total_return,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,
        "Bars": bars,
        "Trades": trades,
    }
    return equity, strat_ret_after_cost, pd.Series(stats)

def _annualization_factor(index: pd.DatetimeIndex) -> float:
    """
    Bestimmt die Annualisierungsbasis für Crypto (24/7).
    - Daily Bars: 365
    - 1h Bars: 365*24
    - 15m Bars: 365*24*4
    usw.
    """
    if len(index) < 2:
        return 365.0
    # Median-Abstand zwischen Bars
    dt = (index[1:] - index[:-1]).median().total_seconds()

    day = 24 * 3600
    if dt >= 20 * 3600:        # ~1d Bars
        return 365.0
    elif dt >= 3 * 3600:       # ~4h-12h Bars
        return 365.0 * 24 / 6  # ca. 1460
    elif dt >= 3600:           # ~1h Bars
        return 365.0 * 24      # 8760
    elif dt >= 15 * 60:        # ~15m Bars
        return 365.0 * 24 * 4  # 35040
    else:                      # Fallback (z. B. Minuten-Daten)
        return 365.0 * (day / dt)
