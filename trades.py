# trades.py
import pandas as pd
from typing import List, Optional, Tuple

def _executed_signal(sig: pd.Series) -> pd.Series:
    """
    Ausgeführtes Signal (Handel am NÄCHSTEN Bar).
    0/1 -> long/flat, int.
    """
    return sig.shift(1).fillna(0).astype(int)

def _turn_points(sig_exec: pd.Series) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Entry-/Exit-Indizes anhand des ausgeführten Signals.
    Entry: 0 -> 1  (turn > 0)
    Exit:  1 -> 0  (turn < 0)
    """
    turns = sig_exec.diff().fillna(sig_exec.iloc[0])
    entries = turns[turns > 0].index
    exits   = turns[turns < 0].index
    return entries, exits

def build_trade_report(
    df: pd.DataFrame,
    sig: pd.Series,
    equity: pd.Series,
) -> pd.DataFrame:
    """
    Baut einen Trade-Report für Long-Only 0/1-Strategien.
    - df: muss 'close' (und DatetimeIndex) enthalten
    - sig: ursprüngliche (UNverschobene) Signal-Serie {0,1}
    - equity: Equity-Kurve aus dem Backtest (nach Kosten)
    Return: DataFrame mit einer Zeile je Trade (inkl. offener Trade, falls vorhanden)
    """
    sig_exec = _executed_signal(sig)
    entries, exits = _turn_points(sig_exec)

    # Zeitlich paaren: jeder Entry bekommt den nächsten späteren Exit (falls vorhanden)
    exits_iter = iter(exits)
    next_exit = next(exits_iter, None)

    rows: List[dict] = []
    for ent in entries:
        # passenden Exit suchen (erstes Exit > Entry)
        while next_exit is not None and next_exit <= ent:
            next_exit = next(exits_iter, None)

        exit_time: Optional[pd.Timestamp] = next_exit
        if exit_time is None:
            # offener Trade -> wir nehmen letzten verfügbaren Zeitpunkt
            exit_time = equity.index[-1]
            trade_open = True
        else:
            trade_open = False
            # für den nächsten Entry gleich den folgenden Exit im Iterator holen
            next_exit = next(exits_iter, None)

        # Preise (Ausführung am Bar selbst)
        entry_price = float(df.loc[ent, "close"])
        exit_price  = float(df.loc[exit_time, "close"])

        # Equity: vor Entry (t-1) und nach Entry (t), nach Kosten
        eq_before_entry = float(equity.shift(1).reindex([ent]).iloc[0]) if ent in equity.index else float(equity.asof(ent).shift(0))
        eq_after_entry  = float(equity.loc[ent])
        eq_at_exit      = float(equity.loc[exit_time])

        # PnL relativ NACH Entry (inkl. Entry-Kosten bereits im eq_after_entry enthalten)
        pnl_pct = (eq_at_exit / eq_after_entry) - 1.0

        # Bars gehalten
        try:
            bars_held = int(equity.index.get_loc(exit_time) - equity.index.get_loc(ent))
        except Exception:
            bars_held = int((exit_time - ent).total_seconds() // (equity.index[1] - equity.index[0]).total_seconds())

        # In-Trade Runup/Drawdown relativ zum Equity nach Entry
        seg = equity.loc[ent:exit_time]
        runup_pct    = float(seg.max() / eq_after_entry - 1.0)
        drawdown_pct = float(seg.min() / eq_after_entry - 1.0)

        rows.append({
            "entry_time": ent,
            "exit_time": exit_time,
            "open": trade_open,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "equity_before_entry": eq_before_entry,
            "equity_after_entry": eq_after_entry,
            "equity_at_exit": eq_at_exit,
            "pnl_pct": pnl_pct,
            "bars_held": bars_held,
            "runup_pct": runup_pct,
            "drawdown_pct": drawdown_pct,
        })

    report = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    return report
