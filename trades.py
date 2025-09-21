# trades.py
import pandas as pd
from typing import List, Optional, Tuple

def _executed_signal(sig: pd.Series) -> pd.Series:
    """Signal so verschieben, dass am NÄCHSTEN Bar gehandelt wird (wie Backtest)."""
    return sig.shift(1).fillna(0)

def _discretize_exposure_to_phases(
    sig_exec: pd.Series,
    enter_level: float = 0.6,
    exit_level: float = 0.4,
) -> pd.Series:
    """
    Wandelt ein kontinuierliches Exposure (0..1) in 0/1-Phasen um (mit Hysterese).
    enter: exposure steigt über enter_level -> Phase=1
    exit : exposure fällt unter exit_level -> Phase=0
    """
    assert 0.0 <= exit_level < enter_level <= 1.0, "need 0 <= exit < enter <= 1"
    phase = []
    in_pos = 0
    for x in sig_exec.fillna(0).astype(float):
        #print("x:", x, "in_pos:", in_pos)   # Debug-Ausgabe
        if in_pos == 0 and x >= enter_level:
            in_pos = 1
        elif in_pos == 1 and x <= exit_level:
            in_pos = 0
        phase.append(in_pos)
    return pd.Series(phase, index=sig_exec.index, dtype=int, name="phase01")

def _turn_points_from_phase(phase01: pd.Series) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    turns = phase01.diff().fillna(phase01.iloc[0])
    entries = turns[turns > 0].index
    exits   = turns[turns < 0].index
    return entries, exits

def build_trade_report(
    df: pd.DataFrame,
    sig: pd.Series,        # kann 0/1 ODER 0..1 sein
    equity: pd.Series,
    *,
    enter_level: float = 0.6,
    exit_level: float = 0.4,
) -> pd.DataFrame:
    """
    Erzeugt einen Trade-Report. Funktioniert für binäre und kontinuierliche Signale.
    - Kontinuierliche Signale werden via Hysterese (enter_level/exit_level) in Phasen umgewandelt.
    - Ausführung am nächsten Bar (konsistent mit Backtest).
    """
    # 1) Ausgeführtes Signal (nächster Bar)
    sig_exec = _executed_signal(sig)

    # 2) Falls binär: direkt; falls kontinuierlich: in Phasen 0/1 diskretisieren
    if set(pd.Series(sig_exec.dropna().unique()).round(6)) <= {0.0, 1.0}:
        phase01 = sig_exec.astype(float)
    else:
        phase01 = _discretize_exposure_to_phases(sig_exec, enter_level=enter_level, exit_level=exit_level)

    # 3) Entry/Exit-Zeitpunkte
    entries, exits = _turn_points_from_phase(phase01)

    rows: List[dict] = []
    if len(entries) == 0 and len(exits) == 0:
        # Keine Trades -> leeres, aber wohlgeformtes DataFrame
        return pd.DataFrame(columns=[
            "entry_time","exit_time","open",
            "entry_price","exit_price",
            "equity_before_entry","equity_after_entry","equity_at_exit",
            "pnl_pct","bars_held","runup_pct","drawdown_pct",
        ])

    # 4) Equity & Preis auf gleichen Index
    df = df.sort_index()
    equity = equity.sort_index()
    eq_aligned = equity.reindex(df.index).ffill()

    # 5) Pairs bilden (jeder Entry bekommt den nächsten Exit; offener Trade erlaubt)
    ex_iter = iter(exits)
    next_ex = next(ex_iter, None)

    for ent in entries:
        while next_ex is not None and next_ex <= ent:
            next_ex = next(ex_iter, None)
        exit_time: Optional[pd.Timestamp] = next_ex if next_ex is not None else df.index[-1]
        trade_open = next_ex is None
        if next_ex is not None:
            next_ex = next(ex_iter, None)

        # Preise (Close zu Bar-Zeitpunkt)
        entry_price = float(df.loc[ent, "close"]) if ent in df.index else float(df["close"].asof(ent))
        exit_price  = float(df.loc[exit_time, "close"]) if exit_time in df.index else float(df["close"].asof(exit_time))

        # Equity vor/nach Entry und am Exit
        eq_before_entry = float(eq_aligned.shift(1).reindex([ent]).iloc[0]) if ent in eq_aligned.index else float(eq_aligned.asof(ent))
        eq_after_entry  = float(eq_aligned.loc[ent]) if ent in eq_aligned.index else float(eq_aligned.asof(ent))
        eq_at_exit      = float(eq_aligned.loc[exit_time]) if exit_time in eq_aligned.index else float(eq_aligned.asof(exit_time))

        pnl_pct = (eq_at_exit / max(eq_after_entry, 1e-12)) - 1.0

        # Bars gehalten
        try:
            bars_held = int(eq_aligned.index.get_loc(exit_time) - eq_aligned.index.get_loc(ent))
        except Exception:
            bars_held = int((exit_time - ent).total_seconds() // (eq_aligned.index[1] - eq_aligned.index[0]).total_seconds())

        seg = eq_aligned.loc[ent:exit_time]
        runup_pct    = float(seg.max() / max(eq_after_entry, 1e-12) - 1.0)
        drawdown_pct = float(seg.min() / max(eq_after_entry, 1e-12) - 1.0)

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
