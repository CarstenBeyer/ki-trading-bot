import matplotlib.pyplot as plt
import pandas as pd

def _trade_marks_from_signal(sig: pd.Series):
    sig_exec = sig.shift(1).fillna(0).astype(int)
    turns = sig_exec.diff().fillna(sig_exec.iloc[0])
    entries = turns[turns > 0].index
    exits   = turns[turns < 0].index
    return entries, exits

def plot_equity_with_trades(equity: pd.Series, sig: pd.Series, title: str = "Equity Curve with Trades"):
    entries, exits = _trade_marks_from_signal(sig)
    plt.figure(figsize=(10, 4))
    plt.plot(equity.index, equity.values, label="Equity (start=1.0)")
    if len(entries):
        plt.scatter(entries, equity.loc[entries], marker="^", s=80, label="Entry")
    if len(exits):
        plt.scatter(exits, equity.loc[exits], marker="v", s=80, label="Exit")
    plt.title(title)
    plt.xlabel("Zeit (UTC)")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.show()
