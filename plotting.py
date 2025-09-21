# plotting.py
import matplotlib.pyplot as plt
import pandas as pd
from trades import build_trade_report

def plot_price_equity_dual_axis(
    df: pd.DataFrame,   # braucht Spalten: close, optional ohlc
    equity: pd.Series,  # Equity-Kurve
    sig: pd.Series,     # unverschobenes Signal {0,1}
    title: str = "Price & Equity (Dual Axis)",
    savefig: str | None = None,
) -> None:
    """
    Plot mit zwei Y-Achsen:
    - links: Preis (Close)
    - rechts: Equity
    - Marker (▲/▼) für Entry/Exit auf beiden Kurven
    """
    # Trade-Report für Marker
    report = build_trade_report(df, sig, equity)
    entries = report["entry_time"]
    exits   = report["exit_time"]

    fig, ax1 = plt.subplots(figsize=(12,6))

    # Preis (linke Achse)
    ax1.set_xlabel("Zeit (UTC)")
    ax1.set_ylabel("Preis", color="tab:blue")
    ax1.plot(df.index, df["close"], color="tab:blue", label="Preis (Close)")
    if len(entries):
        ax1.scatter(entries, df.loc[entries, "close"], marker="^", color="green", s=80, label="Entry")
    if len(exits):
        ax1.scatter(exits, df.loc[exits, "close"], marker="v", color="red", s=80, label="Exit")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Equity (rechte Achse)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Equity", color="tab:orange")
    ax2.plot(equity.index, equity.values, color="tab:orange", label="Equity")
    if len(entries):
        ax2.scatter(entries, equity.loc[entries], marker="^", color="green", s=60)
    if len(exits):
        ax2.scatter(exits, equity.loc[exits], marker="v", color="red", s=60)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Titel & Legende
    plt.title(title)
    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
