# plotting.py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from trades import build_trade_report

def plot_price_equity_dual_axis(
    df: pd.DataFrame,
    equity: pd.Series,
    sig: pd.Series,
    stats: dict | pd.Series | None = None,
    title: str = "Price & Equity (Dual Axis)",
    savefig: str | None = None,
    interactive_cursor: bool = True,   # <<<< Neu
) -> None:
    """
    Plot mit zwei Y-Achsen:
      - links: Preis (hellblaue Fläche)
      - rechts: Equity (orange Linie)
      - Marker (▲/▼) für Entry/Exit
      - Stats-Textbox + optional Live-Cursor-Info (Equity/Preis am Maus-X)
    """
    # --- Vorbereitung
    report = build_trade_report(df, sig, equity)
    df = df.sort_index()
    equity = equity.sort_index()
    equity_aligned = equity.reindex(df.index).ffill()

    fig, ax1 = plt.subplots(figsize=(32, 12))
    ax2 = ax1.twinx()

    # --- Preis
    ax1.set_xlabel("Zeit (UTC)")
    ax1.set_ylabel("Preis", color="tab:blue")
    ax1.fill_between(df.index, df["close"], color="lightblue", alpha=0.5, label="Preis (Close)")
    ax1.plot(df.index, df["close"], color="tab:blue", linewidth=1.0)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # --- Equity
    ax2.set_ylabel("Equity", color="tab:orange")
    ax2.plot(equity_aligned.index, equity_aligned.values, color="tab:orange", label="Equity", linewidth=1.5)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # --- Trade-Overlays
    for _, trade in report.iterrows():
        entry = trade["entry_time"]
        exit_ = trade["exit_time"]
        color = "green" if trade["pnl_pct"] > 0 else "red"
        ax1.axvspan(entry, exit_, color=color, alpha=0.1)

    # --- Stats-Textbox
    if stats is not None:
        if isinstance(stats, pd.Series):
            stats = stats.to_dict()
        text = "\n".join(
            [f"{k}: {v:.4f}" if isinstance(v, (int,float)) else f"{k}: {v}" for k,v in stats.items()]
        )
    else:
        text = ""

    # Textbox erstellen (rechts oben)
    box = ax2.text(
        1.02, 0.98, text,
        transform=ax2.transAxes,
        fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    # --- Interaktive Cursor-Anzeige
    if interactive_cursor:
        def on_mouse_move(event):
            if not event.inaxes:
                return
            if event.xdata is None:
                return
            try:
                # X-Position → nächster Index
                x = mdates.num2date(event.xdata).replace(tzinfo=None)
                nearest_idx = equity_aligned.index.get_indexer([pd.Timestamp(x)], method="nearest")[0]
                ts = equity_aligned.index[nearest_idx]
                px = df["close"].iloc[nearest_idx]
                eq = equity_aligned.iloc[nearest_idx]
                cursor_text = f"\n@ {ts.strftime('%Y-%m-%d %H:%M')}  Price: {px:.2f}  Equity: {eq:.4f}"
            except Exception:
                cursor_text = ""
            # Stats + Cursor zusammen
            full_text = text + cursor_text
            box.set_text(full_text)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # --- Layout
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax1.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    plt.title(title)
    fig.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
