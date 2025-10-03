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
    interactive_cursor: bool = True,
    price_pad_frac: float = 0.1,            # 10% padding above/below
    price_quantiles: tuple[float,float] = (0.01, 0.99),  # ignore extremes
    nice_bounds: bool = True                # round to nice steps
) -> None:
    """
    Plot mit zwei Y-Achsen:
      - links: Preis (hellblaue Fläche, gepaddet und schön skaliert)
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

    # --- Preisdaten vorbereiten (mit Padding) ---
    px = df["close"].dropna()
    qlo, qhi = price_quantiles
    lo = float(px.quantile(qlo)) if 0.0 <= qlo < 0.5 else float(px.min())
    hi = float(px.quantile(qhi)) if 0.5 < qhi <= 1.0 else float(px.max())
    if not (hi > lo):
        lo, hi = float(px.min()), float(px.max())

    rng = hi - lo if hi > lo else max(1e-6, lo * 0.01)
    pad = rng * price_pad_frac
    y_min_raw, y_max_raw = lo - pad, hi + pad

    if nice_bounds:
        import math
        def _nice_step(x: float) -> float:
            # 1–2–5 progression
            mag = 10 ** math.floor(math.log10(x)) if x > 0 else 1
            for k in (1, 2, 5, 10):
                if x <= k * mag:
                    return k * mag
            return 10 * mag
        step = _nice_step((y_max_raw - y_min_raw) / 7)  # aim ~7 ticks
        y_min = math.floor(y_min_raw / step) * step
        y_max = math.ceil(y_max_raw / step) * step
    else:
        y_min, y_max = y_min_raw, y_max_raw

    # --- Preis
    ax1.set_xlabel("Zeit (UTC)")
    ax1.set_ylabel("Preis", color="tab:blue")
    ax1.fill_between(df.index, df["close"], color="lightblue", alpha=0.5, label="Preis (Close)")
    ax1.plot(df.index, df["close"], color="tab:blue", linewidth=1.0)
    ax1.set_ylim(y_min, y_max)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

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
            if not event.inaxes or event.xdata is None:
                return
            try:
                x = mdates.num2date(event.xdata).replace(tzinfo=None)
                nearest_idx = equity_aligned.index.get_indexer([pd.Timestamp(x)], method="nearest")[0]
                ts = equity_aligned.index[nearest_idx]
                px_val = df["close"].iloc[nearest_idx]
                eq_val = equity_aligned.iloc[nearest_idx]
                cursor_text = f"\n@ {ts.strftime('%Y-%m-%d %H:%M')}  Price: {px_val:.2f}  Equity: {eq_val:.4f}"
            except Exception:
                cursor_text = ""
            box.set_text(text + cursor_text)
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
