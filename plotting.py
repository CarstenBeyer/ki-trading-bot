# in plotting.py (patched function)
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from trades import build_trade_report
from strategies import rsi_ewm  # reuse your RSI implementation

def plot_price_equity_dual_axis(
    df: pd.DataFrame,
    equity: pd.Series,
    sig: pd.Series,
    stats: dict | pd.Series | None = None,
    title: str = "Price, Equity, Bollinger & RSI",
    savefig: str | None = None,
    interactive_cursor: bool = True,
    # Price-axis behavior
    price_pad_frac: float = 0.10,
    price_quantiles: tuple[float, float] = (0.01, 0.99),
    nice_bounds: bool = True,
    # Keep plot indicators in sync with strategy (if provided)
    strategy=None,
    # Bollinger settings (used if strategy is None or fields missing)
    show_bbands: bool = True,
    bb_win: int | None = None,
    bb_k: float | None = None,
    # RSI settings
    show_rsi: bool = True,
    rsi_n: int | None = None,
    rsi_lines: tuple[float, float] | None = None,  # (lower, upper)
    # Exposure strip (0..1) for debugging regimes
    show_exposure: bool = True,
    # NEW: vertical mouse cursor line
    show_vcursor: bool = True,
) -> None:
    """
    Top panel: Price (with optional Bollinger Bands) + Equity (twin y-axis) + trade spans + stats box
    Middle panel (optional): Strategy exposure (executed, i.e., shifted by 1 bar)
    Bottom panel (optional): RSI with thresholds
    Adds a vertical cursor line that tracks the mouse across panels when show_vcursor=True.
    """
    # ---- Validate / prepare inputs
    if df is None or len(df) == 0:
        raise ValueError("df is empty.")
    if "close" not in df.columns:
        raise ValueError("df must contain a 'close' column.")
    df = df.sort_index()
    equity = equity.sort_index()
    equity_aligned = equity.reindex(df.index).ffill()
    px = df["close"].astype(float).dropna()

    # ---- Derive indicator params from strategy if provided
    if strategy is not None:
        if bb_win is None:
            bb_win = getattr(strategy, "bb_win", 20)
        if bb_k is None:
            bb_k = getattr(strategy, "bb_k", 2.0)
        if rsi_n is None:
            rsi_n = getattr(strategy, "rsi_n", 14)
        if rsi_lines is None:
            rsi_buy = float(getattr(strategy, "rsi_buy", 30.0))
            rsi_exit = float(getattr(strategy, "rsi_exit", 70.0))
            rsi_lines = (rsi_buy, rsi_exit)

    # ---- Final fallbacks if still None
    bb_win = 20 if bb_win is None else int(bb_win)
    bb_k = 2.0 if bb_k is None else float(bb_k)
    rsi_n = 14 if rsi_n is None else int(rsi_n)
    rsi_lines = (30.0, 70.0) if rsi_lines is None else tuple(rsi_lines)

    # ---- Indicators: Bollinger + RSI
    if show_bbands:
        bb_mid = px.rolling(bb_win, min_periods=bb_win).mean()
        bb_std = px.rolling(bb_win, min_periods=bb_win).std()
        bb_up = bb_mid + bb_k * bb_std
        bb_dn = bb_mid - bb_k * bb_std
    else:
        bb_mid = bb_up = bb_dn = None

    rsi = rsi_ewm(px, n=rsi_n) if show_rsi else None

    # ---- Exposure (executed signal, i.e., next-bar trade like backtest)
    sig_exec = None
    if show_exposure and sig is not None:
        sig_exec = (
            sig.shift(1)
            .astype(float)
            .clip(lower=0.0, upper=1.0)
            .reindex(df.index)
            .fillna(0.0)
        )

    # ---- Price y-axis padding & nice bounds
    qlo, qhi = price_quantiles
    lo = float(px.quantile(qlo)) if 0.0 <= qlo < 0.5 else float(px.min())
    hi = float(px.quantile(qhi)) if 0.5 < qhi <= 1.0 else float(px.max())
    if not (hi > lo):
        lo, hi = float(px.min()), float(px.max())
    rng = hi - lo if hi > lo else max(1e-6, abs(lo) * 0.01)
    pad = rng * price_pad_frac
    y_min_raw, y_max_raw = lo - pad, hi + pad

    if nice_bounds:
        def _nice_step(x: float) -> float:
            mag = 10 ** math.floor(math.log10(x)) if x > 0 else 1
            for k in (1, 2, 5, 10):
                if x <= k * mag:
                    return k * mag
            return 10 * mag
        step = _nice_step((y_max_raw - y_min_raw) / 7)  # target ~7 ticks
        y_min = math.floor(y_min_raw / step) * step
        y_max = math.ceil(y_max_raw / step) * step
    else:
        y_min, y_max = y_min_raw, y_max_raw

    # ---- Build trade report for overlays
    report = build_trade_report(df, sig, equity)

    # ---- Figure layout (use constrained_layout to avoid tight_layout warnings)
    rows = 1 + (1 if show_exposure else 0) + (1 if show_rsi else 0)
    if rows == 1:
        height_ratios = [1.0]
    elif rows == 2:
        height_ratios = [3.0, 1.0]
    else:
        height_ratios = [3.0, 0.8, 1.2]

    fig = plt.figure(
        figsize=(32, 14 if rows > 1 else 12),
        constrained_layout=True
    )
    gs = fig.add_gridspec(rows, 1, height_ratios=height_ratios)

    # Top axes: price + equity
    ax_price = fig.add_subplot(gs[0, 0])
    ax_equity = ax_price.twinx()

    # Optional middle: exposure
    ax_exp = None
    if show_exposure:
        ax_exp = fig.add_subplot(gs[1, 0], sharex=ax_price if rows > 1 else None)

    # Optional bottom: RSI
    ax_rsi = None
    if show_rsi:
        ax_rsi = fig.add_subplot(gs[-1, 0], sharex=ax_price if rows > 1 else None)

    # ---- Price (left y)
    ax_price.set_xlabel("Zeit (UTC)")
    ax_price.set_ylabel("Preis", color="tab:blue")
    ax_price.fill_between(df.index, df["close"], alpha=0.5, label="Preis (Close)")
    ax_price.plot(df.index, df["close"], linewidth=1.0, label="Close")
    ax_price.set_ylim(y_min, y_max)
    ax_price.tick_params(axis="y", labelcolor="tab:blue")
    ax_price.grid(True, which="both", linestyle="--", alpha=0.35)

    # Bollinger overlays
    if show_bbands and bb_mid is not None:
        ax_price.plot(bb_mid.index, bb_mid.values, linewidth=1.0, alpha=0.9, label=f"BB mid ({bb_win})")
        ax_price.plot(bb_up.index,  bb_up.values,  linewidth=0.9, alpha=0.9, linestyle="--", label=f"BB upper ({bb_k}σ)")
        ax_price.plot(bb_dn.index,  bb_dn.values,  linewidth=0.9, alpha=0.9, linestyle="--", label=f"BB lower ({bb_k}σ)")
        ax_price.fill_between(bb_up.index, bb_dn.values, bb_up.values, alpha=0.07, label="BB band")

    ax_price.legend(loc="upper left", fontsize=9, framealpha=0.4)

    # ---- Equity (right y)
    ax_equity.set_ylabel("Equity", color="tab:orange")
    ax_equity.plot(equity_aligned.index, equity_aligned.values, label="Equity", linewidth=1.5)
    ax_equity.tick_params(axis="y", labelcolor="tab:orange")

    # ---- Trade spans on price
    for _, trade in report.iterrows():
        entry = trade["entry_time"]
        exit_ = trade["exit_time"]
        color = "green" if trade["pnl_pct"] > 0 else "red"
        ax_price.axvspan(entry, exit_, color=color, alpha=0.10)

    # ---- Stats (figure-level text, works with constrained_layout)
    if stats is not None:
        if isinstance(stats, pd.Series):
            stats = stats.to_dict()
        base_text = "\n".join(
            [f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in stats.items()]
        )
    else:
        base_text = ""
    stats_text = fig.text(
        0.995, 0.99, base_text,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    # ---- Vertical cursor lines (price/equity + exposure + RSI)
    vlines = []
    if show_vcursor:
        # initialize at first timestamp (invisible until move)
        x0 = mdates.date2num(df.index[0].to_pydatetime())
        v_price = ax_price.axvline(x0, linestyle=":", linewidth=1.0, alpha=0.7, visible=False)
        vlines.append(v_price)
        # optional middle exposure line
        if ax_exp is not None:
            v_exp = ax_exp.axvline(x0, linestyle=":", linewidth=1.0, alpha=0.7, visible=False)
            vlines.append(v_exp)
        # optional RSI line
        if ax_rsi is not None:
            v_rsi = ax_rsi.axvline(x0, linestyle=":", linewidth=1.0, alpha=0.7, visible=False)
            vlines.append(v_rsi)

    # ---- Exposure subplot
    if ax_exp is not None and sig_exec is not None:
        ax_exp.plot(sig_exec.index, sig_exec.values, linewidth=1.0, label="Exposure (executed)")
        ax_exp.set_ylim(-0.05, 1.05)
        ax_exp.set_yticks([0.0, 0.5, 1.0])
        ax_exp.grid(True, linestyle="--", alpha=0.25)
        ax_exp.legend(loc="upper left", fontsize=9, framealpha=0.4)
        ax_exp.set_ylabel("Expo")

    # ---- RSI subplot
    if ax_rsi is not None and rsi is not None:
        lo_thr, hi_thr = rsi_lines
        ax_rsi.plot(rsi.index, rsi.values, linewidth=1.0, label=f"RSI({rsi_n})")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.axhline(lo_thr, linestyle="--", alpha=0.4, label=f"OS {lo_thr:.0f}")
        ax_rsi.axhline(hi_thr, linestyle="--", alpha=0.4, label=f"OB {hi_thr:.0f}")
        ax_rsi.grid(True, which="both", linestyle="--", alpha=0.25)
        ax_rsi.legend(loc="upper left", fontsize=9, framealpha=0.4)
        ax_rsi.set_ylabel("RSI")

    # ---- Interactive cursor: updates stats + vertical lines
    if interactive_cursor:
        def on_mouse_move(event):
            if event.xdata is None:
                # hide lines when off axes
                for ln in vlines:
                    ln.set_visible(False)
                stats_text.set_text(base_text)
                fig.canvas.draw_idle()
                return

            # Snap to nearest timestamp for clean alignment
            try:
                x_dt = mdates.num2date(event.xdata).replace(tzinfo=None)
                idx = equity_aligned.index.get_indexer([pd.Timestamp(x_dt)], method="nearest")[0]
                ts = equity_aligned.index[idx]
                px_val = df["close"].iloc[idx]
                eq_val = equity_aligned.iloc[idx]
                parts = [f"@ {ts.strftime('%Y-%m-%d %H:%M')}  Price: {px_val:.2f}  Equity: {eq_val:.4f}"]
                if rsi is not None and len(rsi) > idx:
                    rsi_val = float(rsi.iloc[idx])
                    if not pd.isna(rsi_val):
                        parts.append(f"RSI: {rsi_val:.1f}")
                if sig_exec is not None and len(sig_exec) > idx:
                    ex_val = float(sig_exec.iloc[idx])
                    if not pd.isna(ex_val):
                        parts.append(f"Expo: {ex_val:.2f}")
                cursor_text = ("\n" + "  ".join(parts)) if parts else ""
            except Exception:
                cursor_text = ""
                ts = None

            # Move/show vertical lines at snapped timestamp
            if show_vcursor and ts is not None:
                xnum = mdates.date2num(ts.to_pydatetime())
                for ln in vlines:
                    ln.set_xdata([xnum, xnum])
                    ln.set_visible(True)

            stats_text.set_text(base_text + cursor_text)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # ---- X-axis formatting
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_price.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax_price.xaxis.get_major_locator()))
    if rows > 1:
        plt.setp(ax_price.get_xticklabels(), visible=False)
        if ax_exp is not None and ax_rsi is not None:
            plt.setp(ax_exp.get_xticklabels(), visible=False)

    plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
