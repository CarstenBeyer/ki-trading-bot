# plotting.py
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from trades import build_trade_report
from strategies import RegimeAdaptiveHybrid, rsi_ewm  # helpers & RSI


def plot_price_equity_dual_axis(
    df: pd.DataFrame,
    equity: pd.Series,
    sig: pd.Series,
    stats: dict | pd.Series | None = None,
    title: str = "Price, Equity, BB, RSI, Trend, Donchian",
    savefig: str | None = None,
    interactive_cursor: bool = True,

    # Price-axis behavior
    price_pad_frac: float = 0.10,
    price_quantiles: tuple[float, float] = (0.01, 0.99),
    nice_bounds: bool = True,

    # Pull params from strategy to keep plots in sync (preferred)
    strategy=None,

    # Bollinger
    show_bbands: bool = True,
    bb_win: int | None = None,
    bb_k: float | None = None,

    # RSI
    show_rsi: bool = True,
    rsi_n: int | None = None,
    rsi_lines: tuple[float, float] | None = None,  # (lower, upper)

    # Exposure strip
    show_exposure: bool = True,

    # Trend diagnostics (slope + z)
    show_trend_diag: bool = True,

    # ---- Vol panel removed; keep these only for regime gating (no plotting) ----
    show_vol_diag: bool = False,         # deprecated, ignored
    vol_win: int | None = None,          # used only to compute ann_vol for regime gate
    vol_cap: float | None = None,        # used only for regime gate

    # Donchian panel (new)
    show_don_panel: bool = True,
    mark_don_breakouts: bool = True,     # upward breakouts (close > HH*(1+pad))
    mark_don_exits: bool = True,         # downward exits  (close < LL*(1-exit_pad))
    don_fill_between: bool = True,       # light band fill between LL and HH
) -> None:
    """
    Panels:
      1) Price (with optional Bollinger) + Equity + trade spans + stats
      2) (optional) Exposure (executed signal)
      3) (optional) Trend diagnostics: slope & z-score (with thresholds)
      4) (optional) Donchian diagnostics: HH(don_entry) & LL(don_exit) + breakout markers
      5) (optional) RSI

    Notes:
    - Annualized volatility is NOT plotted anymore; it's still used internally to compute is_trend.
    - Donchian is shown in its own panel (no overlay on the price panel).
    """

    # ---- Validate / align
    if df is None or len(df) == 0:
        raise ValueError("df is empty.")
    if "close" not in df.columns:
        raise ValueError("df must contain a 'close' column.")

    df = df.sort_index()
    equity = equity.sort_index()
    equity_aligned = equity.reindex(df.index).ffill()
    px = df["close"].astype(float)

    # ---- Parameters from strategy (single source of truth)
    if strategy is not None:
        if bb_win is None: bb_win = int(getattr(strategy, "bb_win", 20))
        if bb_k   is None: bb_k   = float(getattr(strategy, "bb_k", 2.0))
        if rsi_n  is None: rsi_n  = int(getattr(strategy, "rsi_n", 14))
        if rsi_lines is None:
            rsi_lines = (
                float(getattr(strategy, "rsi_buy", 30.0)),
                float(getattr(strategy, "rsi_exit", 70.0)),
            )
        # trend thresholds
        trend_win   = int(getattr(strategy, "trend_win", 200))
        slope_enter = float(getattr(strategy, "slope_enter", 0.0))
        slope_exit  = float(getattr(strategy, "slope_exit", -0.0005))
        z_enter     = float(getattr(strategy, "z_enter", 0.0))
        z_exit      = float(getattr(strategy, "z_exit", -0.25))
        # vol thresholds (regime gating only; no plot)
        if vol_win is None: vol_win = int(getattr(strategy, "vol_win", 30))
        if vol_cap is None: vol_cap = float(getattr(strategy, "max_ann_vol", 1.0))
        # donchian
        don_entry = int(getattr(strategy, "don_entry", 20))
        don_exit  = int(getattr(strategy, "don_exit", 10))
        don_src   = str(getattr(strategy, "don_src", "hl"))
        don_pad   = float(getattr(strategy, "don_break_pad", 0.0))
        don_exit_pad = float(getattr(strategy, "don_exit_pad", 0.0))
    else:
        bb_win = 20 if bb_win is None else int(bb_win)
        bb_k   = 2.0 if bb_k   is None else float(bb_k)
        rsi_n  = 14  if rsi_n  is None else int(rsi_n)
        rsi_lines = (30.0, 70.0) if rsi_lines is None else tuple(rsi_lines)
        trend_win, slope_enter, slope_exit = 200, 0.0, -0.0005
        z_enter, z_exit = 0.0, -0.25
        vol_win, vol_cap = (30 if vol_win is None else int(vol_win)), (1.0 if vol_cap is None else float(vol_cap))
        don_entry, don_exit, don_src, don_pad, don_exit_pad = 20, 10, "hl", 0.0, 0.0

    # ---- Indicators
    # Bollinger
    if show_bbands:
        bb_mid = px.rolling(bb_win, min_periods=bb_win).mean()
        bb_std = px.rolling(bb_win, min_periods=bb_win).std()
        bb_up  = bb_mid + bb_k * bb_std
        bb_dn  = bb_mid - bb_k * bb_std
    else:
        bb_mid = bb_std = bb_up = bb_dn = None

    # RSI
    rsi = rsi_ewm(px, n=rsi_n) if show_rsi else None

    # Executed exposure
    sig_exec = None
    if show_exposure and sig is not None:
        sig_exec = (
            sig.shift(1).astype(float)
               .clip(lower=0.0, upper=1.0)
               .reindex(df.index).fillna(0.0)
        )

    # Trend diagnostics (slope, z, vol, is_trend)
    slope_series = RegimeAdaptiveHybrid._slope_norm(px, win=trend_win)
    sma_tr = px.rolling(trend_win, min_periods=trend_win).mean()
    std_tr = px.rolling(trend_win, min_periods=trend_win).std()
    z_trend = (px - sma_tr) / (std_tr + 1e-12)
    ann_vol = RegimeAdaptiveHybrid._realized_vol(px, win=vol_win)

    enter_cond = (slope_series > slope_enter) & (z_trend > z_enter) & (ann_vol <= vol_cap)
    exit_cond  = (slope_series < slope_exit) | (z_trend < z_exit)  | (ann_vol >  vol_cap)
    is_trend   = RegimeAdaptiveHybrid._hysteresis_bool(enter_cond, exit_cond, index=df.index)

    # Donchian (for its own panel)
    if show_don_panel:
        if don_src.lower() == "hl":
            hh = df["high"].shift(1).rolling(don_entry, min_periods=don_entry).max()
            ll = df["low"] .shift(1).rolling(don_exit,  min_periods=don_exit ).min()
        else:
            hh = df["close"].shift(1).rolling(don_entry, min_periods=don_entry).max()
            ll = df["close"].shift(1).rolling(don_exit,  min_periods=don_exit ).min()
        # breakout/exit booleans
        brk_up = (df["close"] > (hh * (1.0 + don_pad))) & (~hh.isna())
        brk_dn = (df["close"] < (ll * (1.0 - don_exit_pad))) & (~ll.isna())
    else:
        hh = ll = None
        brk_up = brk_dn = pd.Series(False, index=df.index)

    # ---- Price y-axis bounds
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
                if x <= k * mag: return k * mag
            return 10 * mag
        step = _nice_step((y_max_raw - y_min_raw) / 7)
        y_min = math.floor(y_min_raw / step) * step
        y_max = math.ceil(y_max_raw / step) * step
    else:
        y_min, y_max = y_min_raw, y_max_raw

    # ---- Build trade report
    report = build_trade_report(df, sig, equity)

    # ---- Figure layout: Price + (Exposure) + Trend + Donchian + (RSI)
    rows = 1 + (1 if show_exposure else 0) + (1 if show_trend_diag else 0) + (1 if show_don_panel else 0) + (1 if show_rsi else 0)
    if   rows == 1: height_ratios = [1.0]
    elif rows == 2: height_ratios = [3.0, 1.0]
    elif rows == 3: height_ratios = [3.0, 0.9, 1.0]
    elif rows == 4: height_ratios = [3.0, 0.7, 1.0, 1.0]
    else:           height_ratios = [3.0, 0.6, 1.0, 1.0, 1.0]

    fig = plt.figure(figsize=(32, 16 if rows > 1 else 12), constrained_layout=True)
    gs = fig.add_gridspec(rows, 1, height_ratios=height_ratios)

    # Top axes: price + equity
    ax_price = fig.add_subplot(gs[0, 0])
    ax_equity = ax_price.twinx()

    # Middle axes
    row_idx = 1
    ax_exp = ax_trend = ax_trend_z = ax_don = ax_rsi = None

    if show_exposure:
        ax_exp = fig.add_subplot(gs[row_idx, 0], sharex=ax_price)
        row_idx += 1

    if show_trend_diag:
        ax_trend = fig.add_subplot(gs[row_idx, 0], sharex=ax_price)
        ax_trend_z = ax_trend.twinx()
        row_idx += 1

    if show_don_panel:
        ax_don = fig.add_subplot(gs[row_idx, 0], sharex=ax_price)
        row_idx += 1

    if show_rsi:
        ax_rsi = fig.add_subplot(gs[row_idx, 0], sharex=ax_price)

    # ---- Price (left y)
    ax_price.set_xlabel("Zeit (UTC)")
    ax_price.set_ylabel("Preis", color="tab:blue")
    ax_price.fill_between(df.index, df["close"], alpha=0.5, label="Preis (Close)")
    ax_price.plot(df.index, df["close"], linewidth=1.0, label="Close")
    ax_price.set_ylim(y_min, y_max)
    ax_price.tick_params(axis="y", labelcolor="tab:blue")
    ax_price.grid(True, which="both", linestyle="--", alpha=0.35)

    # Bollinger overlays (price panel)
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

    # ---- Trade spans (price axis)
    for _, trade in report.iterrows():
        entry = trade["entry_time"]; exit_ = trade["exit_time"]
        color = "green" if trade["pnl_pct"] > 0 else "red"
        ax_price.axvspan(entry, exit_, color=color, alpha=0.10)

    # ---- Stats box
    if stats is not None:
        if isinstance(stats, pd.Series): stats = stats.to_dict()
        base_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, (int,float)) else f"{k}: {v}" for k,v in stats.items()])
    else:
        base_text = ""
    stats_text = fig.text(
        0.995, 0.99, base_text,
        fontsize=9, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
    )

    # ---- Exposure subplot
    if ax_exp is not None and sig_exec is not None:
        ax_exp.plot(sig_exec.index, sig_exec.values, linewidth=1.0, label="Exposure (executed)")
        ax_exp.set_ylim(-0.05, 1.05)
        ax_exp.set_yticks([0.0, 0.5, 1.0])
        ax_exp.grid(True, linestyle="--", alpha=0.25)
        ax_exp.legend(loc="upper left", fontsize=9, framealpha=0.4)
        ax_exp.set_ylabel("Expo")

    # ---- Trend diagnostics subplot
    if ax_trend is not None:
        # shade trend regime on/off
        in_block = False; start = None
        for t, val in is_trend.items():
            if val and not in_block: in_block, start = True, t
            elif in_block and not val: ax_trend.axvspan(start, t, color="green", alpha=0.06); in_block = False
        if in_block: ax_trend.axvspan(start, df.index[-1], color="green", alpha=0.06)

        # slope (left y)
        ax_trend.plot(slope_series.index, slope_series.values, linewidth=1.0, label="slope_norm", color="tab:green")
        ax_trend.axhline(slope_enter, linestyle="--", alpha=0.5, label=f"slope_enter {slope_enter:g}", color="tab:green")
        ax_trend.axhline(slope_exit,  linestyle="--", alpha=0.5, label=f"slope_exit {slope_exit:g}", color="tab:green")
        ax_trend.set_ylabel("slope (norm)")
        ax_trend.grid(True, linestyle="--", alpha=0.25)
        ax_trend.legend(loc="upper left", fontsize=9, framealpha=0.4)

        # z (right y)
        ax_trend_z.plot(z_trend.index, z_trend.values, linewidth=1.0, alpha=0.9, label="z-score", color="tab:purple")
        ax_trend_z.axhline(z_enter, linestyle="--", alpha=0.5, color="tab:purple", label=f"z_enter {z_enter:g}")
        ax_trend_z.axhline(z_exit,  linestyle="--", alpha=0.5, color="tab:purple", label=f"z_exit {z_exit:g}")
        ax_trend_z.set_ylabel("z (close vs SMA)")
        ax_trend_z.legend(loc="upper right", fontsize=9, framealpha=0.4)

    # ---- Donchian diagnostics subplot
    if ax_don is not None and hh is not None and ll is not None:
        ax_don.set_ylabel("Donchian")
        # optional fill between LL and HH
        if don_fill_between:
            ax_don.fill_between(hh.index, ll.values, hh.values, alpha=0.06, label=f"Band [{don_entry}/{don_exit}]")
        # plot HH/LL
        ax_don.plot(hh.index, hh.values, linestyle="--", linewidth=1.0, label=f"HH ({don_entry})")
        ax_don.plot(ll.index, ll.values, linestyle="--", linewidth=1.0, label=f"LL ({don_exit})")
        # light close overlay for context
        ax_don.plot(px.index, px.values, linewidth=0.8, alpha=0.6, label="Close")
        # markers
        if mark_don_breakouts and brk_up.any():
            ax_don.scatter(df.index[brk_up], px[brk_up], s=18, marker="^", alpha=0.85, label="Breakout ↑")
        if mark_don_exits and brk_dn.any():
            ax_don.scatter(df.index[brk_dn], px[brk_dn], s=18, marker="v", alpha=0.85, label="Exit ↓")
        ax_don.grid(True, linestyle="--", alpha=0.25)
        ax_don.legend(loc="upper left", fontsize=9, framealpha=0.4)

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

    # ---- Interactive cursor: show Price/Equity + RSI + Trend + Donchian flags
    if interactive_cursor:
        def on_mouse_move(event):
            if not event.inaxes or event.xdata is None:
                stats_text.set_text(base_text); fig.canvas.draw_idle(); return
            try:
                x_dt = mdates.num2date(event.xdata).replace(tzinfo=None)
                idx = equity_aligned.index.get_indexer([pd.Timestamp(x_dt)], method="nearest")[0]
                ts = equity_aligned.index[idx]
                px_val = df["close"].iloc[idx]
                eq_val = equity_aligned.iloc[idx]
                parts = [f"@ {ts.strftime('%Y-%m-%d %H:%M')}  Price: {px_val:.2f}  Equity: {eq_val:.4f}"]
                # RSI
                if rsi is not None and len(rsi) > idx:
                    rv = float(rsi.iloc[idx])
                    if not pd.isna(rv): parts.append(f"RSI: {rv:.1f}")
                # Trend diag
                sl = float(slope_series.iloc[idx]) if len(slope_series) > idx else float("nan")
                zv = float(z_trend.iloc[idx]) if len(z_trend) > idx else float("nan")
                it = bool(is_trend.iloc[idx]) if len(is_trend) > idx else False
                parts.append(f"slope: {sl:.4g}  z: {zv:.2f}  trend: {int(it)}")
                # Donchian flags
                if show_don_panel and hh is not None and len(hh) > idx and len(ll) > idx:
                    ub = hh.iloc[idx]
                    lb = ll.iloc[idx]
                    if pd.notna(ub) and pd.notna(lb):
                        br_up = df["close"].iloc[idx] > ub * (1.0 + don_pad)
                        br_dn_now = df["close"].iloc[idx] < lb * (1.0 - don_exit_pad)
                        parts.append(f"Don ↑:{int(br_up)} ↓:{int(br_dn_now)}")
                cursor_text = "\n" + "  ".join(parts)
            except Exception:
                cursor_text = ""
            stats_text.set_text(base_text + cursor_text)
            fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # ---- X-axis formatting
    ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_price.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax_price.xaxis.get_major_locator()))
    if rows > 1:
        plt.setp(ax_price.get_xticklabels(), visible=False)
        if ax_exp is not None:   plt.setp(ax_exp.get_xticklabels(), visible=False)
        if ax_trend is not None: plt.setp(ax_trend.get_xticklabels(), visible=False)
        if ax_don is not None and ax_rsi is not None:
            plt.setp(ax_don.get_xticklabels(), visible=False)

    plt.title(title)
    if savefig is not None:
        plt.savefig(savefig, dpi=150)
    plt.show()
