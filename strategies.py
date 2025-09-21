import pandas as pd
import numpy as np

# --- SMA ---
def sma_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    assert fast > 0 and slow > 0 and fast < slow, "SMA: require 0 < fast < slow"
    ma_fast = df["close"].rolling(fast, min_periods=fast).mean()
    ma_slow = df["close"].rolling(slow, min_periods=slow).mean()
    sig = (ma_fast > ma_slow).astype(int)
    # Warmup -> flat
    sig[(ma_fast.isna()) | (ma_slow.isna())] = 0
    sig.name = "sma_long"
    return sig

# --- Donchian ---
def donchian_breakout_signal(df: pd.DataFrame, entry_n: int = 20, exit_n: int = 10) -> pd.Series:
    assert entry_n > 1 and exit_n > 0 and exit_n <= entry_n, "Donchian: need exit_n <= entry_n"
    close = df["close"]
    # konservativ: Referenzfenster ohne aktuellen Bar
    hh = close.shift(1).rolling(entry_n, min_periods=entry_n).max()
    ll = close.shift(1).rolling(exit_n,  min_periods=exit_n).min()

    up_break   = (close > hh) & (~hh.isna())
    down_break = (close < ll) & (~ll.isna())

    pos = np.zeros(len(df), dtype=int)
    in_pos = 0
    for i in range(len(df)):
        if in_pos == 0 and up_break.iloc[i]:
            in_pos = 1
        elif in_pos == 1 and down_break.iloc[i]:
            in_pos = 0
        pos[i] = in_pos
    sig = pd.Series(pos, index=df.index, name="donchian_long").astype(int)
    return sig

# --- RSI & Mean-Reversion ---
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"rsi_{n}"
    return rsi

def rsi_meanrev_signal(df: pd.DataFrame, n: int = 14, buy_thr: float = 30, exit_thr: float = 55) -> pd.Series:
    assert 0 < buy_thr < exit_thr < 100, "RSI: need 0 < buy_thr < exit_thr < 100"
    r = _rsi(df["close"], n=n)
    buy  = (r < buy_thr) & (~r.isna())
    sell = (r > exit_thr) & (~r.isna())

    pos = []
    in_pos = 0
    for b, s in zip(buy, sell):
        if in_pos == 0 and b:
            in_pos = 1
        elif in_pos == 1 and s:
            in_pos = 0
        pos.append(in_pos)
    sig = pd.Series(pos, index=df.index, name="rsi_meanrev_long").astype(int)
    return sig

# --- ATR & Trend-Breakout ---
def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    atr.name = f"atr_{n}"
    return atr

def trend_breakout_signal(
    df: pd.DataFrame,
    entry_n: int = 20,
    exit_n: int = 10,
    atr_n: int = 14,
    atr_mult: float = 3.0,
) -> pd.Series:
    assert entry_n > 1 and exit_n > 0 and exit_n <= entry_n, "Trend: exit_n <= entry_n"
    assert atr_n > 0 and atr_mult > 0, "Trend: positive ATR params"
    close = df["close"]
    sma200 = close.rolling(200, min_periods=200).mean()
    atr = _atr(df, atr_n)

    don_high = close.shift(1).rolling(entry_n, min_periods=entry_n).max()
    don_low  = close.shift(1).rolling(exit_n,  min_periods=exit_n).min()

    pos = []
    in_pos = False
    entry_price = None
    # optional: Chandelier-Variante – höchster Close seit Entry
    highest_since_entry = None

    for i in range(len(df)):
        c = close.iloc[i]
        if not in_pos:
            if (c > don_high.iloc[i]) and (c > sma200.iloc[i]) and (not np.isnan(atr.iloc[i])):
                # einfacher Hitzefilter: Breakout nicht mehr als 2*ATR über DonHigh
                if (c - don_high.iloc[i]) <= 2 * atr.iloc[i]:
                    in_pos = True
                    entry_price = c
                    highest_since_entry = c
        else:
            highest_since_entry = max(highest_since_entry, c) if highest_since_entry is not None else c
            chandelier_stop = highest_since_entry - atr_mult * atr.iloc[i]  # trailing am Hoch
            rule_stop = max(don_low.iloc[i], entry_price - atr_mult * atr.iloc[i])
            stop = max(chandelier_stop, rule_stop)
            if c < stop:
                in_pos = False
                entry_price = None
                highest_since_entry = None
        pos.append(1 if in_pos else 0)

    sig = pd.Series(pos, index=df.index, name="trend_breakout_long").astype(int)
    # Warmup -> flat
    warmup = (sma200.isna()) | (don_high.isna()) | (don_low.isna()) | (atr.isna())
    sig[warmup] = 0
    return sig

# ---- kleine Helfer ----
def _realized_vol(close: pd.Series, win: int = 30) -> pd.Series:
    ret = np.log(close).diff()
    # Std der Logreturns * sqrt(Anz. Bars/Jahr). Für 6h: 365*4 = 1460 Bars p.a.
    ann_factor = 365*24 // 6 if hasattr(close.index, "inferred_type") else 1460
    return ret.rolling(win).std() * np.sqrt(ann_factor)

def _slope_norm(series: pd.Series, win: int = 200) -> pd.Series:
    # lineare Steigung über Fenster, normiert mit Preisniveau (robust)
    x = np.arange(len(series))
    def sl(xw, yw):
        # einfache OLS-Steigung
        xw = xw - xw.mean()
        denom = (xw**2).sum()
        if denom == 0: return 0.0
        return float((xw * (yw - yw.mean())).sum() / denom)
    out = np.full(len(series), np.nan)
    for i in range(win-1, len(series)):
        xs = x[i-win+1:i+1]
        ys = series.iloc[i-win+1:i+1].values
        out[i] = sl(xs, ys) / (series.iloc[i] + 1e-9)  # normiert
    return pd.Series(out, index=series.index)

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100 / (1 + rs))

def regime_adaptive_hybrid(
    df: pd.DataFrame,
    # Regime
    trend_win: int = 200, trend_slope_thr: float = 0.0, sma_z_thr: float = 0.0,
    vol_win: int = 30, max_ann_vol: float = 1.0,          # 100% p.a. als „zu hoch“
    # Trend-Block
    don_entry: int = 20, don_exit: int = 10,
    # Range-Block
    rsi_n: int = 14, rsi_buy: float = 30, rsi_exit: float = 55, bb_win: int = 20, bb_k: float = 2.0,
    # Risk
    atr_n: int = 14, atr_mult: float = 3.0, time_exit: int = 100,
    # Volatility Targeting
    vol_target_ann: float = 0.12, max_exposure: float = 1.0
) -> pd.Series:
    """
    Liefert eine kontinuierliche Exposure-Serie (0..1), die dein Backtest direkt nutzen kann.
    Handeln am nächsten Bar bitte weiterhin im Backtest (.shift(1)).

    Regime:
      TREND, wenn SMA200-Slope > trend_slope_thr und zScore(Close vs SMA200) > sma_z_thr
      und realisierte AnnVol <= max_ann_vol; sonst RANGE.
    Trend-Block: Donchian-Breakout; Range-Block: RSI Mean-Reversion + Bollinger.
    Risk: ATR-Stop, Chandelier-Trail, Zeit-Exit. Vol-Targeting skaliert Exposure.
    """
    close = df["close"]
    sma200 = close.rolling(trend_win, min_periods=trend_win).mean()
    sma_z = (close - sma200) / (close.rolling(trend_win, min_periods=trend_win).std() + 1e-12)
    slope = _slope_norm(close, win=trend_win)
    ann_vol = _realized_vol(close, win=vol_win)

    # Regime-Maske
    is_trend = (slope > trend_slope_thr) & (sma_z > sma_z_thr) & (ann_vol <= max_ann_vol)
    is_range = ~is_trend

    # Trend-Block: Donchian auf VOR-Bar
    don_hi = close.shift(1).rolling(don_entry, min_periods=don_entry).max()
    don_lo = close.shift(1).rolling(don_exit,  min_periods=don_exit).min()
    mom_strength = (close - don_hi).clip(lower=0) / (close * 1e-12 + (close - don_hi).abs() + 1e-12)  # 0..~1
    trend_sig_raw = (close > don_hi).astype(float) * mom_strength
    trend_sig_raw[~is_trend] = 0.0

    # Range-Block: RSI & Bollinger-z
    rsi = _rsi(close, n=rsi_n)
    bb_mid = close.rolling(bb_win, min_periods=bb_win).mean()
    bb_std = close.rolling(bb_win, min_periods=bb_win).std()
    z = (close - bb_mid) / (bb_std + 1e-12)
    # Einstieg bei Überverkauft (RSI<buy) & unterem Band, Exit bei Normalisierung (RSI>exit oder z>0)
    range_enter = (rsi < rsi_buy) & (z < -1.0)
    range_exit  = (rsi > rsi_exit) | (z > 0)
    # Stateful long/flat im Range-Regime
    pos_range = np.zeros(len(df), dtype=int)
    in_pos = 0; bars_in = 0
    for i in range(len(df)):
        if not is_range.iloc[i]:
            in_pos = 0; bars_in = 0
        else:
            if in_pos == 0 and range_enter.iloc[i]:
                in_pos = 1; bars_in = 0
            elif in_pos == 1 and (range_exit.iloc[i] or bars_in >= time_exit):
                in_pos = 0; bars_in = 0
            bars_in = bars_in + 1 if in_pos else 0
        pos_range[i] = in_pos
    range_sig_raw = pd.Series(pos_range, index=df.index, dtype=float)

    # Kombiniere Signale (max statt sum → keine doppelte Überhebelung)
    raw_sig = pd.concat([trend_sig_raw, range_sig_raw], axis=1).max(axis=1)
    raw_sig = raw_sig.clip(0, 1).fillna(0.0)

    # Volatility Targeting: skaliere Exposure auf Zielvol
    # per-bar Vol ~ ann_vol / sqrt(bars_per_year); wir invertieren als Skalierungsfaktor
    bars_per_year = 365*24 // 6  # 6h-Default; passe an deinen TF an
    per_bar_vol = ann_vol / np.sqrt(bars_per_year)
    vol_scale = (vol_target_ann / (ann_vol + 1e-12)).clip(upper=1.5)  # dämpfen
    sig_vt = (raw_sig * vol_scale).clip(0, max_exposure)

    # Warmup immer flat
    warmup = sma200.isna()
    sig_vt[warmup] = 0.0
    sig_vt.name = "regime_adaptive_hybrid"
    return sig_vt
