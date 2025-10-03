import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

# ---------- Helpers ----------

def bars_per_year_from_index(idx: pd.Index, fallback: int = 1460) -> int:
    """
    Versucht Bars/Jahr aus der Index-Frequenz zu schätzen.
    Fallback=1460 entspricht 6h-TF (365*4).
    """
    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None

    if freq is None:
        return fallback

    mapping = {
        "T": 525600, "5T": 105120, "15T": 35040, "30T": 17520,
        "H": 8760, "2H": 4380, "4H": 2190, "6H": 1460, "8H": 1095, "12H": 730,
        "D": 365, "2D": 182, "W-SUN": 52, "W-MON": 52
    }
    return mapping.get(freq, fallback)

def rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    alpha = 1 / n
    roll_up = up.ewm(alpha=alpha, adjust=False).mean()
    roll_dn = dn.ewm(alpha=alpha, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    out = 100 - (100 / (1 + rs))
    out.name = f"rsi_{n}"
    return out

def atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out = tr.ewm(alpha=1/n, adjust=False).mean()
    out.name = f"atr_{n}"
    return out

def donchian_bounds(close: pd.Series, n_high: int, n_low: int) -> tuple[pd.Series, pd.Series]:
    hh = close.shift(1).rolling(n_high, min_periods=n_high).max()
    ll = close.shift(1).rolling(n_low,  min_periods=n_low).min()
    return hh, ll

def ewma_vol(logret: pd.Series, lam: float = 0.94) -> pd.Series:
    # RiskMetrics-Style Var EWMA
    var = (logret**2).ewm(alpha=1-lam, adjust=False).mean()
    return np.sqrt(var)

# ---------- Einfache Signale ----------

def sma_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    assert 0 < fast < slow
    ma_fast = df["close"].rolling(fast, min_periods=fast).mean()
    ma_slow = df["close"].rolling(slow, min_periods=slow).mean()
    sig = ((ma_fast > ma_slow) & (~ma_fast.isna()) & (~ma_slow.isna())).astype(int)
    sig.name = "sma_long"
    return sig

def donchian_breakout_signal(df: pd.DataFrame, entry_n: int = 20, exit_n: int = 10) -> pd.Series:
    assert entry_n > 1 and 0 < exit_n <= entry_n
    close = df["close"]
    hh, ll = donchian_bounds(close, entry_n, exit_n)
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

def rsi_meanrev_signal(df: pd.DataFrame, n: int = 14, buy_thr: float = 30, exit_thr: float = 55) -> pd.Series:
    assert 0 < buy_thr < exit_thr < 100
    r = rsi_ewm(df["close"], n=n)
    buy  = (r < buy_thr) & (~r.isna())
    sell = (r > exit_thr) & (~r.isna())
    out = []
    in_pos = 0
    for b, s in zip(buy, sell):
        if not in_pos and b: in_pos = 1
        elif in_pos and s:   in_pos = 0
        out.append(in_pos)
    sig = pd.Series(out, index=df.index, name="rsi_meanrev_long").astype(int)
    return sig

# ---------- Regime-adaptiv (verbessert) ----------

@dataclass
class RegimeAdaptiveHybrid:
    """
    Regime-adaptive Trend/Range-Strategie (long-only, binär 0/1).

    Regime-Erkennung (Trend vs. Range):
      - Slope (normiert) & Z-Score von Close vs. SMA(trend_win)
      - Volatilitäts-Gate (annualisierte realisierte Vol)

    Trend-Block (nur wenn is_trend=True):
      - Donchian-Breakout (upper-Band) mit optionalem Breakout-Puffer
      - Exit: max(Donchian-Low [optional gepaddet], Entry-ATR, Chandelier-ATR)

    Range-Block (nur wenn is_trend=False):
      - Mean-Reversion mit RSI + Bollinger-ZScore (Entry z<=-1, Exit z>=0)
      - Optional: Guards, damit BB+RSI nicht gegen „zu schlechten“ Trend kauft
    """

    # -------- Regime-Parameter --------
    trend_win: int = 200
    slope_enter: float = 0.0
    slope_exit: float  = -0.0005
    z_enter: float = 0.0
    z_exit: float  = -0.25
    vol_win: int = 30
    max_ann_vol: float = 1.0

    # -------- Trend-Block (Donchian) --------
    don_entry: int = 20
    don_exit: int  = 10
    don_src: Literal["hl", "close"] = "hl"   # "hl" = High/Low (klassisch), "close" = Close-Kanal
    don_break_pad: float = 0.0               # Breakout-Puffer (z.B. 0.0005 = 5 Bps über Upper)
    don_exit_pad: float = 0.0                # Exit-Puffer: ll*(1+pad) -> strengerer Exit

    atr_n: int = 14
    atr_mult: float = 3.0

    # -------- Range-Block (RSI + Bollinger) --------
    rsi_n: int = 14
    rsi_buy: float = 30.0
    rsi_exit: float = 55.0
    bb_win: int = 20
    bb_k: float = 2.0
    time_exit: int = 100

    # -------- Zusätzliche Guards für Range-Einstieg --------
    range_guard_slope_min: float | None = None      # z.B. 0.0 => nur wenn slope >= 0
    range_guard_z_min: float | None = None          # z.B. -0.2 => nur wenn z >= -0.2
    range_guard_max_ann_vol: float | None = None    # z.B. 1.2 => blocke bei zu hoher Vol

    # ---- feste Entscheidungen ----
    allow_shorts: bool = False
    binary_output: bool = True
    use_vol_targeting: bool = False  # (absichtlich inaktiv in dieser Variante)

    # ========= öffentliche API =========
    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)
        close = df["close"]

        # ----- Regime-Features -----
        sma_tr = close.rolling(self.trend_win, min_periods=self.trend_win).mean()
        std_tr = close.rolling(self.trend_win, min_periods=self.trend_win).std()
        z_tr = (close - sma_tr) / (std_tr + 1e-12)
        slope = self._slope_norm(close, win=self.trend_win)
        ann_vol = self._realized_vol(close, win=self.vol_win)

        # Trend-Regime via Hysterese
        is_trend_enter = (slope > self.slope_enter) & (z_tr > self.z_enter) & (ann_vol <= self.max_ann_vol)
        is_trend_exit  = (slope < self.slope_exit) |  (z_tr < self.z_exit)  | (ann_vol >  self.max_ann_vol)
        is_trend = self._hysteresis_bool(is_trend_enter, is_trend_exit, index=df.index)
        is_range = ~is_trend

        # Range-Guards (optional)
        range_guard = pd.Series(True, index=df.index)
        if self.range_guard_slope_min is not None:
            range_guard &= (slope >= self.range_guard_slope_min)
        if self.range_guard_z_min is not None:
            range_guard &= (z_tr  >= self.range_guard_z_min)
        if self.range_guard_max_ann_vol is not None:
            range_guard &= (ann_vol <= self.range_guard_max_ann_vol)

        # ----- Donchian-Bänder (für Trend-Block & Warmup) -----
        if self.don_src == "hl":
            hh = df["high"].shift(1).rolling(self.don_entry, min_periods=self.don_entry).max()
            ll = df["low"] .shift(1).rolling(self.don_exit,  min_periods=self.don_exit ).min()
        else:  # "close"
            hh = close.shift(1).rolling(self.don_entry, min_periods=self.don_entry).max()
            ll = close.shift(1).rolling(self.don_exit,  min_periods=self.don_exit ).min()

        # ----- ATR (Wilder) -----
        atr = self._atr_wilder(df, self.atr_n)

        # ----- Trend-Block (Breakout + Stops) -----
        pos_trend = np.zeros(len(df), dtype=int)
        in_pos = 0
        entry_price = None
        highest = None

        for i in range(len(df)):
            c = close.iloc[i]
            if not is_trend.iloc[i]:
                in_pos = 0; entry_price = None; highest = None
            else:
                if in_pos == 0:
                    ubreak = (hh.iloc[i] * (1.0 + self.don_break_pad)) if not np.isnan(hh.iloc[i]) else np.nan
                    if (not np.isnan(ubreak)) and (c > ubreak):
                        in_pos = 1; entry_price = c; highest = c
                else:
                    highest = c if highest is None else max(highest, c)
                    # Donchian-Exit (optional gepaddet) + ATR-basierte Stops
                    don_exit_lvl = (ll.iloc[i] * (1.0 + self.don_exit_pad)) if not np.isnan(ll.iloc[i]) else -np.inf
                    stop = max(
                        don_exit_lvl,
                        entry_price - self.atr_mult * atr.iloc[i],
                        highest     - self.atr_mult * atr.iloc[i]
                    )
                    if c < stop:
                        in_pos = 0; entry_price = None; highest = None
            pos_trend[i] = in_pos

        pos_trend = pd.Series(pos_trend, index=df.index, dtype=int)
        pos_trend[~is_trend] = 0  # außerhalb Trend-Regime flat

        # ----- Range-Block (RSI + Bollinger, long-only) -----
        rsi = self._rsi_ewm(close, n=self.rsi_n)
        bb_mid = close.rolling(self.bb_win, min_periods=self.bb_win).mean()
        bb_std = close.rolling(self.bb_win, min_periods=self.bb_win).std()
        z_bb = (close - bb_mid) / (bb_std + 1e-12)

        # Entry/Exit-Regeln (wie zuvor): z<=-1 & RSI<buy, Exit bei z>=0 oder RSI>exit, plus Time-Exit
        enter = is_range & range_guard & (z_bb <= -1.0) & (rsi < self.rsi_buy) & (~z_bb.isna()) & (~rsi.isna())
        exit_  = (z_bb >= 0.0) | (rsi > self.rsi_exit)

        pos_range = np.zeros(len(df), dtype=int)
        in_pos = 0; bars_in = 0
        for i in range(len(df)):
            if not is_range.iloc[i]:
                in_pos = 0; bars_in = 0
            else:
                if in_pos == 0 and enter.iloc[i]:
                    in_pos = 1; bars_in = 0
                elif in_pos == 1 and (exit_.iloc[i] or bars_in >= self.time_exit):
                    in_pos = 0; bars_in = 0
                bars_in = bars_in + 1 if in_pos else 0
            pos_range[i] = in_pos
        pos_range = pd.Series(pos_range, index=df.index, dtype=int)

        # ----- Kombination (ohne Doppelhebelung) -----
        raw_pos = pd.concat([pos_trend, pos_range], axis=1).max(axis=1).fillna(0).astype(int)

        # ----- Warmup: sobald Kern-Features fehlen → flat -----
        warmup = (sma_tr.isna() | bb_mid.isna() | atr.isna() | hh.isna() | ll.isna())

        # ----- finaler Output (binär 0/1, long-only) -----
        expo = (raw_pos > 0).astype(int).astype(float)
        expo[warmup] = 0.0
        expo.name = "regime_adaptive_hybrid"
        return expo

    # ========= Helper =========
    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame fehlt Spalten: {missing}")
        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError("Index muss DatetimeIndex/PeriodIndex sein (UTC empfohlen).")

    @staticmethod
    def _atr_wilder(df: pd.DataFrame, n: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        out = tr.ewm(alpha=1/n, adjust=False).mean()
        out.name = f"atr_{n}"
        return out

    @staticmethod
    def _rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
        d = close.diff()
        up = d.clip(lower=0)
        dn = (-d).clip(lower=0)
        alpha = 1 / n
        roll_up = up.ewm(alpha=alpha, adjust=False).mean()
        roll_dn = dn.ewm(alpha=alpha, adjust=False).mean()
        rs = roll_up / (roll_dn + 1e-12)
        out = 100 - (100 / (1 + rs))
        out.name = f"rsi_{n}"
        return out

    @staticmethod
    def _realized_vol(close: pd.Series, win: int = 30) -> pd.Series:
        ret = np.log(close).diff()
        bpy = RegimeAdaptiveHybrid._bars_per_year_from_index(close.index)
        return ret.rolling(win).std() * np.sqrt(bpy)

    @staticmethod
    def _bars_per_year_from_index(idx: pd.Index, fallback: int = 1460) -> int:
        try:
            freq = pd.infer_freq(idx)
        except Exception:
            freq = None
        if freq is None:
            return fallback
        mapping = {
            "T": 525600, "5T": 105120, "15T": 35040, "30T": 17520,
            "H": 8760, "2H": 4380, "4H": 2190, "6H": 1460, "8H": 1095, "12H": 730,
            "D": 365, "2D": 182, "W-SUN": 52, "W-MON": 52
        }
        return mapping.get(freq, fallback)

    @staticmethod
    def _slope_norm(series: pd.Series, win: int = 200) -> pd.Series:
        x = np.arange(len(series))
        out = np.full(len(series), np.nan)
        for i in range(win-1, len(series)):
            xs = x[i-win+1:i+1]
            ys = series.iloc[i-win+1:i+1].values
            xs_c = xs - xs.mean()
            denom = (xs_c**2).sum()
            out[i] = 0.0 if denom == 0 else float((xs_c * (ys - ys.mean())).sum() / denom) / (series.iloc[i] + 1e-12)
        return pd.Series(out, index=series.index)

    @staticmethod
    def _hysteresis_bool(enter_cond: pd.Series, exit_cond: pd.Series, index: pd.Index) -> pd.Series:
        """Schaltet zwischen False↔True mit getrennten Enter/Exit-Bedingungen (Hysterese)."""
        assert len(enter_cond) == len(exit_cond)
        out = np.zeros(len(enter_cond), dtype=bool)
        state = False
        for i in range(len(enter_cond)):
            if not state and bool(enter_cond.iloc[i]):
                state = True
            elif state and bool(exit_cond.iloc[i]):
                state = False
            out[i] = state
        return pd.Series(out, index=index)
