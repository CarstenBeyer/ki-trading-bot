# strategies.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

# ---------- Helpers ----------

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

def sma_signal(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    assert 0 < fast < slow
    ma_fast = df["close"].rolling(fast, min_periods=fast).mean()
    ma_slow = df["close"].rolling(slow, min_periods=slow).mean()
    sig = ((ma_fast > ma_slow) & (~ma_fast.isna()) & (~ma_slow.isna())).astype(int)
    sig.name = "sma_long"
    return sig

# ---------- Simplified Regime Strategy ----------

@dataclass
class RegimeAdaptiveHybrid:
    """
    Lightweight regime-adaptive strategy (long-only, 0/1):

    Regime (Trend vs Range):
      - Hysteresis over slope_norm and z-score of close vs SMA(trend_win)
        enter  when slope > slope_enter  AND z > z_enter
        exit   when slope < slope_exit   OR  z < z_exit

    Trend block (when is_trend=True):
      - Position = 1 (no extra breakouts/stops).

    Range block (when is_trend=False):
      - Mean reversion using RSI + Bollinger z-score:
          enter if (RSI < rsi_buy) AND (z_bb <= bb_enter_z)
          exit  if (RSI > rsi_exit) OR  (z_bb >= bb_exit_z) OR time_exit
    """

    # Regime params
    trend_win: int = 200
    slope_enter: float = 0.0
    slope_exit: float  = -0.0005
    z_enter: float = 0.0
    z_exit: float  = -0.25

    # Range (RSI + Bollinger)
    rsi_n: int = 14
    rsi_buy: float = 30.0
    rsi_exit: float = 55.0
    bb_win: int = 20
    bb_k: float = 2.0
    bb_enter_z: float = -1.0
    bb_exit_z: float  = 0.0
    time_exit: int = 100

    # Fixed for this variant
    allow_shorts: bool = False
    binary_output: bool = True

    # ========= main API =========
    def generate(self, df: pd.DataFrame) -> pd.Series:
        self._validate_df(df)
        close = df["close"]

        # --- Regime features
        sma_tr = close.rolling(self.trend_win, min_periods=self.trend_win).mean()
        std_tr = close.rolling(self.trend_win, min_periods=self.trend_win).std()
        z_tr   = (close - sma_tr) / (std_tr + 1e-12)
        slope  = self._slope_norm(close, win=self.trend_win)

        # Hysteresis
        is_trend_enter = (slope > self.slope_enter) & (z_tr > self.z_enter)
        is_trend_exit  = (slope < self.slope_exit) | (z_tr < self.z_exit)
        is_trend = self._hysteresis_bool(is_trend_enter, is_trend_exit, index=df.index)
        is_range = ~is_trend

        # --- Trend block: just follow regime
        pos_trend = is_trend.astype(int)

        # --- Range block: RSI + Bollinger + time exit
        rsi   = self._rsi_ewm(close, n=self.rsi_n)
        bb_mid = close.rolling(self.bb_win, min_periods=self.bb_win).mean()
        bb_std = close.rolling(self.bb_win, min_periods=self.bb_win).std()
        z_bb = (close - bb_mid) / (bb_std + 1e-12)

        enter = is_range & (rsi < self.rsi_buy) & (z_bb <= self.bb_enter_z)
        exit_ = (rsi > self.rsi_exit) | (z_bb >= self.bb_exit_z)

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

        # --- Combine (no double leverage)
        raw_pos = pd.concat([pos_trend, pos_range], axis=1).max(axis=1).fillna(0).astype(int)

        # --- Warmup: need SMA/STD and BB
        warmup = (sma_tr.isna() | bb_mid.isna())
        expo = (raw_pos > 0).astype(float)
        expo[warmup] = 0.0
        expo.name = "regime_adaptive_hybrid_lite"
        return expo

    # ========= helpers =========
    @staticmethod
    def _validate_df(df: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame fehlt Spalten: {missing}")
        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError("Index muss DatetimeIndex/PeriodIndex sein (UTC empfohlen).")

    @staticmethod
    def _rsi_ewm(close: pd.Series, n: int = 14) -> pd.Series:
        return rsi_ewm(close, n=n)

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
