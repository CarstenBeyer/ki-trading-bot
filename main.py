#!/usr/bin/env python3

# in deiner bestehenden Datei (unten im __main__)
from strategies import sma_signal, donchian_breakout_signal, rsi_meanrev_signal, trend_breakout_signal, regime_adaptive_hybrid
from backtest import run_backtest
from bitget_loader import fetch_ohlcv
from plotting import plot_price_equity_dual_axis
from trades import build_trade_report

if __name__ == "__main__":
    # Daten holen
    symbol = "SOL/USDT"
    timeframe = "4h"
    limit = 1000

    df = fetch_ohlcv(symbol, timeframe, limit)
    print(df)

    # Strategie-Signal
    #sig = sma_signal(df, fast=20, slow=50)
    #sig = donchian_breakout_signal(df, entry_n=20, exit_n=10)
    #sig = rsi_meanrev_signal(df, n=14, buy_thr=30, exit_thr=50)
    #sig = trend_breakout_signal(df, entry_n=20, exit_n=10, atr_n=14, atr_mult=3.0)
    sig = regime_adaptive_hybrid(df,
        trend_win=200, trend_slope_thr=0.0, sma_z_thr=0.0,
        vol_win=30, max_ann_vol=1.0,         # 100% p.a. Obergrenze
        don_entry=20, don_exit=10,
        rsi_n=14, rsi_buy=30, rsi_exit=55, bb_win=20, bb_k=2.0,
        atr_n=14, atr_mult=3.0, time_exit=100,
        vol_target_ann=0.12, max_exposure=1.0
    )

    sig = sma_signal(df, fast=15, slow=40)

    # Backtest
    equity, rets, stats = run_backtest(df, sig, fee_pct=0.1, slippage_bps=5)

    # ... nachdem du equity, sig und df berechnet hast:
    report = build_trade_report(df, sig, equity, enter_level=0.20, exit_level=0.1)
    print("\n=== Trade Report (gekürzt) ===")
    print(report.round(4).tail(10))   # oder .tail(), oder komplett ausgeben


    print("\n=== Stats ===")
    print(stats.round(4))

    # Plotten
    plot_price_equity_dual_axis(
        df, equity, sig,
        stats=stats,  # <<<< hier Stats übergeben
        title=f"{symbol} {timeframe} — Stats"
    )    
        
    