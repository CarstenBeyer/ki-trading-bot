#!/usr/bin/env python3

# in deiner bestehenden Datei (unten im __main__)
from strategies import *
from backtest import run_backtest
from bitget_loader import fetch_ohlcv
from plotting import plot_price_equity_dual_axis
from trades import build_trade_report

if __name__ == "__main__":
    # Daten holen
    symbol = "ETH/USDT"
    timeframe = "6h"
    limit = 1000

    df = fetch_ohlcv(symbol, timeframe, limit)
    print(df)

    # Strategie-Signal
    #sig = sma_signal(df, fast=20, slow=50)
    #sig = donchian_breakout_signal(df, entry_n=20, exit_n=10)
    #sig = rsi_meanrev_signal(df, n=14, buy_thr=30, exit_thr=50)
    #sig = trend_breakout_signal(df, entry_n=20, exit_n=10, atr_n=14, atr_mult=3.0)

    strategy = RegimeAdaptiveHybrid(
        trend_win=100, slope_enter=0.0, slope_exit=-0.0005,
        z_enter=0.0, z_exit=-0.20,
        vol_win=30, max_ann_vol=0.8,
        
        don_entry=25, don_exit=10,
        
        atr_n=14, atr_mult=3.5,
        rsi_n=14, rsi_buy=40, rsi_exit=60,
        bb_win=20, time_exit=100,
        allow_shorts=False, binary_output=True, use_vol_targeting=False
    )


    sig = strategy.generate(df)
        
    #sig = sma_signal(df, fast=15, slow=40)

    # Backtest
    equity, rets, stats = run_backtest(df, sig, fee_pct=0.1, slippage_bps=5)

    # ... nachdem du equity, sig und df berechnet hast:
    report = build_trade_report(df, sig, equity, enter_level=0.20, exit_level=0.1)
    print("\n=== Trade Report (gekürzt) ===")
    print(report.round(4))   # oder .tail(), oder komplett ausgeben


    print("\n=== Stats ===")
    print(stats.round(4))

    # Plotten
    plot_price_equity_dual_axis(
        df, equity, sig,
        stats=stats,  # <<<< hier Stats übergeben
        title=f"{symbol} {timeframe} — Stats"
    )    
        
    