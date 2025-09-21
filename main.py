#!/usr/bin/env python3

# in deiner bestehenden Datei (unten im __main__)
from strategies import sma_signal
from backtest import run_backtest
from bitget_loader import fetch_ohlcv
from plotting import plot_price_equity_dual_axis
from trades import build_trade_report

if __name__ == "__main__":
    # Daten holen
    df = fetch_ohlcv("ETH/USDT", "6h", 1000)  # nimm gern "1h" o. ä.
    print(df)

    # Strategie-Signal
    sig = sma_signal(df, fast=20, slow=50)

    # Backtest
    equity, rets, stats = run_backtest(df, sig, fee_pct=0.1, slippage_bps=5)

    # ... nachdem du equity, sig und df berechnet hast:
    report = build_trade_report(df, sig, equity)
    print("\n=== Trade Report (gekürzt) ===")
    print(report.round(4).head(10))   # oder .tail(), oder komplett ausgeben


    print("\n=== Stats ===")
    print(stats.round(4))

    # Plotten
    plot_price_equity_dual_axis(
        df, equity, sig,
        stats=stats,  # <<<< hier Stats übergeben
        title="ETH/USDT 6h — SMA(20/50) mit Stats"
    )    
        
    