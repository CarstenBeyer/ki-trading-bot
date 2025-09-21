#!/usr/bin/env python3

# in deiner bestehenden Datei (unten im __main__)
from strategies import sma_signal
from backtest import run_backtest
from bitget_loader import fetch_ohlcv
if __name__ == "__main__":
    # Daten holen
    df = fetch_ohlcv("ETH/USDT", "1d", 1000)  # nimm gern "1h" o. ä.

    # Strategie-Signal
    sig = sma_signal(df, fast=20, slow=50)

    # Backtest
    equity, rets, stats = run_backtest(df, sig, fee_pct=0.1, slippage_bps=5)

    print("\n=== Stats ===")
    print(stats.round(4))

    # (Optional) Equity plotten – schlicht mit matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve — SMA(20/50) ETH/USDT")
    plt.xlabel("Zeit (UTC)"); plt.ylabel("Equity (Start=1.0)")
    plt.tight_layout(); plt.show()