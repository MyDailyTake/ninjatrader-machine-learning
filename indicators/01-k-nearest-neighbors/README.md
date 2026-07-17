# Part 1 — k-Nearest Neighbors Indicator

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-k-nearest-neighbors-indicator-for-ninjatrader-8/) |
| **Class** | `MlKNearestNeighbors` |
| **Series** | Part 1 of 8 — [see all](../../README.md) |
| **License** | **PolyForm Noncommercial 1.0.0** — free for personal use, not commercial. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

Search TradingView's community scripts for terms like "machine learning" or "AI" and you'll find a long list of community-built indicators that lean on one of the simplest and oldest algorithms in ML.

[Read the full write-up →](https://mydailytake.com/ml-k-nearest-neighbors-indicator-for-ninjatrader-8/)

## Install

1. Download **[`MlKNearestNeighbors.cs`](MlKNearestNeighbors.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlKNearestNeighbors** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Search | **Search** | Number of nearest historical bars to consult per prediction. k=1 is one expert (noisy); k=50 polls a broad crowd (smoothed). Sweet spot for futures: 10–30 with a 2000-bar lookback. |
| Features | **Features** | Lookback for the trend moving average. Used in 'distance from MA' (directional position) and as the SMA window for the volatility-regime feature. Default 50. |
| Prediction | **Prediction** | Magnitude gate: signals only fire when |prediction| meets this threshold (in returns, where 0.0005 = 5 basis points). 0 = no magnitude filter. Combined with Min Signal-to-Noise (both must pass). |
| Display | **Display** | Vertical offset of the signal triangles from the bar's high/low, in ticks. Triangle sits this far from the bar. |

## Found a bug?

Probably mine. ML on live bars is fiddly — training windows, feature scaling, and warm-up all shift
results, and a model that looks great historically can behave differently forward. If it errors,
repaints, or the math plainly disagrees with the write-up, [open an issue](../../issues) with your
NT8 version, instrument, bar type, and data provider. More in the [main README](../../README.md#found-a-bug).

## Licensing

© 2026 MyDailyTake.com. **Free for personal use — not for commercial use.** Licensed under
[PolyForm Noncommercial 1.0.0](../../LICENSE). Learn from it, run it on your own charts, tear it
apart. Don't repackage it into a product you sell.

## Disclaimer

Educational and informational only. Nothing here is trading advice, and a machine-learning indicator
is not a trading edge. Futures trading involves substantial risk of loss and is not suitable for
every investor. See the [full disclosure](https://mydailytake.com/disclosure/).

---
[Part 2](../02-online-logistic-regression) →
