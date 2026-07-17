# Part 2 — Online Logistic Regression: Your First Neural Net

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-online-logistic-regression-your-first-neural-net-for-ninjatrader-8/) |
| **Class** | `MlOnlineLogisticRegression` |
| **Series** | Part 2 of 8 — [see all](../../README.md) |
| **License** | **PolyForm Noncommercial 1.0.0** — free for personal use, not commercial. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

The k-Nearest Neighbors post introduced one half of supervised learning: memorizing models that hold onto every historical bar and search through them at prediction time. This post is the other half.

[Read the full write-up →](https://mydailytake.com/ml-online-logistic-regression-your-first-neural-net-for-ninjatrader-8/)

## Install

1. Download **[`MlOnlineLogisticRegression.cs`](MlOnlineLogisticRegression.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlOnlineLogisticRegression** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Learning | **Learning** | Step size for each weight update. Larger = adapts faster but jitters; smaller = smoother but slower to react. 0.01 is a sensible default for normalized features. |
| Features | **Features** | Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio. Smaller = more reactive; larger = trend-anchored. |
| Signal | **Signal** | How far the predicted probability of an up move must be from 0.5 before a signal fires. 0.10 means: long fires when P(up) > 0.60, short fires when P(up) < 0.40. Larger values produce fewer, higher-conviction signals. |
| Display | **Display** | Vertical offset of the signal triangle from the bar's high (shorts) / low (longs), in ticks. |

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
← [Part 1](../01-k-nearest-neighbors) · [Part 3](../03-single-hidden-layer) →
