# Part 8 — Random Forest: An Ensemble of Decision Trees

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-random-forest-ninjatrader-8/) |
| **Class** | `MlRandomForest` |
| **Series** | Part 8 of 8 — [see all](../../README.md) |
| **License** | **All rights reserved** — read it, learn from it, run it on your own charts. Not for commercial use or redistribution. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

Posts 2 through 7 of this series were one long thread. A single neuron became hidden layers, hidden layers became configurable depth, depth got a better optimizer, and then the network grew a memory and finally gated memory.

[Read the full write-up →](https://mydailytake.com/ml-random-forest-ninjatrader-8/)

## Install

1. Download **[`MlRandomForest.cs`](MlRandomForest.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlRandomForest** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Architecture | **Architecture** | Number of decision trees in the forest. Each tree is grown on its own bootstrap sample, and the forest prediction averages all of their votes. More trees = a smoother, lower-variance prediction but proportionally more compute on each retrain. Default 50. |
| Learning | **Learning** | Number of recent look-ahead-safe (feature, label) examples kept for training. Each forest rebuild draws its bootstrap samples from this rolling window, so older examples fall out over time. Default 300. |
| Features | **Features** | Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio. |
| Signal | **Signal** | How far the forest's predicted probability of an up move must be from 0.5 before a signal fires. |
| Display | **Display** | Vertical offset of the signal triangle from the bar's high (shorts) / low (longs), in ticks. |

## Found a bug?

Probably mine. ML on live bars is fiddly — training windows, feature scaling, and warm-up all shift
results, and a model that looks great historically can behave differently forward. If it errors,
repaints, or the math plainly disagrees with the write-up, [open an issue](../../issues) with your
NT8 version, instrument, bar type, and data provider. More in the [main README](../../README.md#found-a-bug).

## Licensing

© 2026 MyDailyTake.com. **All rights reserved.** Published openly so you can read it, learn from it,
and run it on your own charts — not for commercial use, resale, or redistribution. The notice at the
top of the `.cs` governs. See [LICENSE](../../LICENSE). Want to do more? Ask: jack@mydailytake.com

## Disclaimer

Educational and informational only. Nothing here is trading advice, and a machine-learning indicator
is not a trading edge. Futures trading involves substantial risk of loss and is not suitable for
every investor. See the [full disclosure](https://mydailytake.com/disclosure/).

---
← [Part 7](../07-lstm)
