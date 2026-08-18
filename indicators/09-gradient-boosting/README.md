# Part 9 — Gradient Boosting: The Algorithm Behind XGBoost

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-gradient-boosting-ninjatrader-8/) |
| **Class** | `MlGradientBoost` |
| **Series** | Part 9 of 9 — [see all](../../README.md) |
| **License** | **All rights reserved** — read it, learn from it, run it on your own charts. Not for commercial use or redistribution. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

The previous post built a Random Forest — many decision trees, each grown independently on its own slice of the data, their votes averaged. Gradient boosting takes the opposite approach: it grows trees **in sequence**, each one correcting the errors the ensemble has made so far. It's the algorithm behind XGBoost.

[Read the full write-up →](https://mydailytake.com/ml-gradient-boosting-ninjatrader-8/)

## Install

1. Download **[`MlGradientBoost.cs`](MlGradientBoost.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlGradientBoost** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Boosting | **Boosting** | Number of trees grown in sequence and the learning rate that shrinks each tree's contribution to the ensemble. |
| Features | **Features** | Period of the moving average used in the distFromMa feature, and the smoothing window for the ATR regime ratio. |
| Learning | **Learning** | Number of recent look-ahead-safe (feature, label) examples kept in the rolling training window each rebuild draws from. |
| Signal | **Signal** | How far the predicted probability of an up move must be from 0.5 before a signal fires. |
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
← [Part 8](../08-random-forest)
