# Part 5 — Adam Optimizer + Mini-Batch: Stable Training

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-adam-optimizer-mini-batch-stable-training-for-ninjatrader-8/) |
| **Class** | `MlNeuralNetAdam` |
| **Series** | Part 5 of 8 — [see all](../../README.md) |
| **License** | **PolyForm Noncommercial 1.0.0** — free for personal use, not commercial. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

The prior post let you stack any number of hidden layers — but every weight update was still plain stochastic gradient descent: one update per bar, one fixed step size for every weight.

[Read the full write-up →](https://mydailytake.com/ml-adam-optimizer-mini-batch-stable-training-for-ninjatrader-8/)

## Install

1. Download **[`MlNeuralNetAdam.cs`](MlNeuralNetAdam.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlNeuralNetAdam** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Architecture | **Architecture** | Per-layer neuron counts as a list of integers. Any common separator works — comma, space, hyphen, x, semicolon. Examples: '6' = one layer of 6 neurons. '8, 6' = two layers (8 then 6). '10, 6, 4' = three layers. Each value must be ≥ 2. No upper cap; with Adam, slightly deeper networks train more reliably than under plain SGD, but the practical lift on a 3-feature problem still plateaus around 2 layers. Invalid input falls back to default '8, 6' and logs a warning to the NT Output window. |
| Learning | **Learning** | Adam's base learning rate. Adam is much less sensitive to this than plain SGD because of the adaptive per-parameter scaling, but the value still sets the overall step magnitude. 0.001 is the canonical Adam default; for online financial data, 0.001-0.005 works well. |
| Optimizer | **Optimizer** | Number of training-eligible bars to accumulate gradients over before applying an Adam update. 1 = pure online (one update per bar, like the prior posts). 5-20 = small mini-batches that smooth single-bar noise. Larger values trade adaptation speed for stability — recommended 5-10 for online financial data. |
| Features | **Features** | Period of the moving average used in the distFromMa feature. |
| Signal | **Signal** | How far the predicted probability of an up move must be from 0.5 before a signal fires. |
| Display | **Display** | Vertical offset of the signal triangle from the bar's high/low, in ticks. |

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
← [Part 4](../04-multi-hidden-layer) · [Part 6](../06-recurrent-neural-network) →
