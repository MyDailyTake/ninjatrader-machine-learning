# Part 3 — Single-Hidden-Layer Neural Network: Your First Multi-Layer Net

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-single-hidden-layer-neural-network-your-first-multi-layer-net-for-ninjatrader-8/) |
| **Class** | `MlNeuralNetSingleHidden` |
| **Series** | Part 3 of 8 — [see all](../../README.md) |
| **License** | **PolyForm Noncommercial 1.0.0** — free for personal use, not commercial. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

The prior post built a single-neuron model — three weights, a bias, and a sigmoid that turned a weighted sum into P(up). It worked, but its limits were architectural: a single neuron can only carve the feature space with a straight line.

[Read the full write-up →](https://mydailytake.com/ml-single-hidden-layer-neural-network-your-first-multi-layer-net-for-ninjatrader-8/)

## Install

1. Download **[`MlNeuralNetSingleHidden.cs`](MlNeuralNetSingleHidden.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlNeuralNetSingleHidden** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Architecture | **Architecture** | Number of neurons in the hidden layer. 6 (= 2 × input features) is the conventional starting point for shallow networks. Bump up for richer feature interactions; bump down if signals look noisy on small datasets. |
| Learning | **Learning** | Step size for each weight update. Larger = adapts faster but jitters; smaller = smoother but slower to react. With more parameters than the single-neuron sibling, slightly smaller values (0.005-0.01) often produce more stable training. |
| Features | **Features** | Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio. |
| Signal | **Signal** | How far the predicted probability of an up move must be from 0.5 before a signal fires. 0.10 means: long fires when P(up) > 0.60, short fires when P(up) < 0.40. |
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
← [Part 2](../02-online-logistic-regression) · [Part 4](../04-multi-hidden-layer) →
