# Part 6 — Recurrent Neural Network: Bar-to-Bar Memory

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-recurrent-neural-network-ninjatrader-8/) |
| **Class** | `MlNeuralNetRnn` |
| **Series** | Part 6 of 9 — [see all](../../README.md) |
| **License** | **All rights reserved** — read it, learn from it, run it on your own charts. Not for commercial use or redistribution. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

Every model in this series so far — k-Nearest Neighbors, online logistic regression, the single- and multi-hidden-layer nets, and last post's Adam-trained version — has shared one quiet limitation. Each prediction sees only the current bar's feature vector.

[Read the full write-up →](https://mydailytake.com/ml-recurrent-neural-network-ninjatrader-8/)

## Install

1. Download **[`MlNeuralNetRnn.cs`](MlNeuralNetRnn.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlNeuralNetRnn** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Architecture | **Architecture** | Number of neurons in the recurrent hidden layer. Each bar updates an H-dimensional hidden-state vector that persists across bars. Larger H = more memory capacity but more parameters to train. Default 8. |
| Learning | **Learning** | Step size for each weight update. RNN training is more sensitive to learning rate than feedforward training because gradients compound across the BPTT window. Start at 0.005 and lower if training is unstable. |
| Features | **Features** | Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio. |
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
← [Part 5](../05-adam-optimizer) · [Part 7](../07-lstm) →
