# Part 7 — Long Short-Term Memory (LSTM): Gated Memory

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-lstm-neural-network-ninjatrader-8/) |
| **Class** | `MlNeuralNetLstm` |
| **Series** | Part 7 of 8 — [see all](../../README.md) |
| **License** | **All rights reserved** — read it, learn from it, run it on your own charts. Not for commercial use or redistribution. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

The previous post built a vanilla Recurrent Neural Network — a model with a hidden state that carries forward bar to bar — and then ran straight into its defining limitation. Training a recurrent network means multiplying the gradient by a tanh derivative at every step back through time, and a long chain of numbers below one collapses toward zero.

[Read the full write-up →](https://mydailytake.com/ml-lstm-neural-network-ninjatrader-8/)

## Install

1. Download **[`MlNeuralNetLstm.cs`](MlNeuralNetLstm.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlNeuralNetLstm** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Architecture | **Architecture** | Number of neurons in the LSTM hidden layer. Each bar updates an H-dimensional hidden-state vector AND an H-dimensional cell-state vector that both persist across bars. Larger H = more memory capacity but more parameters — the LSTM has four gates, so roughly 4x the weights of a vanilla RNN of the same size. Default 8. |
| Learning | **Learning** | Step size for each weight update. LSTM training is sensitive to learning rate because gradients compound across the BPTT window and four gates. Start at 0.005 and lower it first if training is unstable. |
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
← [Part 6](../06-recurrent-neural-network) · [Part 8](../08-random-forest) →
