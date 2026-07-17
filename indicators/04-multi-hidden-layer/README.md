# Part 4 — Multi-Hidden-Layer Neural Network: Configurable Depth

Machine learning in native NinjaScript for **NinjaTrader 8**. No Python, no external libraries, no
DLLs — it trains and predicts inside the indicator, on your bars.

| | |
|---|---|
| **Full write-up** | [mydailytake.com](https://mydailytake.com/ml-multi-hidden-layer-neural-network-configurable-depth-in-ninjatrader-8/) |
| **Class** | `MlNeuralNetMultiLayer` |
| **Series** | Part 4 of 8 — [see all](../../README.md) |
| **License** | **PolyForm Noncommercial 1.0.0** — free for personal use, not commercial. See [LICENSE](../../LICENSE). |
| **Platform** | NinjaTrader 8 |

## What it does

The prior post built a single-hidden-layer neural network — three inputs, six tanh neurons in one hidden layer, one sigmoid output, 31 parameters. It worked because the hidden layer can compose the raw features into non-linear combinations that a single neuron cannot represent.

[Read the full write-up →](https://mydailytake.com/ml-multi-hidden-layer-neural-network-configurable-depth-in-ninjatrader-8/)

## Install

1. Download **[`MlNeuralNetMultiLayer.cs`](MlNeuralNetMultiLayer.cs)**.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add **MlNeuralNetMultiLayer** to a chart.

## Settings

| Group | Setting | Description |
|---|---|---|
| Architecture | **Architecture** | Per-layer neuron counts as a list of integers. Any common separator works — comma, space, hyphen, x, semicolon. Examples: '6' = one layer of 6 neurons (matches the single-hidden-layer sibling). '8, 6' = two layers (8 then 6). '10, 6, 4' = three layers. Each value must be ≥ 2. No upper cap; the indicator allocates memory in proportion to total weights, so a sane upper limit on this hardware is ~64 neurons per layer. Invalid input falls back to default '8, 6' and logs a warning to the NT Output window. |
| Learning | **Learning** | Step size for each weight update. With more layers, the gradient at early layers is smaller (vanishing-gradient effect), so a slightly larger learning rate compensates — but too large destabilizes the deeper output side. Start at 0.005-0.01 and tune from there. |
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
← [Part 3](../03-single-hidden-layer) · [Part 5](../05-adam-optimizer) →
