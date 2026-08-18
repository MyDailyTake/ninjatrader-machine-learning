# Machine Learning for NinjaTrader 8

Machine learning models written in **native NinjaScript** — no Python, no external libraries, no
DLLs, no bridge. Each one trains and predicts **inside the indicator, on your bars**, in C#.

This is a teaching series. It starts with the simplest thing that can learn and builds, one post at a
time, to a recurrent network with gated memory. Every model is a single self-contained `.cs` file you
can read top to bottom.

**Full write-ups:** <https://mydailytake.com/blog/>

---

## The series

| # | Model | Class | Write-up |
|---|---|---|---|
| 1 | [k-Nearest Neighbors](indicators/01-k-nearest-neighbors) | `MlKNearestNeighbors` | [Read](https://mydailytake.com/ml-k-nearest-neighbors-indicator-for-ninjatrader-8/) |
| 2 | [Online Logistic Regression](indicators/02-online-logistic-regression) | `MlOnlineLogisticRegression` | [Read](https://mydailytake.com/ml-online-logistic-regression-your-first-neural-net-for-ninjatrader-8/) |
| 3 | [Single Hidden Layer](indicators/03-single-hidden-layer) | `MlNeuralNetSingleHidden` | [Read](https://mydailytake.com/ml-single-hidden-layer-neural-network-your-first-multi-layer-net-for-ninjatrader-8/) |
| 4 | [Multi Hidden Layer](indicators/04-multi-hidden-layer) | `MlNeuralNetMultiLayer` | [Read](https://mydailytake.com/ml-multi-hidden-layer-neural-network-configurable-depth-in-ninjatrader-8/) |
| 5 | [Adam Optimizer](indicators/05-adam-optimizer) | `MlNeuralNetAdam` | [Read](https://mydailytake.com/ml-adam-optimizer-mini-batch-stable-training-for-ninjatrader-8/) |
| 6 | [Recurrent Neural Network](indicators/06-recurrent-neural-network) | `MlNeuralNetRnn` | [Read](https://mydailytake.com/ml-recurrent-neural-network-ninjatrader-8/) |
| 7 | [LSTM](indicators/07-lstm) | `MlNeuralNetLstm` | [Read](https://mydailytake.com/ml-lstm-neural-network-ninjatrader-8/) |
| 8 | [Random Forest](indicators/08-random-forest) | `MlRandomForest` | [Read](https://mydailytake.com/ml-random-forest-ninjatrader-8/) |
| 9 | [Gradient Boosting](indicators/09-gradient-boosting) | `MlGradientBoost` | [Read](https://mydailytake.com/ml-gradient-boosting-ninjatrader-8/) |

Read them in order if you're learning. Parts 2–7 are one continuous thread: a single neuron becomes a
hidden layer, becomes depth, gets a better optimizer, gains memory, then gates that memory.

## Install

1. Download the `.cs` for the model you want.
2. Copy it to `Documents\NinjaTrader 8\bin\Custom\Indicators\`.
3. In NinjaTrader: **Control Center → New → NinjaScript Editor**, press **F5** to compile.
4. Add the indicator to a chart.

Every model is standalone — no shared base class, no dependency on anything else in this repo.

## Principles

- **Native NinjaScript.** No Python, no ONNX, no DLL, no socket to a sidecar process. If it can't be
  done in C# inside the indicator, it isn't here.
- **Readable over clever.** These are meant to be understood. The math is written out, not hidden
  behind a library call.
- **Honest about what it is.** A model that fits historical bars is not an edge. See below.

## Found a bug?

Probably mine. ML on live bars is fiddly — training windows, feature scaling, and warm-up all shift
results, and a model that looks great on history can behave differently going forward. Some of that
is the nature of the thing; some of it is me getting it wrong.

[Open an issue](../../issues) with:

- **Which model**, and your NT8 version
- **Instrument, bar type, and your data provider**
- **What you expected vs. what you got** — a screenshot says it faster than three paragraphs

No SLA, no support desk — it's free code from one guy who also has to trade. But I read every one,
and "here's the math, here's why it's wrong" is genuinely welcome. So is "here's a PR."

## Licensing — read this

© 2026 MyDailyTake.com. **All rights reserved.** See [LICENSE](LICENSE).

The source is published openly so you can **read it, learn from it, and run it on your own charts**.
That's the point of the series. It is **not** placed in the public domain and **not** released under
an open-source license.

**Not permitted:** commercial use or resale, redistribution or republication elsewhere, or removing
the copyright notices from the files.

**The notice at the top of each `.cs` is the authoritative statement of these terms** and governs
that file.

Want to use it commercially, or do something these terms don't cover? Just ask —
[jack@mydailytake.com](mailto:jack@mydailytake.com).

## Disclaimer

Educational and informational only. **Nothing here is trading advice, and a machine-learning
indicator is not a trading edge.** These are teaching implementations — they demonstrate how the
algorithms work on market data, not how to make money. Futures trading involves substantial risk of
loss and is not suitable for every investor. Past or simulated performance is not indicative of
future results. See the [full disclosure](https://mydailytake.com/disclosure/).

---

Built by [MyDailyTake](https://mydailytake.com) — NinjaTrader 8 tools and NinjaScript education.
Also: [TradingView → NinjaTrader 8 conversions](https://github.com/MyDailyTake/tradingview-to-ninjatrader-8) ·
[Learn NinjaScript](https://mydailytake.com/learn-ninjascript/) · [Tools](https://mydailytake.com/software/)
