#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

// Write-up: https://mydailytake.com/ml-lstm-neural-network-ninjatrader-8/
// Source: https://github.com/MyDailyTake/ninjatrader-machine-learning
// © 2026 MyDailyTake.com — free for personal use, not for commercial use.
// Licensed under PolyForm Noncommercial 1.0.0: https://polyformproject.org/licenses/noncommercial/1.0.0

// MyDailyTake.com
// Author: Jack <jack@mydailytake.com>
//
// Copyright © 2026 MyDailyTake.com
// All rights reserved.
//
// This code is provided for personal use only.
// No commercial use, distribution, or resale is permitted.
// Unauthorized copying, modification, or redistribution of this code is strictly prohibited.
// Removal or alteration of this notice — including the author and copyright information — is prohibited.
//
// For more information, visit: https://mydailytake.com

public enum MlNeuralNetLstm_WeightInitMode { Zero, Random }
public enum MlNeuralNetLstm_LabelMode { CloseToClose, FavorableExcursion }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Architecture",	10100)]
	[Gui.CategoryOrder("Learning",		10200)]
	[Gui.CategoryOrder("Features",		10300)]
	[Gui.CategoryOrder("Signal",		10400)]
	[Gui.CategoryOrder("Display",		10500)]

	#endregion

	public class MlNeuralNetLstm : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - LSTM Neural Net (Long Short-Term Memory)";
		public string indDescription	= "A Long Short-Term Memory (LSTM) network for NinjaTrader 8 — the gated successor to the vanilla RNN. Where the RNN's single hidden state is overwritten every bar, the LSTM adds a separate cell-state vector that information can flow along almost untouched, governed by three sigmoid gates: a forget gate that decides what to drop from the cell, an input gate that decides what to write, and an output gate that decides what to expose as the hidden state. That gated design is what lets the LSTM hold information across far longer spans than a vanilla RNN, because gradients can travel back along the cell state without being squashed by a tanh derivative at every step. Training uses truncated Backpropagation Through Time (BPTT): every training-eligible bar, the network is unrolled BpttWindow steps back, the forward pass through that sequence is recomputed, then the gradient is backpropagated through every gate at every step. Default features (same as the k-NN, OLR, SHL, MHL, Adam, and RNN siblings for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Z-score normalized using each bar's own local-time stats. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the prior posts. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Architecture

		[NinjaScriptProperty]
		[Range(2, 64)]
		[Display(Order = 01, GroupName = "Architecture", Name = "Hidden Size", Description = "Number of neurons in the LSTM hidden layer. Each bar updates an H-dimensional hidden-state vector AND an H-dimensional cell-state vector that both persist across bars. Larger H = more memory capacity but more parameters — the LSTM has four gates, so roughly 4x the weights of a vanilla RNN of the same size. Default 8.")]
		public int HiddenSize { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Order = 02, GroupName = "Architecture", Name = "BPTT Window (bars)", Description = "Number of past bars to unroll the network through when computing gradients (truncated Backpropagation Through Time). Bigger window = the model can learn longer-range dependencies. Unlike the vanilla RNN, the LSTM's cell-state path lets gradients survive longer windows, so a larger BpttWindow is genuinely more worthwhile here — at a proportional compute cost per training step. Default 10.")]
		public int BpttWindow { get; set; }

		[NinjaScriptProperty]
		[Range(0, 999999)]
		[Display(Order = 03, GroupName = "Architecture", Name = "Random Seed", Description = "Seed for the random weight initialization (when Weight Init = Random). Same seed produces the same starting weights — useful for reproducible testing or comparing different architectures fairly.")]
		public int RandomSeed { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(0.0001, 1.0)]
		[Display(Order = 01, GroupName = "Learning", Name = "Learning Rate", Description = "Step size for each weight update. LSTM training is sensitive to learning rate because gradients compound across the BPTT window and four gates. Start at 0.005 and lower it first if training is unstable.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.1)]
		[Display(Order = 02, GroupName = "Learning", Name = "Regularization Lambda (L2)", Description = "L2 penalty on weight magnitude. Pulls weights gently toward zero each update so they don't drift to extreme values. Applied to the weight matrices only, not the biases. Recommended 0.0001 to 0.001. Default 0.0005.")]
		public double RegularizationLambda { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. The model updates each bar using the prediction from N bars ago, whose forward outcome is now known.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Weight Init", Description = "Zero starts every weight at 0 — but then all four gates would be identical and nothing could propagate (the symmetry problem). Strongly recommend Random. Random uses Xavier/Glorot scaling for the gate weights and starts the forget-gate bias positive (1.0) so the cell remembers by default early in training.")]
		public MlNeuralNetLstm_WeightInitMode WeightInit { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 05, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. CloseToClose: y = 1 if Close at end of LabelHorizon window is above Close at trainBar. FavorableExcursion: y = 1 if MFE_long beat MFE_short during the window (uses bar highs/lows). Skips bars below Min Favorable Move (chop).")]
		public MlNeuralNetLstm_LabelMode LabelMode { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 5.0)]
		[Display(Order = 06, GroupName = "Learning", Name = "Min Favorable Move (ATRs)", Description = "ONLY USED WHEN Label Mode = FavorableExcursion. Minimum favorable excursion (in ATRs at entry) required during the post-bar window for the model to update.")]
		public double MinFavorableMoveAtrs { get; set; }

		#endregion

		#region Features

		[NinjaScriptProperty]
		[Range(2, 500)]
		[Display(Order = 01, GroupName = "Features", Name = "MA Period", Description = "Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio.")]
		public int MaPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 100)]
		[Display(Order = 02, GroupName = "Features", Name = "ATR Period", Description = "Period of the ATR used to scale every feature into volatility units.")]
		public int AtrPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 200)]
		[Display(Order = 03, GroupName = "Features", Name = "Slope Lookback (bars)", Description = "Number of bars over which the slope feature is measured: (Close[0] − Close[N]) / ATR.")]
		public int SlopeLookback { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features (Z-Score)", Description = "Master toggle for z-score normalization. Recommended ON.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Window used to compute the rolling mean / stddev that z-score the features. Each bar uses its own local-time stats.")]
		public int NormalizationLookback { get; set; }

		#endregion

		#region Signal

		[NinjaScriptProperty]
		[Range(0.0, 0.49)]
		[Display(Order = 01, GroupName = "Signal", Name = "Min Probability Edge", Description = "How far the predicted probability of an up move must be from 0.5 before a signal fires.")]
		public double MinProbabilityEdge { get; set; }

		[NinjaScriptProperty]
		[Range(0, 500)]
		[Display(Order = 02, GroupName = "Signal", Name = "Signal Cooldown (bars)", Description = "Minimum bars between consecutive signals.")]
		public int SignalCooldownBars { get; set; }

		#endregion

		#region Display

		[Display(Order = 01, GroupName = "Display", Name = "Marker Offset (ticks)", Description = "Vertical offset of the signal triangle from the bar's high (shorts) / low (longs), in ticks.")]
		[Range(0, 200)]
		public int MarkerOffsetTicks { get; set; }

		[Display(Order = 02, GroupName = "Display", Name = "Label Offset (ticks)", Description = "Distance from the bar to the text label, in ticks.")]
		[Range(0, 500)]
		public int LabelOffsetTicks { get; set; }

		[Display(Order = 03, GroupName = "Display", Name = "Show Labels", Description = "Render the predicted-probability label beside each signal triangle.")]
		public bool ShowLabels { get; set; }

		[Display(Order = 04, GroupName = "Display", Name = "Label Font Size", Description = "Font size for signal labels.")]
		[Range(8, 24)]
		public int LabelFontSize { get; set; }

		#endregion

		#region Private Fields

		private const int NumFeatures = 3;

		// Source indicators
		private SMA	trendMa;
		private ATR	atr;
		private SMA	atrRegimeMa;

		// Per-feature backing series + rolling stats (z-score normalization).
		private Series<double>[]	featureSeries;
		private SMA[]				featureMean;
		private StdDev[]			featureStd;

		// ─── LSTM weights ───
		// Per gate: input-to-gate (Wx*), recurrent hidden-to-gate (Wh*), bias (b*).
		// Gates: f = forget, i = input, g = candidate cell, o = output.
		//   Wx*[h][k] — k indexes the feature vector.
		//   Wh*[h][j] — j indexes the previous hidden state.
		private double[][]	Wxf, Wxi, Wxg, Wxo;
		private double[][]	Whf, Whi, Whg, Who;
		private double[]	bf,  bi,  bg,  bo;

		// Output layer: hidden state → sigmoid.
		private double[]	Wy;
		private double		by;

		// Persistent live state — hidden + cell — carried forward across bars and
		// used for the live prediction at bar 0. Walks forward indefinitely.
		private double[]	hLive;
		private double[]	cLive;

		// Rolling buffer of recent normalized feature vectors. Length = BpttWindow
		// + LabelHorizon. Index 0 = most recent bar; index (length - 1) = oldest.
		private double[][]	featureRing;
		private int			ringFilled;

		// ─── BPTT unrolled-window storage ───
		//   hSeq[0] / cSeq[0]   = anchor (zero state at the start of each window).
		//   hSeq[t+1] / cSeq[t+1] = state after processing bar t (t in 0..BpttWindow-1).
		//   The gate sequences (fSeq/iSeq/gSeq/oSeq) are filled for indices 1..BpttWindow.
		private double[][]	hSeq, cSeq;
		private double[][]	fSeq, iSeq, gSeq, oSeq;

		// Gradient accumulators — one slot per weight, summed across the unrolled window.
		private double[][]	gWxf, gWxi, gWxg, gWxo;
		private double[][]	gWhf, gWhi, gWhg, gWho;
		private double[]	gbf,  gbi,  gbg,  gbo;
		private double[]	gWy;

		// Per-step scratch arrays, allocated once in DataLoaded and reused on every
		// bar / BPTT step.
		private double[]	hPrevScratch;       // snapshot of h before in-place update
		private double[]	dhNext;             // dh flowing back from the next step
		private double[]	dcNext;             // dc flowing back from the next step
		private double[]	dzf, dzi, dzg, dzo; // gate pre-activation gradients
		private double[]	dhPrevScratch;      // dh propagated to the previous step
		private double[]	dcPrevScratch;      // dc propagated to the previous step

		// Scratch for feature extraction / normalization.
		private double[]	scratchRaw;
		private double[]	scratchNorm;

		// Cooldown tracker — bar index of the last fired signal. -1 = no signal yet.
		private int			lastSignalBar	= -1;

		// Public Series backing fields.
		private Series<double>	sProbabilityUp;
		private Series<double>	sConfidence;
		private Series<bool>	sIsLongSignal;
		private Series<bool>	sIsShortSignal;

		// TickSize-derived offsets, resolved once in DataLoaded.
		private double	markerOffsetPts;
		private double	labelOffsetPts;

		// Cached label font (re-built only when LabelFontSize changes).
		private SimpleFont	labelFont;
		private int			labelFontSizeCached = -1;

		#endregion

		#region OnStateChange

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				#region Settings

				Description					= indDescription;
				Name						= indName;
				Calculate					= Calculate.OnBarClose;

				IsOverlay					= true;
				DisplayInDataBox			= true;
				DrawHorizontalGridLines		= true;
				DrawVerticalGridLines		= true;
				PaintPriceMarkers			= true;
				ScaleJustification			= NinjaTrader.Gui.Chart.ScaleJustification.Right;
				IsSuspendedWhileInactive	= true;

				#endregion

				#region Defaults

				HiddenSize				= 8;
				BpttWindow				= 10;
				RandomSeed				= 42;

				LearningRate			= 0.005;
				RegularizationLambda	= 0.0005;
				LabelHorizon			= 2;
				WeightInit				= MlNeuralNetLstm_WeightInitMode.Random;
				LabelMode				= MlNeuralNetLstm_LabelMode.CloseToClose;
				MinFavorableMoveAtrs	= 1.0;

				MaPeriod				= 8;
				AtrPeriod				= 50;
				SlopeLookback			= 2;
				NormalizeFeatures		= true;
				NormalizationLookback	= 200;

				MinProbabilityEdge		= 0.10;
				SignalCooldownBars		= 3;

				MarkerOffsetTicks		= 4;
				LabelOffsetTicks		= 20;
				ShowLabels				= true;
				LabelFontSize			= 12;

				AddPlot(new Stroke(Brushes.LimeGreen, 5),	PlotStyle.TriangleUp,	"NN Long");
				AddPlot(new Stroke(Brushes.OrangeRed, 5),	PlotStyle.TriangleDown,	"NN Short");

				#endregion
			}
			else if (State == State.DataLoaded)
			{
				sProbabilityUp	= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sConfidence		= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sIsLongSignal	= new Series<bool>(this,   MaximumBarsLookBack.Infinite);
				sIsShortSignal	= new Series<bool>(this,   MaximumBarsLookBack.Infinite);

				trendMa		= SMA(MaPeriod);
				atr			= ATR(AtrPeriod);
				atrRegimeMa	= SMA(atr, MaPeriod);

				featureSeries	= new Series<double>[NumFeatures];
				featureMean		= new SMA[NumFeatures];
				featureStd		= new StdDev[NumFeatures];
				for (int k = 0; k < NumFeatures; k++)
				{
					featureSeries[k]	= new Series<double>(this, MaximumBarsLookBack.Infinite);
					featureMean[k]		= SMA(featureSeries[k], NormalizationLookback);
					featureStd[k]		= StdDev(featureSeries[k], NormalizationLookback);
				}

				// Gate weights
				Wxf = NewJagged(HiddenSize, NumFeatures);
				Wxi = NewJagged(HiddenSize, NumFeatures);
				Wxg = NewJagged(HiddenSize, NumFeatures);
				Wxo = NewJagged(HiddenSize, NumFeatures);
				Whf = NewJagged(HiddenSize, HiddenSize);
				Whi = NewJagged(HiddenSize, HiddenSize);
				Whg = NewJagged(HiddenSize, HiddenSize);
				Who = NewJagged(HiddenSize, HiddenSize);
				bf  = new double[HiddenSize];
				bi  = new double[HiddenSize];
				bg  = new double[HiddenSize];
				bo  = new double[HiddenSize];
				Wy  = new double[HiddenSize];

				InitializeWeights();

				// Live state + ring buffer
				hLive		= new double[HiddenSize];
				cLive		= new double[HiddenSize];
				int ringSize = BpttWindow + LabelHorizon;
				featureRing	= new double[ringSize][];
				for (int t = 0; t < ringSize; t++) featureRing[t] = new double[NumFeatures];
				ringFilled	= 0;

				// BPTT unrolled-window storage
				hSeq = NewJagged(BpttWindow + 1, HiddenSize);
				cSeq = NewJagged(BpttWindow + 1, HiddenSize);
				fSeq = NewJagged(BpttWindow + 1, HiddenSize);
				iSeq = NewJagged(BpttWindow + 1, HiddenSize);
				gSeq = NewJagged(BpttWindow + 1, HiddenSize);
				oSeq = NewJagged(BpttWindow + 1, HiddenSize);

				// Gradient accumulators
				gWxf = NewJagged(HiddenSize, NumFeatures);
				gWxi = NewJagged(HiddenSize, NumFeatures);
				gWxg = NewJagged(HiddenSize, NumFeatures);
				gWxo = NewJagged(HiddenSize, NumFeatures);
				gWhf = NewJagged(HiddenSize, HiddenSize);
				gWhi = NewJagged(HiddenSize, HiddenSize);
				gWhg = NewJagged(HiddenSize, HiddenSize);
				gWho = NewJagged(HiddenSize, HiddenSize);
				gbf  = new double[HiddenSize];
				gbi  = new double[HiddenSize];
				gbg  = new double[HiddenSize];
				gbo  = new double[HiddenSize];
				gWy  = new double[HiddenSize];

				// Per-step scratch
				hPrevScratch	= new double[HiddenSize];
				dhNext			= new double[HiddenSize];
				dcNext			= new double[HiddenSize];
				dzf				= new double[HiddenSize];
				dzi				= new double[HiddenSize];
				dzg				= new double[HiddenSize];
				dzo				= new double[HiddenSize];
				dhPrevScratch	= new double[HiddenSize];
				dcPrevScratch	= new double[HiddenSize];

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];

				lastSignalBar	= -1;
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;
			}
		}

		#endregion

		// ─── ALLOCATION HELPER ────────────────────────────────────────────────────
		// Allocates a rows×cols jagged array. Called only from DataLoaded.

		private static double[][] NewJagged(int rows, int cols)
		{
			double[][] m = new double[rows][];
			for (int r = 0; r < rows; r++) m[r] = new double[cols];
			return m;
		}

		// ─── WEIGHT INITIALIZATION ────────────────────────────────────────────────
		// Xavier/Glorot scaling, applied per matrix:
		//   Wx*:  sqrt(2 / (NumFeatures + HiddenSize))   — input → gate
		//   Wh*:  sqrt(2 / (HiddenSize + HiddenSize))    — hidden → gate
		//   Wy:   sqrt(2 / (HiddenSize + 1))             — hidden → output sigmoid
		// The forget-gate bias starts at 1.0 (a standard LSTM practice): a positive
		// forget bias means the cell defaults to remembering, so gradients can flow
		// along the cell state from the first training step instead of being shut
		// off by a near-zero forget gate.

		private void InitializeWeights()
		{
			if (WeightInit == MlNeuralNetLstm_WeightInitMode.Random)
			{
				var rng = new System.Random(RandomSeed);

				double scaleX = Math.Sqrt(2.0 / (NumFeatures + HiddenSize));
				double scaleH = Math.Sqrt(2.0 / (HiddenSize + HiddenSize));
				double scaleY = Math.Sqrt(2.0 / (HiddenSize + 1));

				for (int h = 0; h < HiddenSize; h++)
				{
					for (int k = 0; k < NumFeatures; k++)
					{
						Wxf[h][k] = SampleNormal(rng) * scaleX;
						Wxi[h][k] = SampleNormal(rng) * scaleX;
						Wxg[h][k] = SampleNormal(rng) * scaleX;
						Wxo[h][k] = SampleNormal(rng) * scaleX;
					}
					for (int j = 0; j < HiddenSize; j++)
					{
						Whf[h][j] = SampleNormal(rng) * scaleH;
						Whi[h][j] = SampleNormal(rng) * scaleH;
						Whg[h][j] = SampleNormal(rng) * scaleH;
						Who[h][j] = SampleNormal(rng) * scaleH;
					}
					bf[h] = 1.0;   // forget-gate bias starts positive — cell remembers by default
					bi[h] = 0.0;
					bg[h] = 0.0;
					bo[h] = 0.0;
					Wy[h] = SampleNormal(rng) * scaleY;
				}
				by = 0.0;
			}
			else // Zero — strongly discouraged for an LSTM; included for completeness.
			{
				for (int h = 0; h < HiddenSize; h++)
				{
					for (int k = 0; k < NumFeatures; k++)
					{
						Wxf[h][k] = 0; Wxi[h][k] = 0; Wxg[h][k] = 0; Wxo[h][k] = 0;
					}
					for (int j = 0; j < HiddenSize; j++)
					{
						Whf[h][j] = 0; Whi[h][j] = 0; Whg[h][j] = 0; Who[h][j] = 0;
					}
					bf[h] = 0; bi[h] = 0; bg[h] = 0; bo[h] = 0;
					Wy[h] = 0;
				}
				by = 0.0;
			}
		}

		private static double SampleNormal(System.Random rng)
		{
			// Box-Muller transform — converts two uniform random samples into one normal.
			double u1 = 1.0 - rng.NextDouble();
			double u2 = 1.0 - rng.NextDouble();
			return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
		}

		// ─── ON BAR UPDATE ────────────────────────────────────────────────────────

		protected override void OnBarUpdate()
		{
			Values[0].Reset();
			Values[1].Reset();
			sIsLongSignal[0]	= false;
			sIsShortSignal[0]	= false;

			int sourceWarmup = Math.Max(Math.Max(MaPeriod, AtrPeriod), SlopeLookback) + 1;
			if (CurrentBar < sourceWarmup) return;

			GetFeaturesInto(scratchRaw, 0);
			for (int k = 0; k < NumFeatures; k++)
				featureSeries[k][0] = scratchRaw[k];

			int predictionWarmup = sourceWarmup + NormalizationLookback;
			if (CurrentBar < predictionWarmup) return;

			// 1) Forward pass on the current bar — used for the live prediction.
			//    Walks the persistent hidden + cell state one step forward.
			NormalizeInto(scratchNorm, scratchRaw, 0);
			StepCell(hLive, cLive, scratchNorm);
			double pUp = OutputSigmoid(hLive);

			// 2) Push the new normalized feature into the ring buffer (index 0 = newest).
			PushFeatureRing(scratchNorm);

			// 3) Training step — when LabelHorizon ago is observable AND we've buffered
			//    enough bars for a full BPTT window.
			int needForTraining = BpttWindow + LabelHorizon;
			if (ringFilled >= needForTraining && CurrentBar >= predictionWarmup + LabelHorizon)
			{
				bool   trainThisBar	= false;
				double y			= 0.0;
				int    trainBar		= LabelHorizon;

				if (LabelMode == MlNeuralNetLstm_LabelMode.CloseToClose)
				{
					y = Close[0] > Close[trainBar] ? 1.0 : 0.0;
					trainThisBar = true;
				}
				else // FavorableExcursion
				{
					double closeAtTrain	= Close[trainBar];
					double atrAtTrain	= atr[trainBar];
					double safeAtrTrain	= atrAtTrain > 1e-9 ? atrAtTrain : TickSize;

					double maxHigh = double.MinValue;
					double minLow  = double.MaxValue;
					for (int b = 0; b < LabelHorizon; b++)
					{
						double hi = High[b];
						double lo = Low[b];
						if (hi > maxHigh) maxHigh = hi;
						if (lo < minLow)  minLow  = lo;
					}

					double mfeLong  = (maxHigh    - closeAtTrain) / safeAtrTrain;
					double mfeShort = (closeAtTrain - minLow)     / safeAtrTrain;

					if (Math.Max(mfeLong, mfeShort) >= MinFavorableMoveAtrs)
					{
						y = mfeLong > mfeShort ? 1.0 : 0.0;
						trainThisBar = true;
					}
				}

				if (trainThisBar)
					TruncatedBptt(y);
			}

			// 4) Public-Series outputs.
			sProbabilityUp[0] = pUp;
			sConfidence[0]    = Math.Abs(pUp - 0.5) * 2.0;

			// 5) Signal gate — wait for at least 50 update steps after the first
			//    label is observable so the gate weights have time to settle.
			int signalWarmup = predictionWarmup + LabelHorizon + 50;
			if (CurrentBar < signalWarmup) return;

			if (lastSignalBar >= 0 && CurrentBar - lastSignalBar < SignalCooldownBars) return;

			if (pUp > 0.5 + MinProbabilityEdge)
			{
				sIsLongSignal[0]	= true;
				Values[0][0]		= Low[0] - markerOffsetPts;
				lastSignalBar		= CurrentBar;
				if (ShowLabels) DrawSignalLabel(true, pUp);
			}
			else if (pUp < 0.5 - MinProbabilityEdge)
			{
				sIsShortSignal[0]	= true;
				Values[1][0]		= High[0] + markerOffsetPts;
				lastSignalBar		= CurrentBar;
				if (ShowLabels) DrawSignalLabel(false, pUp);
			}
		}

		// ─── FEATURE RING BUFFER ──────────────────────────────────────────────────
		// Index 0 = most recent bar; index (length - 1) = oldest. Push shifts older
		// entries up by one and writes the new vector at index 0.

		private void PushFeatureRing(double[] x)
		{
			int n = featureRing.Length;
			// Shift older entries: ring[i+1] = ring[i] for i = n-2..0 (in place).
			// Re-use the array at the back to hold the new entry — avoids allocation.
			double[] recycled = featureRing[n - 1];
			for (int i = n - 1; i > 0; i--)
				featureRing[i] = featureRing[i - 1];
			featureRing[0] = recycled;
			for (int k = 0; k < NumFeatures; k++) featureRing[0][k] = x[k];

			if (ringFilled < n) ringFilled++;
		}

		// ─── FORWARD STEP (one bar) ───────────────────────────────────────────────
		// Advances the LSTM cell one bar, updating h and c in place:
		//   f = sigmoid(Wxf·x + Whf·h_prev + bf)     forget gate
		//   i = sigmoid(Wxi·x + Whi·h_prev + bi)     input gate
		//   g = tanh   (Wxg·x + Whg·h_prev + bg)     candidate cell
		//   o = sigmoid(Wxo·x + Who·h_prev + bo)     output gate
		//   c = f ⊙ c_prev + i ⊙ g                   new cell state
		//   h = o ⊙ tanh(c)                          new hidden state
		// h_prev is snapshotted first so every neuron reads a consistent copy of it.
		// The cell update is element-wise in the cell dimension, so reading and
		// writing c[hh] within the same iteration is correct.

		private void StepCell(double[] h, double[] c, double[] x)
		{
			for (int j = 0; j < HiddenSize; j++) hPrevScratch[j] = h[j];

			for (int hh = 0; hh < HiddenSize; hh++)
			{
				double zf = bf[hh], zi = bi[hh], zg = bg[hh], zo = bo[hh];
				for (int k = 0; k < NumFeatures; k++)
				{
					double xk = x[k];
					zf += Wxf[hh][k] * xk;
					zi += Wxi[hh][k] * xk;
					zg += Wxg[hh][k] * xk;
					zo += Wxo[hh][k] * xk;
				}
				for (int j = 0; j < HiddenSize; j++)
				{
					double hj = hPrevScratch[j];
					zf += Whf[hh][j] * hj;
					zi += Whi[hh][j] * hj;
					zg += Whg[hh][j] * hj;
					zo += Who[hh][j] * hj;
				}
				double f  = Sigmoid(zf);
				double ig = Sigmoid(zi);
				double gg = Math.Tanh(zg);
				double og = Sigmoid(zo);
				double cv = f * c[hh] + ig * gg;
				c[hh] = cv;
				h[hh] = og * Math.Tanh(cv);
			}
		}

		// ─── OUTPUT (final sigmoid) ───────────────────────────────────────────────

		private double OutputSigmoid(double[] h)
		{
			double z = by;
			for (int hh = 0; hh < HiddenSize; hh++) z += Wy[hh] * h[hh];
			return Sigmoid(z);
		}

		// ─── TRUNCATED BPTT ───────────────────────────────────────────────────────
		// Recompute the forward pass over the BPTT window ending at trainBar, then
		// backprop the gradient through every gate at every time step. The anchor
		// hidden + cell state at the start of the window is reset to zero each call
		// — a standard simplification for online truncated BPTT. The live persistent
		// state (hLive / cLive) is unaffected; it continues forward outside training.
		//
		// Window indexing (the feature ring is 0 = newest):
		//   trainBar lives at ring index LabelHorizon.
		//   The window covers ring indices (LabelHorizon + BpttWindow - 1) down to
		//   LabelHorizon — BpttWindow bars total, with trainBar at the newest end.

		private void TruncatedBptt(double y)
		{
			// ── Forward pass over the BPTT window ──
			// hSeq[0] / cSeq[0] = zero anchor; index t+1 = state after processing bar t.
			for (int j = 0; j < HiddenSize; j++)
			{
				hSeq[0][j] = 0.0;
				cSeq[0][j] = 0.0;
			}

			for (int t = 0; t < BpttWindow; t++)
			{
				int ringIdx = LabelHorizon + BpttWindow - 1 - t;   // oldest first
				double[] xt    = featureRing[ringIdx];
				double[] hPrev = hSeq[t];
				double[] cPrev = cSeq[t];
				double[] hCur  = hSeq[t + 1];
				double[] cCur  = cSeq[t + 1];
				double[] fCur  = fSeq[t + 1];
				double[] iCur  = iSeq[t + 1];
				double[] gCur  = gSeq[t + 1];
				double[] oCur  = oSeq[t + 1];

				for (int hh = 0; hh < HiddenSize; hh++)
				{
					double zf = bf[hh], zi = bi[hh], zg = bg[hh], zo = bo[hh];
					for (int k = 0; k < NumFeatures; k++)
					{
						double xk = xt[k];
						zf += Wxf[hh][k] * xk;
						zi += Wxi[hh][k] * xk;
						zg += Wxg[hh][k] * xk;
						zo += Wxo[hh][k] * xk;
					}
					for (int j = 0; j < HiddenSize; j++)
					{
						double hj = hPrev[j];
						zf += Whf[hh][j] * hj;
						zi += Whi[hh][j] * hj;
						zg += Whg[hh][j] * hj;
						zo += Who[hh][j] * hj;
					}
					double f  = Sigmoid(zf);
					double ig = Sigmoid(zi);
					double gg = Math.Tanh(zg);
					double og = Sigmoid(zo);
					double cv = f * cPrev[hh] + ig * gg;

					fCur[hh] = f;
					iCur[hh] = ig;
					gCur[hh] = gg;
					oCur[hh] = og;
					cCur[hh] = cv;
					hCur[hh] = og * Math.Tanh(cv);
				}
			}

			// ── Output and error at the end of the window ──
			double pTrain = OutputSigmoid(hSeq[BpttWindow]);
			double error  = pTrain - y;   // dL/dzOut for cross-entropy + sigmoid

			// ── Zero the gate-weight gradient accumulators ──
			for (int hh = 0; hh < HiddenSize; hh++)
			{
				for (int k = 0; k < NumFeatures; k++)
				{
					gWxf[hh][k] = 0.0; gWxi[hh][k] = 0.0;
					gWxg[hh][k] = 0.0; gWxo[hh][k] = 0.0;
				}
				for (int j = 0; j < HiddenSize; j++)
				{
					gWhf[hh][j] = 0.0; gWhi[hh][j] = 0.0;
					gWhg[hh][j] = 0.0; gWho[hh][j] = 0.0;
				}
				gbf[hh] = 0.0; gbi[hh] = 0.0; gbg[hh] = 0.0; gbo[hh] = 0.0;
			}

			// ── Output-layer gradients (using OLD Wy before it is updated) ──
			// Seed dhNext = gradient flowing back from the output into the last
			// hidden state hSeq[BpttWindow]; dcNext starts at zero (no future cell).
			for (int hh = 0; hh < HiddenSize; hh++)
			{
				gWy[hh]    = error * hSeq[BpttWindow][hh] + RegularizationLambda * Wy[hh];
				dhNext[hh] = error * Wy[hh];
				dcNext[hh] = 0.0;
			}
			double gby = error;

			// ── BPTT — walk backward through the unrolled time steps ──
			// At step t we know dh_t (dhNext) and the cell gradient carried from
			// step t+1 (dcNext). The per-step backward math:
			//   doG = dh_t ⊙ tanh(c_t)
			//   dc  = dh_t ⊙ o_t ⊙ (1 - tanh(c_t)^2)  +  dcNext        [total dc_t]
			//   df  = dc ⊙ c_{t-1} ;  di = dc ⊙ g_t ;  dg = dc ⊙ i_t
			//   dz* = d* ⊙ (gate activation derivative)
			//   accumulate gW* and gb* ; propagate dh_{t-1} and dc_{t-1}.
			for (int t = BpttWindow; t >= 1; t--)
			{
				int ringIdx = LabelHorizon + BpttWindow - 1 - (t - 1);
				double[] xt    = featureRing[ringIdx];
				double[] hPrev = hSeq[t - 1];
				double[] cPrev = cSeq[t - 1];
				double[] fCur  = fSeq[t];
				double[] iCur  = iSeq[t];
				double[] gCur  = gSeq[t];
				double[] oCur  = oSeq[t];
				double[] cCur  = cSeq[t];

				// Gate pre-activation gradients for this step.
				for (int hh = 0; hh < HiddenSize; hh++)
				{
					double tanhC = Math.Tanh(cCur[hh]);

					// h = o ⊙ tanh(c)
					double dh  = dhNext[hh];
					double doG = dh * tanhC;
					double dc  = dh * oCur[hh] * (1.0 - tanhC * tanhC) + dcNext[hh];

					// c = f ⊙ c_prev + i ⊙ g
					double fv = fCur[hh], iv = iCur[hh], gv = gCur[hh], ov = oCur[hh];
					double df = dc * cPrev[hh];
					double di = dc * gv;
					double dg = dc * iv;
					dcPrevScratch[hh] = dc * fv;     // gradient carried to c_{t-1}

					// Gate activation derivatives (sigmoid: s(1-s); tanh: 1-g^2).
					dzf[hh] = df  * fv * (1.0 - fv);
					dzi[hh] = di  * iv * (1.0 - iv);
					dzg[hh] = dg  * (1.0 - gv * gv);
					dzo[hh] = doG * ov * (1.0 - ov);
				}

				// Accumulate gate-weight gradients for this step.
				for (int hh = 0; hh < HiddenSize; hh++)
				{
					double dzfh = dzf[hh], dzih = dzi[hh], dzgh = dzg[hh], dzoh = dzo[hh];
					for (int k = 0; k < NumFeatures; k++)
					{
						double xk = xt[k];
						gWxf[hh][k] += dzfh * xk;
						gWxi[hh][k] += dzih * xk;
						gWxg[hh][k] += dzgh * xk;
						gWxo[hh][k] += dzoh * xk;
					}
					for (int j = 0; j < HiddenSize; j++)
					{
						double hj = hPrev[j];
						gWhf[hh][j] += dzfh * hj;
						gWhi[hh][j] += dzih * hj;
						gWhg[hh][j] += dzgh * hj;
						gWho[hh][j] += dzoh * hj;
					}
					gbf[hh] += dzfh;
					gbi[hh] += dzih;
					gbg[hh] += dzgh;
					gbo[hh] += dzoh;
				}

				// Propagate gradients to the previous step. Uses OLD recurrent
				// weights — weight updates happen AFTER this loop completes.
				// Skip at t == 1: hSeq[0] / cSeq[0] are the fixed zero anchor.
				if (t > 1)
				{
					for (int j = 0; j < HiddenSize; j++)
					{
						double s = 0.0;
						for (int hh = 0; hh < HiddenSize; hh++)
							s += Whf[hh][j] * dzf[hh] + Whi[hh][j] * dzi[hh]
							   + Whg[hh][j] * dzg[hh] + Who[hh][j] * dzo[hh];
						dhPrevScratch[j] = s;
					}
					for (int j = 0; j < HiddenSize; j++)
					{
						dhNext[j] = dhPrevScratch[j];
						dcNext[j] = dcPrevScratch[j];
					}
				}
			}

			// ── Apply weight updates — plain SGD, L2 on weight matrices only ──
			for (int hh = 0; hh < HiddenSize; hh++)
			{
				for (int k = 0; k < NumFeatures; k++)
				{
					Wxf[hh][k] -= LearningRate * (gWxf[hh][k] + RegularizationLambda * Wxf[hh][k]);
					Wxi[hh][k] -= LearningRate * (gWxi[hh][k] + RegularizationLambda * Wxi[hh][k]);
					Wxg[hh][k] -= LearningRate * (gWxg[hh][k] + RegularizationLambda * Wxg[hh][k]);
					Wxo[hh][k] -= LearningRate * (gWxo[hh][k] + RegularizationLambda * Wxo[hh][k]);
				}
				for (int j = 0; j < HiddenSize; j++)
				{
					Whf[hh][j] -= LearningRate * (gWhf[hh][j] + RegularizationLambda * Whf[hh][j]);
					Whi[hh][j] -= LearningRate * (gWhi[hh][j] + RegularizationLambda * Whi[hh][j]);
					Whg[hh][j] -= LearningRate * (gWhg[hh][j] + RegularizationLambda * Whg[hh][j]);
					Who[hh][j] -= LearningRate * (gWho[hh][j] + RegularizationLambda * Who[hh][j]);
				}
				bf[hh] -= LearningRate * gbf[hh];
				bi[hh] -= LearningRate * gbi[hh];
				bg[hh] -= LearningRate * gbg[hh];
				bo[hh] -= LearningRate * gbo[hh];
				Wy[hh] -= LearningRate * gWy[hh];
			}
			by -= LearningRate * gby;
		}

		// ─── FEATURE EXTRACTION + NORMALIZATION + SIGMOID ─────────────────────────

		private void GetFeaturesInto(double[] dst, int barsAgo)
		{
			double atrVal	= atr[barsAgo];
			double safeAtr	= atrVal > 1e-9 ? atrVal : TickSize;

			dst[0] = (Close[barsAgo] - trendMa[barsAgo]) / safeAtr;

			int slopeBack = barsAgo + SlopeLookback;
			dst[1] = slopeBack <= CurrentBar
				? (Close[barsAgo] - Close[slopeBack]) / safeAtr
				: 0;

			double atrSmaVal = atrRegimeMa[barsAgo];
			dst[2] = atrSmaVal > 1e-9 ? atrVal / atrSmaVal : 1.0;
		}

		private void NormalizeInto(double[] dst, double[] raw, int barsAgo)
		{
			if (!NormalizeFeatures)
			{
				for (int k = 0; k < NumFeatures; k++) dst[k] = raw[k];
				return;
			}

			for (int k = 0; k < NumFeatures; k++)
			{
				double mean = featureMean[k][barsAgo];
				double std  = featureStd[k][barsAgo];
				dst[k] = std > 1e-9 ? (raw[k] - mean) / std : (raw[k] - mean);
			}
		}

		private static double Sigmoid(double z)
		{
			if (z > 35.0)  return 1.0;
			if (z < -35.0) return 0.0;
			return 1.0 / (1.0 + Math.Exp(-z));
		}

		// ─── SIGNAL LABEL ─────────────────────────────────────────────────────────

		private void DrawSignalLabel(bool isLong, double pUp)
		{
			if (labelFontSizeCached != LabelFontSize)
			{
				labelFont			= new SimpleFont("Arial", LabelFontSize);
				labelFontSizeCached	= LabelFontSize;
			}

			string pctText = string.Format("P(up)={0:0.00}", pUp);
			string dirText = isLong ? "Long" : "Short";

			string text = isLong
				? pctText + "\n" + dirText
				: dirText + "\n" + pctText;

			double halfBlockHeight	= LabelFontSize * 0.6 * TickSize;
			double yPx = isLong
				? Low[0]  - labelOffsetPts - halfBlockHeight
				: High[0] + labelOffsetPts + halfBlockHeight;

			Brush brush = isLong ? Brushes.LimeGreen : Brushes.OrangeRed;
			string tag  = (isLong ? "lstm-long-lbl-" : "lstm-short-lbl-") + CurrentBar;

			Draw.Text(this, tag, false, text, 0, yPx, 0, brush,
				labelFont, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
		}

		#region Public Series

		[Browsable(false)] [XmlIgnore] public Series<double>	ProbabilityUpSeries		{ get { Update(); return sProbabilityUp;	} }
		[Browsable(false)] [XmlIgnore] public Series<double>	ConfidenceSeries		{ get { Update(); return sConfidence;		} }
		[Browsable(false)] [XmlIgnore] public Series<bool>		IsLongSignalSeries		{ get { Update(); return sIsLongSignal;		} }
		[Browsable(false)] [XmlIgnore] public Series<bool>		IsShortSignalSeries		{ get { Update(); return sIsShortSignal;	} }

		#endregion
	}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private indMyDailyTake.MlNeuralNetLstm[] cacheMlNeuralNetLstm;
		public indMyDailyTake.MlNeuralNetLstm MlNeuralNetLstm(int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetLstm_WeightInitMode weightInit, MlNeuralNetLstm_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlNeuralNetLstm(Input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlNeuralNetLstm MlNeuralNetLstm(ISeries<double> input, int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetLstm_WeightInitMode weightInit, MlNeuralNetLstm_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlNeuralNetLstm != null)
				for (int idx = 0; idx < cacheMlNeuralNetLstm.Length; idx++)
					if (cacheMlNeuralNetLstm[idx] != null && cacheMlNeuralNetLstm[idx].HiddenSize == hiddenSize && cacheMlNeuralNetLstm[idx].BpttWindow == bpttWindow && cacheMlNeuralNetLstm[idx].RandomSeed == randomSeed && cacheMlNeuralNetLstm[idx].LearningRate == learningRate && cacheMlNeuralNetLstm[idx].RegularizationLambda == regularizationLambda && cacheMlNeuralNetLstm[idx].LabelHorizon == labelHorizon && cacheMlNeuralNetLstm[idx].WeightInit == weightInit && cacheMlNeuralNetLstm[idx].LabelMode == labelMode && cacheMlNeuralNetLstm[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlNeuralNetLstm[idx].MaPeriod == maPeriod && cacheMlNeuralNetLstm[idx].AtrPeriod == atrPeriod && cacheMlNeuralNetLstm[idx].SlopeLookback == slopeLookback && cacheMlNeuralNetLstm[idx].NormalizeFeatures == normalizeFeatures && cacheMlNeuralNetLstm[idx].NormalizationLookback == normalizationLookback && cacheMlNeuralNetLstm[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlNeuralNetLstm[idx].SignalCooldownBars == signalCooldownBars && cacheMlNeuralNetLstm[idx].EqualsInput(input))
						return cacheMlNeuralNetLstm[idx];
			return CacheIndicator<indMyDailyTake.MlNeuralNetLstm>(new indMyDailyTake.MlNeuralNetLstm(){ HiddenSize = hiddenSize, BpttWindow = bpttWindow, RandomSeed = randomSeed, LearningRate = learningRate, RegularizationLambda = regularizationLambda, LabelHorizon = labelHorizon, WeightInit = weightInit, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlNeuralNetLstm);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetLstm MlNeuralNetLstm(int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetLstm_WeightInitMode weightInit, MlNeuralNetLstm_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetLstm(Input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetLstm MlNeuralNetLstm(ISeries<double> input , int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetLstm_WeightInitMode weightInit, MlNeuralNetLstm_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetLstm(input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetLstm MlNeuralNetLstm(int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetLstm_WeightInitMode weightInit, MlNeuralNetLstm_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetLstm(Input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetLstm MlNeuralNetLstm(ISeries<double> input , int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetLstm_WeightInitMode weightInit, MlNeuralNetLstm_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetLstm(input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
