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

// Write-up: https://mydailytake.com/ml-recurrent-neural-network-ninjatrader-8/
// Source: https://github.com/MyDailyTake/ninjatrader-machine-learning

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

public enum MlNeuralNetRnn_WeightInitMode { Zero, Random }
public enum MlNeuralNetRnn_LabelMode { CloseToClose, FavorableExcursion }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Architecture",	10100)]
	[Gui.CategoryOrder("Learning",		10200)]
	[Gui.CategoryOrder("Features",		10300)]
	[Gui.CategoryOrder("Signal",		10400)]
	[Gui.CategoryOrder("Display",		10500)]

	#endregion

	public class MlNeuralNetRnn : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Recurrent Neural Net (Vanilla RNN)";
		public string indDescription	= "A vanilla recurrent neural network for NinjaTrader 8. Unlike the feedforward siblings (which see only the current bar's features), the RNN maintains a hidden-state vector that carries information from bar to bar — so each prediction can depend on what happened many bars ago, not just on what's encoded in the current feature vector. Training uses truncated Backpropagation Through Time (BPTT): every training-eligible bar, we unroll the network BpttWindow steps back, recompute the forward pass through that sequence, then backprop the gradient through every step in the unrolled chain. Tanh activation on the hidden state keeps values in [-1, 1] to prevent the recurrent loop from blowing up. Default features (same as the k-NN, OLR, SHL, MHL, and Adam siblings for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Z-score normalized using each bar's own local-time stats. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the prior posts. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Architecture

		[NinjaScriptProperty]
		[Range(2, 64)]
		[Display(Order = 01, GroupName = "Architecture", Name = "Hidden Size", Description = "Number of neurons in the recurrent hidden layer. Each bar updates an H-dimensional hidden-state vector that persists across bars. Larger H = more memory capacity but more parameters to train. Default 8.")]
		public int HiddenSize { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Order = 02, GroupName = "Architecture", Name = "BPTT Window (bars)", Description = "Number of past bars to unroll the network through when computing gradients (truncated Backpropagation Through Time). Bigger window = the model can learn longer-range dependencies but each training step costs more compute. Smaller window = fast training but the model can only learn short-range patterns. Default 10.")]
		public int BpttWindow { get; set; }

		[NinjaScriptProperty]
		[Range(0, 999999)]
		[Display(Order = 03, GroupName = "Architecture", Name = "Random Seed", Description = "Seed for the random weight initialization (when Weight Init = Random). Same seed produces the same starting weights — useful for reproducible testing or comparing different architectures fairly.")]
		public int RandomSeed { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(0.0001, 1.0)]
		[Display(Order = 01, GroupName = "Learning", Name = "Learning Rate", Description = "Step size for each weight update. RNN training is more sensitive to learning rate than feedforward training because gradients compound across the BPTT window. Start at 0.005 and lower if training is unstable.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.1)]
		[Display(Order = 02, GroupName = "Learning", Name = "Regularization Lambda (L2)", Description = "L2 penalty on weight magnitude. Pulls weights gently toward zero each update so they don't drift to extreme values. Recommended 0.0001 to 0.001. Default 0.0005.")]
		public double RegularizationLambda { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. The model updates each bar using the prediction from N bars ago, whose forward outcome is now known.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Weight Init", Description = "Zero starts every weight at 0 — but the recurrent matrix would then have nothing to propagate, and all neurons would learn identical weights (symmetry problem). Strongly recommend Random. Random uses Xavier/Glorot scaling for tanh activations.")]
		public MlNeuralNetRnn_WeightInitMode WeightInit { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 05, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. CloseToClose: y = 1 if Close at end of LabelHorizon window is above Close at trainBar. FavorableExcursion: y = 1 if MFE_long beat MFE_short during the window (uses bar highs/lows). Skips bars below Min Favorable Move (chop).")]
		public MlNeuralNetRnn_LabelMode LabelMode { get; set; }

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

		// ─── RNN state ───
		// Input-to-hidden weights:  Wxh[h][k]  — k indexes the feature vector
		// Recurrent (hidden-to-hidden) weights: Wrec[h][j] — j indexes the previous hidden state
		// Hidden bias:              bh[h]
		// Output weights:           Wo[h]
		// Output bias:              bo
		private double[][]	Wxh;
		private double[][]	Wrec;
		private double[]	bh;
		private double[]	Wo;
		private double		bo;

		// Persistent live hidden state — carried forward across bars and used for
		// the live prediction at bar 0. Walks forward indefinitely.
		private double[]	hLive;

		// Rolling buffer of recent normalized feature vectors. Length = BpttWindow
		// + LabelHorizon. Index 0 = most recent bar; index (BpttWindow + LabelHorizon - 1)
		// = oldest bar still tracked.
		private double[][]	featureRing;
		private int			ringFilled;     // how many entries are currently filled (cap = buffer length)

		// Hidden-state sequence for the BPTT unrolled window:
		//   hSeq[0]     = anchor (zero state at the start of each training window).
		//   hSeq[t + 1] = hidden state after processing bar t of the window (t in 0..BpttWindow-1).
		//   hSeq[BpttWindow] = final state, fed into the output sigmoid.
		private double[][]	hSeq;

		// Gradient accumulators for BPTT — one slot per weight, summed across the unrolled window.
		private double[][]	gWxh;
		private double[][]	gWrec;
		private double[]	gbh;
		private double[]	gWo;

		// Per-step scratch arrays, allocated once in DataLoaded and reused on every bar / BPTT step.
		private double[]	hPrevScratch;       // snapshot of hLive before in-place update in StepHidden
		private double[]	dhNext;             // dh flowing backward from the next BPTT step
		private double[]	dzScratch;          // dz_t for the current BPTT step
		private double[]	dhPrevScratch;      // dh propagated to the previous step (Wrec^T · dz)

		// Scratch for feature extraction / normalization.
		private double[]	scratchRaw;
		private double[]	scratchNorm;

		// Cooldown tracker — bar index of the last fired signal. -1 means "no signal yet."
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
				WeightInit				= MlNeuralNetRnn_WeightInitMode.Random;
				LabelMode				= MlNeuralNetRnn_LabelMode.CloseToClose;
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

				// Network weights
				Wxh	= new double[HiddenSize][];
				for (int h = 0; h < HiddenSize; h++) Wxh[h] = new double[NumFeatures];
				Wrec = new double[HiddenSize][];
				for (int h = 0; h < HiddenSize; h++) Wrec[h] = new double[HiddenSize];
				bh = new double[HiddenSize];
				Wo = new double[HiddenSize];

				InitializeWeights();

				// State + buffers
				hLive		= new double[HiddenSize];
				int ringSize = BpttWindow + LabelHorizon;
				featureRing	= new double[ringSize][];
				for (int t = 0; t < ringSize; t++) featureRing[t] = new double[NumFeatures];
				ringFilled	= 0;

				// BPTT scratch
				hSeq	= new double[BpttWindow + 1][];
				for (int t = 0; t < BpttWindow + 1; t++) hSeq[t] = new double[HiddenSize];
				gWxh	= new double[HiddenSize][];
				for (int h = 0; h < HiddenSize; h++) gWxh[h] = new double[NumFeatures];
				gWrec	= new double[HiddenSize][];
				for (int h = 0; h < HiddenSize; h++) gWrec[h] = new double[HiddenSize];
				gbh		= new double[HiddenSize];
				gWo		= new double[HiddenSize];
				hPrevScratch	= new double[HiddenSize];
				dhNext			= new double[HiddenSize];
				dzScratch		= new double[HiddenSize];
				dhPrevScratch	= new double[HiddenSize];

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];

				lastSignalBar	= -1;
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;
			}
		}

		#endregion

		// ─── WEIGHT INITIALIZATION ────────────────────────────────────────────────
		// Xavier/Glorot scaling for tanh activations:
		//   Wxh:  sqrt(2 / (NumFeatures + HiddenSize))   — input → hidden
		//   Wrec: sqrt(2 / (HiddenSize + HiddenSize))    — hidden → hidden
		//   Wo:   sqrt(2 / (HiddenSize + 1))             — hidden → output sigmoid
		// The recurrent matrix Wrec is the trickiest — values too large make the
		// recurrent loop amplify on each step (exploding state); too small and the
		// network forgets immediately. Xavier scaling lands in a workable range
		// for vanilla RNNs at this depth.

		private void InitializeWeights()
		{
			if (WeightInit == MlNeuralNetRnn_WeightInitMode.Random)
			{
				var rng = new System.Random(RandomSeed);

				double scaleXh   = Math.Sqrt(2.0 / (NumFeatures + HiddenSize));
				double scaleRec  = Math.Sqrt(2.0 / (HiddenSize + HiddenSize));
				double scaleOut  = Math.Sqrt(2.0 / (HiddenSize + 1));

				for (int h = 0; h < HiddenSize; h++)
				{
					for (int k = 0; k < NumFeatures; k++)
						Wxh[h][k] = SampleNormal(rng) * scaleXh;
					for (int j = 0; j < HiddenSize; j++)
						Wrec[h][j] = SampleNormal(rng) * scaleRec;
					bh[h] = 0;
					Wo[h] = SampleNormal(rng) * scaleOut;
				}
				bo = 0;
			}
			else // Zero — strongly discouraged for RNN; included for completeness.
			{
				for (int h = 0; h < HiddenSize; h++)
				{
					for (int k = 0; k < NumFeatures; k++) Wxh[h][k] = 0;
					for (int j = 0; j < HiddenSize; j++) Wrec[h][j] = 0;
					bh[h] = 0;
					Wo[h] = 0;
				}
				bo = 0;
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
			//    Walks the persistent hidden state hLive one step forward.
			NormalizeInto(scratchNorm, scratchRaw, 0);
			StepHidden(hLive, scratchNorm);     // hLive ← tanh(Wxh·x + Wrec·hLive + bh)
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

				if (LabelMode == MlNeuralNetRnn_LabelMode.CloseToClose)
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
			//    label is observable so the recurrent weights have time to settle.
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
		// Updates h in place: h ← tanh(Wxh · x + Wrec · h + bh).
		// h is snapshotted into hPrevScratch first so reads of h_prev stay consistent
		// while the new h values are written element-by-element.

		private void StepHidden(double[] h, double[] x)
		{
			for (int j = 0; j < HiddenSize; j++) hPrevScratch[j] = h[j];

			for (int hh = 0; hh < HiddenSize; hh++)
			{
				double z = bh[hh];
				for (int k = 0; k < NumFeatures; k++) z += Wxh[hh][k] * x[k];
				for (int j = 0; j < HiddenSize;  j++) z += Wrec[hh][j] * hPrevScratch[j];
				h[hh] = Math.Tanh(z);
			}
		}

		// ─── OUTPUT (final sigmoid) ───────────────────────────────────────────────

		private double OutputSigmoid(double[] h)
		{
			double z = bo;
			for (int hh = 0; hh < HiddenSize; hh++) z += Wo[hh] * h[hh];
			return Sigmoid(z);
		}

		// ─── TRUNCATED BPTT ───────────────────────────────────────────────────────
		// Recompute the forward pass over the BPTT window ending at trainBar, then
		// backprop the gradient through every time step in that window. The "anchor"
		// hidden state at the start of the window is reset to zero each call — a
		// common simplification for online truncated BPTT. The live persistent state
		// (hLive) is unaffected; it continues forward indefinitely outside training.
		//
		// Window indexing (feature ring is 0 = newest):
		//   trainBar lives at ring index LabelHorizon.
		//   The BPTT window covers ring indices (LabelHorizon + BpttWindow - 1) down to
		//   LabelHorizon — BpttWindow bars total, with the trainBar at the right end.

		private void TruncatedBptt(double y)
		{
			// Forward pass over the BPTT window.
			// hSeq[0] = anchor (zero); hSeq[t+1] = hidden state after processing bar t.
			for (int j = 0; j < HiddenSize; j++) hSeq[0][j] = 0.0;

			for (int t = 0; t < BpttWindow; t++)
			{
				int ringIdx = LabelHorizon + BpttWindow - 1 - t;   // oldest first
				double[] xt = featureRing[ringIdx];
				double[] hPrev = hSeq[t];
				double[] hCur  = hSeq[t + 1];
				for (int hh = 0; hh < HiddenSize; hh++)
				{
					double z = bh[hh];
					for (int k = 0; k < NumFeatures; k++) z += Wxh[hh][k] * xt[k];
					for (int j = 0; j < HiddenSize;  j++) z += Wrec[hh][j] * hPrev[j];
					hCur[hh] = Math.Tanh(z);
				}
			}

			// Output and error at the end of the window.
			double pTrain = OutputSigmoid(hSeq[BpttWindow]);
			double error  = pTrain - y;   // dL/dzOut for cross-entropy + sigmoid

			// Zero the gradient accumulators.
			for (int hh = 0; hh < HiddenSize; hh++)
			{
				for (int k = 0; k < NumFeatures; k++) gWxh[hh][k]  = 0.0;
				for (int j = 0; j < HiddenSize;  j++) gWrec[hh][j] = 0.0;
				gbh[hh] = 0.0;
			}

			// Output-layer gradients (computed using OLD Wo before we update it).
			// Also seed dhNext[h] = error * Wo[h] — gradient flowing back from output
			// into the LAST hidden state hSeq[BpttWindow].
			for (int hh = 0; hh < HiddenSize; hh++)
			{
				gWo[hh] = error * hSeq[BpttWindow][hh] + RegularizationLambda * Wo[hh];
				dhNext[hh] = error * Wo[hh];
			}
			double gbo = error;

			// BPTT — walk backward through the unrolled time steps.
			// At step t (1..BpttWindow), we know dh at hSeq[t] (call it dh_t).
			// Compute dz_t = dh_t * (1 - h_t^2)  [tanh derivative]
			// Accumulate gradients into gWxh[h][k] += dz_t[h] * x_t[k]
			//                          gWrec[h][j] += dz_t[h] * h_{t-1}[j]
			//                          gbh[h]      += dz_t[h]
			// Propagate dh_{t-1}[j] = sum_h(Wrec[h][j] * dz_t[h]).
			for (int t = BpttWindow; t >= 1; t--)
			{
				int ringIdx = LabelHorizon + BpttWindow - 1 - (t - 1);  // x at step t = ring[LabelHorizon + BpttWindow - t]
				double[] xt     = featureRing[ringIdx];
				double[] hPrev  = hSeq[t - 1];
				double[] hCur   = hSeq[t];

				// dz_t[h] = dhNext[h] * (1 - h_t[h]^2)
				for (int hh = 0; hh < HiddenSize; hh++)
					dzScratch[hh] = dhNext[hh] * (1.0 - hCur[hh] * hCur[hh]);

				// Accumulate weight gradients for this step.
				for (int hh = 0; hh < HiddenSize; hh++)
				{
					double dzh = dzScratch[hh];
					for (int k = 0; k < NumFeatures; k++)
						gWxh[hh][k] += dzh * xt[k];
					for (int j = 0; j < HiddenSize; j++)
						gWrec[hh][j] += dzh * hPrev[j];
					gbh[hh] += dzh;
				}

				// Propagate gradient to hPrev → becomes dhNext for the next iteration.
				// Uses OLD Wrec — weight updates happen AFTER this loop completes.
				// Skip at t == 1 because hSeq[0] is the fixed zero anchor (no gradient flows past it).
				if (t > 1)
				{
					for (int j = 0; j < HiddenSize; j++)
					{
						double s = 0.0;
						for (int hh = 0; hh < HiddenSize; hh++)
							s += Wrec[hh][j] * dzScratch[hh];
						dhPrevScratch[j] = s;
					}
					for (int j = 0; j < HiddenSize; j++) dhNext[j] = dhPrevScratch[j];
				}
			}

			// Apply weight updates with L2 regularization on weight matrices
			// (biases get no L2, per convention).
			for (int hh = 0; hh < HiddenSize; hh++)
			{
				for (int k = 0; k < NumFeatures; k++)
				{
					double grad = gWxh[hh][k] + RegularizationLambda * Wxh[hh][k];
					Wxh[hh][k] -= LearningRate * grad;
				}
				for (int j = 0; j < HiddenSize; j++)
				{
					double grad = gWrec[hh][j] + RegularizationLambda * Wrec[hh][j];
					Wrec[hh][j] -= LearningRate * grad;
				}
				bh[hh] -= LearningRate * gbh[hh];
				Wo[hh] -= LearningRate * gWo[hh];
			}
			bo -= LearningRate * gbo;
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
			string tag  = (isLong ? "rnn-long-lbl-" : "rnn-short-lbl-") + CurrentBar;

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
		private indMyDailyTake.MlNeuralNetRnn[] cacheMlNeuralNetRnn;
		public indMyDailyTake.MlNeuralNetRnn MlNeuralNetRnn(int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetRnn_WeightInitMode weightInit, MlNeuralNetRnn_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlNeuralNetRnn(Input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlNeuralNetRnn MlNeuralNetRnn(ISeries<double> input, int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetRnn_WeightInitMode weightInit, MlNeuralNetRnn_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlNeuralNetRnn != null)
				for (int idx = 0; idx < cacheMlNeuralNetRnn.Length; idx++)
					if (cacheMlNeuralNetRnn[idx] != null && cacheMlNeuralNetRnn[idx].HiddenSize == hiddenSize && cacheMlNeuralNetRnn[idx].BpttWindow == bpttWindow && cacheMlNeuralNetRnn[idx].RandomSeed == randomSeed && cacheMlNeuralNetRnn[idx].LearningRate == learningRate && cacheMlNeuralNetRnn[idx].RegularizationLambda == regularizationLambda && cacheMlNeuralNetRnn[idx].LabelHorizon == labelHorizon && cacheMlNeuralNetRnn[idx].WeightInit == weightInit && cacheMlNeuralNetRnn[idx].LabelMode == labelMode && cacheMlNeuralNetRnn[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlNeuralNetRnn[idx].MaPeriod == maPeriod && cacheMlNeuralNetRnn[idx].AtrPeriod == atrPeriod && cacheMlNeuralNetRnn[idx].SlopeLookback == slopeLookback && cacheMlNeuralNetRnn[idx].NormalizeFeatures == normalizeFeatures && cacheMlNeuralNetRnn[idx].NormalizationLookback == normalizationLookback && cacheMlNeuralNetRnn[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlNeuralNetRnn[idx].SignalCooldownBars == signalCooldownBars && cacheMlNeuralNetRnn[idx].EqualsInput(input))
						return cacheMlNeuralNetRnn[idx];
			return CacheIndicator<indMyDailyTake.MlNeuralNetRnn>(new indMyDailyTake.MlNeuralNetRnn(){ HiddenSize = hiddenSize, BpttWindow = bpttWindow, RandomSeed = randomSeed, LearningRate = learningRate, RegularizationLambda = regularizationLambda, LabelHorizon = labelHorizon, WeightInit = weightInit, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlNeuralNetRnn);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetRnn MlNeuralNetRnn(int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetRnn_WeightInitMode weightInit, MlNeuralNetRnn_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetRnn(Input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetRnn MlNeuralNetRnn(ISeries<double> input , int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetRnn_WeightInitMode weightInit, MlNeuralNetRnn_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetRnn(input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetRnn MlNeuralNetRnn(int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetRnn_WeightInitMode weightInit, MlNeuralNetRnn_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetRnn(Input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetRnn MlNeuralNetRnn(ISeries<double> input , int hiddenSize, int bpttWindow, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetRnn_WeightInitMode weightInit, MlNeuralNetRnn_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetRnn(input, hiddenSize, bpttWindow, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
