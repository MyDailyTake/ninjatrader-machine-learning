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

// Write-up: https://mydailytake.com/ml-single-hidden-layer-neural-network-your-first-multi-layer-net-for-ninjatrader-8/
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

public enum MlNeuralNetSingleHidden_WeightInitMode { Zero, Random }
public enum MlNeuralNetSingleHidden_LabelMode { CloseToClose, FavorableExcursion }
public enum MlNeuralNetSingleHidden_Activation { Tanh, ReLU, Sigmoid }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Architecture",	10100)]
	[Gui.CategoryOrder("Learning",		10200)]
	[Gui.CategoryOrder("Features",		10300)]
	[Gui.CategoryOrder("Signal",		10400)]
	[Gui.CategoryOrder("Display",		10500)]

	#endregion

	public class MlNeuralNetSingleHidden : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Neural Net (Single Hidden Layer)";
		public string indDescription	= "Your first multi-layer neural network for NinjaTrader 8 — three input features feed into a hidden layer of H neurons (default 6, tanh activation), which feeds a single sigmoid output that produces P(up). Trained online via backpropagation: every bar, the prediction error flows backward through both layers, and each weight gets nudged by its share of the blame. Adds non-linear capacity over the single-neuron sibling — the hidden layer can learn cross-feature interactions like 'extreme distFromMa AND positive slope means continuation, but extreme distFromMa AND negative slope means reversal' — patterns a single neuron cannot represent. Default features (same as the k-NN and OLR siblings for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Features are z-score normalized using each bar's own local-time stats. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the OLR sibling. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Architecture

		[NinjaScriptProperty]
		[Range(2, 32)]
		[Display(Order = 01, GroupName = "Architecture", Name = "Hidden Neurons", Description = "Number of neurons in the hidden layer. 6 (= 2 × input features) is the conventional starting point for shallow networks. Bump up for richer feature interactions; bump down if signals look noisy on small datasets.")]
		public int HiddenNeurons { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 02, GroupName = "Architecture", Name = "Hidden Activation", Description = "Activation function on the hidden layer. Tanh: zero-centered, smooth, the safest default for online learning. ReLU: modern default for deep nets but can suffer from dead neurons under online single-step gradient descent. Sigmoid: classic but suffers from vanishing-gradient at extremes. For a single hidden layer, all three work; Tanh is recommended.")]
		public MlNeuralNetSingleHidden_Activation HiddenActivation { get; set; }

		[NinjaScriptProperty]
		[Range(0, 999999)]
		[Display(Order = 03, GroupName = "Architecture", Name = "Random Seed", Description = "Seed for the random weight initialization (when Weight Init = Random). Same seed produces the same starting weights — useful for reproducible testing or comparing different architectures fairly. Change to explore whether the model converges differently from a different starting point. Has no effect when Weight Init = Zero.")]
		public int RandomSeed { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(0.0001, 1.0)]
		[Display(Order = 01, GroupName = "Learning", Name = "Learning Rate", Description = "Step size for each weight update. Larger = adapts faster but jitters; smaller = smoother but slower to react. With more parameters than the single-neuron sibling, slightly smaller values (0.005-0.01) often produce more stable training.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.1)]
		[Display(Order = 02, GroupName = "Learning", Name = "Regularization Lambda (L2)", Description = "L2 penalty on weight magnitude. Pulls weights gently toward zero each update so they don't drift to extreme values. With more parameters, regularization matters more — recommended 0.0001 to 0.001.")]
		public double RegularizationLambda { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. The model updates each bar using the feature vector from N bars ago, whose forward direction is now known.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Weight Init", Description = "Zero starts every weight at 0 — but ALL hidden neurons would learn identical weights (symmetry problem). Strongly recommend Random for multi-neuron networks. Random uses activation-aware scaling: He scaling for ReLU, Xavier/Glorot for tanh and sigmoid.")]
		public MlNeuralNetSingleHidden_WeightInitMode WeightInit { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 05, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. CloseToClose: y = 1 if Close at end of LabelHorizon window is above Close at trainBar. FavorableExcursion: y = 1 if MFE_long beat MFE_short during the window (uses bar highs/lows). Skips bars below Min Favorable Move (chop).")]
		public MlNeuralNetSingleHidden_LabelMode LabelMode { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 5.0)]
		[Display(Order = 06, GroupName = "Learning", Name = "Min Favorable Move (ATRs)", Description = "ONLY USED WHEN Label Mode = FavorableExcursion. Minimum favorable excursion (in ATRs at entry) required during the post-bar window for the model to update. Has no effect when Label Mode = CloseToClose.")]
		public double MinFavorableMoveAtrs { get; set; }

		#endregion

		#region Features

		[NinjaScriptProperty]
		[Range(2, 500)]
		[Display(Order = 01, GroupName = "Features", Name = "MA Period", Description = "Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio.")]
		public int MaPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 100)]
		[Display(Order = 02, GroupName = "Features", Name = "ATR Period", Description = "Period of the ATR used to scale every feature into volatility units, so distance comparisons stay consistent across regimes.")]
		public int AtrPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 200)]
		[Display(Order = 03, GroupName = "Features", Name = "Slope Lookback (bars)", Description = "Number of bars over which the slope feature is measured: (Close[0] − Close[N]) / ATR.")]
		public int SlopeLookback { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features (Z-Score)", Description = "Master toggle for z-score normalization. When ON, each feature is rescaled against its own historical rolling stats. Recommended ON.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Window used to compute the rolling mean / stddev that z-score the features. Each bar uses its own local-time stats.")]
		public int NormalizationLookback { get; set; }

		#endregion

		#region Signal

		[NinjaScriptProperty]
		[Range(0.0, 0.49)]
		[Display(Order = 01, GroupName = "Signal", Name = "Min Probability Edge", Description = "How far the predicted probability of an up move must be from 0.5 before a signal fires. 0.10 means: long fires when P(up) > 0.60, short fires when P(up) < 0.40.")]
		public double MinProbabilityEdge { get; set; }

		[NinjaScriptProperty]
		[Range(0, 500)]
		[Display(Order = 02, GroupName = "Signal", Name = "Signal Cooldown (bars)", Description = "Minimum bars between consecutive signals. Higher values space signals out so the chart stays readable. Set to 0 to fire on every qualifying bar.")]
		public int SignalCooldownBars { get; set; }

		#endregion

		#region Display

		[Display(Order = 01, GroupName = "Display", Name = "Marker Offset (ticks)", Description = "Vertical offset of the signal triangle from the bar's high (shorts) / low (longs), in ticks.")]
		[Range(0, 200)]
		public int MarkerOffsetTicks { get; set; }

		[Display(Order = 02, GroupName = "Display", Name = "Label Offset (ticks)", Description = "Distance from the bar to the text label, in ticks. Should be larger than Marker Offset so the label sits beyond the triangle.")]
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

		// ─── Network state ───
		// Hidden-layer weights: Wh[h][i] = weight from input i to hidden neuron h.
		private double[][]	Wh;
		private double[]	bh;        // hidden biases, length = HiddenNeurons

		// Output-layer weights: Wo[h] = weight from hidden neuron h to output.
		private double[]	Wo;
		private double		bo;        // output bias

		// Forward-pass scratch — preserved across the bar so the backward pass can use them.
		private double[]	scratchRaw;
		private double[]	scratchNorm;       // normalized inputs (NumFeatures)
		private double[]	hiddenPre;         // pre-activation z values for hidden layer (HiddenNeurons)
		private double[]	hiddenAct;         // post-activation values for hidden layer (HiddenNeurons)

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

				HiddenNeurons			= 6;
				HiddenActivation		= MlNeuralNetSingleHidden_Activation.Tanh;
				RandomSeed				= 42;

				LearningRate			= 0.01;
				RegularizationLambda	= 0.0001;
				LabelHorizon			= 2;
				WeightInit				= MlNeuralNetSingleHidden_WeightInitMode.Random;
				LabelMode				= MlNeuralNetSingleHidden_LabelMode.CloseToClose;
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

				// Allocate network state.
				Wh = new double[HiddenNeurons][];
				for (int h = 0; h < HiddenNeurons; h++)
					Wh[h] = new double[NumFeatures];
				bh = new double[HiddenNeurons];
				Wo = new double[HiddenNeurons];

				InitializeWeights();

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];
				hiddenPre	= new double[HiddenNeurons];
				hiddenAct	= new double[HiddenNeurons];

				lastSignalBar	= -1;
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;
			}
		}

		#endregion

		// ─── WEIGHT INITIALIZATION ────────────────────────────────────────────────
		// Zero init causes a symmetry problem: all hidden neurons get identical
		// gradients on the first update step and learn identical weights forever.
		// Random init breaks symmetry. Init scale is activation-aware:
		//   - He scaling     (sqrt(2 / fan_in))             for ReLU hidden layers
		//   - Xavier scaling (sqrt(2 / (fan_in + fan_out))) for tanh / sigmoid hidden layers
		// The output layer is always sigmoid for binary classification, so it uses
		// Xavier scaling regardless of which hidden activation was selected.

		private void InitializeWeights()
		{
			if (WeightInit == MlNeuralNetSingleHidden_WeightInitMode.Random)
			{
				var rng = new System.Random(RandomSeed);

				// Hidden-layer init scale — He for ReLU, Xavier for tanh / sigmoid.
				double scaleH = HiddenActivation == MlNeuralNetSingleHidden_Activation.ReLU
					? Math.Sqrt(2.0 / NumFeatures)
					: Math.Sqrt(2.0 / (NumFeatures + HiddenNeurons));

				for (int h = 0; h < HiddenNeurons; h++)
				{
					for (int i = 0; i < NumFeatures; i++)
						Wh[h][i] = SampleNormal(rng) * scaleH;
					bh[h] = 0;
				}

				// Output layer: sigmoid activation → Xavier scaling.
				double scaleO = Math.Sqrt(2.0 / (HiddenNeurons + 1));
				for (int h = 0; h < HiddenNeurons; h++)
					Wo[h] = SampleNormal(rng) * scaleO;
				bo = 0;
			}
			else // Zero — included for completeness, but breaks multi-neuron training.
			{
				for (int h = 0; h < HiddenNeurons; h++)
				{
					for (int i = 0; i < NumFeatures; i++) Wh[h][i] = 0;
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

			// 1) Forward pass on the current bar — used for the prediction the user sees.
			NormalizeInto(scratchNorm, scratchRaw, 0);
			double pUp = ForwardPass(scratchNorm);

			// 2) Update step — applied to the bar from LabelHorizon ago, whose forward
			//    window is now observable.
			int trainBar = LabelHorizon;
			if (CurrentBar >= predictionWarmup + LabelHorizon)
			{
				bool   trainThisBar	= false;
				double y			= 0.0;

				if (LabelMode == MlNeuralNetSingleHidden_LabelMode.CloseToClose)
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
						double h = High[b];
						double l = Low[b];
						if (h > maxHigh) maxHigh = h;
						if (l < minLow)  minLow  = l;
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
				{
					GetFeaturesInto(scratchRaw, trainBar);
					NormalizeInto(scratchNorm, scratchRaw, trainBar);

					// Forward pass at trainBar with current weights (reuses scratchNorm).
					double pTrain = ForwardPass(scratchNorm);

					// Backward pass — applies one step of gradient descent.
					Backprop(scratchNorm, pTrain, y);
				}
			}

			// 3) Public-Series outputs.
			sProbabilityUp[0] = pUp;
			sConfidence[0]    = Math.Abs(pUp - 0.5) * 2.0;

			// 4) Signal gate — wait for at least 50 update steps after the first
			//    label is observable so the network has time to settle.
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

		// ─── FORWARD PASS ─────────────────────────────────────────────────────────
		// inputs[NumFeatures] → hiddenPre[HiddenNeurons] → hiddenAct[HiddenNeurons]
		// → output → sigmoid → P(up).
		// Caches hiddenPre and hiddenAct on instance fields so the backward pass can
		// reuse them without recomputing.

		private double ForwardPass(double[] inputs)
		{
			// Hidden layer: pre-activation = Wh · inputs + bh, then activation.
			for (int h = 0; h < HiddenNeurons; h++)
			{
				double z = bh[h];
				for (int i = 0; i < NumFeatures; i++)
					z += Wh[h][i] * inputs[i];
				hiddenPre[h] = z;
				hiddenAct[h] = Activate(z);
			}

			// Output layer: zOut = Wo · hiddenAct + bo, then sigmoid.
			double zOut = bo;
			for (int h = 0; h < HiddenNeurons; h++)
				zOut += Wo[h] * hiddenAct[h];

			return Sigmoid(zOut);
		}

		// ─── BACKPROP ─────────────────────────────────────────────────────────────
		// Standard chain-rule gradient computation through both layers using the
		// cross-entropy loss with sigmoid output (which simplifies the output-layer
		// gradient to error = pTrain - y).

		private void Backprop(double[] inputs, double pTrain, double y)
		{
			double error = pTrain - y;   // dL/dzOut for cross-entropy + sigmoid

			// Output-layer gradients: dL/dWo[h] = error * hiddenAct[h], dL/dbo = error.
			// Update output weights AND simultaneously compute the gradient flowing
			// back into the hidden layer for each h: dL/dhiddenAct[h] = error * Wo[h].
			// We need the OLD Wo[h] for the back-propagation, so cache before update.
			for (int h = 0; h < HiddenNeurons; h++)
			{
				double dHiddenAct = error * Wo[h];   // gradient on hiddenAct[h] from output side
				double dHiddenPre = dHiddenAct * ActivationDerivative(hiddenPre[h], hiddenAct[h]);

				// Hidden-layer weight updates: dL/dWh[h][i] = dHiddenPre * inputs[i]
				for (int i = 0; i < NumFeatures; i++)
				{
					double grad = dHiddenPre * inputs[i] + RegularizationLambda * Wh[h][i];
					Wh[h][i] -= LearningRate * grad;
				}
				bh[h] -= LearningRate * dHiddenPre;

				// Output weight update (after the back-prop above is computed).
				double gradWo = error * hiddenAct[h] + RegularizationLambda * Wo[h];
				Wo[h] -= LearningRate * gradWo;
			}

			bo -= LearningRate * error;
		}

		// ─── ACTIVATION + DERIVATIVE ──────────────────────────────────────────────
		// Tanh: f'(z) = 1 - tanh(z)^2 = 1 - act^2  (uses post-activation value)
		// ReLU: f'(z) = 1 if z > 0 else 0  (uses pre-activation value)
		// Sigmoid: f'(z) = sigmoid(z) * (1 - sigmoid(z)) = act * (1 - act)

		private double Activate(double z)
		{
			switch (HiddenActivation)
			{
				case MlNeuralNetSingleHidden_Activation.ReLU:
					return z > 0 ? z : 0;
				case MlNeuralNetSingleHidden_Activation.Sigmoid:
					return Sigmoid(z);
				case MlNeuralNetSingleHidden_Activation.Tanh:
				default:
					return Math.Tanh(z);
			}
		}

		private double ActivationDerivative(double pre, double act)
		{
			switch (HiddenActivation)
			{
				case MlNeuralNetSingleHidden_Activation.ReLU:
					return pre > 0 ? 1.0 : 0.0;
				case MlNeuralNetSingleHidden_Activation.Sigmoid:
					return act * (1.0 - act);
				case MlNeuralNetSingleHidden_Activation.Tanh:
				default:
					return 1.0 - act * act;
			}
		}

		// ─── FEATURE EXTRACTION + NORMALIZATION + SIGMOID ─────────────────────────
		// Same three features as the k-NN and OLR siblings.

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
			double y = isLong
				? Low[0]  - labelOffsetPts - halfBlockHeight
				: High[0] + labelOffsetPts + halfBlockHeight;

			Brush brush = isLong ? Brushes.LimeGreen : Brushes.OrangeRed;
			string tag  = (isLong ? "nn-long-lbl-" : "nn-short-lbl-") + CurrentBar;

			Draw.Text(this, tag, false, text, 0, y, 0, brush,
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
		private indMyDailyTake.MlNeuralNetSingleHidden[] cacheMlNeuralNetSingleHidden;
		public indMyDailyTake.MlNeuralNetSingleHidden MlNeuralNetSingleHidden(int hiddenNeurons, MlNeuralNetSingleHidden_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetSingleHidden_WeightInitMode weightInit, MlNeuralNetSingleHidden_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlNeuralNetSingleHidden(Input, hiddenNeurons, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlNeuralNetSingleHidden MlNeuralNetSingleHidden(ISeries<double> input, int hiddenNeurons, MlNeuralNetSingleHidden_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetSingleHidden_WeightInitMode weightInit, MlNeuralNetSingleHidden_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlNeuralNetSingleHidden != null)
				for (int idx = 0; idx < cacheMlNeuralNetSingleHidden.Length; idx++)
					if (cacheMlNeuralNetSingleHidden[idx] != null && cacheMlNeuralNetSingleHidden[idx].HiddenNeurons == hiddenNeurons && cacheMlNeuralNetSingleHidden[idx].HiddenActivation == hiddenActivation && cacheMlNeuralNetSingleHidden[idx].RandomSeed == randomSeed && cacheMlNeuralNetSingleHidden[idx].LearningRate == learningRate && cacheMlNeuralNetSingleHidden[idx].RegularizationLambda == regularizationLambda && cacheMlNeuralNetSingleHidden[idx].LabelHorizon == labelHorizon && cacheMlNeuralNetSingleHidden[idx].WeightInit == weightInit && cacheMlNeuralNetSingleHidden[idx].LabelMode == labelMode && cacheMlNeuralNetSingleHidden[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlNeuralNetSingleHidden[idx].MaPeriod == maPeriod && cacheMlNeuralNetSingleHidden[idx].AtrPeriod == atrPeriod && cacheMlNeuralNetSingleHidden[idx].SlopeLookback == slopeLookback && cacheMlNeuralNetSingleHidden[idx].NormalizeFeatures == normalizeFeatures && cacheMlNeuralNetSingleHidden[idx].NormalizationLookback == normalizationLookback && cacheMlNeuralNetSingleHidden[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlNeuralNetSingleHidden[idx].SignalCooldownBars == signalCooldownBars && cacheMlNeuralNetSingleHidden[idx].EqualsInput(input))
						return cacheMlNeuralNetSingleHidden[idx];
			return CacheIndicator<indMyDailyTake.MlNeuralNetSingleHidden>(new indMyDailyTake.MlNeuralNetSingleHidden(){ HiddenNeurons = hiddenNeurons, HiddenActivation = hiddenActivation, RandomSeed = randomSeed, LearningRate = learningRate, RegularizationLambda = regularizationLambda, LabelHorizon = labelHorizon, WeightInit = weightInit, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlNeuralNetSingleHidden);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetSingleHidden MlNeuralNetSingleHidden(int hiddenNeurons, MlNeuralNetSingleHidden_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetSingleHidden_WeightInitMode weightInit, MlNeuralNetSingleHidden_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetSingleHidden(Input, hiddenNeurons, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetSingleHidden MlNeuralNetSingleHidden(ISeries<double> input , int hiddenNeurons, MlNeuralNetSingleHidden_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetSingleHidden_WeightInitMode weightInit, MlNeuralNetSingleHidden_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetSingleHidden(input, hiddenNeurons, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetSingleHidden MlNeuralNetSingleHidden(int hiddenNeurons, MlNeuralNetSingleHidden_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetSingleHidden_WeightInitMode weightInit, MlNeuralNetSingleHidden_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetSingleHidden(Input, hiddenNeurons, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetSingleHidden MlNeuralNetSingleHidden(ISeries<double> input , int hiddenNeurons, MlNeuralNetSingleHidden_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetSingleHidden_WeightInitMode weightInit, MlNeuralNetSingleHidden_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetSingleHidden(input, hiddenNeurons, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
