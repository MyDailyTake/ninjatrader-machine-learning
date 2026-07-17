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

// Write-up: https://mydailytake.com/ml-multi-hidden-layer-neural-network-configurable-depth-in-ninjatrader-8/
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

public enum MlNeuralNetMultiLayer_WeightInitMode { Zero, Random }
public enum MlNeuralNetMultiLayer_LabelMode { CloseToClose, FavorableExcursion }
public enum MlNeuralNetMultiLayer_Activation { Tanh, ReLU, Sigmoid }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Architecture",	10100)]
	[Gui.CategoryOrder("Learning",		10200)]
	[Gui.CategoryOrder("Features",		10300)]
	[Gui.CategoryOrder("Signal",		10400)]
	[Gui.CategoryOrder("Display",		10500)]

	#endregion

	public class MlNeuralNetMultiLayer : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Neural Net (Multi-Hidden Layer)";
		public string indDescription	= "A configurable-depth feedforward neural network for NinjaTrader 8. The architecture is a list of integers like '8, 6' (two hidden layers with 8 and 6 neurons) or '10, 6, 4' (three hidden layers); any common separator works — comma, space, hyphen, semicolon, x. The same three input features feed forward through every hidden layer to a single sigmoid output that produces P(up). Trained online via backpropagation: the prediction error flows backward through every layer, multiplied by each layer's activation derivative — this is the chain rule made literal. Adds compositional capacity over the single-hidden-layer sibling: layer 1 detects raw feature combinations, layer 2 composes those into higher-order patterns. The trade-off is vanishing gradients — at depth, tanh/sigmoid derivatives compound and updates to the earliest layer weights become tiny. ReLU partially fixes that. Default features (same as the k-NN, OLR, and SHL siblings for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Z-score normalized using each bar's own local-time stats. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the prior posts. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Architecture

		[NinjaScriptProperty]
		[Display(Order = 01, GroupName = "Architecture", Name = "Hidden Layer Sizes", Description = "Per-layer neuron counts as a list of integers. Any common separator works — comma, space, hyphen, x, semicolon. Examples: '6' = one layer of 6 neurons (matches the single-hidden-layer sibling). '8, 6' = two layers (8 then 6). '10, 6, 4' = three layers. Each value must be ≥ 2. No upper cap; the indicator allocates memory in proportion to total weights, so a sane upper limit on this hardware is ~64 neurons per layer. Invalid input falls back to default '8, 6' and logs a warning to the NT Output window.")]
		public string HiddenLayerSizes { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 02, GroupName = "Architecture", Name = "Hidden Activation", Description = "Activation function on EVERY hidden layer. Tanh: zero-centered, smooth, the safest default for shallow networks. ReLU: better at depth because its derivative is 1 (not <1), so the chain-rule product through many layers doesn't shrink. Sigmoid: classic but worst at depth — its derivative caps at 0.25, so a 3-layer net reduces signal by ~64x just from the activation. For 2+ hidden layers, ReLU is recommended.")]
		public MlNeuralNetMultiLayer_Activation HiddenActivation { get; set; }

		[NinjaScriptProperty]
		[Range(0, 999999)]
		[Display(Order = 03, GroupName = "Architecture", Name = "Random Seed", Description = "Seed for the random weight initialization (when Weight Init = Random). Same seed produces the same starting weights — useful for comparing different architectures fairly.")]
		public int RandomSeed { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(0.0001, 1.0)]
		[Display(Order = 01, GroupName = "Learning", Name = "Learning Rate", Description = "Step size for each weight update. With more layers, the gradient at early layers is smaller (vanishing-gradient effect), so a slightly larger learning rate compensates — but too large destabilizes the deeper output side. Start at 0.005-0.01 and tune from there.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.1)]
		[Display(Order = 02, GroupName = "Learning", Name = "Regularization Lambda (L2)", Description = "L2 penalty on weight magnitude. Pulls weights gently toward zero each update. With many parameters across multiple layers, regularization matters more — recommended 0.0001 to 0.001.")]
		public double RegularizationLambda { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. The model updates each bar using the feature vector from N bars ago, whose forward direction is now known.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Weight Init", Description = "Zero starts every weight at 0 — but ALL hidden neurons in a layer would learn identical weights (symmetry problem) AND all-zero activations would zero out gradients in deeper layers. Strongly recommend Random. Random uses activation-aware scaling: He scaling for ReLU, Xavier/Glorot for tanh and sigmoid.")]
		public MlNeuralNetMultiLayer_WeightInitMode WeightInit { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 05, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. CloseToClose: y = 1 if Close at end of LabelHorizon window is above Close at trainBar. FavorableExcursion: y = 1 if MFE_long beat MFE_short during the window (uses bar highs/lows). Skips bars below Min Favorable Move (chop).")]
		public MlNeuralNetMultiLayer_LabelMode LabelMode { get; set; }

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
		[Display(Order = 01, GroupName = "Signal", Name = "Min Probability Edge", Description = "How far the predicted probability of an up move must be from 0.5 before a signal fires. 0.10 means: long fires when P(up) > 0.60, short fires when P(up) < 0.40.")]
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

		// ─── Network state ───
		// Parsed architecture: layerSizes[l] = neuron count in hidden layer l.
		// numHiddenLayers = layerSizes.Length.
		private int[]	layerSizes;
		private int		numHiddenLayers;

		// Hidden weights: Wh[l][h][k] = weight from input k of layer l to neuron h of layer l.
		// For layer 0, "input k" indexes scratchNorm (size NumFeatures).
		// For layer l > 0, "input k" indexes hiddenAct[l-1] (size layerSizes[l-1]).
		private double[][][]	Wh;
		private double[][]		bh;        // bh[l][h] = bias of neuron h in layer l

		// Output-layer weights: Wo[h] = weight from last hidden layer's neuron h to output.
		private double[]	Wo;
		private double		bo;        // output bias

		// Forward-pass scratch — preserved across the bar so the backward pass can reuse them.
		private double[]	scratchRaw;
		private double[]	scratchNorm;       // normalized inputs (NumFeatures)
		private double[][]	hiddenPre;         // hiddenPre[l][h] = pre-activation z value
		private double[][]	hiddenAct;         // hiddenAct[l][h] = post-activation value

		// Backprop scratch — gradient signal at each layer's activations / pre-activations.
		private double[][]	dHiddenAct;
		private double[][]	dHiddenPre;

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

				HiddenLayerSizes		= "8, 6";
				HiddenActivation		= MlNeuralNetMultiLayer_Activation.ReLU;
				RandomSeed				= 42;

				LearningRate			= 0.005;
				RegularizationLambda	= 0.0005;
				LabelHorizon			= 2;
				WeightInit				= MlNeuralNetMultiLayer_WeightInitMode.Random;
				LabelMode				= MlNeuralNetMultiLayer_LabelMode.CloseToClose;
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
				layerSizes			= ParseHiddenLayerSizes(HiddenLayerSizes);
				numHiddenLayers		= layerSizes.Length;

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

				// Allocate network state for each hidden layer.
				Wh			= new double[numHiddenLayers][][];
				bh			= new double[numHiddenLayers][];
				hiddenPre	= new double[numHiddenLayers][];
				hiddenAct	= new double[numHiddenLayers][];
				dHiddenAct	= new double[numHiddenLayers][];
				dHiddenPre	= new double[numHiddenLayers][];

				for (int l = 0; l < numHiddenLayers; l++)
				{
					int sizeIn = (l == 0) ? NumFeatures : layerSizes[l - 1];
					int sizeOut = layerSizes[l];

					Wh[l]			= new double[sizeOut][];
					for (int h = 0; h < sizeOut; h++) Wh[l][h] = new double[sizeIn];
					bh[l]			= new double[sizeOut];
					hiddenPre[l]	= new double[sizeOut];
					hiddenAct[l]	= new double[sizeOut];
					dHiddenAct[l]	= new double[sizeOut];
					dHiddenPre[l]	= new double[sizeOut];
				}

				// Output layer reads from the last hidden layer.
				int lastHidden = layerSizes[numHiddenLayers - 1];
				Wo = new double[lastHidden];

				InitializeWeights();

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];

				lastSignalBar	= -1;
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;
			}
		}

		#endregion

		// ─── ARCHITECTURE PARSING ─────────────────────────────────────────────────
		// Forgiving parser: any non-digit run (comma, space, hyphen, semicolon, x, "by",
		// any combination thereof) is treated as a separator. Each parsed integer must
		// be ≥ 2. Invalid input falls back to default and logs a warning to the NT
		// Output window — the indicator never throws on bad property values.

		private static readonly int[] DefaultLayers = new int[] { 8, 6 };

		private int[] ParseHiddenLayerSizes(string s)
		{
			if (string.IsNullOrWhiteSpace(s))
			{
				Print(string.Format("{0}: Hidden Layer Sizes is empty, using default '8, 6'.", indName));
				return (int[])DefaultLayers.Clone();
			}

			// Walk the string, accumulating digit runs into tokens.
			var tokens = new System.Collections.Generic.List<string>();
			var sb = new System.Text.StringBuilder();
			foreach (char c in s)
			{
				if (c >= '0' && c <= '9') sb.Append(c);
				else if (sb.Length > 0) { tokens.Add(sb.ToString()); sb.Clear(); }
			}
			if (sb.Length > 0) tokens.Add(sb.ToString());

			if (tokens.Count == 0)
			{
				Print(string.Format("{0}: Hidden Layer Sizes '{1}' contains no integers, using default '8, 6'.", indName, s));
				return (int[])DefaultLayers.Clone();
			}

			var sizes = new int[tokens.Count];
			for (int i = 0; i < tokens.Count; i++)
			{
				if (!int.TryParse(tokens[i], out sizes[i]) || sizes[i] < 2)
				{
					Print(string.Format("{0}: Hidden Layer Sizes '{1}' has invalid value '{2}' (must be integer ≥ 2), using default '8, 6'.", indName, s, tokens[i]));
					return (int[])DefaultLayers.Clone();
				}
			}
			return sizes;
		}

		// ─── WEIGHT INITIALIZATION ────────────────────────────────────────────────
		// Per-layer activation-aware scaling:
		//   - He scaling     (sqrt(2 / fan_in))             for ReLU hidden layers
		//   - Xavier scaling (sqrt(2 / (fan_in + fan_out))) for tanh / sigmoid hidden layers
		// The output layer is always sigmoid and uses Xavier regardless.

		private void InitializeWeights()
		{
			if (WeightInit == MlNeuralNetMultiLayer_WeightInitMode.Random)
			{
				var rng = new System.Random(RandomSeed);

				for (int l = 0; l < numHiddenLayers; l++)
				{
					int fanIn  = (l == 0) ? NumFeatures : layerSizes[l - 1];
					int fanOut = layerSizes[l];
					double scale = HiddenActivation == MlNeuralNetMultiLayer_Activation.ReLU
						? Math.Sqrt(2.0 / fanIn)
						: Math.Sqrt(2.0 / (fanIn + fanOut));

					for (int h = 0; h < fanOut; h++)
					{
						for (int k = 0; k < fanIn; k++)
							Wh[l][h][k] = SampleNormal(rng) * scale;
						bh[l][h] = 0;
					}
				}

				int lastHidden = layerSizes[numHiddenLayers - 1];
				double scaleO = Math.Sqrt(2.0 / (lastHidden + 1));
				for (int h = 0; h < lastHidden; h++)
					Wo[h] = SampleNormal(rng) * scaleO;
				bo = 0;
			}
			else // Zero — strongly discouraged for multi-layer; included for completeness.
			{
				for (int l = 0; l < numHiddenLayers; l++)
				{
					for (int h = 0; h < layerSizes[l]; h++)
					{
						for (int k = 0; k < Wh[l][h].Length; k++) Wh[l][h][k] = 0;
						bh[l][h] = 0;
					}
				}
				for (int h = 0; h < Wo.Length; h++) Wo[h] = 0;
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

				if (LabelMode == MlNeuralNetMultiLayer_LabelMode.CloseToClose)
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
				{
					GetFeaturesInto(scratchRaw, trainBar);
					NormalizeInto(scratchNorm, scratchRaw, trainBar);

					// Forward pass at trainBar with current weights (reuses scratchNorm).
					double pTrain = ForwardPass(scratchNorm);

					// Backward pass — applies one step of gradient descent through every layer.
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
		// inputs (NumFeatures) → hidden layer 0 → hidden layer 1 → ... → output sigmoid.
		// Caches every layer's pre- and post-activation values so the backward pass
		// can compute gradients without re-running the forward pass.

		private double ForwardPass(double[] inputs)
		{
			for (int l = 0; l < numHiddenLayers; l++)
			{
				double[] inputToLayer = (l == 0) ? inputs : hiddenAct[l - 1];
				int sizeIn = inputToLayer.Length;
				int sizeOut = layerSizes[l];

				for (int h = 0; h < sizeOut; h++)
				{
					double z = bh[l][h];
					for (int k = 0; k < sizeIn; k++)
						z += Wh[l][h][k] * inputToLayer[k];
					hiddenPre[l][h] = z;
					hiddenAct[l][h] = Activate(z);
				}
			}

			// Output layer reads from the last hidden layer.
			int last = numHiddenLayers - 1;
			double zOut = bo;
			for (int h = 0; h < layerSizes[last]; h++)
				zOut += Wo[h] * hiddenAct[last][h];

			return Sigmoid(zOut);
		}

		// ─── BACKPROP ─────────────────────────────────────────────────────────────
		// Chain-rule gradient descent through every layer.
		// Cross-entropy loss + sigmoid output simplifies dL/dzOut to (pTrain - y).
		// Then for each hidden layer, working from the output side back to the input:
		//   1. Compute dHiddenAct[l] (gradient on this layer's outputs).
		//   2. Multiply by activation derivative to get dHiddenPre[l] (gradient on this
		//      layer's pre-activations).
		//   3. If l > 0, propagate to dHiddenAct[l-1] using the OLD Wh[l] (BEFORE updating).
		//   4. Update Wh[l] and bh[l] using dHiddenPre[l] and the layer's input.
		// Output weights Wo and bo are updated last using the (now consumed) error signal.

		private void Backprop(double[] inputs, double pTrain, double y)
		{
			double error = pTrain - y;   // dL/dzOut for cross-entropy + sigmoid

			int last = numHiddenLayers - 1;

			// Step 1: Seed the gradient at the LAST hidden layer's activations from the
			// output side. We need the OLD Wo for this — it's used both to back-propagate
			// AND to compute its own gradient — so we read OLD values into dHiddenAct
			// and store the new values for Wo separately.
			for (int h = 0; h < layerSizes[last]; h++)
				dHiddenAct[last][h] = error * Wo[h];

			// Step 2: Walk backward through the hidden layers, propagating gradient.
			for (int l = last; l >= 0; l--)
			{
				int sizeOut = layerSizes[l];
				int sizeIn  = (l == 0) ? NumFeatures : layerSizes[l - 1];
				double[] inputToLayer = (l == 0) ? inputs : hiddenAct[l - 1];

				// Convert dHiddenAct to dHiddenPre via activation derivative.
				for (int h = 0; h < sizeOut; h++)
					dHiddenPre[l][h] = dHiddenAct[l][h] * ActivationDerivative(hiddenPre[l][h], hiddenAct[l][h]);

				// If a layer below exists, propagate gradient to dHiddenAct[l-1] using the
				// CURRENT (old) Wh[l] — must do this BEFORE updating Wh[l].
				if (l > 0)
				{
					for (int k = 0; k < sizeIn; k++) dHiddenAct[l - 1][k] = 0;
					for (int h = 0; h < sizeOut; h++)
					{
						double dpre = dHiddenPre[l][h];
						for (int k = 0; k < sizeIn; k++)
							dHiddenAct[l - 1][k] += dpre * Wh[l][h][k];
					}
				}

				// Now update Wh[l] and bh[l].
				for (int h = 0; h < sizeOut; h++)
				{
					double dpre = dHiddenPre[l][h];
					for (int k = 0; k < sizeIn; k++)
					{
						double grad = dpre * inputToLayer[k] + RegularizationLambda * Wh[l][h][k];
						Wh[l][h][k] -= LearningRate * grad;
					}
					bh[l][h] -= LearningRate * dpre;
				}
			}

			// Step 3: Update output layer weights and bias.
			for (int h = 0; h < layerSizes[last]; h++)
			{
				double gradWo = error * hiddenAct[last][h] + RegularizationLambda * Wo[h];
				Wo[h] -= LearningRate * gradWo;
			}
			bo -= LearningRate * error;
		}

		// ─── ACTIVATION + DERIVATIVE ──────────────────────────────────────────────

		private double Activate(double z)
		{
			switch (HiddenActivation)
			{
				case MlNeuralNetMultiLayer_Activation.ReLU:
					return z > 0 ? z : 0;
				case MlNeuralNetMultiLayer_Activation.Sigmoid:
					return Sigmoid(z);
				case MlNeuralNetMultiLayer_Activation.Tanh:
				default:
					return Math.Tanh(z);
			}
		}

		private double ActivationDerivative(double pre, double act)
		{
			switch (HiddenActivation)
			{
				case MlNeuralNetMultiLayer_Activation.ReLU:
					return pre > 0 ? 1.0 : 0.0;
				case MlNeuralNetMultiLayer_Activation.Sigmoid:
					return act * (1.0 - act);
				case MlNeuralNetMultiLayer_Activation.Tanh:
				default:
					return 1.0 - act * act;
			}
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
			string tag  = (isLong ? "nnml-long-lbl-" : "nnml-short-lbl-") + CurrentBar;

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
		private indMyDailyTake.MlNeuralNetMultiLayer[] cacheMlNeuralNetMultiLayer;
		public indMyDailyTake.MlNeuralNetMultiLayer MlNeuralNetMultiLayer(string hiddenLayerSizes, MlNeuralNetMultiLayer_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetMultiLayer_WeightInitMode weightInit, MlNeuralNetMultiLayer_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlNeuralNetMultiLayer(Input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlNeuralNetMultiLayer MlNeuralNetMultiLayer(ISeries<double> input, string hiddenLayerSizes, MlNeuralNetMultiLayer_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetMultiLayer_WeightInitMode weightInit, MlNeuralNetMultiLayer_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlNeuralNetMultiLayer != null)
				for (int idx = 0; idx < cacheMlNeuralNetMultiLayer.Length; idx++)
					if (cacheMlNeuralNetMultiLayer[idx] != null && cacheMlNeuralNetMultiLayer[idx].HiddenLayerSizes == hiddenLayerSizes && cacheMlNeuralNetMultiLayer[idx].HiddenActivation == hiddenActivation && cacheMlNeuralNetMultiLayer[idx].RandomSeed == randomSeed && cacheMlNeuralNetMultiLayer[idx].LearningRate == learningRate && cacheMlNeuralNetMultiLayer[idx].RegularizationLambda == regularizationLambda && cacheMlNeuralNetMultiLayer[idx].LabelHorizon == labelHorizon && cacheMlNeuralNetMultiLayer[idx].WeightInit == weightInit && cacheMlNeuralNetMultiLayer[idx].LabelMode == labelMode && cacheMlNeuralNetMultiLayer[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlNeuralNetMultiLayer[idx].MaPeriod == maPeriod && cacheMlNeuralNetMultiLayer[idx].AtrPeriod == atrPeriod && cacheMlNeuralNetMultiLayer[idx].SlopeLookback == slopeLookback && cacheMlNeuralNetMultiLayer[idx].NormalizeFeatures == normalizeFeatures && cacheMlNeuralNetMultiLayer[idx].NormalizationLookback == normalizationLookback && cacheMlNeuralNetMultiLayer[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlNeuralNetMultiLayer[idx].SignalCooldownBars == signalCooldownBars && cacheMlNeuralNetMultiLayer[idx].EqualsInput(input))
						return cacheMlNeuralNetMultiLayer[idx];
			return CacheIndicator<indMyDailyTake.MlNeuralNetMultiLayer>(new indMyDailyTake.MlNeuralNetMultiLayer(){ HiddenLayerSizes = hiddenLayerSizes, HiddenActivation = hiddenActivation, RandomSeed = randomSeed, LearningRate = learningRate, RegularizationLambda = regularizationLambda, LabelHorizon = labelHorizon, WeightInit = weightInit, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlNeuralNetMultiLayer);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetMultiLayer MlNeuralNetMultiLayer(string hiddenLayerSizes, MlNeuralNetMultiLayer_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetMultiLayer_WeightInitMode weightInit, MlNeuralNetMultiLayer_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetMultiLayer(Input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetMultiLayer MlNeuralNetMultiLayer(ISeries<double> input , string hiddenLayerSizes, MlNeuralNetMultiLayer_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetMultiLayer_WeightInitMode weightInit, MlNeuralNetMultiLayer_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetMultiLayer(input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetMultiLayer MlNeuralNetMultiLayer(string hiddenLayerSizes, MlNeuralNetMultiLayer_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetMultiLayer_WeightInitMode weightInit, MlNeuralNetMultiLayer_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetMultiLayer(Input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetMultiLayer MlNeuralNetMultiLayer(ISeries<double> input , string hiddenLayerSizes, MlNeuralNetMultiLayer_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetMultiLayer_WeightInitMode weightInit, MlNeuralNetMultiLayer_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetMultiLayer(input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
