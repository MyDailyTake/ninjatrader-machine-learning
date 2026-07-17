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

// Write-up: https://mydailytake.com/ml-adam-optimizer-mini-batch-stable-training-for-ninjatrader-8/
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

public enum MlNeuralNetAdam_WeightInitMode { Zero, Random }
public enum MlNeuralNetAdam_LabelMode { CloseToClose, FavorableExcursion }
public enum MlNeuralNetAdam_Activation { Tanh, ReLU, Sigmoid }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Architecture",	10100)]
	[Gui.CategoryOrder("Learning",		10200)]
	[Gui.CategoryOrder("Optimizer",		10250)]
	[Gui.CategoryOrder("Features",		10300)]
	[Gui.CategoryOrder("Signal",		10400)]
	[Gui.CategoryOrder("Display",		10500)]

	#endregion

	public class MlNeuralNetAdam : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Neural Net (Adam + Mini-Batch)";
		public string indDescription	= "A multi-hidden-layer neural network trained with the Adam optimizer and mini-batch updates. Same configurable architecture and same three input features as the multi-hidden-layer sibling, but instead of plain stochastic gradient descent on every bar, this version (a) accumulates gradients across BatchSize bars before applying an update, smoothing out single-bar noise, and (b) uses Adam — adaptive per-parameter learning rates that combine momentum (an exponential running average of the gradient) and variance scaling (an exponential running average of the squared gradient). The result is a substantially more stable trainer than plain SGD: large gradients in noisy directions don't blow up small gradient signals; persistent gradient directions accumulate momentum and converge faster. The two changes together turn 'wobbly online learner' into 'stably trained model.' Default features (same as the k-NN, OLR, SHL, and Multi-Layer siblings): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Z-score normalized using each bar's own local-time stats. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the prior posts. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Architecture

		[NinjaScriptProperty]
		[Display(Order = 01, GroupName = "Architecture", Name = "Hidden Layer Sizes", Description = "Per-layer neuron counts as a list of integers. Any common separator works — comma, space, hyphen, x, semicolon. Examples: '6' = one layer of 6 neurons. '8, 6' = two layers (8 then 6). '10, 6, 4' = three layers. Each value must be ≥ 2. No upper cap; with Adam, slightly deeper networks train more reliably than under plain SGD, but the practical lift on a 3-feature problem still plateaus around 2 layers. Invalid input falls back to default '8, 6' and logs a warning to the NT Output window.")]
		public string HiddenLayerSizes { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 02, GroupName = "Architecture", Name = "Hidden Activation", Description = "Activation function on EVERY hidden layer. ReLU is the recommended default — it pairs naturally with Adam and avoids vanishing-gradient problems at depth.")]
		public MlNeuralNetAdam_Activation HiddenActivation { get; set; }

		[NinjaScriptProperty]
		[Range(0, 999999)]
		[Display(Order = 03, GroupName = "Architecture", Name = "Random Seed", Description = "Seed for the random weight initialization (when Weight Init = Random). Same seed = same starting weights. Useful for fair comparisons between architectures or optimizer settings.")]
		public int RandomSeed { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(0.00001, 1.0)]
		[Display(Order = 01, GroupName = "Learning", Name = "Learning Rate", Description = "Adam's base learning rate. Adam is much less sensitive to this than plain SGD because of the adaptive per-parameter scaling, but the value still sets the overall step magnitude. 0.001 is the canonical Adam default; for online financial data, 0.001-0.005 works well.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.1)]
		[Display(Order = 02, GroupName = "Learning", Name = "Regularization Lambda (L2)", Description = "L2 penalty on weight magnitude. Recommended 0.0001 to 0.001.")]
		public double RegularizationLambda { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Weight Init", Description = "Random init is essentially required — Adam's adaptive scaling can't recover from the symmetry of all-zero weights.")]
		public MlNeuralNetAdam_WeightInitMode WeightInit { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 05, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. Same semantics as the prior posts.")]
		public MlNeuralNetAdam_LabelMode LabelMode { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 5.0)]
		[Display(Order = 06, GroupName = "Learning", Name = "Min Favorable Move (ATRs)", Description = "ONLY USED WHEN Label Mode = FavorableExcursion. Minimum favorable excursion (in ATRs at entry) required during the post-bar window for the model to update.")]
		public double MinFavorableMoveAtrs { get; set; }

		#endregion

		#region Optimizer

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 01, GroupName = "Optimizer", Name = "Mini-Batch Size", Description = "Number of training-eligible bars to accumulate gradients over before applying an Adam update. 1 = pure online (one update per bar, like the prior posts). 5-20 = small mini-batches that smooth single-bar noise. Larger values trade adaptation speed for stability — recommended 5-10 for online financial data.")]
		public int BatchSize { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.999)]
		[Display(Order = 02, GroupName = "Optimizer", Name = "Beta1 (momentum decay)", Description = "Exponential decay rate for Adam's first-moment estimate (the momentum / running gradient mean). Higher values = longer memory of past gradients. 0.9 is the canonical default; rarely needs tuning.")]
		public double Beta1 { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.99999)]
		[Display(Order = 03, GroupName = "Optimizer", Name = "Beta2 (variance decay)", Description = "Exponential decay rate for Adam's second-moment estimate (the running gradient variance). Higher values = longer memory of past gradient magnitudes. 0.999 is the canonical default; rarely needs tuning.")]
		public double Beta2 { get; set; }

		[NinjaScriptProperty]
		[Range(1e-10, 1e-3)]
		[Display(Order = 04, GroupName = "Optimizer", Name = "Epsilon", Description = "Small constant added to Adam's variance term denominator to prevent division by zero. 1e-8 is the canonical default and rarely needs tuning.")]
		public double Epsilon { get; set; }

		#endregion

		#region Features

		[NinjaScriptProperty]
		[Range(2, 500)]
		[Display(Order = 01, GroupName = "Features", Name = "MA Period", Description = "Period of the moving average used in the distFromMa feature.")]
		public int MaPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 100)]
		[Display(Order = 02, GroupName = "Features", Name = "ATR Period", Description = "Period of the ATR used to scale every feature into volatility units.")]
		public int AtrPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 200)]
		[Display(Order = 03, GroupName = "Features", Name = "Slope Lookback (bars)", Description = "Number of bars over which the slope feature is measured.")]
		public int SlopeLookback { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features (Z-Score)", Description = "Master toggle for z-score normalization. Recommended ON.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Window used to compute the rolling mean / stddev that z-score the features.")]
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

		[Display(Order = 01, GroupName = "Display", Name = "Marker Offset (ticks)", Description = "Vertical offset of the signal triangle from the bar's high/low, in ticks.")]
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
		private int[]	layerSizes;
		private int		numHiddenLayers;

		// Hidden weights & biases (same layout as MlNeuralNetMultiLayer).
		private double[][][]	Wh;
		private double[][]		bh;

		// Output weights & bias.
		private double[]	Wo;
		private double		bo;

		// ─── Adam optimizer state ───
		// First moment (m) and second moment (v) per parameter, parallel to weights.
		private double[][][]	mWh, vWh;
		private double[][]		mbh, vbh;
		private double[]		mWo, vWo;
		private double			mbo, vbo;

		// Adam time step — increments by 1 with each weight update (each mini-batch flush).
		private long	adamStep;

		// ─── Mini-batch gradient accumulators ───
		// Parallel to weights, accumulated across BatchSize training-eligible bars.
		private double[][][]	gWh;
		private double[][]		gbh;
		private double[]		gWo;
		private double			gbo;
		private int				gAccumCount;   // training-eligible bars contributed to the current accumulator

		// Forward-pass scratch — preserved across the bar so backprop can reuse them.
		private double[]	scratchRaw;
		private double[]	scratchNorm;
		private double[][]	hiddenPre;
		private double[][]	hiddenAct;

		// Backprop scratch — gradient signal at each layer.
		private double[][]	dHiddenAct;
		private double[][]	dHiddenPre;

		// Cooldown tracker.
		private int			lastSignalBar	= -1;

		// Public Series backing fields.
		private Series<double>	sProbabilityUp;
		private Series<double>	sConfidence;
		private Series<bool>	sIsLongSignal;
		private Series<bool>	sIsShortSignal;

		// TickSize-derived offsets.
		private double	markerOffsetPts;
		private double	labelOffsetPts;

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
				HiddenActivation		= MlNeuralNetAdam_Activation.ReLU;
				RandomSeed				= 42;

				LearningRate			= 0.001;       // Adam-canonical default
				RegularizationLambda	= 0.0001;
				LabelHorizon			= 2;
				WeightInit				= MlNeuralNetAdam_WeightInitMode.Random;
				LabelMode				= MlNeuralNetAdam_LabelMode.CloseToClose;
				MinFavorableMoveAtrs	= 1.0;

				BatchSize				= 8;
				Beta1					= 0.9;         // Adam canonical
				Beta2					= 0.999;       // Adam canonical
				Epsilon					= 1e-8;        // Adam canonical

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

				// Allocate network state, Adam state, and gradient accumulators in parallel.
				Wh			= new double[numHiddenLayers][][];
				bh			= new double[numHiddenLayers][];
				mWh			= new double[numHiddenLayers][][];
				vWh			= new double[numHiddenLayers][][];
				mbh			= new double[numHiddenLayers][];
				vbh			= new double[numHiddenLayers][];
				gWh			= new double[numHiddenLayers][][];
				gbh			= new double[numHiddenLayers][];
				hiddenPre	= new double[numHiddenLayers][];
				hiddenAct	= new double[numHiddenLayers][];
				dHiddenAct	= new double[numHiddenLayers][];
				dHiddenPre	= new double[numHiddenLayers][];

				for (int l = 0; l < numHiddenLayers; l++)
				{
					int sizeIn  = (l == 0) ? NumFeatures : layerSizes[l - 1];
					int sizeOut = layerSizes[l];

					Wh[l]	= new double[sizeOut][];
					mWh[l]	= new double[sizeOut][];
					vWh[l]	= new double[sizeOut][];
					gWh[l]	= new double[sizeOut][];
					for (int h = 0; h < sizeOut; h++)
					{
						Wh[l][h]	= new double[sizeIn];
						mWh[l][h]	= new double[sizeIn];
						vWh[l][h]	= new double[sizeIn];
						gWh[l][h]	= new double[sizeIn];
					}
					bh[l]	= new double[sizeOut];
					mbh[l]	= new double[sizeOut];
					vbh[l]	= new double[sizeOut];
					gbh[l]	= new double[sizeOut];

					hiddenPre[l]	= new double[sizeOut];
					hiddenAct[l]	= new double[sizeOut];
					dHiddenAct[l]	= new double[sizeOut];
					dHiddenPre[l]	= new double[sizeOut];
				}

				int lastHidden = layerSizes[numHiddenLayers - 1];
				Wo	= new double[lastHidden];
				mWo	= new double[lastHidden];
				vWo	= new double[lastHidden];
				gWo	= new double[lastHidden];

				InitializeWeights();
				adamStep    = 0;
				gAccumCount = 0;

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];

				lastSignalBar	= -1;
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;
			}
		}

		#endregion

		// ─── ARCHITECTURE PARSING ─────────────────────────────────────────────────
		// Forgiving parser: any non-digit run (comma, space, hyphen, semicolon, x, etc.)
		// is treated as a separator. Each parsed integer must be ≥ 2. Invalid input
		// falls back to default and logs a warning to the NT Output window.

		private static readonly int[] DefaultLayers = new int[] { 8, 6 };

		private int[] ParseHiddenLayerSizes(string s)
		{
			if (string.IsNullOrWhiteSpace(s))
			{
				Print(string.Format("{0}: Hidden Layer Sizes is empty, using default '8, 6'.", indName));
				return (int[])DefaultLayers.Clone();
			}

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

		private void InitializeWeights()
		{
			if (WeightInit == MlNeuralNetAdam_WeightInitMode.Random)
			{
				var rng = new System.Random(RandomSeed);

				for (int l = 0; l < numHiddenLayers; l++)
				{
					int fanIn  = (l == 0) ? NumFeatures : layerSizes[l - 1];
					int fanOut = layerSizes[l];
					double scale = HiddenActivation == MlNeuralNetAdam_Activation.ReLU
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
			else
			{
				for (int l = 0; l < numHiddenLayers; l++)
					for (int h = 0; h < layerSizes[l]; h++)
					{
						for (int k = 0; k < Wh[l][h].Length; k++) Wh[l][h][k] = 0;
						bh[l][h] = 0;
					}
				for (int h = 0; h < Wo.Length; h++) Wo[h] = 0;
				bo = 0;
			}
		}

		private static double SampleNormal(System.Random rng)
		{
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

			// 2) Update step — accumulate gradients across BatchSize training-eligible bars,
			//    then apply one Adam update.
			int trainBar = LabelHorizon;
			if (CurrentBar >= predictionWarmup + LabelHorizon)
			{
				bool   trainThisBar	= false;
				double y			= 0.0;

				if (LabelMode == MlNeuralNetAdam_LabelMode.CloseToClose)
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

					// Forward pass at trainBar with current weights.
					double pTrain = ForwardPass(scratchNorm);

					// Backward pass — accumulates this bar's gradients into the mini-batch.
					AccumulateGradients(scratchNorm, pTrain, y);
					gAccumCount++;

					// Mini-batch flush: apply Adam update once we've accumulated BatchSize bars.
					if (gAccumCount >= BatchSize)
					{
						ApplyAdamUpdate();
						gAccumCount = 0;
					}
				}
			}

			// 3) Public-Series outputs.
			sProbabilityUp[0] = pUp;
			sConfidence[0]    = Math.Abs(pUp - 0.5) * 2.0;

			// 4) Signal gate — wait for at least 50 update steps after the first label
			//    is observable. With BatchSize > 1, that's 50 * BatchSize bars.
			int signalWarmup = predictionWarmup + LabelHorizon + 50 * Math.Max(1, BatchSize);
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

		private double ForwardPass(double[] inputs)
		{
			for (int l = 0; l < numHiddenLayers; l++)
			{
				double[] inputToLayer = (l == 0) ? inputs : hiddenAct[l - 1];
				int sizeIn  = inputToLayer.Length;
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

			int last = numHiddenLayers - 1;
			double zOut = bo;
			for (int h = 0; h < layerSizes[last]; h++)
				zOut += Wo[h] * hiddenAct[last][h];

			return Sigmoid(zOut);
		}

		// ─── ACCUMULATE GRADIENTS (no weight update) ──────────────────────────────
		// Standard chain-rule backprop, but the gradients are added to the running
		// accumulators rather than applied directly. The L2 regularization term is
		// also accumulated so the final per-parameter gradient is the mini-batch
		// mean of (data gradient + λ·w).

		private void AccumulateGradients(double[] inputs, double pTrain, double y)
		{
			double error = pTrain - y;   // dL/dzOut for cross-entropy + sigmoid

			int last = numHiddenLayers - 1;

			// Seed gradient at last hidden layer's activations from the output side.
			for (int h = 0; h < layerSizes[last]; h++)
				dHiddenAct[last][h] = error * Wo[h];

			// Walk backward through hidden layers.
			for (int l = last; l >= 0; l--)
			{
				int sizeOut = layerSizes[l];
				int sizeIn  = (l == 0) ? NumFeatures : layerSizes[l - 1];
				double[] inputToLayer = (l == 0) ? inputs : hiddenAct[l - 1];

				for (int h = 0; h < sizeOut; h++)
					dHiddenPre[l][h] = dHiddenAct[l][h] * ActivationDerivative(hiddenPre[l][h], hiddenAct[l][h]);

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

				// Accumulate gradients into gWh[l] / gbh[l].
				for (int h = 0; h < sizeOut; h++)
				{
					double dpre = dHiddenPre[l][h];
					for (int k = 0; k < sizeIn; k++)
						gWh[l][h][k] += dpre * inputToLayer[k] + RegularizationLambda * Wh[l][h][k];
					gbh[l][h] += dpre;
				}
			}

			// Accumulate output-layer gradients.
			for (int h = 0; h < layerSizes[last]; h++)
				gWo[h] += error * hiddenAct[last][h] + RegularizationLambda * Wo[h];
			gbo += error;
		}

		// ─── ADAM UPDATE ──────────────────────────────────────────────────────────
		// Standard Adam:
		//   m_t = β1 * m_{t-1} + (1 - β1) * g
		//   v_t = β2 * v_{t-1} + (1 - β2) * g²
		//   m_hat = m_t / (1 - β1^t)
		//   v_hat = v_t / (1 - β2^t)
		//   w -= lr * m_hat / (sqrt(v_hat) + ε)
		//
		// Bias-correction terms (m_hat, v_hat) compensate for the running averages
		// being initialized to zero — without them, early updates would be tiny.
		//
		// The accumulated gradients are mean-normalized by dividing by the actual
		// number of bars contributed (gAccumCount), which can be less than BatchSize
		// if the user changes BatchSize on a partial accumulator.

		private void ApplyAdamUpdate()
		{
			adamStep++;
			double invN = 1.0 / Math.Max(1, gAccumCount);
			double biasCorr1 = 1.0 - Math.Pow(Beta1, adamStep);
			double biasCorr2 = 1.0 - Math.Pow(Beta2, adamStep);

			// Hidden layers.
			for (int l = 0; l < numHiddenLayers; l++)
			{
				int sizeIn  = (l == 0) ? NumFeatures : layerSizes[l - 1];
				int sizeOut = layerSizes[l];

				for (int h = 0; h < sizeOut; h++)
				{
					for (int k = 0; k < sizeIn; k++)
					{
						double g = gWh[l][h][k] * invN;
						mWh[l][h][k] = Beta1 * mWh[l][h][k] + (1.0 - Beta1) * g;
						vWh[l][h][k] = Beta2 * vWh[l][h][k] + (1.0 - Beta2) * g * g;
						double mHat = mWh[l][h][k] / biasCorr1;
						double vHat = vWh[l][h][k] / biasCorr2;
						Wh[l][h][k] -= LearningRate * mHat / (Math.Sqrt(vHat) + Epsilon);
						gWh[l][h][k] = 0;
					}
					double gb = gbh[l][h] * invN;
					mbh[l][h] = Beta1 * mbh[l][h] + (1.0 - Beta1) * gb;
					vbh[l][h] = Beta2 * vbh[l][h] + (1.0 - Beta2) * gb * gb;
					double mbHat = mbh[l][h] / biasCorr1;
					double vbHat = vbh[l][h] / biasCorr2;
					bh[l][h] -= LearningRate * mbHat / (Math.Sqrt(vbHat) + Epsilon);
					gbh[l][h] = 0;
				}
			}

			// Output layer.
			int last = numHiddenLayers - 1;
			for (int h = 0; h < layerSizes[last]; h++)
			{
				double g = gWo[h] * invN;
				mWo[h] = Beta1 * mWo[h] + (1.0 - Beta1) * g;
				vWo[h] = Beta2 * vWo[h] + (1.0 - Beta2) * g * g;
				double mHat = mWo[h] / biasCorr1;
				double vHat = vWo[h] / biasCorr2;
				Wo[h] -= LearningRate * mHat / (Math.Sqrt(vHat) + Epsilon);
				gWo[h] = 0;
			}
			double goB = gbo * invN;
			mbo = Beta1 * mbo + (1.0 - Beta1) * goB;
			vbo = Beta2 * vbo + (1.0 - Beta2) * goB * goB;
			double mboHat = mbo / biasCorr1;
			double vboHat = vbo / biasCorr2;
			bo -= LearningRate * mboHat / (Math.Sqrt(vboHat) + Epsilon);
			gbo = 0;
		}

		// ─── ACTIVATION + DERIVATIVE ──────────────────────────────────────────────

		private double Activate(double z)
		{
			switch (HiddenActivation)
			{
				case MlNeuralNetAdam_Activation.ReLU:
					return z > 0 ? z : 0;
				case MlNeuralNetAdam_Activation.Sigmoid:
					return Sigmoid(z);
				case MlNeuralNetAdam_Activation.Tanh:
				default:
					return Math.Tanh(z);
			}
		}

		private double ActivationDerivative(double pre, double act)
		{
			switch (HiddenActivation)
			{
				case MlNeuralNetAdam_Activation.ReLU:
					return pre > 0 ? 1.0 : 0.0;
				case MlNeuralNetAdam_Activation.Sigmoid:
					return act * (1.0 - act);
				case MlNeuralNetAdam_Activation.Tanh:
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
			string tag  = (isLong ? "nnadam-long-lbl-" : "nnadam-short-lbl-") + CurrentBar;

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
		private indMyDailyTake.MlNeuralNetAdam[] cacheMlNeuralNetAdam;
		public indMyDailyTake.MlNeuralNetAdam MlNeuralNetAdam(string hiddenLayerSizes, MlNeuralNetAdam_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetAdam_WeightInitMode weightInit, MlNeuralNetAdam_LabelMode labelMode, double minFavorableMoveAtrs, int batchSize, double beta1, double beta2, double epsilon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlNeuralNetAdam(Input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, batchSize, beta1, beta2, epsilon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlNeuralNetAdam MlNeuralNetAdam(ISeries<double> input, string hiddenLayerSizes, MlNeuralNetAdam_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetAdam_WeightInitMode weightInit, MlNeuralNetAdam_LabelMode labelMode, double minFavorableMoveAtrs, int batchSize, double beta1, double beta2, double epsilon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlNeuralNetAdam != null)
				for (int idx = 0; idx < cacheMlNeuralNetAdam.Length; idx++)
					if (cacheMlNeuralNetAdam[idx] != null && cacheMlNeuralNetAdam[idx].HiddenLayerSizes == hiddenLayerSizes && cacheMlNeuralNetAdam[idx].HiddenActivation == hiddenActivation && cacheMlNeuralNetAdam[idx].RandomSeed == randomSeed && cacheMlNeuralNetAdam[idx].LearningRate == learningRate && cacheMlNeuralNetAdam[idx].RegularizationLambda == regularizationLambda && cacheMlNeuralNetAdam[idx].LabelHorizon == labelHorizon && cacheMlNeuralNetAdam[idx].WeightInit == weightInit && cacheMlNeuralNetAdam[idx].LabelMode == labelMode && cacheMlNeuralNetAdam[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlNeuralNetAdam[idx].BatchSize == batchSize && cacheMlNeuralNetAdam[idx].Beta1 == beta1 && cacheMlNeuralNetAdam[idx].Beta2 == beta2 && cacheMlNeuralNetAdam[idx].Epsilon == epsilon && cacheMlNeuralNetAdam[idx].MaPeriod == maPeriod && cacheMlNeuralNetAdam[idx].AtrPeriod == atrPeriod && cacheMlNeuralNetAdam[idx].SlopeLookback == slopeLookback && cacheMlNeuralNetAdam[idx].NormalizeFeatures == normalizeFeatures && cacheMlNeuralNetAdam[idx].NormalizationLookback == normalizationLookback && cacheMlNeuralNetAdam[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlNeuralNetAdam[idx].SignalCooldownBars == signalCooldownBars && cacheMlNeuralNetAdam[idx].EqualsInput(input))
						return cacheMlNeuralNetAdam[idx];
			return CacheIndicator<indMyDailyTake.MlNeuralNetAdam>(new indMyDailyTake.MlNeuralNetAdam(){ HiddenLayerSizes = hiddenLayerSizes, HiddenActivation = hiddenActivation, RandomSeed = randomSeed, LearningRate = learningRate, RegularizationLambda = regularizationLambda, LabelHorizon = labelHorizon, WeightInit = weightInit, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, BatchSize = batchSize, Beta1 = beta1, Beta2 = beta2, Epsilon = epsilon, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlNeuralNetAdam);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetAdam MlNeuralNetAdam(string hiddenLayerSizes, MlNeuralNetAdam_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetAdam_WeightInitMode weightInit, MlNeuralNetAdam_LabelMode labelMode, double minFavorableMoveAtrs, int batchSize, double beta1, double beta2, double epsilon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetAdam(Input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, batchSize, beta1, beta2, epsilon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetAdam MlNeuralNetAdam(ISeries<double> input , string hiddenLayerSizes, MlNeuralNetAdam_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetAdam_WeightInitMode weightInit, MlNeuralNetAdam_LabelMode labelMode, double minFavorableMoveAtrs, int batchSize, double beta1, double beta2, double epsilon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetAdam(input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, batchSize, beta1, beta2, epsilon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlNeuralNetAdam MlNeuralNetAdam(string hiddenLayerSizes, MlNeuralNetAdam_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetAdam_WeightInitMode weightInit, MlNeuralNetAdam_LabelMode labelMode, double minFavorableMoveAtrs, int batchSize, double beta1, double beta2, double epsilon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetAdam(Input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, batchSize, beta1, beta2, epsilon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlNeuralNetAdam MlNeuralNetAdam(ISeries<double> input , string hiddenLayerSizes, MlNeuralNetAdam_Activation hiddenActivation, int randomSeed, double learningRate, double regularizationLambda, int labelHorizon, MlNeuralNetAdam_WeightInitMode weightInit, MlNeuralNetAdam_LabelMode labelMode, double minFavorableMoveAtrs, int batchSize, double beta1, double beta2, double epsilon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlNeuralNetAdam(input, hiddenLayerSizes, hiddenActivation, randomSeed, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, batchSize, beta1, beta2, epsilon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
