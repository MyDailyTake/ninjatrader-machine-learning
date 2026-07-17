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

// Write-up: https://mydailytake.com/ml-online-logistic-regression-your-first-neural-net-for-ninjatrader-8/
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

public enum MlOnlineLogisticRegression_WeightInitMode { Zero, Random }
public enum MlOnlineLogisticRegression_LabelMode { CloseToClose, FavorableExcursion }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Learning",	10100)]
	[Gui.CategoryOrder("Features",	10200)]
	[Gui.CategoryOrder("Signal",	10300)]
	[Gui.CategoryOrder("Display",	10400)]

	#endregion

	public class MlOnlineLogisticRegression : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Online Logistic Regression";
		public string indDescription	= "Your first neural-net indicator for NinjaTrader 8 — a single neuron with sigmoid activation, trained online via gradient descent. Unlike the k-NN sibling that does a fresh search every bar, this model carries a small set of weights (one per feature, plus a bias) and updates them incrementally as each new bar's forward direction becomes observable. The weights start near zero and adapt over time, so the model is inherently regime-aware: when market character shifts, the weights migrate to fit the new regime. Default features (same as the k-NN sibling for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Features are z-score normalized using each bar's own local-time stats. Two label modes drive how the model is trained: CloseToClose (y = direction of close-vs-close over the LabelHorizon window — simplest, trend-aligned in sustained moves), or FavorableExcursion (y = which side had the bigger MFE within the window using bar highs/lows, with chop-window skipping — more trader-aligned but biased mean-reversion at short horizons). The label choice fundamentally shapes the model's character. Renders as a chart overlay: green triangle below the bar with a P(up) label when a high-conviction long signal fires, red triangle above the bar with P(up) label for shorts. Signals fire only when the predicted probability crosses a configurable edge above 0.5 (or below) AND a cooldown window has elapsed since the previous signal — so the chart stays readable instead of stamping a label on every bar. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, weight-per-feature, BiasSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(0.0001, 1.0)]
		[Display(Order = 01, GroupName = "Learning", Name = "Learning Rate", Description = "Step size for each weight update. Larger = adapts faster but jitters; smaller = smoother but slower to react. 0.01 is a sensible default for normalized features.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 0.1)]
		[Display(Order = 02, GroupName = "Learning", Name = "Regularization Lambda (L2)", Description = "L2 penalty on weight magnitude. Pulls weights gently toward zero each update so they don't drift to extreme values when a feature is noisy. 0 disables regularization.")]
		public double RegularizationLambda { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. The model updates each bar using the feature vector from N bars ago (whose forward direction is now known) — this is what makes the training step look-ahead-safe.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Weight Init", Description = "Zero starts every weight at 0 (model takes longer to commit but has no built-in bias). Random starts with small random values around 0 (faster to commit but the initial state is non-deterministic).")]
		public MlOnlineLogisticRegression_WeightInitMode WeightInit { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 05, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined for each historical bar. CloseToClose: y=1 if Close at the end of the LabelHorizon window is above Close at trainBar (simplest, ignores intra-window action). FavorableExcursion: y=1 if max favorable excursion in the LONG direction beat that of the SHORT direction during the window — uses bar highs/lows so wicks count, and skips bars whose post-window was below Min Favorable Move (chop). The MFE mode is more trader-aligned; the close-to-close mode is simpler and useful for comparing the two paradigms.")]
		public MlOnlineLogisticRegression_LabelMode LabelMode { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 5.0)]
		[Display(Order = 06, GroupName = "Learning", Name = "Min Favorable Move (ATRs)", Description = "ONLY USED WHEN Label Mode = FavorableExcursion. Minimum favorable excursion (in ATRs at entry) required during the post-bar window for the model to update on that bar. If max(MFE_long, MFE_short) is below this threshold, the bar's follow-on was just chop and we skip the weight update. Set to 0 to train on every observable bar regardless of move size. Has no effect when Label Mode = CloseToClose.")]
		public double MinFavorableMoveAtrs { get; set; }

		#endregion

		#region Features

		[NinjaScriptProperty]
		[Range(2, 500)]
		[Display(Order = 01, GroupName = "Features", Name = "MA Period", Description = "Period of the moving average used in the distFromMa feature, and used as the smoothing window for the ATR regime ratio. Smaller = more reactive; larger = trend-anchored.")]
		public int MaPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 100)]
		[Display(Order = 02, GroupName = "Features", Name = "ATR Period", Description = "Period of the ATR used to scale every feature into volatility units, so distance comparisons stay consistent across regimes.")]
		public int AtrPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 200)]
		[Display(Order = 03, GroupName = "Features", Name = "Slope Lookback (bars)", Description = "Number of bars over which the slope feature is measured: (Close[0] − Close[N]) / ATR. Smaller = momentum thrust; larger = trend persistence.")]
		public int SlopeLookback { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features (Z-Score)", Description = "Master toggle for z-score normalization. When ON, each feature is rescaled against its own historical rolling stats so a 1-σ extreme reading means the same thing across regimes. Recommended ON.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Window used to compute the rolling mean / stddev that z-score the features. Each bar uses its OWN local-time stats — a 2008 bar against 2008's distribution, a 2024 bar against 2024's. Default 200.")]
		public int NormalizationLookback { get; set; }

		#endregion

		#region Signal

		[NinjaScriptProperty]
		[Range(0.0, 0.49)]
		[Display(Order = 01, GroupName = "Signal", Name = "Min Probability Edge", Description = "How far the predicted probability of an up move must be from 0.5 before a signal fires. 0.10 means: long fires when P(up) > 0.60, short fires when P(up) < 0.40. Larger values produce fewer, higher-conviction signals.")]
		public double MinProbabilityEdge { get; set; }

		[NinjaScriptProperty]
		[Range(0, 500)]
		[Display(Order = 02, GroupName = "Signal", Name = "Signal Cooldown (bars)", Description = "Minimum bars between consecutive signals. Higher values space signals out so the chart stays readable; lower values let signals come in clusters during sustained moves. Set to 0 to fire on every qualifying bar.")]
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

		// Model state — three weights + bias.
		private double[]	weights;
		private double		bias;

		// Reusable scratch (no per-bar allocation).
		private double[]	scratchRaw;
		private double[]	scratchNorm;

		// Cooldown tracker — bar index of the last fired signal. -1 means "no signal yet."
		private int			lastSignalBar	= -1;

		// Public Series backing fields.
		private Series<double>	sProbabilityUp;
		private Series<double>	sConfidence;
		private Series<double>	sWeightDistFromMa;
		private Series<double>	sWeightSlope;
		private Series<double>	sWeightAtrRegime;
		private Series<double>	sBias;
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

				IsOverlay					= true;		// signals + labels render on the price panel
				DisplayInDataBox			= true;
				DrawHorizontalGridLines		= true;
				DrawVerticalGridLines		= true;
				PaintPriceMarkers			= true;
				ScaleJustification			= NinjaTrader.Gui.Chart.ScaleJustification.Right;
				IsSuspendedWhileInactive	= true;

				#endregion

				#region Defaults

				LearningRate			= 0.01;
				RegularizationLambda	= 0.0001;
				LabelHorizon			= 2;
				WeightInit				= MlOnlineLogisticRegression_WeightInitMode.Zero;
				LabelMode				= MlOnlineLogisticRegression_LabelMode.FavorableExcursion;
				MinFavorableMoveAtrs	= 2.0;

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

				// Plot indices (chart overlay — Y values are PRICES):
				//   0 Long signal triangle  (green, plotted below bar's Low)
				//   1 Short signal triangle (red, plotted above bar's High)
				AddPlot(new Stroke(Brushes.LimeGreen, 5),	PlotStyle.TriangleUp,	"OLR Long");
				AddPlot(new Stroke(Brushes.OrangeRed, 5),	PlotStyle.TriangleDown,	"OLR Short");

				#endregion
			}
			else if (State == State.DataLoaded)
			{
				sProbabilityUp		= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sConfidence			= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sWeightDistFromMa	= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sWeightSlope		= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sWeightAtrRegime	= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sBias				= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sIsLongSignal		= new Series<bool>(this,   MaximumBarsLookBack.Infinite);
				sIsShortSignal		= new Series<bool>(this,   MaximumBarsLookBack.Infinite);

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

				weights = new double[NumFeatures];
				if (WeightInit == MlOnlineLogisticRegression_WeightInitMode.Random)
				{
					var rng = new System.Random(42);	// fixed seed so weight evolution is reproducible across reloads
					for (int k = 0; k < NumFeatures; k++)
						weights[k] = (rng.NextDouble() - 0.5) * 0.1;
					bias = (rng.NextDouble() - 0.5) * 0.1;
				}
				else
				{
					for (int k = 0; k < NumFeatures; k++) weights[k] = 0;
					bias = 0;
				}

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];

				lastSignalBar	= int.MinValue;
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;
			}
		}

		#endregion

		// ─── ON BAR UPDATE ────────────────────────────────────────────────────────

		protected override void OnBarUpdate()
		{
			Values[0].Reset();
			Values[1].Reset();
			sIsLongSignal[0]	= false;
			sIsShortSignal[0]	= false;

			// PHASE 1 — source-indicator warmup.
			int sourceWarmup = Math.Max(Math.Max(MaPeriod, AtrPeriod), SlopeLookback) + 1;
			if (CurrentBar < sourceWarmup) return;

			// PHASE 2 — populate featureSeries every bar so the rolling SMA / StdDev
			// pre-loads stable normalization stats before prediction starts.
			GetFeaturesInto(scratchRaw, 0);
			for (int k = 0; k < NumFeatures; k++)
				featureSeries[k][0] = scratchRaw[k];

			// PHASE 3 — prediction warmup. Need NormalizationLookback bars of feature
			// history so per-bar z-score stats are meaningful.
			int predictionWarmup = sourceWarmup + NormalizationLookback;
			if (CurrentBar < predictionWarmup) return;

			// 1) Forward pass on the current bar — used for the prediction the user sees.
			NormalizeInto(scratchNorm, scratchRaw, 0);
			double zCurrent = bias;
			for (int k = 0; k < NumFeatures; k++) zCurrent += weights[k] * scratchNorm[k];
			double pUp = Sigmoid(zCurrent);

			// 2) Update step — applied to the bar from LabelHorizon ago, whose forward
			//    window is now observable. This is what makes the training step
			//    look-ahead-safe: we never train on a label we haven't yet seen.
			//
			//    Two label modes:
			//      CloseToClose:        y = 1 if Close[0] > Close[trainBar], else 0.
			//                           Trains on every observable bar.
			//      FavorableExcursion:  Scan the LabelHorizon-bar window AFTER trainBar,
			//                           measure max favorable excursion in each direction
			//                           using bar highs/lows. Skip the update if neither
			//                           side reached MinFavorableMoveAtrs (chop window —
			//                           no tradeable move to learn). Otherwise y = 1 if
			//                           MFE_long was strictly larger than MFE_short.
			int trainBar = LabelHorizon;
			if (CurrentBar >= predictionWarmup + LabelHorizon)
			{
				bool   trainThisBar	= false;
				double y			= 0.0;

				if (LabelMode == MlOnlineLogisticRegression_LabelMode.CloseToClose)
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

					double zTrain = bias;
					for (int k = 0; k < NumFeatures; k++) zTrain += weights[k] * scratchNorm[k];
					double pTrain = Sigmoid(zTrain);

					// Cross-entropy gradient: dL/dw_i = (p − y) · f_i, dL/db = (p − y).
					// L2 regularization adds λ·w_i to each weight gradient (bias is not regularized).
					double error = pTrain - y;
					for (int k = 0; k < NumFeatures; k++)
						weights[k] -= LearningRate * (error * scratchNorm[k] + RegularizationLambda * weights[k]);
					bias -= LearningRate * error;
				}
			}

			// 3) Write the public-Series outputs.
			sProbabilityUp[0]		= pUp;
			sConfidence[0]			= Math.Abs(pUp - 0.5) * 2.0;	// 0 (coin flip) → 1 (certain)
			sWeightDistFromMa[0]	= weights[0];
			sWeightSlope[0]			= weights[1];
			sWeightAtrRegime[0]		= weights[2];
			sBias[0]				= bias;

			// 4) Signal gate. The model needs some bars of training before the
			//    weights are meaningful — wait for at least 50 update steps.
			int signalWarmup = predictionWarmup + LabelHorizon + 50;
			if (CurrentBar < signalWarmup) return;

			// Cooldown — don't fire if we're within SignalCooldownBars of the last signal.
			// Skip the check until at least one signal has fired (lastSignalBar >= 0).
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

		// ─── FEATURE EXTRACTION ───────────────────────────────────────────────────
		// Three features: distance from MA, N-bar slope, and ATR-regime ratio.
		// All ATR-normalized so distance comparisons stay consistent across regimes.

		private void GetFeaturesInto(double[] dst, int barsAgo)
		{
			double atrVal	= atr[barsAgo];
			double safeAtr	= atrVal > 1e-9 ? atrVal : TickSize;

			// Distance from MA, in ATRs — directional position scaled by volatility.
			dst[0] = (Close[barsAgo] - trendMa[barsAgo]) / safeAtr;

			// N-bar slope, in ATRs — recent momentum thrust, normalized.
			int slopeBack = barsAgo + SlopeLookback;
			dst[1] = slopeBack <= CurrentBar
				? (Close[barsAgo] - Close[slopeBack]) / safeAtr
				: 0;

			// Volatility regime — current ATR vs typical ATR.
			double atrSmaVal = atrRegimeMa[barsAgo];
			dst[2] = atrSmaVal > 1e-9 ? atrVal / atrSmaVal : 1.0;
		}

		// ─── NORMALIZATION ────────────────────────────────────────────────────────
		// Each bar is z-scored against ITS OWN local-time stats — regime-relative
		// comparison so a 1-σ extreme from one era is comparable to a 1-σ extreme
		// from another even though raw values differ.

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

		// ─── SIGMOID ──────────────────────────────────────────────────────────────
		// Standard logistic. Clamps the input to avoid double overflow on extreme z.

		private static double Sigmoid(double z)
		{
			if (z > 35.0)  return 1.0;
			if (z < -35.0) return 0.0;
			return 1.0 / (1.0 + Math.Exp(-z));
		}

		// ─── DRAW: signal label on the price panel ────────────────────────────────
		// Two-line label. P(up) is always closest to the triangle so the visual
		// hierarchy is consistent (bar → triangle → P(up) → direction).

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
			string tag  = (isLong ? "olr-long-lbl-" : "olr-short-lbl-") + CurrentBar;

			Draw.Text(this, tag, false, text, 0, y, 0, brush,
				labelFont, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
		}

		#region Public Series

		[Browsable(false)] [XmlIgnore] public Series<double>	ProbabilityUpSeries		{ get { Update(); return sProbabilityUp;		} }
		[Browsable(false)] [XmlIgnore] public Series<double>	ConfidenceSeries		{ get { Update(); return sConfidence;			} }
		[Browsable(false)] [XmlIgnore] public Series<double>	WeightDistFromMaSeries	{ get { Update(); return sWeightDistFromMa;		} }
		[Browsable(false)] [XmlIgnore] public Series<double>	WeightSlopeSeries		{ get { Update(); return sWeightSlope;			} }
		[Browsable(false)] [XmlIgnore] public Series<double>	WeightAtrRegimeSeries	{ get { Update(); return sWeightAtrRegime;		} }
		[Browsable(false)] [XmlIgnore] public Series<double>	BiasSeries				{ get { Update(); return sBias;					} }
		[Browsable(false)] [XmlIgnore] public Series<bool>		IsLongSignalSeries		{ get { Update(); return sIsLongSignal;			} }
		[Browsable(false)] [XmlIgnore] public Series<bool>		IsShortSignalSeries		{ get { Update(); return sIsShortSignal;		} }

		#endregion
	}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private indMyDailyTake.MlOnlineLogisticRegression[] cacheMlOnlineLogisticRegression;
		public indMyDailyTake.MlOnlineLogisticRegression MlOnlineLogisticRegression(double learningRate, double regularizationLambda, int labelHorizon, MlOnlineLogisticRegression_WeightInitMode weightInit, MlOnlineLogisticRegression_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlOnlineLogisticRegression(Input, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlOnlineLogisticRegression MlOnlineLogisticRegression(ISeries<double> input, double learningRate, double regularizationLambda, int labelHorizon, MlOnlineLogisticRegression_WeightInitMode weightInit, MlOnlineLogisticRegression_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlOnlineLogisticRegression != null)
				for (int idx = 0; idx < cacheMlOnlineLogisticRegression.Length; idx++)
					if (cacheMlOnlineLogisticRegression[idx] != null && cacheMlOnlineLogisticRegression[idx].LearningRate == learningRate && cacheMlOnlineLogisticRegression[idx].RegularizationLambda == regularizationLambda && cacheMlOnlineLogisticRegression[idx].LabelHorizon == labelHorizon && cacheMlOnlineLogisticRegression[idx].WeightInit == weightInit && cacheMlOnlineLogisticRegression[idx].LabelMode == labelMode && cacheMlOnlineLogisticRegression[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlOnlineLogisticRegression[idx].MaPeriod == maPeriod && cacheMlOnlineLogisticRegression[idx].AtrPeriod == atrPeriod && cacheMlOnlineLogisticRegression[idx].SlopeLookback == slopeLookback && cacheMlOnlineLogisticRegression[idx].NormalizeFeatures == normalizeFeatures && cacheMlOnlineLogisticRegression[idx].NormalizationLookback == normalizationLookback && cacheMlOnlineLogisticRegression[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlOnlineLogisticRegression[idx].SignalCooldownBars == signalCooldownBars && cacheMlOnlineLogisticRegression[idx].EqualsInput(input))
						return cacheMlOnlineLogisticRegression[idx];
			return CacheIndicator<indMyDailyTake.MlOnlineLogisticRegression>(new indMyDailyTake.MlOnlineLogisticRegression(){ LearningRate = learningRate, RegularizationLambda = regularizationLambda, LabelHorizon = labelHorizon, WeightInit = weightInit, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlOnlineLogisticRegression);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlOnlineLogisticRegression MlOnlineLogisticRegression(double learningRate, double regularizationLambda, int labelHorizon, MlOnlineLogisticRegression_WeightInitMode weightInit, MlOnlineLogisticRegression_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlOnlineLogisticRegression(Input, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlOnlineLogisticRegression MlOnlineLogisticRegression(ISeries<double> input , double learningRate, double regularizationLambda, int labelHorizon, MlOnlineLogisticRegression_WeightInitMode weightInit, MlOnlineLogisticRegression_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlOnlineLogisticRegression(input, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlOnlineLogisticRegression MlOnlineLogisticRegression(double learningRate, double regularizationLambda, int labelHorizon, MlOnlineLogisticRegression_WeightInitMode weightInit, MlOnlineLogisticRegression_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlOnlineLogisticRegression(Input, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlOnlineLogisticRegression MlOnlineLogisticRegression(ISeries<double> input , double learningRate, double regularizationLambda, int labelHorizon, MlOnlineLogisticRegression_WeightInitMode weightInit, MlOnlineLogisticRegression_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlOnlineLogisticRegression(input, learningRate, regularizationLambda, labelHorizon, weightInit, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
