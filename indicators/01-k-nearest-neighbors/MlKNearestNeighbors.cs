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

// Write-up: https://mydailytake.com/ml-k-nearest-neighbors-indicator-for-ninjatrader-8/
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

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Search",		10100)]
	[Gui.CategoryOrder("Features",		10200)]
	[Gui.CategoryOrder("Prediction",	10300)]
	[Gui.CategoryOrder("Display",		10400)]

	#endregion

	public class MlKNearestNeighbors : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - k-Nearest Neighbors";
		public string indDescription	= "Your first machine-learning indicator for NinjaTrader 8. k-Nearest Neighbors finds the K most similar past bars to the current one (in feature space) and predicts the average forward return of those neighbors. Default features (pure trend-following): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio (ATR vs SMA-of-ATR). Features are z-score normalized using each bar's OWN local-time stats — a regime-aware comparison so a 1-σ extreme reading from one era matches a 1-σ extreme from another, even when raw values differ. The search loop is capped at barsAgo >= ForwardHorizon so neighbor outcomes are always known data — no look-ahead. Renders as a chart overlay: green triangle below the bar with a percentage + 'Long' label when a high-conviction long signal fires, red triangle above the bar with percentage + 'Short' for a short. Signals fire only when the prediction passes BOTH a magnitude gate (|prediction| >= MinPredictedReturn) and a signal-to-noise gate (|prediction| / std-dev >= MinSignalToNoise) so tiny predictions and coin-flip neighbor splits are filtered out. Public Series<double> outputs (PredictionSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Search

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Order = 01, GroupName = "Search", Name = "K (Neighbors)", Description = "Number of nearest historical bars to consult per prediction. k=1 is one expert (noisy); k=50 polls a broad crowd (smoothed). Sweet spot for futures: 10–30 with a 2000-bar lookback.")]
		public int KNeighbors { get; set; }

		[NinjaScriptProperty]
		[Range(100, 20000)]
		[Display(Order = 02, GroupName = "Search", Name = "Search Lookback (bars)", Description = "How many past bars to search through. More bars = more candidates but slower computation. Cost scales linearly with this value.")]
		public int SearchLookbackBars { get; set; }

		[NinjaScriptProperty]
		[Range(1, 50)]
		[Display(Order = 03, GroupName = "Search", Name = "Forward Horizon (bars)", Description = "How many bars ahead the model predicts. Each neighbor's forward outcome is its (Close after horizon - Close at neighbor) / Close at neighbor. The search is capped at barsAgo >= ForwardHorizon so all neighbor outcomes are already known — no look-ahead.")]
		public int ForwardHorizon { get; set; }

		#endregion

		#region Features

		[NinjaScriptProperty]
		[Range(2, 500)]
		[Display(Order = 01, GroupName = "Features", Name = "MA Period", Description = "Lookback for the trend moving average. Used in 'distance from MA' (directional position) and as the SMA window for the volatility-regime feature. Default 50.")]
		public int MaPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 100)]
		[Display(Order = 02, GroupName = "Features", Name = "ATR Period", Description = "Lookback for the ATR used to normalize distance and slope features so they're comparable across instruments and timeframes. Default 14.")]
		public int AtrPeriod { get; set; }

		[NinjaScriptProperty]
		[Range(2, 200)]
		[Display(Order = 03, GroupName = "Features", Name = "Slope Lookback", Description = "Bars over which to measure recent momentum: (Close - Close[N]) / ATR. Default 20 — captures short-term trend thrust in volatility-normalized units.")]
		public int SlopeLookback { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features", Description = "Z-score normalize each feature using a rolling window mean / standard deviation. CRITICAL when feature ranges differ. Without normalization, the wider-range feature dominates the distance metric.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Rolling window for the per-feature mean / standard deviation used in z-score normalization. 200 bars is a reasonable default — long enough to be stable, short enough to adapt to regime change.")]
		public int NormalizationLookback { get; set; }

		#endregion

		#region Prediction

		[NinjaScriptProperty]
		[Range(0.0, 0.05)]
		[Display(Order = 01, GroupName = "Prediction", Name = "Min Predicted Return", Description = "Magnitude gate: signals only fire when |prediction| meets this threshold (in returns, where 0.0005 = 5 basis points). 0 = no magnitude filter. Combined with Min Signal-to-Noise (both must pass).")]
		public double MinPredictedReturn { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 5.0)]
		[Display(Order = 02, GroupName = "Prediction", Name = "Min Signal-to-Noise", Description = "Noise gate: signals only fire when |prediction| / std-dev meets this threshold. 0 = no filter. 1.0 = prediction must exceed one std-dev of disagreement across the neighbors. Self-scaling — high-volatility regimes auto-raise the bar.")]
		public double MinSignalToNoise { get; set; }

		#endregion

		#region Display

		[Range(0, 200)]
		[Display(Order = 01, GroupName = "Display", Name = "Marker Offset (ticks)", Description = "Vertical offset of the signal triangles from the bar's high/low, in ticks. Triangle sits this far from the bar.")]
		public int MarkerOffsetTicks { get; set; }

		[Range(0, 500)]
		[Display(Order = 02, GroupName = "Display", Name = "Label Offset (ticks)", Description = "Vertical offset of the text label from the bar's high/low, in ticks. Should be GREATER than Marker Offset so the text sits beyond the triangle (bar → triangle → text). Default 16.")]
		public int LabelOffsetTicks { get; set; }

		[Display(Order = 03, GroupName = "Display", Name = "Show Labels", Description = "Render text labels (% return on line 1, Long/Short on line 2) next to each signal triangle.")]
		public bool ShowLabels { get; set; }

		[Range(8, 24)]
		[Display(Order = 04, GroupName = "Display", Name = "Label Font Size", Description = "Font size for the signal labels.")]
		public int LabelFontSize { get; set; }

		#endregion

		#region Variables

		// Fixed feature count — bumping this requires updating GetFeaturesInto.
		private const int NumFeatures = 3;

		// Public Series outputs (consumed by strategies / chained indicators).
		private Series<double>	sPrediction;
		private Series<double>	sConfidence;
		private Series<double>	sNeighborAvgFwdReturn;
		private Series<bool>	sIsLongSignal;
		private Series<bool>	sIsShortSignal;

		// Per-feature input series + rolling SMA / StdDev for z-score normalization.
		// Infinite lookback is required because SMA/StdDev wrap the custom Series.
		private Series<double>[]	featureSeries;
		private SMA[]				featureMean;
		private StdDev[]			featureStd;

		// Source indicators.
		private SMA	trendMa;
		private ATR	atr;
		private SMA	atrRegimeMa;

		// TickSize-derived offsets, resolved once in DataLoaded.
		private double markerOffsetPts;
		private double labelOffsetPts;

		// Cached label font (re-built only when LabelFontSize changes).
		private SimpleFont	labelFont;
		private int			labelFontSizeCached = -1;

		// Reusable scratch buffers — avoids allocating per candidate inside the
		// search loop. `currentNormBuf` is preserved across the loop; `scratchRaw`
		// and `candidateNormBuf` are overwritten per candidate.
		private double[]	scratchRaw;
		private double[]	currentNormBuf;
		private double[]	candidateNormBuf;

		// Top-k tracking arrays — sized at KNeighbors, allocated once in
		// DataLoaded so we don't reallocate per OnBarUpdate.
		private double[]	bestDist;
		private double[]	bestFwd;
		private int[]		bestBarsAgo;

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

				IsOverlay					= true;		// chart overlay — triangles + labels render at price levels
				DisplayInDataBox			= true;
				DrawHorizontalGridLines		= true;
				DrawVerticalGridLines		= true;
				PaintPriceMarkers			= true;
				ScaleJustification			= NinjaTrader.Gui.Chart.ScaleJustification.Right;
				IsSuspendedWhileInactive	= true;

				#endregion

				#region Defaults

				KNeighbors				= 2;
				SearchLookbackBars		= 2000;
				ForwardHorizon			= 30;

				MaPeriod				= 8;
				AtrPeriod				= 50;
				SlopeLookback			= 2;
				NormalizeFeatures		= true;
				NormalizationLookback	= 200;

				MinPredictedReturn		= 0.0015;
				MinSignalToNoise		= 0.3;

				MarkerOffsetTicks		= 4;
				LabelOffsetTicks		= 20;
				ShowLabels				= true;
				LabelFontSize			= 12;

				// Plot indices (chart overlay — Y values are PRICES):
				//   0 Long signal triangle  (green, plotted below bar's Low)
				//   1 Short signal triangle (red, plotted above bar's High)
				AddPlot(new Stroke(Brushes.LimeGreen, 5),	PlotStyle.TriangleUp,	"k-NN Long");
				AddPlot(new Stroke(Brushes.OrangeRed, 5),	PlotStyle.TriangleDown,	"k-NN Short");

				#endregion
			}
			else if (State == State.DataLoaded)
			{
				sPrediction				= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sConfidence				= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sNeighborAvgFwdReturn	= new Series<double>(this, MaximumBarsLookBack.Infinite);
				sIsLongSignal			= new Series<bool>(this,   MaximumBarsLookBack.Infinite);
				sIsShortSignal			= new Series<bool>(this,   MaximumBarsLookBack.Infinite);

				trendMa		= SMA(MaPeriod);
				atr			= ATR(AtrPeriod);
				atrRegimeMa	= SMA(atr, MaPeriod);

				// Per-feature backing Series + rolling stats. Infinite lookback
				// because SMA / StdDev wrap these and look back NormalizationLookback.
				featureSeries	= new Series<double>[NumFeatures];
				featureMean		= new SMA[NumFeatures];
				featureStd		= new StdDev[NumFeatures];
				for (int k = 0; k < NumFeatures; k++)
				{
					featureSeries[k]	= new Series<double>(this, MaximumBarsLookBack.Infinite);
					featureMean[k]		= SMA(featureSeries[k], NormalizationLookback);
					featureStd[k]		= StdDev(featureSeries[k], NormalizationLookback);
				}

				// TickSize is only valid here (not in Configure — instrument isn't loaded yet).
				markerOffsetPts	= MarkerOffsetTicks * TickSize;
				labelOffsetPts	= LabelOffsetTicks  * TickSize;

				// Scratch + top-k buffers, allocated once.
				scratchRaw			= new double[NumFeatures];
				currentNormBuf		= new double[NumFeatures];
				candidateNormBuf	= new double[NumFeatures];
				int kBest			= Math.Max(1, KNeighbors);
				bestDist			= new double[kBest];
				bestFwd				= new double[kBest];
				bestBarsAgo			= new int[kBest];
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

			// PHASE 1 — source-indicator warmup. Wait until trendMa, atr, atrRegimeMa
			// and the slope lookback have valid data before we can even COMPUTE features.
			int sourceWarmup = Math.Max(Math.Max(MaPeriod, AtrPeriod), SlopeLookback) + 1;
			if (CurrentBar < sourceWarmup) return;

			// PHASE 2 — populate featureSeries every bar (even when we can't yet
			// predict). This pre-loads the rolling SMA / StdDev so when prediction
			// starts the per-bar normalization stats are already meaningful.
			GetFeaturesInto(scratchRaw, 0);
			for (int k = 0; k < NumFeatures; k++)
				featureSeries[k][0] = scratchRaw[k];

			// PHASE 3 — prediction warmup. Need NormalizationLookback bars of feature
			// history (so stats are stable) AND ForwardHorizon bars beyond that for
			// candidates' forward outcomes.
			int predictionWarmup = sourceWarmup + NormalizationLookback + ForwardHorizon;
			if (CurrentBar < predictionWarmup) return;

			// 1) Normalize the current bar using its own local-time stats. Keep
			//    in currentNormBuf so it survives the loop's writes to candidateNormBuf.
			NormalizeInto(currentNormBuf, scratchRaw, 0);

			// 2) Search past bars for k nearest neighbors.
			//    INVARIANT: barsAgo in [ForwardHorizon, min(CurrentBar − ForwardHorizon, CurrentBar − SlopeLookback)]
			//      - lower bound: forward return must be observed data (no look-ahead)
			//      - upper bound: slope feature requires Close[barsAgo + SlopeLookback]
			int kBest = Math.Max(1, KNeighbors);
			for (int i = 0; i < kBest; i++) bestDist[i] = double.MaxValue;
			int filled = 0;

			int maxBarsAgo = Math.Min(SearchLookbackBars,
									   Math.Min(CurrentBar - ForwardHorizon, CurrentBar - SlopeLookback));
			for (int barsAgo = ForwardHorizon; barsAgo <= maxBarsAgo; barsAgo++)
			{
				GetFeaturesInto(scratchRaw, barsAgo);
				NormalizeInto(candidateNormBuf, scratchRaw, barsAgo);

				double dist = SquaredDistance(currentNormBuf, candidateNormBuf);

				double closeNeighbor	= Close[barsAgo];
				double closeFuture		= Close[barsAgo - ForwardHorizon];
				if (closeNeighbor <= 0) continue;
				double fwdReturn = (closeFuture - closeNeighbor) / closeNeighbor;

				InsertTopK(bestDist, bestFwd, bestBarsAgo, ref filled, kBest, dist, fwdReturn, barsAgo);
			}

			if (filled == 0) return;

			// 4) Aggregate: prediction = mean fwd return, confidence = std-dev.
			double sum = 0, sumSq = 0;
			for (int i = 0; i < filled; i++)
			{
				sum   += bestFwd[i];
				sumSq += bestFwd[i] * bestFwd[i];
			}
			double prediction	= sum / filled;
			double variance		= (sumSq / filled) - (prediction * prediction);
			double std			= variance > 0 ? Math.Sqrt(variance) : 0;

			sPrediction[0]				= prediction;
			sConfidence[0]				= std;
			sNeighborAvgFwdReturn[0]	= prediction;

			// 5) Two-gate signal qualification:
			//    - Magnitude gate: |prediction| >= MinPredictedReturn — rejects
			//      tiny-but-confident moves that aren't worth trading.
			//    - SNR gate: |prediction| / std >= MinSignalToNoise — rejects
			//      coin-flip neighbor splits.
			double absPrediction	= Math.Abs(prediction);
			double snr				= std > 1e-12 ? absPrediction / std : double.PositiveInfinity;
			bool meetsMagnitude		= absPrediction >= MinPredictedReturn;
			bool meetsSnr			= snr >= MinSignalToNoise;
			if (!(meetsMagnitude && meetsSnr)) return;

			// 6) Plot the triangle and (optionally) draw a text label.
			if (prediction > 0)
			{
				sIsLongSignal[0]	= true;
				Values[0][0]		= Low[0] - markerOffsetPts;
				if (ShowLabels) DrawSignalLabel(true, prediction);
			}
			else if (prediction < 0)
			{
				sIsShortSignal[0]	= true;
				Values[1][0]		= High[0] + markerOffsetPts;
				if (ShowLabels) DrawSignalLabel(false, prediction);
			}
		}

		// ─── FEATURE EXTRACTION ───────────────────────────────────────────────────
		// To change what the model considers "similar," edit this method.
		// Add or swap features here, recompile, done. NumFeatures must match.
		// Writes into a caller-supplied buffer to avoid allocating per call.

		private void GetFeaturesInto(double[] dst, int barsAgo)
		{
			double atrVal	= atr[barsAgo];
			double safeAtr	= atrVal > 1e-9 ? atrVal : TickSize;

			// Distance from MA in ATRs — directional position scaled by volatility.
			dst[0] = (Close[barsAgo] - trendMa[barsAgo]) / safeAtr;

			// N-bar slope in ATRs — recent momentum thrust, normalized.
			int slopeBack = barsAgo + SlopeLookback;
			dst[1] = slopeBack <= CurrentBar
				? (Close[barsAgo] - Close[slopeBack]) / safeAtr
				: 0;

			// Volatility regime — current ATR vs typical.
			double atrSmaVal = atrRegimeMa[barsAgo];
			dst[2] = atrSmaVal > 1e-9 ? atrVal / atrSmaVal : 1.0;
		}

		// ─── NORMALIZATION ────────────────────────────────────────────────────────
		// Z-score per feature using each bar's OWN local-time stats — the rolling
		// mean / std-dev computed over the NormalizationLookback bars before that
		// bar. A "1-σ extreme" reading from one era is comparable to a "1-σ extreme"
		// from another even though raw values differ. Without this regime-relative
		// scaling, candidates from different vol/trend regimes get judged against
		// today's distribution — which they don't belong to. Stats are drawn from
		// NT's SMA / StdDev over the per-feature input series, so
		// `featureMean[k][barsAgo]` is the local-time mean at that historical bar
		// (no look-ahead — SMA only sees bars [0..CurrentBar]).
		// Writes into a caller-supplied buffer to avoid allocating per call.

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

		private double SquaredDistance(double[] a, double[] b)
		{
			double sum = 0;
			for (int k = 0; k < NumFeatures; k++)
			{
				double d = a[k] - b[k];
				sum += d * d;
			}
			return sum;
		}

		// Top-k tracking via insertion sort. With k ≤ 100 and search lookback in
		// the thousands, the O(k) insert per candidate is negligible.
		private void InsertTopK(double[] bestDist, double[] bestFwd, int[] bestBarsAgo,
								ref int filled, int kBest,
								double dist, double fwd, int barsAgo)
		{
			if (filled < kBest)
			{
				int pos = filled;
				while (pos > 0 && bestDist[pos - 1] > dist) { pos--; }
				for (int i = filled; i > pos; i--)
				{
					bestDist[i]		= bestDist[i - 1];
					bestFwd[i]		= bestFwd[i - 1];
					bestBarsAgo[i]	= bestBarsAgo[i - 1];
				}
				bestDist[pos]		= dist;
				bestFwd[pos]		= fwd;
				bestBarsAgo[pos]	= barsAgo;
				filled++;
				return;
			}

			if (dist >= bestDist[kBest - 1]) return;
			int p = kBest - 1;
			while (p > 0 && bestDist[p - 1] > dist) { p--; }
			for (int i = kBest - 1; i > p; i--)
			{
				bestDist[i]		= bestDist[i - 1];
				bestFwd[i]		= bestFwd[i - 1];
				bestBarsAgo[i]	= bestBarsAgo[i - 1];
			}
			bestDist[p]		= dist;
			bestFwd[p]		= fwd;
			bestBarsAgo[p]	= barsAgo;
		}

		// Two-line signal label. Percent is always the line CLOSEST to the
		// triangle so the visual hierarchy is consistent: bar → triangle → %
		// → direction. For longs (label below bar) that means "% / direction";
		// for shorts (label above bar) that means "direction / %".
		// Tag includes CurrentBar so each signal persists historically without
		// overwriting earlier ones.
		private void DrawSignalLabel(bool isLong, double prediction)
		{
			if (labelFontSizeCached != LabelFontSize)
			{
				labelFont				= new SimpleFont("Arial", LabelFontSize);
				labelFontSizeCached		= LabelFontSize;
			}

			string pctText = string.Format("{0}{1:0.00}%",
				prediction >= 0 ? "+" : "",
				prediction * 100.0);
			string dirText = isLong ? "Long" : "Short";

			// Always put % on the line closest to the triangle.
			string text = isLong
				? pctText + "\n" + dirText		// label below bar — top line is closest to triangle
				: dirText + "\n" + pctText;		// label above bar — bottom line is closest to triangle

			// Draw.Text centers the text block vertically on y. To make
			// LabelOffsetTicks measure from the bar to the EDGE of the label
			// (not the center), shift y by half the block's height.
			double halfBlockHeight	= LabelFontSize * 0.6 * TickSize;
			double y = isLong
				? Low[0]  - labelOffsetPts - halfBlockHeight
				: High[0] + labelOffsetPts + halfBlockHeight;

			Brush brush = isLong ? Brushes.LimeGreen : Brushes.OrangeRed;
			string tag  = (isLong ? "knn-long-" : "knn-short-") + CurrentBar;

			Draw.Text(this, tag, false, text, 0, y, 0, brush,
				labelFont, TextAlignment.Center, Brushes.Transparent, Brushes.Transparent, 0);
		}

		#region Public Series

		[Browsable(false)] [XmlIgnore] public Series<double> PredictionSeries           { get { Update(); return sPrediction;           } }
		[Browsable(false)] [XmlIgnore] public Series<double> ConfidenceSeries           { get { Update(); return sConfidence;           } }
		[Browsable(false)] [XmlIgnore] public Series<double> NeighborAvgFwdReturnSeries { get { Update(); return sNeighborAvgFwdReturn; } }
		[Browsable(false)] [XmlIgnore] public Series<bool>   IsLongSignalSeries         { get { Update(); return sIsLongSignal;         } }
		[Browsable(false)] [XmlIgnore] public Series<bool>   IsShortSignalSeries        { get { Update(); return sIsShortSignal;        } }

		#endregion
	}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private indMyDailyTake.MlKNearestNeighbors[] cacheMlKNearestNeighbors;
		public indMyDailyTake.MlKNearestNeighbors MlKNearestNeighbors(int kNeighbors, int searchLookbackBars, int forwardHorizon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minPredictedReturn, double minSignalToNoise)
		{
			return MlKNearestNeighbors(Input, kNeighbors, searchLookbackBars, forwardHorizon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minPredictedReturn, minSignalToNoise);
		}

		public indMyDailyTake.MlKNearestNeighbors MlKNearestNeighbors(ISeries<double> input, int kNeighbors, int searchLookbackBars, int forwardHorizon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minPredictedReturn, double minSignalToNoise)
		{
			if (cacheMlKNearestNeighbors != null)
				for (int idx = 0; idx < cacheMlKNearestNeighbors.Length; idx++)
					if (cacheMlKNearestNeighbors[idx] != null && cacheMlKNearestNeighbors[idx].KNeighbors == kNeighbors && cacheMlKNearestNeighbors[idx].SearchLookbackBars == searchLookbackBars && cacheMlKNearestNeighbors[idx].ForwardHorizon == forwardHorizon && cacheMlKNearestNeighbors[idx].MaPeriod == maPeriod && cacheMlKNearestNeighbors[idx].AtrPeriod == atrPeriod && cacheMlKNearestNeighbors[idx].SlopeLookback == slopeLookback && cacheMlKNearestNeighbors[idx].NormalizeFeatures == normalizeFeatures && cacheMlKNearestNeighbors[idx].NormalizationLookback == normalizationLookback && cacheMlKNearestNeighbors[idx].MinPredictedReturn == minPredictedReturn && cacheMlKNearestNeighbors[idx].MinSignalToNoise == minSignalToNoise && cacheMlKNearestNeighbors[idx].EqualsInput(input))
						return cacheMlKNearestNeighbors[idx];
			return CacheIndicator<indMyDailyTake.MlKNearestNeighbors>(new indMyDailyTake.MlKNearestNeighbors(){ KNeighbors = kNeighbors, SearchLookbackBars = searchLookbackBars, ForwardHorizon = forwardHorizon, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinPredictedReturn = minPredictedReturn, MinSignalToNoise = minSignalToNoise }, input, ref cacheMlKNearestNeighbors);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlKNearestNeighbors MlKNearestNeighbors(int kNeighbors, int searchLookbackBars, int forwardHorizon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minPredictedReturn, double minSignalToNoise)
		{
			return indicator.MlKNearestNeighbors(Input, kNeighbors, searchLookbackBars, forwardHorizon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minPredictedReturn, minSignalToNoise);
		}

		public Indicators.indMyDailyTake.MlKNearestNeighbors MlKNearestNeighbors(ISeries<double> input , int kNeighbors, int searchLookbackBars, int forwardHorizon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minPredictedReturn, double minSignalToNoise)
		{
			return indicator.MlKNearestNeighbors(input, kNeighbors, searchLookbackBars, forwardHorizon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minPredictedReturn, minSignalToNoise);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlKNearestNeighbors MlKNearestNeighbors(int kNeighbors, int searchLookbackBars, int forwardHorizon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minPredictedReturn, double minSignalToNoise)
		{
			return indicator.MlKNearestNeighbors(Input, kNeighbors, searchLookbackBars, forwardHorizon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minPredictedReturn, minSignalToNoise);
		}

		public Indicators.indMyDailyTake.MlKNearestNeighbors MlKNearestNeighbors(ISeries<double> input , int kNeighbors, int searchLookbackBars, int forwardHorizon, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minPredictedReturn, double minSignalToNoise)
		{
			return indicator.MlKNearestNeighbors(input, kNeighbors, searchLookbackBars, forwardHorizon, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minPredictedReturn, minSignalToNoise);
		}
	}
}

#endregion
