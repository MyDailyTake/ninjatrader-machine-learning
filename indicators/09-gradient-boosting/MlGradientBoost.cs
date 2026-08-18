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

public enum MlGradientBoost_LabelMode { CloseToClose, FavorableExcursion }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Boosting",	10100)]
	[Gui.CategoryOrder("Learning",	10200)]
	[Gui.CategoryOrder("Features",	10300)]
	[Gui.CategoryOrder("Signal",	10400)]
	[Gui.CategoryOrder("Display",	10500)]

	#endregion

	public class MlGradientBoost : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Gradient-Boosted Trees";
		public string indDescription	= "A gradient-boosted decision-tree model for NinjaTrader 8 — the algorithm at the heart of XGBoost. Where the Random Forest grew many independent trees and averaged them, gradient boosting grows trees one at a time, in sequence: each new tree is fitted to the errors the current ensemble still makes. It works in log-odds space — every tree adds a small correction to a running score, and the prediction is the sigmoid of that score. Each tree is built with the second-order (Newton) objective XGBoost uses: every split is scored by a gain computed from the gradient and hessian of the logistic loss, leaf values are the regularized optimal weights -G/(H+lambda), and a shrinkage learning rate keeps each tree's contribution small so later trees can keep correcting. Like the Random Forest, a boosted ensemble is a batch learner — it rebuilds from a rolling window of look-ahead-safe (feature, label) examples every RetrainInterval bars. Default features (same as the rest of the series for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the prior posts. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Boosting

		[NinjaScriptProperty]
		[Range(1, 500)]
		[Display(Order = 01, GroupName = "Boosting", Name = "Number of Trees", Description = "Number of boosting rounds — one tree is added per round. Each new tree corrects the errors the current ensemble still makes. More trees fit the training data more closely; paired with a small Learning Rate, more rounds generalize better at the cost of compute. Default 60.")]
		public int NumTrees { get; set; }

		[NinjaScriptProperty]
		[Range(1, 10)]
		[Display(Order = 02, GroupName = "Boosting", Name = "Max Tree Depth", Description = "Maximum depth of each boosted tree. Gradient boosting works best with shallow trees — weak learners — and many of them. Depth 2 to 4 is typical; the default is 3.")]
		public int MaxDepth { get; set; }

		[NinjaScriptProperty]
		[Range(0.01, 1.0)]
		[Display(Order = 03, GroupName = "Boosting", Name = "Learning Rate", Description = "Shrinkage applied to every tree's contribution (XGBoost's 'eta'). A small rate makes each tree a gentle correction so later trees can keep refining the fit. Lower rate = needs more trees but generalizes better. Default 0.10.")]
		public double LearningRate { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 100.0)]
		[Display(Order = 04, GroupName = "Boosting", Name = "Min Child Weight", Description = "Minimum total hessian a split's child node must hold. The hessian of the logistic loss measures prediction confidence, so this stops the model from carving leaves out of a few uncertain samples. Higher = more conservative. Default 1.0.")]
		public double MinChildWeight { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 20.0)]
		[Display(Order = 05, GroupName = "Boosting", Name = "L2 Lambda", Description = "L2 regularization on the leaf weights (XGBoost's 'lambda'). It appears in the denominator of both the leaf-weight and split-gain formulas, shrinking leaf values toward zero and damping over-confident trees. Default 1.0.")]
		public double L2Lambda { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 10.0)]
		[Display(Order = 06, GroupName = "Boosting", Name = "Gamma (min split gain)", Description = "Minimum loss reduction required to make a split (XGBoost's 'gamma'). A split is only kept if its gain exceeds this penalty, so larger values prune weak splits and grow simpler trees. Default 0.0.")]
		public double Gamma { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(50, 3000)]
		[Display(Order = 01, GroupName = "Learning", Name = "Training Window (bars)", Description = "Number of recent look-ahead-safe (feature, label) examples kept for training. Each ensemble rebuild is fitted to this rolling window, so older examples fall out over time. Default 300.")]
		public int TrainingWindow { get; set; }

		[NinjaScriptProperty]
		[Range(1, 500)]
		[Display(Order = 02, GroupName = "Learning", Name = "Retrain Interval (bars)", Description = "How often the whole ensemble is rebuilt. A boosted ensemble cannot be updated one bar at a time, so the model retrains in batches every N bars from the current Training Window. Smaller = more responsive but more compute. Default 25.")]
		public int RetrainInterval { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. Each bar contributes one training example using the features from N bars ago, whose forward outcome is now known — this keeps training look-ahead-safe.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. CloseToClose: y = 1 if Close at the end of the Label Horizon window is above Close at the training bar. FavorableExcursion: y = 1 if the long-side favorable move beat the short-side move during the window (uses bar highs/lows); bars below Min Favorable Move are skipped as chop.")]
		public MlGradientBoost_LabelMode LabelMode { get; set; }

		[NinjaScriptProperty]
		[Range(0.0, 5.0)]
		[Display(Order = 05, GroupName = "Learning", Name = "Min Favorable Move (ATRs)", Description = "ONLY USED WHEN Label Mode = FavorableExcursion. Minimum favorable excursion (in ATRs at entry) required during the post-bar window for the bar to become a training example.")]
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
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features (Z-Score)", Description = "Master toggle for z-score normalization. Boosted trees are scale-invariant to fixed transforms, but the rolling z-score still helps by making each feature stationary against its own recent distribution. Recommended ON for parity with the rest of the series.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Window used to compute the rolling mean / stddev that z-score the features. Each bar uses its own local-time stats.")]
		public int NormalizationLookback { get; set; }

		#endregion

		#region Signal

		[NinjaScriptProperty]
		[Range(0.0, 0.49)]
		[Display(Order = 01, GroupName = "Signal", Name = "Min Probability Edge", Description = "How far the ensemble's predicted probability of an up move must be from 0.5 before a signal fires.")]
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

		#region Tree storage

		// One node of a boosted regression tree. Internal nodes use
		// Feature/Threshold/Left/Right; leaves are marked by Feature == -1 and carry
		// LeafWeight — the (shrunk) log-odds contribution this tree adds.
		private struct GbNode
		{
			public int		Feature;
			public double	Threshold;
			public int		Left;
			public int		Right;
			public double	LeafWeight;
		}

		// (feature value, gradient, hessian) triple — sorted by value when searching
		// for a split.
		private struct GbSplitItem : IComparable<GbSplitItem>
		{
			public double	Value;
			public double	Grad;
			public double	Hess;
			public int CompareTo(GbSplitItem other) { return Value.CompareTo(other.Value); }
		}

		#endregion

		#region Private Fields

		private const int NumFeatures		= 3;
		private const int MinTrainToBuild	= 30;   // examples required before the first rebuild

		// Source indicators
		private SMA	trendMa;
		private ATR	atr;
		private SMA	atrRegimeMa;

		// Per-feature backing series + rolling stats (z-score normalization).
		private Series<double>[]	featureSeries;
		private SMA[]				featureMean;
		private StdDev[]			featureStd;

		// ─── Training buffer ───
		// A circular store of recent (normalized feature vector, label) examples.
		// trainCount caps at TrainingWindow; trainWp is the write pointer. The
		// ensemble is fitted to all physical slots [0, trainCount).
		private double[][]	trainFeat;
		private int[]		trainLabel;
		private int			trainCount;
		private int			trainWp;

		// ─── Boosting work buffers (one slot per training example) ───
		private double[]	scoreBuf;   // running ensemble score F_i (log-odds)
		private double[]	gradBuf;    // per-round gradient  g_i = p_i - y_i
		private double[]	hessBuf;    // per-round hessian   h_i = p_i (1 - p_i)

		// ─── Ensemble ───
		// trees[m] is a flat node array for tree m; treeNodeCount[m] is its node count.
		private GbNode[][]	trees;
		private int[]		treeNodeCount;
		private int			maxNodes;
		private double		baseScore;      // initial log-odds, set from the base rate
		private bool		ensembleReady;
		private int			barsSinceRetrain;

		// Build scratch — allocated once, reused on every rebuild.
		private int[]			sampleIdx;      // example indices for the tree under construction
		private GbSplitItem[]	splitScratch;   // (value, grad, hess) triples sorted during split search

		// Per-bar feature scratch.
		private double[]	scratchRaw;
		private double[]	scratchNorm;
		private double[]	trainRaw;
		private double[]	trainNorm;

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

				NumTrees				= 60;
				MaxDepth				= 3;
				LearningRate			= 0.10;
				MinChildWeight			= 1.0;
				L2Lambda				= 1.0;
				Gamma					= 0.0;

				TrainingWindow			= 300;
				RetrainInterval			= 25;
				LabelHorizon			= 2;
				LabelMode				= MlGradientBoost_LabelMode.CloseToClose;
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

				AddPlot(new Stroke(Brushes.LimeGreen, 5),	PlotStyle.TriangleUp,	"GB Long");
				AddPlot(new Stroke(Brushes.OrangeRed, 5),	PlotStyle.TriangleDown,	"GB Short");

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

				// Training buffer
				trainFeat	= new double[TrainingWindow][];
				for (int i = 0; i < TrainingWindow; i++) trainFeat[i] = new double[NumFeatures];
				trainLabel	= new int[TrainingWindow];
				trainCount	= 0;
				trainWp		= 0;

				// Boosting work buffers
				scoreBuf	= new double[TrainingWindow];
				gradBuf		= new double[TrainingWindow];
				hessBuf		= new double[TrainingWindow];

				// Ensemble — a tree has at most 2^(MaxDepth+1)-1 nodes by depth, and at
				// most 2*trainCount-1 by sample count; size to the smaller bound.
				maxNodes		= Math.Min((1 << (MaxDepth + 1)) - 1, 2 * TrainingWindow - 1);
				trees			= new GbNode[NumTrees][];
				for (int m = 0; m < NumTrees; m++) trees[m] = new GbNode[maxNodes];
				treeNodeCount	= new int[NumTrees];
				baseScore		= 0.0;
				ensembleReady	= false;
				barsSinceRetrain = 0;

				// Build scratch
				sampleIdx		= new int[TrainingWindow];
				splitScratch	= new GbSplitItem[TrainingWindow];

				scratchRaw	= new double[NumFeatures];
				scratchNorm	= new double[NumFeatures];
				trainRaw	= new double[NumFeatures];
				trainNorm	= new double[NumFeatures];

				lastSignalBar	= -1;
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

			int sourceWarmup = Math.Max(Math.Max(MaPeriod, AtrPeriod), SlopeLookback) + 1;
			if (CurrentBar < sourceWarmup) return;

			GetFeaturesInto(scratchRaw, 0);
			for (int k = 0; k < NumFeatures; k++)
				featureSeries[k][0] = scratchRaw[k];

			int predictionWarmup = sourceWarmup + NormalizationLookback;
			if (CurrentBar < predictionWarmup) return;

			NormalizeInto(scratchNorm, scratchRaw, 0);

			// 1) Collect the look-ahead-safe training example from LabelHorizon ago,
			//    then rebuild the ensemble every RetrainInterval bars.
			if (CurrentBar >= predictionWarmup + LabelHorizon)
			{
				CollectTrainingExample();

				barsSinceRetrain++;
				if (barsSinceRetrain >= RetrainInterval && trainCount >= MinTrainToBuild)
				{
					RebuildEnsemble();
					ensembleReady		= true;
					barsSinceRetrain	= 0;
				}
			}

			if (!ensembleReady) return;

			// 2) Live prediction — run the current bar's features through the ensemble.
			double pUp = EnsemblePredict(scratchNorm);
			sProbabilityUp[0]	= pUp;
			sConfidence[0]		= Math.Abs(pUp - 0.5) * 2.0;

			// 3) Signal gate — wait for a usable dataset to have accumulated.
			int signalWarmup = predictionWarmup + LabelHorizon + 100;
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

		// ─── TRAINING EXAMPLE COLLECTION ──────────────────────────────────────────
		// The bar from LabelHorizon ago is now fully resolved: its forward outcome is
		// observable. Pair that bar's normalized features with the realized label and
		// append it to the rolling training buffer.

		private void CollectTrainingExample()
		{
			GetFeaturesInto(trainRaw, LabelHorizon);
			NormalizeInto(trainNorm, trainRaw, LabelHorizon);

			bool include	= false;
			int  y			= 0;

			if (LabelMode == MlGradientBoost_LabelMode.CloseToClose)
			{
				y		= Close[0] > Close[LabelHorizon] ? 1 : 0;
				include	= true;
			}
			else // FavorableExcursion
			{
				double closeAtTrain	= Close[LabelHorizon];
				double atrAtTrain	= atr[LabelHorizon];
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
					y		= mfeLong > mfeShort ? 1 : 0;
					include	= true;
				}
			}

			if (!include) return;

			for (int k = 0; k < NumFeatures; k++) trainFeat[trainWp][k] = trainNorm[k];
			trainLabel[trainWp] = y;
			trainWp = (trainWp + 1) % TrainingWindow;
			if (trainCount < TrainingWindow) trainCount++;
		}

		// ─── ENSEMBLE REBUILD (the boosting loop) ─────────────────────────────────
		// Start from a flat base score, then add NumTrees trees one at a time. Before
		// each tree, recompute the gradient and hessian of the logistic loss at the
		// current scores; the tree is fitted to those, and its (shrunk) output is
		// added back into every example's running score.

		private void RebuildEnsemble()
		{
			int n = trainCount;

			// Base score — the global log-odds of the training set's up-rate.
			int posCount = 0;
			for (int i = 0; i < n; i++) posCount += trainLabel[i];
			double p0 = (double)posCount / n;
			if (p0 < 0.01) p0 = 0.01;
			if (p0 > 0.99) p0 = 0.99;
			baseScore = Math.Log(p0 / (1.0 - p0));

			for (int i = 0; i < n; i++) scoreBuf[i] = baseScore;

			for (int m = 0; m < NumTrees; m++)
			{
				// Gradient + hessian of the logistic loss at the current scores.
				// The hessian is floored so the second-order objective stays
				// well-defined even when a prediction saturates.
				for (int i = 0; i < n; i++)
				{
					double p = Sigmoid(scoreBuf[i]);
					gradBuf[i] = p - trainLabel[i];
					hessBuf[i] = Math.Max(p * (1.0 - p), 1e-6);
				}

				// Fit tree m to (g, h) over all n examples.
				for (int i = 0; i < n; i++) sampleIdx[i] = i;
				treeNodeCount[m] = 0;
				BuildNode(m, 0, n, 0);

				// Add the new tree's contribution to every example's score.
				for (int i = 0; i < n; i++)
					scoreBuf[i] += PredictTree(m, trainFeat[i]);
			}
		}

		// ─── TREE BUILD ───────────────────────────────────────────────────────────
		// Recursively grow tree `treeIdx`. sampleIdx[lo..hi) holds the example slots
		// for this node; the node is partitioned in place by the chosen split. A node
		// becomes a leaf at the depth cap or when no split yields positive gain.

		private int BuildNode(int treeIdx, int lo, int hi, int depth)
		{
			GbNode[] tree	= trees[treeIdx];
			int nodeIdx		= treeNodeCount[treeIdx]++;

			double G = 0.0, H = 0.0;
			for (int s = lo; s < hi; s++)
			{
				int ex = sampleIdx[s];
				G += gradBuf[ex];
				H += hessBuf[ex];
			}

			int    bestFeat	= -1;
			double bestThr	= 0.0;
			int    mid		= lo;
			if (depth < MaxDepth && FindBestSplit(lo, hi, G, H, out bestFeat, out bestThr))
			{
				mid = Partition(lo, hi, bestFeat, bestThr);
				if (mid <= lo || mid >= hi) bestFeat = -1;   // degenerate split — make a leaf
			}

			if (bestFeat < 0)
			{
				// Leaf weight = the regularized optimal Newton step, shrunk by the
				// learning rate:  w = LearningRate * ( -G / (H + lambda) ).
				tree[nodeIdx].Feature    = -1;
				tree[nodeIdx].LeafWeight = LearningRate * (-G / (H + L2Lambda));
				return nodeIdx;
			}

			int left  = BuildNode(treeIdx, lo,  mid, depth + 1);
			int right = BuildNode(treeIdx, mid, hi,  depth + 1);

			tree[nodeIdx].Feature   = bestFeat;
			tree[nodeIdx].Threshold = bestThr;
			tree[nodeIdx].Left      = left;
			tree[nodeIdx].Right     = right;
			return nodeIdx;
		}

		// ─── SPLIT SEARCH ─────────────────────────────────────────────────────────
		// Score every candidate split across all features by the XGBoost gain:
		//   gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
		// A split is kept only if its gain is positive and both children hold at
		// least MinChildWeight of total hessian.

		private bool FindBestSplit(int lo, int hi, double G, double H, out int bestFeat, out double bestThr)
		{
			bestFeat = -1;
			bestThr  = 0.0;

			int    n          = hi - lo;
			double parentTerm = G * G / (H + L2Lambda);
			double bestGain   = 0.0;   // a split must yield strictly positive gain

			for (int f = 0; f < NumFeatures; f++)
			{
				for (int s = 0; s < n; s++)
				{
					int ex = sampleIdx[lo + s];
					splitScratch[s].Value = trainFeat[ex][f];
					splitScratch[s].Grad  = gradBuf[ex];
					splitScratch[s].Hess  = hessBuf[ex];
				}
				Array.Sort(splitScratch, 0, n);

				double GL = 0.0, HL = 0.0;
				for (int i = 0; i < n - 1; i++)
				{
					GL += splitScratch[i].Grad;
					HL += splitScratch[i].Hess;

					if (splitScratch[i].Value == splitScratch[i + 1].Value) continue;

					double GR = G - GL;
					double HR = H - HL;
					if (HL < MinChildWeight || HR < MinChildWeight) continue;

					double gain = 0.5 * (GL * GL / (HL + L2Lambda)
									   + GR * GR / (HR + L2Lambda)
									   - parentTerm) - Gamma;

					if (gain > bestGain)
					{
						bestGain = gain;
						bestFeat = f;
						bestThr  = 0.5 * (splitScratch[i].Value + splitScratch[i + 1].Value);
					}
				}
			}

			return bestFeat >= 0;
		}

		// Rearrange sampleIdx[lo..hi) so examples with feature[feat] <= thr come
		// first; the returned index is the start of the right child.

		private int Partition(int lo, int hi, int feat, double thr)
		{
			int mid = lo;
			for (int s = lo; s < hi; s++)
			{
				if (trainFeat[sampleIdx[s]][feat] <= thr)
				{
					int tmp = sampleIdx[mid];
					sampleIdx[mid] = sampleIdx[s];
					sampleIdx[s]   = tmp;
					mid++;
				}
			}
			return mid;
		}

		// ─── PREDICTION ───────────────────────────────────────────────────────────
		// The ensemble score is the base score plus every tree's leaf weight; the
		// probability is the sigmoid of that score.

		private double EnsemblePredict(double[] feat)
		{
			double score = baseScore;
			for (int m = 0; m < NumTrees; m++) score += PredictTree(m, feat);
			return Sigmoid(score);
		}

		private double PredictTree(int t, double[] feat)
		{
			GbNode[] tree = trees[t];
			int idx = 0;
			while (tree[idx].Feature >= 0)
				idx = feat[tree[idx].Feature] <= tree[idx].Threshold ? tree[idx].Left : tree[idx].Right;
			return tree[idx].LeafWeight;
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
			string tag  = (isLong ? "gb-long-lbl-" : "gb-short-lbl-") + CurrentBar;

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
		private indMyDailyTake.MlGradientBoost[] cacheMlGradientBoost;
		public indMyDailyTake.MlGradientBoost MlGradientBoost(int numTrees, int maxDepth, double learningRate, double minChildWeight, double l2Lambda, double gamma, int trainingWindow, int retrainInterval, int labelHorizon, MlGradientBoost_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlGradientBoost(Input, numTrees, maxDepth, learningRate, minChildWeight, l2Lambda, gamma, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlGradientBoost MlGradientBoost(ISeries<double> input, int numTrees, int maxDepth, double learningRate, double minChildWeight, double l2Lambda, double gamma, int trainingWindow, int retrainInterval, int labelHorizon, MlGradientBoost_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlGradientBoost != null)
				for (int idx = 0; idx < cacheMlGradientBoost.Length; idx++)
					if (cacheMlGradientBoost[idx] != null && cacheMlGradientBoost[idx].NumTrees == numTrees && cacheMlGradientBoost[idx].MaxDepth == maxDepth && cacheMlGradientBoost[idx].LearningRate == learningRate && cacheMlGradientBoost[idx].MinChildWeight == minChildWeight && cacheMlGradientBoost[idx].L2Lambda == l2Lambda && cacheMlGradientBoost[idx].Gamma == gamma && cacheMlGradientBoost[idx].TrainingWindow == trainingWindow && cacheMlGradientBoost[idx].RetrainInterval == retrainInterval && cacheMlGradientBoost[idx].LabelHorizon == labelHorizon && cacheMlGradientBoost[idx].LabelMode == labelMode && cacheMlGradientBoost[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlGradientBoost[idx].MaPeriod == maPeriod && cacheMlGradientBoost[idx].AtrPeriod == atrPeriod && cacheMlGradientBoost[idx].SlopeLookback == slopeLookback && cacheMlGradientBoost[idx].NormalizeFeatures == normalizeFeatures && cacheMlGradientBoost[idx].NormalizationLookback == normalizationLookback && cacheMlGradientBoost[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlGradientBoost[idx].SignalCooldownBars == signalCooldownBars && cacheMlGradientBoost[idx].EqualsInput(input))
						return cacheMlGradientBoost[idx];
			return CacheIndicator<indMyDailyTake.MlGradientBoost>(new indMyDailyTake.MlGradientBoost(){ NumTrees = numTrees, MaxDepth = maxDepth, LearningRate = learningRate, MinChildWeight = minChildWeight, L2Lambda = l2Lambda, Gamma = gamma, TrainingWindow = trainingWindow, RetrainInterval = retrainInterval, LabelHorizon = labelHorizon, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlGradientBoost);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlGradientBoost MlGradientBoost(int numTrees, int maxDepth, double learningRate, double minChildWeight, double l2Lambda, double gamma, int trainingWindow, int retrainInterval, int labelHorizon, MlGradientBoost_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlGradientBoost(Input, numTrees, maxDepth, learningRate, minChildWeight, l2Lambda, gamma, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlGradientBoost MlGradientBoost(ISeries<double> input , int numTrees, int maxDepth, double learningRate, double minChildWeight, double l2Lambda, double gamma, int trainingWindow, int retrainInterval, int labelHorizon, MlGradientBoost_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlGradientBoost(input, numTrees, maxDepth, learningRate, minChildWeight, l2Lambda, gamma, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlGradientBoost MlGradientBoost(int numTrees, int maxDepth, double learningRate, double minChildWeight, double l2Lambda, double gamma, int trainingWindow, int retrainInterval, int labelHorizon, MlGradientBoost_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlGradientBoost(Input, numTrees, maxDepth, learningRate, minChildWeight, l2Lambda, gamma, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlGradientBoost MlGradientBoost(ISeries<double> input , int numTrees, int maxDepth, double learningRate, double minChildWeight, double l2Lambda, double gamma, int trainingWindow, int retrainInterval, int labelHorizon, MlGradientBoost_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlGradientBoost(input, numTrees, maxDepth, learningRate, minChildWeight, l2Lambda, gamma, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
