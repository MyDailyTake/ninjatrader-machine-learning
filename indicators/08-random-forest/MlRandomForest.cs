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

// Write-up: https://mydailytake.com/ml-random-forest-ninjatrader-8/
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

public enum MlRandomForest_LabelMode { CloseToClose, FavorableExcursion }

namespace NinjaTrader.NinjaScript.Indicators.indMyDailyTake
{
	#region Categories

	[Gui.CategoryOrder("Architecture",	10100)]
	[Gui.CategoryOrder("Learning",		10200)]
	[Gui.CategoryOrder("Features",		10300)]
	[Gui.CategoryOrder("Signal",		10400)]
	[Gui.CategoryOrder("Display",		10500)]

	#endregion

	public class MlRandomForest : Indicator
	{
		#region Versioning

		public string indVersion		= "v1.0";
		public string indName			= "ML - Random Forest";
		public string indDescription	= "A Random Forest classifier for NinjaTrader 8 — the first non-neural model in the Learn NinjaScript ML series. Instead of nudging weights with gradient descent, a Random Forest is an ensemble of decision trees: each tree is grown on a bootstrap sample of the training data, and at every split it considers a random subset of features, so the trees disagree in useful ways. The forest's prediction is the average of the trees' votes. A decision tree is a fixed structure — it cannot be updated one bar at a time the way a neural net can — so this indicator retrains in batches: every RetrainInterval bars it rebuilds the whole forest from the most recent TrainingWindow look-ahead-safe (feature, label) examples. Splits are chosen by Gini impurity. Default features (same as the rest of the series for direct comparability): distance from MA in ATRs, N-bar slope in ATRs, and a volatility regime ratio. Two label modes (CloseToClose / FavorableExcursion) with the same semantics as the prior posts. Renders as a chart overlay with green/red triangle markers and P(up) labels. Public Series<double> outputs (ProbabilityUpSeries, ConfidenceSeries, IsLongSignalSeries, IsShortSignalSeries) let strategies consume the model directly.";

		public override string DisplayName { get { return string.Format("{0} {1}", indName, indVersion); } }

		#endregion

		#region Architecture

		[NinjaScriptProperty]
		[Range(1, 300)]
		[Display(Order = 01, GroupName = "Architecture", Name = "Number of Trees", Description = "Number of decision trees in the forest. Each tree is grown on its own bootstrap sample, and the forest prediction averages all of their votes. More trees = a smoother, lower-variance prediction but proportionally more compute on each retrain. Default 50.")]
		public int NumTrees { get; set; }

		[NinjaScriptProperty]
		[Range(1, 20)]
		[Display(Order = 02, GroupName = "Architecture", Name = "Max Tree Depth", Description = "Maximum depth of each decision tree. Deeper trees can carve more detailed regions but are more prone to fitting noise. With only three features and noisy financial data, shallow trees in a large forest tend to generalize better. Default 6.")]
		public int MaxDepth { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Architecture", Name = "Min Samples per Leaf", Description = "Minimum number of training samples a node must keep on each side of a split. Larger values force bigger, more stable leaves and resist overfitting; smaller values let trees carve finer regions. Default 8.")]
		public int MinSamplesLeaf { get; set; }

		[NinjaScriptProperty]
		[Range(1, 3)]
		[Display(Order = 04, GroupName = "Architecture", Name = "Features per Split", Description = "Number of features (out of 3) randomly considered at each split — the 'mtry' of a Random Forest. Values below 3 decorrelate the trees, which is what makes the ensemble work. Default 2.")]
		public int FeaturesPerSplit { get; set; }

		[NinjaScriptProperty]
		[Range(0, 999999)]
		[Display(Order = 05, GroupName = "Architecture", Name = "Random Seed", Description = "Seed for the bootstrap sampling and the per-split feature selection. The same seed and the same bar history produce the same forest — useful for reproducible testing.")]
		public int RandomSeed { get; set; }

		#endregion

		#region Learning

		[NinjaScriptProperty]
		[Range(50, 3000)]
		[Display(Order = 01, GroupName = "Learning", Name = "Training Window (bars)", Description = "Number of recent look-ahead-safe (feature, label) examples kept for training. Each forest rebuild draws its bootstrap samples from this rolling window, so older examples fall out over time. Default 300.")]
		public int TrainingWindow { get; set; }

		[NinjaScriptProperty]
		[Range(1, 500)]
		[Display(Order = 02, GroupName = "Learning", Name = "Retrain Interval (bars)", Description = "How often the whole forest is rebuilt. A decision tree cannot be updated one bar at a time, so the model retrains in batches every N bars from the current Training Window. Smaller = more responsive but more compute. Default 25.")]
		public int RetrainInterval { get; set; }

		[NinjaScriptProperty]
		[Range(1, 200)]
		[Display(Order = 03, GroupName = "Learning", Name = "Label Horizon (bars)", Description = "How many bars ahead the realized direction is observed. Each bar contributes one training example using the features from N bars ago, whose forward outcome is now known — this keeps training look-ahead-safe.")]
		public int LabelHorizon { get; set; }

		[NinjaScriptProperty]
		[Display(Order = 04, GroupName = "Learning", Name = "Label Mode", Description = "How the training label is defined. CloseToClose: y = 1 if Close at the end of the Label Horizon window is above Close at the training bar. FavorableExcursion: y = 1 if the long-side favorable move beat the short-side move during the window (uses bar highs/lows); bars below Min Favorable Move are skipped as chop.")]
		public MlRandomForest_LabelMode LabelMode { get; set; }

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
		[Display(Order = 04, GroupName = "Features", Name = "Normalize Features (Z-Score)", Description = "Master toggle for z-score normalization. Decision trees are scale-invariant to fixed transforms, but the rolling z-score still helps by making each feature stationary against its own recent distribution. Recommended ON for parity with the rest of the series.")]
		public bool NormalizeFeatures { get; set; }

		[NinjaScriptProperty]
		[Range(50, 2000)]
		[Display(Order = 05, GroupName = "Features", Name = "Normalization Lookback (bars)", Description = "Window used to compute the rolling mean / stddev that z-score the features. Each bar uses its own local-time stats.")]
		public int NormalizationLookback { get; set; }

		#endregion

		#region Signal

		[NinjaScriptProperty]
		[Range(0.0, 0.49)]
		[Display(Order = 01, GroupName = "Signal", Name = "Min Probability Edge", Description = "How far the forest's predicted probability of an up move must be from 0.5 before a signal fires.")]
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

		// One node of a decision tree. Internal nodes use Feature/Threshold/Left/Right;
		// leaves are marked by Feature == -1 and carry LeafProb (the leaf's P(up)).
		private struct RfNode
		{
			public int		Feature;
			public double	Threshold;
			public int		Left;
			public int		Right;
			public double	LeafProb;
		}

		// (feature value, label) pair — sorted by value when searching for a split.
		private struct SplitItem : IComparable<SplitItem>
		{
			public double	Value;
			public int		Label;
			public int CompareTo(SplitItem other) { return Value.CompareTo(other.Value); }
		}

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

		// ─── Training buffer ───
		// A circular store of recent (normalized feature vector, label) examples.
		// trainCount caps at TrainingWindow; trainWp is the write pointer. The forest
		// treats all physical slots [0, trainCount) as the dataset — order is irrelevant.
		private double[][]	trainFeat;
		private int[]		trainLabel;
		private int			trainCount;
		private int			trainWp;
		private int			minToBuild;     // examples required before the first rebuild

		// ─── Forest ───
		// trees[t] is a flat node array for tree t; treeNodeCount[t] is its node count.
		private RfNode[][]	trees;
		private int[]		treeNodeCount;
		private int			maxNodes;
		private bool		forestReady;
		private int			barsSinceRetrain;

		// Build scratch — all allocated once, reused on every rebuild.
		private int[]			bootstrapIdx;   // sample slot indices for the tree under construction
		private SplitItem[]		sortScratch;    // (value, label) pairs sorted during split search
		private int[]			featureBag;     // {0,1,2}, shuffled to pick the per-split feature subset
		private System.Random	rng;

		// Per-bar feature scratch.
		private double[]	scratchRaw;     // current bar, raw
		private double[]	scratchNorm;    // current bar, normalized
		private double[]	trainRaw;       // training bar, raw
		private double[]	trainNorm;      // training bar, normalized

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

				NumTrees				= 50;
				MaxDepth				= 6;
				MinSamplesLeaf			= 8;
				FeaturesPerSplit		= 2;
				RandomSeed				= 42;

				TrainingWindow			= 300;
				RetrainInterval			= 25;
				LabelHorizon			= 2;
				LabelMode				= MlRandomForest_LabelMode.CloseToClose;
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

				AddPlot(new Stroke(Brushes.LimeGreen, 5),	PlotStyle.TriangleUp,	"RF Long");
				AddPlot(new Stroke(Brushes.OrangeRed, 5),	PlotStyle.TriangleDown,	"RF Short");

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
				minToBuild	= Math.Max(2 * MinSamplesLeaf, 20);

				// Forest — a tree has at most 2^(MaxDepth+1)-1 nodes by depth, and at
				// most 2*trainCount-1 by sample count; size to the smaller bound.
				maxNodes		= Math.Min((1 << (MaxDepth + 1)) - 1, 2 * TrainingWindow + 1);
				trees			= new RfNode[NumTrees][];
				for (int t = 0; t < NumTrees; t++) trees[t] = new RfNode[maxNodes];
				treeNodeCount	= new int[NumTrees];
				forestReady		= false;
				barsSinceRetrain = 0;

				// Build scratch
				bootstrapIdx	= new int[TrainingWindow];
				sortScratch		= new SplitItem[TrainingWindow];
				featureBag		= new int[NumFeatures];
				for (int k = 0; k < NumFeatures; k++) featureBag[k] = k;
				rng				= new System.Random(RandomSeed);

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
			//    then rebuild the forest every RetrainInterval bars.
			if (CurrentBar >= predictionWarmup + LabelHorizon)
			{
				CollectTrainingExample();

				barsSinceRetrain++;
				if (barsSinceRetrain >= RetrainInterval && trainCount >= minToBuild)
				{
					RebuildForest();
					forestReady			= true;
					barsSinceRetrain	= 0;
				}
			}

			if (!forestReady) return;

			// 2) Live prediction — run the current bar's features through the forest.
			double pUp = ForestPredict(scratchNorm);
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

			if (LabelMode == MlRandomForest_LabelMode.CloseToClose)
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

		// ─── FOREST REBUILD ───────────────────────────────────────────────────────
		// Build every tree from scratch. Each tree draws its own bootstrap sample —
		// trainCount indices drawn WITH replacement — then grows a decision tree on it.

		private void RebuildForest()
		{
			int n = trainCount;
			for (int t = 0; t < NumTrees; t++)
			{
				for (int s = 0; s < n; s++)
					bootstrapIdx[s] = rng.Next(n);

				treeNodeCount[t] = 0;
				BuildNode(t, 0, n, 0);
			}
		}

		// ─── TREE BUILD ───────────────────────────────────────────────────────────
		// Recursively grow tree `treeIdx`. bootstrapIdx[lo..hi) holds the sample slots
		// for this node; the node is partitioned in place by the chosen split. Returns
		// the index of the node written into trees[treeIdx].

		private int BuildNode(int treeIdx, int lo, int hi, int depth)
		{
			RfNode[] tree	= trees[treeIdx];
			int nodeIdx		= treeNodeCount[treeIdx]++;
			int n			= hi - lo;

			int pos = 0;
			for (int s = lo; s < hi; s++) pos += trainLabel[bootstrapIdx[s]];

			bool makeLeaf = depth >= MaxDepth || n < 2 * MinSamplesLeaf || pos == 0 || pos == n;

			int    bestFeat	= -1;
			double bestThr	= 0.0;
			int    mid		= lo;
			if (!makeLeaf && FindBestSplit(lo, hi, n, pos, out bestFeat, out bestThr))
			{
				mid = Partition(lo, hi, bestFeat, bestThr);
				if (mid <= lo || mid >= hi) bestFeat = -1;   // degenerate split — make a leaf
			}

			if (makeLeaf || bestFeat < 0)
			{
				tree[nodeIdx].Feature	= -1;
				tree[nodeIdx].LeafProb	= (double)pos / n;
				return nodeIdx;
			}

			int left  = BuildNode(treeIdx, lo,  mid, depth + 1);
			int right = BuildNode(treeIdx, mid, hi,  depth + 1);

			tree[nodeIdx].Feature	= bestFeat;
			tree[nodeIdx].Threshold	= bestThr;
			tree[nodeIdx].Left		= left;
			tree[nodeIdx].Right		= right;
			return nodeIdx;
		}

		// ─── SPLIT SEARCH ─────────────────────────────────────────────────────────
		// For a random subset of FeaturesPerSplit features, find the (feature,
		// threshold) that minimizes the Gini impurity of the two child nodes. A split
		// is only accepted if it beats the parent's impurity and leaves at least
		// MinSamplesLeaf samples on each side.

		private bool FindBestSplit(int lo, int hi, int n, int totalPos, out int bestFeat, out double bestThr)
		{
			bestFeat = -1;
			bestThr  = 0.0;

			double bestWeighted = Gini(totalPos, n);   // a split must improve on the parent
			bool   found        = false;

			ShuffleFeatureBag();

			for (int fi = 0; fi < FeaturesPerSplit; fi++)
			{
				int f = featureBag[fi];

				for (int s = 0; s < n; s++)
				{
					int ex = bootstrapIdx[lo + s];
					sortScratch[s].Value = trainFeat[ex][f];
					sortScratch[s].Label = trainLabel[ex];
				}
				Array.Sort(sortScratch, 0, n);

				int leftPos = 0;
				for (int i = 0; i < n - 1; i++)
				{
					leftPos += sortScratch[i].Label;
					int leftN = i + 1;

					if (sortScratch[i].Value == sortScratch[i + 1].Value) continue;
					if (leftN < MinSamplesLeaf || (n - leftN) < MinSamplesLeaf) continue;

					int rightN   = n - leftN;
					int rightPos = totalPos - leftPos;
					double weighted = (leftN * Gini(leftPos, leftN) + rightN * Gini(rightPos, rightN)) / n;

					if (weighted < bestWeighted)
					{
						bestWeighted	= weighted;
						bestFeat		= f;
						bestThr			= 0.5 * (sortScratch[i].Value + sortScratch[i + 1].Value);
						found			= true;
					}
				}
			}

			return found;
		}

		// Partition bootstrapIdx[lo..hi) so samples with feature[feat] <= thr come
		// first; returns the boundary index (start of the right child).

		private int Partition(int lo, int hi, int feat, double thr)
		{
			int mid = lo;
			for (int s = lo; s < hi; s++)
			{
				if (trainFeat[bootstrapIdx[s]][feat] <= thr)
				{
					int tmp = bootstrapIdx[mid];
					bootstrapIdx[mid] = bootstrapIdx[s];
					bootstrapIdx[s]   = tmp;
					mid++;
				}
			}
			return mid;
		}

		// Gini impurity of a node with `pos` positive labels out of `n`.
		private static double Gini(int pos, int n)
		{
			double p = (double)pos / n;
			return 2.0 * p * (1.0 - p);
		}

		// Partial Fisher-Yates shuffle of featureBag — the first FeaturesPerSplit
		// entries become the random feature subset for one split.
		private void ShuffleFeatureBag()
		{
			for (int i = 0; i < NumFeatures; i++)
			{
				int j = i + rng.Next(NumFeatures - i);
				int tmp = featureBag[i];
				featureBag[i] = featureBag[j];
				featureBag[j] = tmp;
			}
		}

		// ─── PREDICTION ───────────────────────────────────────────────────────────

		private double ForestPredict(double[] feat)
		{
			double sum = 0.0;
			for (int t = 0; t < NumTrees; t++) sum += PredictTree(t, feat);
			return sum / NumTrees;
		}

		private double PredictTree(int t, double[] feat)
		{
			RfNode[] tree = trees[t];
			int idx = 0;
			while (tree[idx].Feature >= 0)
				idx = feat[tree[idx].Feature] <= tree[idx].Threshold ? tree[idx].Left : tree[idx].Right;
			return tree[idx].LeafProb;
		}

		// ─── FEATURE EXTRACTION + NORMALIZATION ───────────────────────────────────

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
			string tag  = (isLong ? "rf-long-lbl-" : "rf-short-lbl-") + CurrentBar;

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
		private indMyDailyTake.MlRandomForest[] cacheMlRandomForest;
		public indMyDailyTake.MlRandomForest MlRandomForest(int numTrees, int maxDepth, int minSamplesLeaf, int featuresPerSplit, int randomSeed, int trainingWindow, int retrainInterval, int labelHorizon, MlRandomForest_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return MlRandomForest(Input, numTrees, maxDepth, minSamplesLeaf, featuresPerSplit, randomSeed, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public indMyDailyTake.MlRandomForest MlRandomForest(ISeries<double> input, int numTrees, int maxDepth, int minSamplesLeaf, int featuresPerSplit, int randomSeed, int trainingWindow, int retrainInterval, int labelHorizon, MlRandomForest_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			if (cacheMlRandomForest != null)
				for (int idx = 0; idx < cacheMlRandomForest.Length; idx++)
					if (cacheMlRandomForest[idx] != null && cacheMlRandomForest[idx].NumTrees == numTrees && cacheMlRandomForest[idx].MaxDepth == maxDepth && cacheMlRandomForest[idx].MinSamplesLeaf == minSamplesLeaf && cacheMlRandomForest[idx].FeaturesPerSplit == featuresPerSplit && cacheMlRandomForest[idx].RandomSeed == randomSeed && cacheMlRandomForest[idx].TrainingWindow == trainingWindow && cacheMlRandomForest[idx].RetrainInterval == retrainInterval && cacheMlRandomForest[idx].LabelHorizon == labelHorizon && cacheMlRandomForest[idx].LabelMode == labelMode && cacheMlRandomForest[idx].MinFavorableMoveAtrs == minFavorableMoveAtrs && cacheMlRandomForest[idx].MaPeriod == maPeriod && cacheMlRandomForest[idx].AtrPeriod == atrPeriod && cacheMlRandomForest[idx].SlopeLookback == slopeLookback && cacheMlRandomForest[idx].NormalizeFeatures == normalizeFeatures && cacheMlRandomForest[idx].NormalizationLookback == normalizationLookback && cacheMlRandomForest[idx].MinProbabilityEdge == minProbabilityEdge && cacheMlRandomForest[idx].SignalCooldownBars == signalCooldownBars && cacheMlRandomForest[idx].EqualsInput(input))
						return cacheMlRandomForest[idx];
			return CacheIndicator<indMyDailyTake.MlRandomForest>(new indMyDailyTake.MlRandomForest(){ NumTrees = numTrees, MaxDepth = maxDepth, MinSamplesLeaf = minSamplesLeaf, FeaturesPerSplit = featuresPerSplit, RandomSeed = randomSeed, TrainingWindow = trainingWindow, RetrainInterval = retrainInterval, LabelHorizon = labelHorizon, LabelMode = labelMode, MinFavorableMoveAtrs = minFavorableMoveAtrs, MaPeriod = maPeriod, AtrPeriod = atrPeriod, SlopeLookback = slopeLookback, NormalizeFeatures = normalizeFeatures, NormalizationLookback = normalizationLookback, MinProbabilityEdge = minProbabilityEdge, SignalCooldownBars = signalCooldownBars }, input, ref cacheMlRandomForest);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.indMyDailyTake.MlRandomForest MlRandomForest(int numTrees, int maxDepth, int minSamplesLeaf, int featuresPerSplit, int randomSeed, int trainingWindow, int retrainInterval, int labelHorizon, MlRandomForest_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlRandomForest(Input, numTrees, maxDepth, minSamplesLeaf, featuresPerSplit, randomSeed, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlRandomForest MlRandomForest(ISeries<double> input , int numTrees, int maxDepth, int minSamplesLeaf, int featuresPerSplit, int randomSeed, int trainingWindow, int retrainInterval, int labelHorizon, MlRandomForest_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlRandomForest(input, numTrees, maxDepth, minSamplesLeaf, featuresPerSplit, randomSeed, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.indMyDailyTake.MlRandomForest MlRandomForest(int numTrees, int maxDepth, int minSamplesLeaf, int featuresPerSplit, int randomSeed, int trainingWindow, int retrainInterval, int labelHorizon, MlRandomForest_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlRandomForest(Input, numTrees, maxDepth, minSamplesLeaf, featuresPerSplit, randomSeed, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}

		public Indicators.indMyDailyTake.MlRandomForest MlRandomForest(ISeries<double> input , int numTrees, int maxDepth, int minSamplesLeaf, int featuresPerSplit, int randomSeed, int trainingWindow, int retrainInterval, int labelHorizon, MlRandomForest_LabelMode labelMode, double minFavorableMoveAtrs, int maPeriod, int atrPeriod, int slopeLookback, bool normalizeFeatures, int normalizationLookback, double minProbabilityEdge, int signalCooldownBars)
		{
			return indicator.MlRandomForest(input, numTrees, maxDepth, minSamplesLeaf, featuresPerSplit, randomSeed, trainingWindow, retrainInterval, labelHorizon, labelMode, minFavorableMoveAtrs, maPeriod, atrPeriod, slopeLookback, normalizeFeatures, normalizationLookback, minProbabilityEdge, signalCooldownBars);
		}
	}
}

#endregion
