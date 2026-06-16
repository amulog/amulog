#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog.eval import cluster_metrics


class TestPrecisionRecallFscore(unittest.TestCase):

    def test_perfect_clustering(self):
        p, r, f = cluster_metrics.precision_recall_fscore(
            [0, 0, 1, 1], [0, 0, 1, 1])
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_all_singletons_no_nan(self):
        # CR-60: degenerate input has no pairs (tp+fp == tp+fn == 0), which
        # used to produce nan (or ZeroDivisionError). Should be 0.0.
        p, r, f = cluster_metrics.precision_recall_fscore(
            [0, 1, 2], [0, 1, 2])
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_single_element(self):
        p, r, f = cluster_metrics.precision_recall_fscore([0], [0])
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))

    def test_recall_zero_when_no_true_pairs_match(self):
        # true has pairs, pred splits them all -> recall 0, no nan
        p, r, f = cluster_metrics.precision_recall_fscore(
            [0, 0, 0], [0, 1, 2])
        self.assertEqual((p, r, f), (0.0, 0.0, 0.0))


class TestOverDivisionAggregation(unittest.TestCase):

    def test_over_division(self):
        # true cluster 0 is split across pred {0,1}; true cluster 1 intact.
        r = cluster_metrics.over_division_cluster_ratio(
            [0, 0, 1, 1], [0, 1, 2, 2])
        self.assertAlmostEqual(r, 0.5)

    def test_over_aggregation(self):
        # pred cluster 11 aggregates true {0,1,2}; pred 10 holds only true 0.
        r = cluster_metrics.over_aggregation_cluster_ratio(
            [0, 0, 1, 2], [10, 11, 11, 11])
        self.assertAlmostEqual(r, 0.5)

    def test_over_aggregation_label_invariant(self):
        # CR-61: a clustering metric must be invariant to cluster relabeling.
        # The old implementation depended on which pred label sorted first.
        r1 = cluster_metrics.over_aggregation_cluster_ratio(
            [0, 0, 1, 2], [10, 11, 11, 11])
        r2 = cluster_metrics.over_aggregation_cluster_ratio(
            [0, 0, 1, 2], [11, 10, 10, 10])
        self.assertAlmostEqual(r1, r2)

    def test_perfect_clustering_no_failures(self):
        self.assertAlmostEqual(
            cluster_metrics.over_division_cluster_ratio(
                [0, 0, 1, 1], [5, 5, 6, 6]), 0.0)
        self.assertAlmostEqual(
            cluster_metrics.over_aggregation_cluster_ratio(
                [0, 0, 1, 1], [5, 5, 6, 6]), 0.0)


if __name__ == "__main__":
    unittest.main()
