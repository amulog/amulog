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


if __name__ == "__main__":
    unittest.main()
