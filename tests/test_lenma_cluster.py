#!/usr/bin/env python
# coding: utf-8

"""Characterization tests for amulog.alg.lenma.lenma.Cluster.

Cluster is the core similarity unit of the LenMa template generator. These
tests pin the similarity/merge behaviour, in particular the exact-match and
wildcard-template matching that CR-05 broke (integer-vs-list comparison and
similarity computed against the stale initial word list).
"""

import unittest

from amulog import lt_common
from amulog.alg.lenma.lenma import Cluster


class TestLenMaCluster(unittest.TestCase):

    def test_exact_match(self):
        c = Cluster(["proc", "alpha", "beta"])
        self.assertEqual(c.get_similarity_score(["proc", "alpha", "beta"]), 1.0)

    def test_head_rule_mismatch(self):
        # the first word (process name) must be equal
        c = Cluster(["proc", "alpha", "beta"])
        self.assertEqual(
            c.get_similarity_score(["other", "alpha", "beta"]), 0.0)

    def test_wildcard_template_matches_fitting_line(self):
        # CR-05 regression: after a merge the 3rd position is a wildcard, so a
        # line that fits the template ["proc", "X", *] must score 1.0.
        # The buggy code returned 0.0 because cnt_same was compared against the
        # stale initial word list and the exact-match path (int == list) was
        # always False.
        c = Cluster(["proc", "X", "Y"])
        c.update(["proc", "X", "Z"])   # 3rd position becomes a wildcard
        score = c.get_similarity_score(["proc", "X", "W"], n_same_count=3)
        self.assertEqual(score, 1.0)

    def test_update_introduces_wildcard(self):
        c = Cluster(["proc", "X", "Y"])
        c.update(["proc", "X", "Z"])
        self.assertEqual(c._words, ["proc", "X", lt_common.REPLACER])

    def test_dissimilar_low_score(self):
        # same head and word count, but the other words differ in length
        c = Cluster(["proc", "aaaa", "bbbb"])
        score = c.get_similarity_score(["proc", "x", "y"], n_same_count=3)
        self.assertLess(score, 0.9)

    def test_score_is_plain_float(self):
        # the similarity score is a scalar float (not a numpy array)
        c = Cluster(["proc", "aa", "bb"])
        score = c.get_similarity_score(["proc", "ab", "cd"], n_same_count=1)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
