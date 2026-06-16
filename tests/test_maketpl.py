#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog import config
from amulog.eval import maketpl


def _make_mlt(answer_tids, trial_tids):
    mlt = maketpl.MeasureLTGen(config.open_config(verbose=False), 1)
    mlt._d_answer = {"l_tid": list(answer_tids)}
    mlt._d_trial = {"l_tid": list(trial_tids)}
    return mlt


class TestValidTidListsJointNone(unittest.TestCase):
    """CR-62: None must be excluded jointly so the answer/trial label arrays
    stay aligned and equal-length before being fed to pairwise metrics."""

    def test_joint_none_exclusion(self):
        # answer has None at positions 3,4; trial has None at position 1.
        mlt = _make_mlt([0, 0, 1, None, None], [0, None, 1, 1, 2])
        a = mlt.valid_tid_list_answer()
        t = mlt.valid_tid_list_trial()
        # only positions 0 and 2 have both non-None
        self.assertEqual(len(a), len(t))
        self.assertEqual(a, [0, 1])
        self.assertEqual(t, [0, 1])

    def test_metric_does_not_crash_on_differing_nones(self):
        mlt = _make_mlt([0, 0, 1, None, None], [0, None, 1, 1, 2])
        # f1_score feeds the pair to a sklearn-style metric; differing-length
        # arrays used to raise.
        mlt.f1_score()  # must not raise

    def test_no_none_unchanged(self):
        # when neither has None, joint filtering is a no-op (published runs
        # with clean labels are unaffected)
        mlt = _make_mlt([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(mlt.valid_tid_list_answer(), [0, 0, 1, 1])
        self.assertEqual(mlt.valid_tid_list_trial(), [0, 1, 1, 1])


class TestIterTplPairJoint(unittest.TestCase):
    """CR-64: the template-level metrics must align answer/trial jointly,
    skipping a line when either side is None (the structure_metrics path
    used to filter answer and trial independently and misalign them)."""

    def _make_mlt(self, answer_labels, trial_labels):
        mlt = maketpl.MeasureLTGen(config.open_config(verbose=False), 1)
        n = len(answer_labels)
        mlt._d_answer = {"l_tid": list(range(10, 10 + n))}
        mlt._d_trial = {"l_tid": list(range(20, 20 + n))}
        mlt.iter_org = lambda: iter([["w"] for _ in range(n)])
        mlt.iter_label_answer = lambda: iter(answer_labels)
        mlt.iter_label_trial = lambda: iter(trial_labels)
        mlt.restore_tpl = lambda labels, l_w: labels  # identity for the test
        return mlt

    def test_joint_alignment(self):
        # answer None at positions 3,4; trial None at position 1.
        mlt = self._make_mlt(["a", "a", "b", None, None],
                             ["a", None, "b", "b", "c"])
        pairs = list(mlt.iter_tpl_pair())
        # only positions 0 and 2 have both sides present, and they stay paired
        self.assertEqual(pairs, [(10, "a", "a"), (12, "b", "b")])

    def test_no_none_keeps_all(self):
        mlt = self._make_mlt(["a", "b", "c"], ["a", "x", "c"])
        tids, l_answer, l_trial = mlt._aligned_tpl_lists()
        self.assertEqual(l_answer, ["a", "b", "c"])
        self.assertEqual(l_trial, ["a", "x", "c"])
        self.assertEqual(tids, [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
