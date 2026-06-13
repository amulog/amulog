#!/usr/bin/env python
# coding: utf-8

"""Regression tests for amulog.lt_va.LTGenVA offline processing.

CR-29: in offline mode the base process_offline calls preprocess() (which adds
every line to the word-count dict) AND then process_line() for each line (which
adds again), double-counting word/line frequencies and distorting the
description/variable decision. The offline path must build the statistics once
and then label against the complete statistics.
"""

import unittest

from amulog import lt_common
from amulog import lt_va


def _plines(lines):
    return {i: {"words": words} for i, words in enumerate(lines)}


LINES = [
    ["proc", "started", "ok"],
    ["proc", "started", "ok"],
    ["proc", "stopped", "err"],
]


class TestLTGenVAOffline(unittest.TestCase):

    def _ltgen(self):
        return lt_va.LTGenVA(lt_common.TemplateTable(),
                             method="relative-line", threshold=0.5)

    def test_offline_line_count_not_doubled(self):
        ltgen = self._ltgen()
        ltgen.process_offline(_plines(LINES))
        # line count must equal the number of input lines (buggy: 6)
        self.assertEqual(ltgen._d_wordcnt[None], len(LINES))

    def test_offline_word_counts_are_true_frequencies(self):
        ltgen = self._ltgen()
        ltgen.process_offline(_plines(LINES))
        self.assertEqual(ltgen._d_wordcnt["proc"], 3)     # in all 3 lines
        self.assertEqual(ltgen._d_wordcnt["started"], 2)  # in 2 lines
        self.assertEqual(ltgen._d_wordcnt["stopped"], 1)  # in 1 line

    def test_offline_counts_match_single_pass(self):
        # offline statistics must equal a single preprocess pass (no doubling)
        ref = self._ltgen()
        ref.preprocess([{"words": w} for w in LINES])

        ltgen = self._ltgen()
        ltgen.process_offline(_plines(LINES))

        self.assertEqual(dict(ltgen._d_wordcnt), dict(ref._d_wordcnt))


if __name__ == "__main__":
    unittest.main()
