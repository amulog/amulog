#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog import lt_common
from amulog.alg.shiso import shiso


def _make_lttable(items):
    """items: list of (ltid, ltw)"""
    table = lt_common.LTTable()
    for ltid, ltw in items:
        table.restore_lt(ltid, None, ltw, [""] * (len(ltw) + 1), 1)
    return table


class TestLTGenSHISO(unittest.TestCase):

    def test_explicit_cfunc_is_used(self):
        # regression (CR-16): passing cfunc left self._cfunc unset (no else
        # branch), so process_line raised AttributeError.
        table = lt_common.TemplateTable()
        ltgen = shiso.LTGenSHISO(table, threshold=0.9, max_child=4,
                                 cfunc=shiso.LTGenSHISO.c_original)
        # first line just creates a node; the second forces a comparison
        # against the first via self._cfunc (where the bug surfaces)
        ltgen.process_line({"words": ["foo", "bar", "baz"]})
        tid, state = ltgen.process_line({"words": ["foo", "bar", "qux"]})
        self.assertIsInstance(tid, int)


class TestLTGroupSHISO(unittest.TestCase):

    def test_construct_and_make(self):
        # regression (CR-17): super().__init__() was called without lttable,
        # so construction raised TypeError and self.lttable was never set.
        table = _make_lttable([
            (0, ["alpha", "beta", "gamma", "delta", "epsilon"]),
            (1, ["alpha", "beta", "gamma", "delta", "zeta"]),
        ])
        group = shiso.LTGroupSHISO(table)
        result = group.make()  # exercises remake_all -> self.lttable
        self.assertEqual(len(result), 2)
        for ltline in result:
            self.assertIsNotNone(ltline.ltgid)

    def test_add_with_ltid_gap(self):
        # regression (CR-18): l_cnt = [0] * len(lttable) indexed by ltid,
        # so a gap in ltids (e.g. after edits) raised IndexError.
        table = _make_lttable([
            (5, ["alpha", "beta", "gamma", "delta", "epsilon"]),
            (9, ["alpha", "beta", "gamma", "delta", "zeta"]),
        ])
        group = shiso.LTGroupSHISO(table)
        for ltline in table:
            gid = group.add(ltline)
            self.assertIsInstance(gid, int)


if __name__ == "__main__":
    unittest.main()
