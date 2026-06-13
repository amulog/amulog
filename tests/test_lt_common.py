#!/usr/bin/env python
# coding: utf-8

"""Unit tests for amulog.lt_common (LogTemplate and helpers)."""

import unittest

from amulog import lt_common
from amulog.lt_common import LogTemplate


class TestLogTemplateIter(unittest.TestCase):

    @staticmethod
    def _lt(ltw):
        return LogTemplate(ltid=0, ltgid=0, ltw=ltw, lts=None, count=1)

    def test_iter_yields_words(self):
        # CR-41: __iter__ returned the list itself (not an iterator), so
        # iterating raised "iter() returned non-iterator of type 'list'".
        lt = self._lt(["a", lt_common.REPLACER, "b"])
        self.assertEqual(list(lt), ["a", lt_common.REPLACER, "b"])

    def test_for_loop(self):
        lt = self._lt(["x", "y", "z"])
        out = [w for w in lt]
        self.assertEqual(out, ["x", "y", "z"])

    def test_iter_returns_real_iterator(self):
        lt = self._lt(["a", "b"])
        it = iter(lt)
        # an iterator yields itself from iter() and supports next()
        self.assertIs(iter(it), it)
        self.assertEqual(next(it), "a")


if __name__ == "__main__":
    unittest.main()
