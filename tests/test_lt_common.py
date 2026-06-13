#!/usr/bin/env python
# coding: utf-8

"""Unit tests for amulog.lt_common (LogTemplate and helpers)."""

import unittest
from types import SimpleNamespace

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


class TestTemplateFromMessages(unittest.TestCase):

    @staticmethod
    def _lm(words):
        return SimpleNamespace(l_w=words)

    def test_common_words(self):
        tpl = lt_common.template_from_messages(
            [self._lm(["a", "b", "c"]), self._lm(["a", "x", "c"])])
        self.assertEqual(tpl, ["a", lt_common.REPLACER, "c"])

    def test_replacer_does_not_override_real_word(self):
        # CR-30: a real word must win over REPLACER even when the first
        # message carries REPLACER at that position.
        r = lt_common.REPLACER
        tpl = lt_common.template_from_messages(
            [self._lm([r, "b"]), self._lm(["a", "b"])])
        self.assertEqual(tpl, ["a", "b"])

    def test_all_replacer_stays_replacer(self):
        r = lt_common.REPLACER
        tpl = lt_common.template_from_messages(
            [self._lm([r, "x"]), self._lm([r, "x"])])
        self.assertEqual(tpl, [r, "x"])


if __name__ == "__main__":
    unittest.main()
