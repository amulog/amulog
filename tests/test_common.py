#!/usr/bin/env python
# coding: utf-8

"""Unit tests for amulog.common helpers."""

import os
import tempfile
import unittest

from amulog import common


class TestIsEmpty(unittest.TestCase):

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(common.is_empty(d))

    def test_dir_with_one_entry(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a"), "w"):
                pass
            self.assertFalse(common.is_empty(d))

    def test_dir_with_multiple_entries(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a", "b", "c"):
                with open(os.path.join(d, name), "w"):
                    pass
            self.assertFalse(common.is_empty(d))

    def test_nonexistent_path(self):
        self.assertFalse(common.is_empty("/no/such/path/should/exist/xyz"))


class TestTimerStat(unittest.TestCase):

    def test_stat_outputs_values_not_header_twice(self):
        # CR-97: the format string used "{0}" twice, so the value (lap_times /
        # average / standard error) was replaced by the header and never shown.
        timer = common.Timer("HDR")
        captured = []
        timer._output = captured.append  # collect output lines
        timer.start()
        timer.lap("a")
        timer.lap("b")
        timer.stat()

        for label in ("lap times", "average", "standard error"):
            lines = [s for s in captured if label in s]
            self.assertEqual(len(lines), 1)
            line = lines[0]
            self.assertTrue(line.startswith("HDR {0}:".format(label)))
            # the header must appear once (not twice in place of the value)
            self.assertEqual(line.count("HDR"), 1)


if __name__ == "__main__":
    unittest.main()
