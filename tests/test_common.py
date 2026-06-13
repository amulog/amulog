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


if __name__ == "__main__":
    unittest.main()
