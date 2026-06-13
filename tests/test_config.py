#!/usr/bin/env python
# coding: utf-8

"""Unit tests for amulog.config helpers."""

import datetime
import unittest

from amulog import config


class TestStr2Dur(unittest.TestCase):

    def test_valid_units(self):
        self.assertEqual(config.str2dur("10s"), datetime.timedelta(seconds=10))
        self.assertEqual(config.str2dur("5m"), datetime.timedelta(minutes=5))
        self.assertEqual(config.str2dur("2h"), datetime.timedelta(hours=2))
        self.assertEqual(config.str2dur("3d"), datetime.timedelta(days=3))
        self.assertEqual(config.str2dur("1w"), datetime.timedelta(days=7))

    def test_surrounding_whitespace(self):
        self.assertEqual(config.str2dur(" 5m "), datetime.timedelta(minutes=5))

    def test_malformed_rejected(self):
        # "1month" was silently parsed as 1 minute by the old substring logic.
        for s in ["1month", "2days", "abc", "", "10", "s", "1.5h", "m5"]:
            with self.assertRaises(ValueError):
                config.str2dur(s)

    def test_error_message_has_value_and_example(self):
        with self.assertRaises(ValueError) as cm:
            config.str2dur("1month")
        msg = str(cm.exception)
        self.assertIn("1month", msg)  # the offending value
        # the message hints the expected format / an example
        self.assertTrue(
            any(hint in msg for hint in ("s/m/h/d/w", "10s", "2h")))


if __name__ == "__main__":
    unittest.main()
