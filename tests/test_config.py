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


class TestLoadImports(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._dir)

    def _write(self, name, body):
        import os
        path = os.path.join(self._dir, name)
        with open(path, "w") as f:
            f.write(body)
        return path

    def test_circular_import_detected(self):
        # CR-50: A imports B and B imports A used to loop forever; it must
        # now raise instead of hanging. (The pre-fix code cannot be exercised
        # safely because it never returns.)
        a = self._write("a.conf", "[general]\nimport = {0}\n")
        b = self._write("b.conf", "[general]\nimport = {0}\n".format(a))
        # rewrite a now that we know b's path
        with open(a, "w") as f:
            f.write("[general]\nimport = {0}\n".format(b))
        with self.assertRaises(ValueError):
            config.open_config(a, base_default=False)

    def test_import_chain_merges(self):
        # a non-circular import must still merge the imported option
        b = self._write("b.conf", "[general]\nsrc_recur = true\n")
        a = self._write("a.conf", "[general]\nimport = {0}\n".format(b))
        conf = config.open_config(a, base_default=False)
        self.assertEqual(conf["general"]["src_recur"], "true")


class TestReleaseCommonLogging(unittest.TestCase):

    def test_logger_name_list(self):
        # regression (CR-51): a list passed as logger_name hit the wrong
        # isinstance(logger, ...) check and raised TypeError, so handlers
        # on named loggers were never removed.
        import logging
        names = ["amulog_test_cr51_a", "amulog_test_cr51_b"]
        loggers = [logging.getLogger(n) for n in names]
        ch = logging.StreamHandler()
        for lg in loggers:
            lg.addHandler(ch)

        config.release_common_logging(ch, logger_name=names)

        for lg in loggers:
            self.assertNotIn(ch, lg.handlers)


if __name__ == "__main__":
    unittest.main()
