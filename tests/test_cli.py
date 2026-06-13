#!/usr/bin/env python
# coding: utf-8

import sys
import unittest

from amulog import cli


def _dummy_func(ns):
    pass


DUMMY_ARGSET = {
    "do-something": ["do something", [], _dummy_func],
}


class TestCliMain(unittest.TestCase):

    def setUp(self):
        self._argv = sys.argv

    def tearDown(self):
        sys.argv = self._argv

    def test_unknown_subcommand_shows_usage(self):
        # regression: an unknown subcommand raised a raw KeyError traceback
        sys.argv = ["amulog", "no-such-subcommand"]
        with self.assertRaises(SystemExit) as cm:
            cli.main(dict(DUMMY_ARGSET))
        msg = str(cm.exception.code)
        self.assertIn("usage:", msg)
        self.assertIn("subcommands:", msg)

    def test_no_subcommand_shows_usage(self):
        sys.argv = ["amulog"]
        with self.assertRaises(SystemExit) as cm:
            cli.main(dict(DUMMY_ARGSET))
        self.assertIn("usage:", str(cm.exception.code))


if __name__ == "__main__":
    unittest.main()
