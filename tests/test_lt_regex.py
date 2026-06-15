#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog import common
from amulog import lt_regex


def _vre():
    return lt_regex.VariableRegex(common.filepath_local(__file__, "test_re.conf"))


class TestVariableRegex(unittest.TestCase):

    def test_label(self):
        vre = _vre()
        self.assertEqual(vre.label("192.168.1.1"), "ipaddr")
        self.assertEqual(vre.label("eth0"), "ifname")
        self.assertEqual(vre.label("12345"), "digit")
        self.assertEqual(vre.label("spam"), vre.label_unknown)

    def test_match_consistent_with_label(self):
        # match() is defined in terms of label() (dedup); they must stay
        # consistent: match is True iff label is not the unknown label.
        vre = _vre()
        for w in ["192.168.1.1", "eth0", "12345", "/var/log/x",
                  "spam", "2001:db8::1", ""]:
            self.assertEqual(vre.match(w),
                             vre.label(w) != vre.label_unknown,
                             "match/label disagree for {0!r}".format(w))


if __name__ == "__main__":
    unittest.main()
