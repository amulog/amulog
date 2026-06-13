#!/usr/bin/env python
# coding: utf-8

"""Characterization tests for amulog.host_alias.HostAlias.

HostAlias is, in practice, a hostname/IP *alias* resolver: it maps host
tokens (hostnames or IP addresses) to a canonical alias and a group.
These tests pin its basic behaviour so that refactoring does not change
the resolution contract.

Notes on the intended contract:
- IP-address tokens are matched by their canonical form (no subnet
  containment); equivalent notations (e.g. expanded IPv6) match.
- CIDR / subnet notation is NOT interpreted as a network; such a token is
  treated as a plain literal string and only matches itself exactly.
- Hostname queries are matched case-insensitively (query is lowercased).
"""

import os
import tempfile
import unittest

from amulog import host_alias


FIXTURE = """\
# host alias characterization fixture
[routers]
<rt0> rt0.example.com rt0-mgmt.example.com
<rt1> rt1.example.com

[switches]
sw0
sw1

[servers]
<db> 10.0.0.5

[mixed]
<gw> 150.99.112.10 gw.example.com

[v6]
<v6host> 2001:db8::1

[literal-cidr]
<office> 192.168.1.0/24
"""


class TestHostAlias(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        fd, cls.fn = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(FIXTURE)
        cls.ha = host_alias.HostAlias(cls.fn)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.fn)

    # --- alias resolution (the core, actually-used path) ---

    def test_resolve_alias_members(self):
        self.assertEqual(self.ha.resolve_host("rt0.example.com"), "rt0")
        self.assertEqual(self.ha.resolve_host("rt0-mgmt.example.com"), "rt0")
        self.assertEqual(self.ha.resolve_host("rt1.example.com"), "rt1")

    def test_resolve_alias_itself(self):
        # the alias token resolves to itself
        self.assertEqual(self.ha.resolve_host("rt0"), "rt0")

    def test_resolve_plain_member(self):
        # a plain (non-aliased) member resolves to itself
        self.assertEqual(self.ha.resolve_host("sw0"), "sw0")
        self.assertEqual(self.ha.resolve_host("sw1"), "sw1")

    def test_resolve_hostname_case_insensitive(self):
        self.assertEqual(self.ha.resolve_host("RT0.EXAMPLE.COM"), "rt0")

    def test_resolve_unknown(self):
        self.assertIsNone(self.ha.resolve_host("unknown.example.com"))

    # --- isknown ---

    def test_isknown(self):
        self.assertTrue(self.ha.isknown("rt0.example.com"))
        self.assertTrue(self.ha.isknown("sw0"))
        self.assertFalse(self.ha.isknown("unknown.example.com"))

    def test_isknown_consistent_with_resolve(self):
        for q in ["rt0.example.com", "sw0", "10.0.0.5", "unknown.example.com"]:
            self.assertEqual(self.ha.isknown(q),
                             self.ha.resolve_host(q) is not None)

    # --- groups ---

    def test_get_group(self):
        self.assertEqual(self.ha.get_group("rt0.example.com"), "routers")
        self.assertEqual(self.ha.get_group("sw0"), "switches")
        self.assertEqual(self.ha.get_group("10.0.0.5"), "servers")

    def test_get_group_unknown(self):
        self.assertIsNone(self.ha.get_group("unknown.example.com"))

    def test_group_members(self):
        self.assertEqual(self.ha.group("switches"), ["sw0", "sw1"])

    # --- IP addresses: canonical exact match, no subnet containment ---

    def test_exact_ip(self):
        self.assertEqual(self.ha.resolve_host("10.0.0.5"), "db")
        self.assertTrue(self.ha.isknown("10.0.0.5"))
        self.assertEqual(self.ha.get_group("10.0.0.5"), "servers")

    def test_alias_with_ip_and_hostname_members(self):
        # mirrors real configs (amulog-config def_s4_host_alias.txt):
        # `<alias> <exact-ip> <hostname>` -- both members resolve to the alias.
        self.assertEqual(self.ha.resolve_host("150.99.112.10"), "gw")
        self.assertEqual(self.ha.resolve_host("gw.example.com"), "gw")
        self.assertEqual(self.ha.get_group("150.99.112.10"), "mixed")

    def test_ipv6_canonicalization(self):
        # an expanded IPv6 form matches the canonical definition
        expanded = "2001:0db8:0000:0000:0000:0000:0000:0001"
        self.assertEqual(self.ha.resolve_host(expanded), "v6host")

    def test_cidr_is_literal_not_a_network(self):
        # the CIDR token only matches itself exactly ...
        self.assertEqual(self.ha.resolve_host("192.168.1.0/24"), "office")
        # ... and a host inside that subnet does NOT resolve (no containment)
        self.assertIsNone(self.ha.resolve_host("192.168.1.10"))

    def test_no_substring_false_positive(self):
        # "92.168.1.0" is a substring of "192.168.1.0/24" but is unrelated;
        # it must NOT resolve to the CIDR's alias (regression: the old
        # `addr in net` substring check returned "office" here).
        self.assertIsNone(self.ha.resolve_host("92.168.1.0"))
        self.assertFalse(self.ha.isknown("92.168.1.0"))


class TestHostAliasEmpty(unittest.TestCase):

    def test_empty_filename(self):
        ha = host_alias.HostAlias("")
        self.assertFalse(ha.isknown("anything"))
        self.assertIsNone(ha.resolve_host("anything"))
        self.assertIsNone(ha.resolve_host("10.0.0.1"))


if __name__ == "__main__":
    unittest.main()
