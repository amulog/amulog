#!/usr/bin/env python
# coding: utf-8

"""Resolver-contract tests for amulog.host_group.HostGroup.

These are DB-independent, pure-function tests that pin the resolution
contract of the host stratification (host group) resolver:

- hostalias scheme: mirrors tests/test_host_alias.py so that host_alias
  compatibility (IP canonicalization, case-insensitive hostnames, CIDR
  literal, IPv6 expansion, no substring mismatch) is pinned.
- regex scheme: capture -> hgid, multi-scheme first-match, BGL chip + IP
  mixed selection, replacement formatting (IP -> /24).
- unmatched policy: keep / drop / drop_alert (warning checked via log
  capture).
- builtin ipaddr scheme.

Temp ini and host_alias files are created under the sandbox-writable
temp directory (tempfile, honoring $TMPDIR).
"""

import os
import re
import datetime
import tempfile
import unittest

from amulog import host_group


def _conf(host_group_filename):
    """Minimal conf stand-in: HostGroup only reads
    conf["manager"]["host_group_filename"]."""
    return {"manager": {"host_group_filename": host_group_filename}}


def _write_temp(content, suffix=".txt"):
    fd, fn = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return fn


# host_alias-format fixture reused from test_host_alias.py contract
HOSTALIAS_FIXTURE = """\
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


class TestHostGroupDisabled(unittest.TestCase):
    """An empty host_group_filename disables stratification."""

    def test_empty_filename(self):
        hg = host_group.HostGroup(_conf(""))
        self.assertEqual(hg.tiers(), [])
        self.assertIsNone(hg.label_tier())

    def test_init_hostgroup(self):
        hg = host_group.init_hostgroup(_conf(""))
        self.assertIsInstance(hg, host_group.HostGroup)


class TestHostGroupHostAliasScheme(unittest.TestCase):
    """hostalias scheme reuses host_alias parsing -> identical contract."""

    @classmethod
    def setUpClass(cls):
        cls.ha_fn = _write_temp(HOSTALIAS_FIXTURE)
        ini = """\
[tiers]
order = default
unmatched = drop

[tier_default]
schemes = s4
s4.type = hostalias
s4.file = {fn}
s4.ignorecase = true
""".format(fn=cls.ha_fn)
        cls.ini_fn = _write_temp(ini, suffix=".ini")
        cls.hg = host_group.HostGroup(_conf(cls.ini_fn))

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.ha_fn)
        os.remove(cls.ini_fn)

    def r(self, host):
        return self.hg.resolve(host, "default")

    def test_tiers(self):
        self.assertEqual(self.hg.tiers(), ["default"])

    def test_resolve_alias_members(self):
        self.assertEqual(self.r("rt0.example.com"), "rt0")
        self.assertEqual(self.r("rt0-mgmt.example.com"), "rt0")
        self.assertEqual(self.r("rt1.example.com"), "rt1")

    def test_resolve_alias_itself(self):
        self.assertEqual(self.r("rt0"), "rt0")

    def test_resolve_plain_member(self):
        self.assertEqual(self.r("sw0"), "sw0")
        self.assertEqual(self.r("sw1"), "sw1")

    def test_resolve_hostname_case_insensitive(self):
        self.assertEqual(self.r("RT0.EXAMPLE.COM"), "rt0")

    def test_exact_ip(self):
        self.assertEqual(self.r("10.0.0.5"), "db")

    def test_alias_with_ip_and_hostname_members(self):
        self.assertEqual(self.r("150.99.112.10"), "gw")
        self.assertEqual(self.r("gw.example.com"), "gw")

    def test_ipv6_canonicalization(self):
        expanded = "2001:0db8:0000:0000:0000:0000:0000:0001"
        self.assertEqual(self.r(expanded), "v6host")

    def test_cidr_is_literal_not_a_network(self):
        self.assertEqual(self.r("192.168.1.0/24"), "office")
        # unmatched policy is drop -> None inside a subnet (no containment)
        self.assertIsNone(self.r("192.168.1.10"))

    def test_no_substring_false_positive(self):
        # "92.168.1.0" is a substring of "192.168.1.0/24" but unrelated
        self.assertIsNone(self.r("92.168.1.0"))

    def test_resolve_unknown_drops(self):
        # unmatched = drop -> unknown host yields None
        self.assertIsNone(self.r("unknown.example.com"))


class TestHostGroupRegexScheme(unittest.TestCase):
    """regex scheme: capture -> hgid, first-match, formatting."""

    def _hg(self, ini_body):
        fn = _write_temp(ini_body, suffix=".ini")
        self.addCleanup(os.remove, fn)
        return host_group.HostGroup(_conf(fn))

    def test_capture_default_replacement(self):
        # default replacement = \1 (first capture group)
        hg = self._hg("""\
[tiers]
order = midplane
unmatched = keep

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
""")
        # BGL chip naming: R00-M0-N0-C:J02-U01 -> midplane R00-M0
        self.assertEqual(hg.resolve("R00-M0-N0-C:J02-U01", "midplane"),
                         "R00-M0")
        self.assertEqual(hg.resolve("R01-M1-N2-C:J05-U10", "midplane"),
                         "R01-M1")

    def test_unmatched_keep_returns_host(self):
        hg = self._hg("""\
[tiers]
order = midplane
unmatched = keep

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
""")
        # a host not matching the regex is kept as-is (lossless)
        self.assertEqual(hg.resolve("somehost", "midplane"), "somehost")

    def test_explicit_replacement_ip_to_24(self):
        # replacement formatting: IP -> /24 network label
        hg = self._hg("""\
[tiers]
order = subnet
unmatched = drop

[tier_subnet]
schemes = ip
ip.type = regex
ip.regex = ^(\\d+\\.\\d+\\.\\d+)\\.\\d+$
ip.replacement = \\1.0/24
""")
        self.assertEqual(hg.resolve("10.20.30.40", "subnet"), "10.20.30.0/24")
        self.assertEqual(hg.resolve("192.168.1.200", "subnet"),
                         "192.168.1.0/24")

    def test_ignorecase(self):
        hg = self._hg("""\
[tiers]
order = role
unmatched = drop

[tier_role]
schemes = web
web.type = regex
web.regex = ^(WEB)\\d+
web.ignorecase = true
""")
        self.assertEqual(hg.resolve("web01", "role"), "web")
        self.assertEqual(hg.resolve("WEB42", "role"), "WEB")

    def test_multi_scheme_first_match(self):
        # two schemes; first matching scheme wins
        hg = self._hg("""\
[tiers]
order = mixed
unmatched = drop

[tier_mixed]
schemes = bgl, ipaddr
bgl.type = regex
bgl.regex = ^(R\\d+-M\\d)
ipaddr.type = builtin
ipaddr.name = ipaddr
""")
        # BGL chip matches bgl first
        self.assertEqual(hg.resolve("R00-M0-N0-C:J02-U01", "mixed"), "R00-M0")
        # IP does not match bgl regex, falls through to builtin ipaddr
        self.assertEqual(hg.resolve("10.0.0.5", "mixed"), "10.0.0.5")
        # neither matches -> drop
        self.assertIsNone(hg.resolve("plainhost", "mixed"))


class TestHostGroupBuiltinIpaddr(unittest.TestCase):

    def _hg(self):
        fn = _write_temp("""\
[tiers]
order = ip
unmatched = drop

[tier_ip]
schemes = ipaddr
ipaddr.type = builtin
ipaddr.name = ipaddr
""", suffix=".ini")
        self.addCleanup(os.remove, fn)
        return host_group.HostGroup(_conf(fn))

    def test_canonicalizes_ipv4(self):
        hg = self._hg()
        self.assertEqual(hg.resolve("10.0.0.5", "ip"), "10.0.0.5")

    def test_canonicalizes_ipv6_expanded(self):
        hg = self._hg()
        expanded = "2001:0db8:0000:0000:0000:0000:0000:0001"
        self.assertEqual(hg.resolve(expanded, "ip"), "2001:db8::1")

    def test_non_ip_does_not_match(self):
        hg = self._hg()
        # non-IP token -> no match -> drop -> None
        self.assertIsNone(hg.resolve("hostname", "ip"))


class TestHostGroupUnmatchedPolicy(unittest.TestCase):

    def _hg(self, policy):
        fn = _write_temp("""\
[tiers]
order = t
unmatched = {policy}

[tier_t]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
""".format(policy=policy), suffix=".ini")
        self.addCleanup(os.remove, fn)
        return host_group.HostGroup(_conf(fn))

    def test_keep(self):
        hg = self._hg("keep")
        self.assertEqual(hg.resolve("nomatch", "t"), "nomatch")

    def test_drop(self):
        hg = self._hg("drop")
        self.assertIsNone(hg.resolve("nomatch", "t"))

    def test_drop_alert_returns_none_and_warns(self):
        hg = self._hg("drop_alert")
        logger_name = host_group._logger.name
        with self.assertLogs(logger_name, level="WARNING") as cm:
            result = hg.resolve("nomatch", "t")
        self.assertIsNone(result)
        self.assertTrue(any("nomatch" in msg for msg in cm.output))

    def test_default_policy_is_keep(self):
        # omitting unmatched defaults to keep
        fn = _write_temp("""\
[tiers]
order = t

[tier_t]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
""", suffix=".ini")
        self.addCleanup(os.remove, fn)
        hg = host_group.HostGroup(_conf(fn))
        self.assertEqual(hg.unmatched_policy(), host_group.UNMATCHED_KEEP)
        self.assertEqual(hg.resolve("nomatch", "t"), "nomatch")


class TestHostGroupTiersAndLabel(unittest.TestCase):

    def test_tier_order_and_label_tier(self):
        fn = _write_temp("""\
[tiers]
order = default, midplane, rack
label_tier = default

[tier_default]
schemes = ipaddr
ipaddr.type = builtin
ipaddr.name = ipaddr

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)

[tier_rack]
schemes = rack
rack.regex = ^(R\\d+)
""", suffix=".ini")
        self.addCleanup(os.remove, fn)
        hg = host_group.HostGroup(_conf(fn))
        self.assertEqual(hg.tiers(), ["default", "midplane", "rack"])
        self.assertEqual(hg.label_tier(), "default")
        # tier resolution at each level
        self.assertEqual(hg.resolve("R00-M0-N0-C:J02-U01", "rack"), "R00")
        self.assertEqual(hg.resolve("R00-M0-N0-C:J02-U01", "midplane"),
                         "R00-M0")

    def test_unknown_tier_raises(self):
        fn = _write_temp("""\
[tiers]
order = default

[tier_default]
schemes = ipaddr
ipaddr.type = builtin
ipaddr.name = ipaddr
""", suffix=".ini")
        self.addCleanup(os.remove, fn)
        hg = host_group.HostGroup(_conf(fn))
        with self.assertRaises(KeyError):
            hg.resolve("anything", "nosuchtier")


class TestHostGroupConfigErrors(unittest.TestCase):

    def _build(self, body):
        fn = _write_temp(body, suffix=".ini")
        self.addCleanup(os.remove, fn)
        return host_group.HostGroup(_conf(fn))

    def test_missing_tier_section(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = default
""")

    def test_invalid_unmatched(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = default
unmatched = bogus

[tier_default]
schemes = ipaddr
ipaddr.type = builtin
ipaddr.name = ipaddr
""")

    def test_regex_scheme_requires_regex(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = t

[tier_t]
schemes = x
x.type = regex
""")

    def test_hostalias_scheme_requires_file(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = t

[tier_t]
schemes = x
x.type = hostalias
""")

    def test_unknown_builtin(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = t

[tier_t]
schemes = x
x.type = builtin
x.name = nope
""")

    def test_unknown_scheme_type(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = t

[tier_t]
schemes = x
x.type = weird
""")

    def test_label_tier_not_in_order(self):
        with self.assertRaises(ValueError):
            self._build("""\
[tiers]
order = default
label_tier = ghost

[tier_default]
schemes = ipaddr
ipaddr.type = builtin
ipaddr.name = ipaddr
""")


# ---------------------------------------------------------------------------
# DB integration tests (spec section 7 items 3, 4, 5).
#
# A small synthetic sqlite DB is built directly via the public LogData API
# (a single log template + a handful of BGL-style chip hosts), then the hg
# table API (update_hg / members / iter_hg) and the iter_lines(hg=) /
# host_in / restore_line behaviours are asserted. These mirror the
# tests/test_db.py fixture style but stay self-contained (no log parsing).
# ---------------------------------------------------------------------------

from amulog import config  # noqa: E402
from amulog import log_db  # noqa: E402
from amulog import lt_common  # noqa: E402


# BGL-style chip hosts: R<rack>-M<midplane>-N..-C:.. (cf. spec section 0).
_BGL_HOSTS = [
    "R00-M0-N0-C:J02-U01",
    "R00-M0-N1-C:J04-U05",
    "R00-M1-N0-C:J00-U00",
    "R01-M0-N2-C:J05-U10",
    "R01-M1-N3-C:J07-U11",
]

_MIDPLANE_INI = """\
[tiers]
order = midplane
unmatched = keep

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
"""


def _build_synthetic_db(db_path, hosts, logging_off=True):
    """Build a minimal sqlite DB with one template and one line per host.

    Returns nothing; the DB file at db_path is populated and committed.
    """
    conf = config.open_config(verbose=False)
    if logging_off:
        # avoid auto.log creation in cwd from CLI-style logging setup
        conf["general"]["logging"] = ""
    conf["database"]["database"] = "sqlite3"
    conf["database"]["sqlite3_filename"] = db_path

    ld = log_db.LogData(conf, edit=True, reset_db=True)
    # one trivial template "node ** event" (ltid=0, ltgid=0)
    ltobj = lt_common.LogTemplate(
        0, 0, ["node", lt_common.REPLACER, "event"], None, len(hosts))
    ld.add_lt(ltobj)
    dt = datetime.datetime(2020, 1, 1, 0, 0, 0)
    for i, host in enumerate(hosts):
        ld.add_line(lid=i, ltid=0, dt=dt + datetime.timedelta(minutes=i),
                    host=host, l_w=["node", "alpha", "event"])
    ld.commit_db()
    return conf


class TestHostGroupDBIntegration(unittest.TestCase):
    """update_hg / members / iter_hg + iter_lines(hg=) on a synthetic DB."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, self.db_path)
        self.conf = _build_synthetic_db(self.db_path, _BGL_HOSTS)

        fd2, self.ini_path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd2, "w") as f:
            f.write(_MIDPLANE_INI)
        self.addCleanup(os.remove, self.ini_path)
        self.conf["manager"]["host_group_filename"] = self.ini_path

    def _ld(self, edit=False):
        return log_db.LogData(self.conf, edit=edit, reset_db=False)

    def test_update_hg_materializes_rows(self):
        ld = self._ld(edit=True)
        hg = host_group.HostGroup(self.conf)
        ld.update_hg(hg)

        # expected midplane grouping from the regex ^(R\d+-M\d)
        expected = {
            "R00-M0": ["R00-M0-N0-C:J02-U01", "R00-M0-N1-C:J04-U05"],
            "R00-M1": ["R00-M1-N0-C:J00-U00"],
            "R01-M0": ["R01-M0-N2-C:J05-U10"],
            "R01-M1": ["R01-M1-N3-C:J07-U11"],
        }
        got = ld.iter_hg("midplane")
        self.assertEqual({k: sorted(v) for k, v in got.items()},
                         {k: sorted(v) for k, v in expected.items()})
        # hg table is the materialized tier
        self.assertEqual(ld.hg_tiers(), ["midplane"])

    def test_update_hg_is_idempotent(self):
        # update_hg resets each tier (DELETE WHERE tier=?) before re-INSERT,
        # so repeated calls must not duplicate rows. The hg table has no
        # DB-level unique constraint (db_common cannot emit a composite PK),
        # so idempotency is enforced only in update_hg -- pin it here.
        # get_hg_members has no DISTINCT, hence a double INSERT would surface
        # as duplicate member hosts below.
        ld = self._ld(edit=True)
        hg = host_group.HostGroup(self.conf)
        ld.update_hg(hg)
        ld.update_hg(hg)  # second pass must not accumulate rows

        members = ld.members("midplane", "R00-M0")
        self.assertEqual(sorted(members),
                         ["R00-M0-N0-C:J02-U01", "R00-M0-N1-C:J04-U05"])
        self.assertEqual(len(members), len(set(members)))  # no duplicates

        # total materialized rows == one per host (no accumulation), and no
        # host group has a duplicated member.
        got = ld.iter_hg("midplane")
        total_rows = sum(len(v) for v in got.values())
        self.assertEqual(total_rows, len(_BGL_HOSTS))
        for hosts in got.values():
            self.assertEqual(len(hosts), len(set(hosts)))
        self.assertEqual(ld.hg_tiers(), ["midplane"])

    def test_members(self):
        ld = self._ld(edit=True)
        ld.update_hg(host_group.HostGroup(self.conf))
        self.assertEqual(sorted(ld.members("midplane", "R00-M0")),
                         ["R00-M0-N0-C:J02-U01", "R00-M0-N1-C:J04-U05"])
        self.assertEqual(ld.members("midplane", "R00-M1"),
                         ["R00-M1-N0-C:J00-U00"])
        # unknown hgid -> empty
        self.assertEqual(ld.members("midplane", "R99-M9"), [])

    def test_members_iter_lines_host_in_matches_hg_axis(self):
        ld = self._ld(edit=True)
        ld.update_hg(host_group.HostGroup(self.conf))

        # extraction via members -> iter_lines(host_in=...) must equal the
        # iter_lines(hg=(tier, hgid)) convenience axis.
        hosts = ld.members("midplane", "R00-M0")
        via_host_in = sorted(lm.lid for lm in ld.iter_lines(host_in=hosts))
        via_hg = sorted(
            lm.lid for lm in ld.iter_lines(hg=("midplane", "R00-M0")))
        self.assertEqual(via_host_in, via_hg)
        # and it really is the two R00-M0 lines (lid 0 and 1)
        self.assertEqual(via_hg, [0, 1])

    def test_iter_lines_hg_returns_original_hosts(self):
        ld = self._ld(edit=True)
        ld.update_hg(host_group.HostGroup(self.conf))
        hosts = {lm.host for lm in ld.iter_lines(hg=("midplane", "R00-M0"))}
        # host column is the ORIGINAL chip host, not the hgid
        self.assertEqual(hosts,
                         {"R00-M0-N0-C:J02-U01", "R00-M0-N1-C:J04-U05"})

    def test_reset_hg(self):
        ld = self._ld(edit=True)
        ld.update_hg(host_group.HostGroup(self.conf))
        self.assertTrue(len(ld.members("midplane", "R00-M0")) > 0)
        ld.reset_hg("midplane")
        # after reset the materialized rows are gone; iter_hg falls back to
        # identity synthesis from whole_host (one group per original host)
        self.assertEqual(ld.db.get_hg_members("midplane", "R00-M0"), [])
        ident = ld.iter_hg("midplane")
        self.assertEqual(set(ident.keys()), set(_BGL_HOSTS))

    def test_host_axis_unchanged(self):
        # the plain host= axis still does raw original-host lookup
        ld = self._ld()
        lids = sorted(lm.lid
                      for lm in ld.iter_lines(host="R00-M0-N0-C:J02-U01"))
        self.assertEqual(lids, [0])


class TestHostGroupBackwardCompat(unittest.TestCase):
    """A DB with no host_group -> default = identity; current behaviour."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, self.db_path)
        self.conf = _build_synthetic_db(self.db_path, _BGL_HOSTS)
        # no host_group_filename configured
        self.conf["manager"]["host_group_filename"] = ""

    def test_whole_host_unchanged(self):
        ld = log_db.LogData(self.conf)
        self.assertEqual(set(ld.whole_host()), set(_BGL_HOSTS))

    def test_iter_lines_host_unchanged(self):
        ld = log_db.LogData(self.conf)
        # current behaviour: host= returns exactly that original host's lines
        for i, host in enumerate(_BGL_HOSTS):
            lids = [lm.lid for lm in ld.iter_lines(host=host)]
            self.assertEqual(lids, [i])

    def test_identity_iter_hg_default(self):
        # default identity tier with no hg rows -> synthesized from whole_host
        ld = log_db.LogData(self.conf)
        ident = ld.iter_hg("default")
        self.assertEqual(set(ident.keys()), set(_BGL_HOSTS))
        for host in _BGL_HOSTS:
            self.assertEqual(ident[host], [host])

    def test_identity_members_default(self):
        ld = log_db.LogData(self.conf)
        # identity fallback: members(default, host) == [host]
        self.assertEqual(ld.members("default", "R00-M0-N0-C:J02-U01"),
                         ["R00-M0-N0-C:J02-U01"])
        # a host not in the DB has no members
        self.assertEqual(ld.members("default", "no-such-host"), [])


class TestHostGroupHgTableAddedToOldDB(unittest.TestCase):
    """A DB created WITHOUT the hg table still opens; repair_tables adds it
    without forcing a rebuild (additive, backward compatible)."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, self.db_path)
        self.conf = _build_synthetic_db(self.db_path, _BGL_HOSTS)

    def _drop_hg_table(self):
        import sqlite3
        con = sqlite3.connect(self.db_path)
        try:
            con.execute("drop index if exists hg_index_members")
            con.execute("drop table if exists hg")
            con.commit()
        finally:
            con.close()

    def test_opens_and_repairs(self):
        self._drop_hg_table()

        import sqlite3
        con = sqlite3.connect(self.db_path)
        try:
            names = [r[0] for r in
                     con.execute("select name from sqlite_master")]
        finally:
            con.close()
        self.assertNotIn("hg", names)

        # opening the DB must still work (data intact, no rebuild)
        ld = log_db.LogData(self.conf)
        # count_lines() == max(lid); lids are 0..len-1
        self.assertEqual(ld.count_lines(), len(_BGL_HOSTS) - 1)
        self.assertEqual(set(ld.whole_host()), set(_BGL_HOSTS))

        # repair_tables re-creates the hg table/index without touching log
        ld_edit = log_db.LogData(self.conf, edit=True, reset_db=False)
        ld_edit.db.repair_tables()

        con = sqlite3.connect(self.db_path)
        try:
            names = [r[0] for r in
                     con.execute("select name from sqlite_master")]
        finally:
            con.close()
        self.assertIn("hg", names)
        self.assertIn("hg_index_members", names)
        # log data untouched by the repair
        self.assertEqual(set(log_db.LogData(self.conf).whole_host()),
                         set(_BGL_HOSTS))


class TestHostGroupRestoreLine(unittest.TestCase):
    """restore_line with host=original emits the original host (spec
    section 2 / log_db.py restore_line)."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, self.db_path)
        self.conf = _build_synthetic_db(self.db_path, _BGL_HOSTS)

    def test_restore_line_round_trips_original_host(self):
        ld = log_db.LogData(self.conf)
        for host in _BGL_HOSTS:
            (lm,) = list(ld.iter_lines(host=host))
            line = lm.restore_line()
            # the original chip host appears verbatim in the restored line
            self.assertIn(host, line)
            self.assertEqual(lm.host, host)


# ---------------------------------------------------------------------------
# CLI handler tests (spec section 5 + section 7 item 2)
# ---------------------------------------------------------------------------

import io  # noqa: E402
import contextlib  # noqa: E402
import types  # noqa: E402

from amulog import __main__ as cli_main  # noqa: E402


def _ns(conf_path, **kwargs):
    """Build an argparse-like Namespace for a CLI handler call."""
    base = {"conf_path": conf_path, "debug": False}
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _write_conf_file(conf, path):
    """Persist a conf to a real file so a CLI handler can reopen it.

    CLI handlers re-open the config from ns.conf_path; logging='' must be
    persisted so the handler does not create auto.log in cwd.
    """
    config.write(path, conf)


def _run(handler, ns):
    """Run a CLI handler, capturing stdout. Returns the printed text."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        handler(ns)
    return buf.getvalue()


_NESTED_INI = """\
[tiers]
order = midplane, rack
unmatched = keep

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)

[tier_rack]
schemes = bgl
bgl.regex = ^(R\\d+)
"""


class _CLIHostGroupBase(unittest.TestCase):

    HOSTS = _BGL_HOSTS
    INI = _NESTED_INI

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, self.db_path)

        conf = _build_synthetic_db(self.db_path, self.HOSTS)

        fd2, self.ini_path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd2, "w") as f:
            f.write(self.INI)
        self.addCleanup(os.remove, self.ini_path)
        conf["manager"]["host_group_filename"] = self.ini_path
        # CLI handler tests MUST keep logging='' so no auto.log is written
        # to cwd when the handler re-opens the config (project memory).
        conf["general"]["logging"] = ""

        fd3, self.conf_path = tempfile.mkstemp(suffix=".conf")
        os.close(fd3)
        self.addCleanup(os.remove, self.conf_path)
        _write_conf_file(conf, self.conf_path)

    def _reload(self):
        c = config.open_config(self.conf_path, base_default=True,
                               verbose=False)
        self.assertEqual(c["general"]["logging"], "",
                         "logging must stay '' to avoid auto.log in cwd")
        return c


class TestCliHostGroupUpdate(_CLIHostGroupBase):

    def test_update_materializes_and_reports(self):
        # logging='' is persisted (guards against auto.log in cwd)
        self._reload()
        out = _run(cli_main.host_group_update, _ns(self.conf_path))
        # both tiers reported with host group + host counts
        self.assertIn("tier midplane:", out)
        self.assertIn("tier rack:", out)

        # the hg table is actually materialized for both tiers
        ld = log_db.LogData(self._reload())
        self.assertEqual(sorted(ld.hg_tiers()), ["midplane", "rack"])
        self.assertEqual(sorted(ld.members("midplane", "R00-M0")),
                         ["R00-M0-N0-C:J02-U01", "R00-M0-N1-C:J04-U05"])
        # rack R00 covers all three R00 chips
        self.assertEqual(len(ld.members("rack", "R00")), 3)

    def test_update_no_tiers_message(self):
        # rewrite config with empty host_group_filename
        conf = self._reload()
        conf["manager"]["host_group_filename"] = ""
        _write_conf_file(conf, self.conf_path)
        out = _run(cli_main.host_group_update, _ns(self.conf_path))
        self.assertIn("no tiers", out)


class TestCliShowHostGroup(_CLIHostGroupBase):

    def test_show_all_tiers(self):
        _run(cli_main.host_group_update, _ns(self.conf_path))
        out = _run(cli_main.show_host_group, _ns(self.conf_path, tier=None))
        self.assertIn("tier midplane: 4 host group(s)", out)
        self.assertIn("tier rack: 2 host group(s)", out)
        # per-hg summary lines include host and event counts
        self.assertIn("R00-M0:", out)
        self.assertIn("host(s)", out)
        self.assertIn("event(s)", out)

    def test_show_single_tier(self):
        _run(cli_main.host_group_update, _ns(self.conf_path))
        out = _run(cli_main.show_host_group,
                   _ns(self.conf_path, tier="rack"))
        self.assertIn("tier rack:", out)
        self.assertNotIn("tier midplane:", out)
        # R00 rack has 3 member hosts, each with one event
        self.assertIn("R00: 3 host(s), 3 event(s)", out)

    def test_show_events_count_matches_whole_host_lt(self):
        _run(cli_main.host_group_update, _ns(self.conf_path))
        out = _run(cli_main.show_host_group,
                   _ns(self.conf_path, tier="midplane"))
        # R00-M0 has 2 member hosts, each contributing one (host, ltid) pair
        self.assertIn("R00-M0: 2 host(s), 2 event(s)", out)


# nesting-check fixtures ----------------------------------------------------

# correct nesting: fine (midplane) refines coarse (rack)
_CHECK_HOSTS_OK = _BGL_HOSTS

# violation: a fine host group straddles two coarse host groups.
# fine = first letter group, coarse = trailing digit group, chosen so that
# one fine group (prefix "A") maps to two coarse groups ("0" and "1").
_VIOLATION_HOSTS = ["A-x0", "A-y1", "B-z0"]
_VIOLATION_INI = """\
[tiers]
order = fine, coarse
unmatched = keep

[tier_fine]
schemes = s
s.regex = ^([A-Z])

[tier_coarse]
schemes = s
s.regex = (\\d)$
"""

# ordering breakage: a host matched by the coarse tier but unmatched (kept as
# self) by the fine tier under a stricter fine regex.
_ORDERING_HOSTS = ["R00-M0-N0", "weird-host-1"]
_ORDERING_INI = """\
[tiers]
order = fine, coarse
unmatched = drop

[tier_fine]
schemes = s
s.regex = ^(R\\d+-M\\d)

[tier_coarse]
schemes = s
s.regex = (\\d)$
"""

# unmatched enumeration: keep policy, one host matches no scheme
_UNMATCHED_HOSTS = ["R00-M0-N0-C:J02-U01", "mystery-box"]
_UNMATCHED_INI = """\
[tiers]
order = midplane
unmatched = keep

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
"""


class TestCliHostGroupCheck(unittest.TestCase):

    def _setup(self, hosts, ini):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.remove, db_path)
        conf = _build_synthetic_db(db_path, hosts)

        fd2, ini_path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd2, "w") as f:
            f.write(ini)
        self.addCleanup(os.remove, ini_path)
        conf["manager"]["host_group_filename"] = ini_path
        conf["general"]["logging"] = ""

        fd3, conf_path = tempfile.mkstemp(suffix=".conf")
        os.close(fd3)
        self.addCleanup(os.remove, conf_path)
        config.write(conf_path, conf)
        return conf_path

    def test_correct_nesting_passes(self):
        conf_path = self._setup(_CHECK_HOSTS_OK, _NESTED_INI)
        out = _run(cli_main.host_group_check, _ns(conf_path))
        self.assertIn("nesting check passed", out)
        self.assertNotIn("nesting violation", out)
        self.assertNotIn("ordering breakage", out)

    def test_nesting_violation_flags_offending_hosts(self):
        conf_path = self._setup(_VIOLATION_HOSTS, _VIOLATION_INI)
        out = _run(cli_main.host_group_check, _ns(conf_path))
        self.assertIn("nesting violation", out)
        self.assertNotIn("nesting check passed", out)
        # the offending fine group "A" spans coarse "0" and "1"; both A hosts
        # are listed as offenders
        self.assertIn("A-x0", out)
        self.assertIn("A-y1", out)

    def test_ordering_breakage_detected(self):
        conf_path = self._setup(_ORDERING_HOSTS, _ORDERING_INI)
        out = _run(cli_main.host_group_check, _ns(conf_path))
        # weird-host-1 is dropped by the fine tier but matched by coarse (ends
        # in a digit), so the coarse group is not cleanly covered by fine.
        self.assertIn("ordering breakage", out)
        self.assertIn("weird-host-1", out)

    def test_unmatched_hosts_enumerated(self):
        conf_path = self._setup(_UNMATCHED_HOSTS, _UNMATCHED_INI)
        out = _run(cli_main.host_group_check, _ns(conf_path))
        # under keep policy, mystery-box matches no scheme -> listed unmatched
        self.assertIn("unmatched", out)
        self.assertIn("mystery-box", out)
        # the BGL chip host did match, so it is not listed
        self.assertNotIn("R00-M0-N0-C:J02-U01\n    R00", out)

    def test_no_tiers_message(self):
        conf_path = self._setup(_BGL_HOSTS, _NESTED_INI)
        conf = config.open_config(conf_path, base_default=True,
                                  verbose=False)
        conf["manager"]["host_group_filename"] = ""
        config.write(conf_path, conf)
        out = _run(cli_main.host_group_check, _ns(conf_path))
        self.assertIn("no tiers", out)


# ---------------------------------------------------------------------------
# Public 2k real-data verification (spec section 7 item 6).
#
# Drive a real amulog build over the bundled public BGL 2k sample
# (example/loghub_BGL/, 2000 lines, public loghub data; no private data and
# no MySQL), then confirm midplane resolution and hgid counts via
# HostGroup / update_hg. The real chip hostnames (e.g.
# R00-M0-N0-C:J02-U01 -> R00-M0) are what motivate the whole feature
# (spec section 0): the chip naming is too fine for the PC algorithm, and
# midplane aggregation collapses it. The full real BGL DB and MySQL backend
# stay in tests-ext / out of scope.
#
# This test is CI-safe: it uses only the public example shipped in the repo
# and a temporary sqlite DB; the build is offline and fast (~0.1s).
# ---------------------------------------------------------------------------

from amulog import manager  # noqa: E402
from amulog import __main__ as amulog_main  # noqa: E402

_EXAMPLE_BGL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir,
                 "example", "loghub_BGL"))

# midplane regex, same as the example would use; chip R<rack>-M<midplane>-...
_BGL_MIDPLANE_INI = """\
[tiers]
order = midplane
unmatched = keep

[tier_midplane]
schemes = bgl
bgl.regex = ^(R\\d+-M\\d)
"""


@unittest.skipUnless(
    os.path.isfile(os.path.join(_EXAMPLE_BGL_DIR, "BGL_2k.log")),
    "public example/loghub_BGL/BGL_2k.log not present")
class TestHostGroupPublicBGL2k(unittest.TestCase):
    """End-to-end midplane resolution on the public BGL 2k sample."""

    @classmethod
    def setUpClass(cls):
        fd, cls.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        # ltgen state dump default is .amulog.dump in cwd; redirect to a
        # temp file so the build never writes into the working tree.
        fd_dump, cls.dump_path = tempfile.mkstemp(suffix=".dump")
        os.close(fd_dump)

        conf = config.open_config(verbose=False)
        # CLI-style logging would create auto.log in cwd; disable it.
        conf["general"]["logging"] = ""
        conf["general"]["src_path"] = os.path.join(
            _EXAMPLE_BGL_DIR, "BGL_2k.log")
        conf["general"]["src_recur"] = "false"
        conf["database"]["database"] = "sqlite3"
        conf["database"]["sqlite3_filename"] = cls.db_path
        conf["manager"]["parser_script"] = os.path.join(
            _EXAMPLE_BGL_DIR, "parser.py")
        conf["manager"]["indata_filename"] = cls.dump_path
        conf["log_template"]["lt_methods"] = "drain"
        conf["log_template"]["ltgroup_alg"] = "none"

        # build the DB from the real public log (offline, single pass)
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_offline(conf, targets, reset_db=True)

        fd2, cls.ini_path = tempfile.mkstemp(suffix=".ini")
        with os.fdopen(fd2, "w") as f:
            f.write(_BGL_MIDPLANE_INI)
        conf["manager"]["host_group_filename"] = cls.ini_path

        cls.conf = conf

    @classmethod
    def tearDownClass(cls):
        for p in (cls.db_path, cls.ini_path, cls.dump_path):
            if os.path.isfile(p):
                os.remove(p)

    def test_build_loaded_real_chip_hosts(self):
        # The build really ingested the 2k sample and the host column holds
        # the original BGL chip names (the truth source).
        ld = log_db.LogData(self.conf)
        self.assertEqual(ld.count_lines(), 2000)
        hosts = set(ld.whole_host())
        # chip-level naming is very fine: well over a thousand distinct hosts
        # in just 2k lines (this fineness is the problem midplane solves).
        self.assertGreater(len(hosts), 1000)
        # a concrete real chip host from the sample is present
        chip_present = any(h.startswith("R") and "-M" in h and "-C:" in h
                           for h in hosts)
        self.assertTrue(chip_present,
                        "no BGL chip-style host found in the built DB")

    def test_resolve_real_chip_to_midplane(self):
        # The headline assertion required by the spec: a real BGL chip
        # hostname resolves to its midplane.
        hg = host_group.HostGroup(self.conf)
        self.assertEqual(
            hg.resolve("R00-M0-N0-C:J02-U01", "midplane"), "R00-M0")
        # every actual chip host in the DB resolves to an R<r>-M<m> midplane,
        # and the midplane is a strict prefix of the chip name.
        ld = log_db.LogData(self.conf)
        chip_re = re.compile(r"^R\d+-M\d")
        for host in ld.whole_host():
            if chip_re.match(host):
                hgid = hg.resolve(host, "midplane")
                self.assertTrue(host.startswith(hgid),
                                "{0!r} -> {1!r} not a prefix".format(
                                    host, hgid))
                self.assertRegex(hgid, r"^R\d+-M\d$")

    def test_update_hg_collapses_chips_to_midplanes(self):
        # update_hg materializes the midplane tier; the chip-level host set
        # collapses to a much smaller number of midplane host groups.
        ld = log_db.LogData(self.conf, edit=True, reset_db=False)
        hg = host_group.HostGroup(self.conf)
        ld.update_hg(hg)

        num_hosts = len(set(ld.whole_host()))
        groups = ld.iter_hg("midplane")
        num_groups = len(groups)
        # real collapse: many chips -> few midplanes (a strict, large drop).
        self.assertGreater(num_hosts, num_groups)
        self.assertLess(num_groups, num_hosts // 2)
        # every materialized midplane hgid that came from a chip matches the
        # R<rack>-M<midplane> shape, and members are a subset of whole_host.
        all_hosts = set(ld.whole_host())
        chip_re = re.compile(r"^R\d+-M\d$")
        chip_midplanes = [g for g in groups if chip_re.match(g)]
        self.assertGreater(len(chip_midplanes), 1)
        for hgid in chip_midplanes:
            members = ld.members("midplane", hgid)
            self.assertTrue(members)
            self.assertTrue(set(members).issubset(all_hosts))
            for m in members:
                self.assertTrue(m.startswith(hgid))

    def test_members_iter_lines_consistent_on_real_data(self):
        # members(tier, hgid) -> iter_lines(host_in=...) equals the
        # iter_lines(hg=(tier, hgid)) convenience axis on real data.
        ld = log_db.LogData(self.conf, edit=True, reset_db=False)
        ld.update_hg(host_group.HostGroup(self.conf))
        groups = ld.iter_hg("midplane")
        # pick the midplane with the most member hosts for a meaningful check
        chip_re = re.compile(r"^R\d+-M\d$")
        chip_groups = {g: ms for g, ms in groups.items() if chip_re.match(g)}
        hgid = max(chip_groups, key=lambda g: len(chip_groups[g]))

        members = ld.members("midplane", hgid)
        via_host_in = sorted(lm.lid for lm in ld.iter_lines(host_in=members))
        via_hg = sorted(lm.lid
                        for lm in ld.iter_lines(hg=("midplane", hgid)))
        self.assertEqual(via_host_in, via_hg)
        self.assertTrue(via_hg, "expected at least one line for the midplane")
        # the host column on those lines is the ORIGINAL chip host (not hgid)
        line_hosts = {lm.host for lm in ld.iter_lines(hg=("midplane", hgid))}
        self.assertTrue(all(h.startswith(hgid) for h in line_hosts))
        self.assertNotIn(hgid, line_hosts)  # hgid itself is not a raw host


if __name__ == "__main__":
    unittest.main()
