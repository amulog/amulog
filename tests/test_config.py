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


class TestHostGroupConfigWiring(unittest.TestCase):
    """[manager] host_group_filename default + main-config -> HostGroup path.

    The key is declared in amulog/data/config.conf.default next to
    host_alias_filename; its default is empty (host stratification disabled).
    A real config that points the key at a host_group ini must load into a
    HostGroup with the expected tiers()/resolve().
    """

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

    def test_default_is_empty_and_disabled(self):
        # default config: key present, empty -> stratification disabled
        from amulog import host_group
        conf = config.open_config(None, base_default=True, verbose=False)
        self.assertTrue(conf.has_option("manager", "host_group_filename"))
        self.assertEqual(conf["manager"]["host_group_filename"], "")
        hg = host_group.init_hostgroup(conf)
        self.assertEqual(hg.tiers(), [])
        self.assertIsNone(hg.label_tier())

    def test_ini_loads_into_hostgroup(self):
        from amulog import host_group
        ini = self._write("hg.ini", """\
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
schemes = bgl
bgl.regex = ^(R\\d+)
""")
        main = self._write("amulog.conf", """\
[general]

[manager]
host_group_filename = {0}
""".format(ini))
        conf = config.open_config(main, base_default=True, verbose=False)
        # the explicit value overrides the empty default
        self.assertEqual(conf["manager"]["host_group_filename"], ini)
        hg = host_group.init_hostgroup(conf)
        self.assertEqual(hg.tiers(), ["default", "midplane", "rack"])
        self.assertEqual(hg.label_tier(), "default")
        self.assertEqual(hg.resolve("R00-M0-N0-C:J02-U01", "midplane"),
                         "R00-M0")
        self.assertEqual(hg.resolve("R00-M0-N0-C:J02-U01", "rack"), "R00")
        # default tier (builtin ipaddr): IP canonicalized, non-IP kept
        self.assertEqual(hg.resolve("10.0.0.5", "default"), "10.0.0.5")

    def test_shipped_sample_parses(self):
        # the shipped sample (sans the s4 hostalias file, which is private)
        # must parse and resolve the BGL tiers.
        import os
        from amulog import host_group
        sample = os.path.join(os.path.dirname(config.DEFAULT_CONFIG),
                              "host_group.conf.sample")
        self.assertTrue(os.path.exists(sample))
        body = open(sample).read()
        # drop the hostalias scheme: its file (def_s4_host_alias.txt) is a
        # private companion artifact, not shipped in the public tree.
        body = body.replace("schemes = ipaddr, s4", "schemes = ipaddr")
        ini = self._write("hg_sample.ini", body)
        conf = {"manager": {"host_group_filename": ini}}
        hg = host_group.HostGroup(conf)
        self.assertEqual(hg.tiers(), ["default", "midplane", "rack"])
        self.assertEqual(hg.label_tier(), "default")
        self.assertEqual(hg.resolve("R02-M1-N0-C:J12-U11", "midplane"),
                         "R02-M1")
        self.assertEqual(hg.resolve("R02-M1-N0-C:J12-U11", "rack"), "R02")


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


class TestTimezone(unittest.TestCase):
    """[general] timezone selection (CR-1). Default = system local; the option
    only labels returned datetimes and query bounds, never the wall-clock."""

    @staticmethod
    def _conf(tz):
        import configparser
        c = configparser.ConfigParser()
        c.read_dict({"general": {"timezone": tz},
                     "t": {"dt": "2020-01-01 00:00:00",
                           "term": "2020-01-01 00:00:00, 2020-01-02 03:00:00"}})
        return c

    def test_default_and_local_are_system_local(self):
        from dateutil.tz import tzlocal
        ref = datetime.datetime(2020, 1, 1)
        for tz in ("", "local", "LOCAL"):
            self.assertEqual(config.get_timezone(self._conf(tz)).utcoffset(ref),
                             tzlocal().utcoffset(ref))

    def test_missing_option_defaults_local(self):
        import configparser
        from dateutil.tz import tzlocal
        c = configparser.ConfigParser()
        c.read_dict({"general": {}})
        ref = datetime.datetime(2020, 1, 1)
        self.assertEqual(config.get_timezone(c).utcoffset(ref),
                         tzlocal().utcoffset(ref))

    def test_utc_and_named_zone(self):
        ref = datetime.datetime(2020, 6, 1)
        self.assertEqual(
            config.get_timezone(self._conf("utc")).utcoffset(ref),
            datetime.timedelta(0))
        self.assertEqual(
            config.get_timezone(self._conf("Asia/Tokyo")).utcoffset(ref),
            datetime.timedelta(hours=9))

    def test_invalid_zone_rejected(self):
        with self.assertRaises(ValueError):
            config.get_timezone(self._conf("Not/AZone"))

    def test_getdt_labels_without_shifting_wallclock(self):
        dt = config.getdt(self._conf("utc"), "t", "dt")
        # wall-clock is exactly the stored string; only the label is UTC
        self.assertEqual((dt.year, dt.month, dt.day, dt.hour, dt.minute),
                         (2020, 1, 1, 0, 0))
        self.assertEqual(dt.utcoffset(), datetime.timedelta(0))

    def test_getterm_uses_timezone(self):
        dts, dte = config.getterm(self._conf("Asia/Tokyo"), "t", "term")
        self.assertEqual(dts.utcoffset(), datetime.timedelta(hours=9))
        self.assertEqual((dte.day, dte.hour), (2, 3))  # wall-clock preserved


if __name__ == "__main__":
    unittest.main()
