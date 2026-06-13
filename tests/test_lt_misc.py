#!/usr/bin/env python
# coding: utf-8

import os
import sys
import tempfile
import unittest

from amulog import lt_common
from amulog import lt_misc


try:
    import ssdeep as _real_ssdeep
except ImportError:
    _real_ssdeep = None


class _FakeSsdeep:
    """Minimal ssdeep stand-in: identical strings score 100, else 0.

    Only exposes the attributes the real ssdeep module has (hash, compare)
    so that calling a non-existent API surfaces as an error here too.
    """

    @staticmethod
    def hash(s):
        return s

    @staticmethod
    def compare(h1, h2):
        return 100 if h1 == h2 else 0


def _make_lttable(items):
    """items: list of (ltid, ltw)"""
    table = lt_common.LTTable()
    for ltid, ltw in items:
        table.restore_lt(ltid, None, ltw, [""] * (len(ltw) + 1), 1)
    return table


class TestLTGroupFuzzyHash(unittest.TestCase):

    def setUp(self):
        # lt_misc._calc_score does `import ssdeep` lazily; inject a fake one
        self._real = sys.modules.get("ssdeep")
        sys.modules["ssdeep"] = _FakeSsdeep

    def tearDown(self):
        if self._real is None:
            sys.modules.pop("ssdeep", None)
        else:
            sys.modules["ssdeep"] = self._real

    def _assert_grouping(self, table, mem_hash):
        # ltids 10 and 20 share an identical template -> same group;
        # ltid 11 is distinct -> its own group.
        group = lt_misc.LTGroupFuzzyHash(table, th=1, mem_hash=mem_hash)
        result = group.make()
        self.assertEqual(len(result), 3)
        gid = {ltline.ltid: ltline.ltgid for ltline in result}
        self.assertEqual(gid[10], gid[20])
        self.assertNotEqual(gid[10], gid[11])

    def test_make_mem_hash(self):
        # regression: class was uninstantiable (abstract make) and the
        # mem_hash branch used enumerate index as ltid (CR-28), so gapped
        # ltids grouped incorrectly.
        table = _make_lttable([
            (10, ["alpha", "beta", "gamma", "delta"]),
            (11, ["foo", "bar", "baz", "qux"]),
            (20, ["alpha", "beta", "gamma", "delta"]),
        ])
        self._assert_grouping(table, mem_hash=True)

    def test_make_no_mem_hash(self):
        table = _make_lttable([
            (10, ["alpha", "beta", "gamma", "delta"]),
            (11, ["foo", "bar", "baz", "qux"]),
            (20, ["alpha", "beta", "gamma", "delta"]),
        ])
        self._assert_grouping(table, mem_hash=False)


@unittest.skipUnless(_real_ssdeep is not None, "ssdeep not installed")
class TestLTGroupFuzzyHashReal(unittest.TestCase):
    """Integration tests against the real ssdeep package, to catch API
    drift the mocked tests cannot (e.g. ssdeep has no hash_score())."""

    def _make_table(self, items):
        return _make_lttable(items)

    def _run(self, mem_hash):
        # identical templates (ltids 3 and 9) must land in the same group;
        # a distinct one (ltid 4) in its own. Long enough for ssdeep chunks.
        words_a = ["sshd", "accepted", "password", "for", "root", "from",
                   lt_common.REPLACER, "port", lt_common.REPLACER, "ssh2"]
        words_b = ["kernel", "usb", "device", "number", lt_common.REPLACER,
                   "using", "ehci", "host", "controller", "reset"]
        table = self._make_table([(3, words_a), (4, words_b), (9, words_a)])
        group = lt_misc.LTGroupFuzzyHash(table, th=80, mem_hash=mem_hash)
        result = group.make()
        gid = {ltline.ltid: ltline.ltgid for ltline in result}
        self.assertEqual(gid[3], gid[9])
        self.assertNotEqual(gid[3], gid[4])

    def test_real_mem_hash(self):
        self._run(mem_hash=True)

    def test_real_no_mem_hash(self):
        self._run(mem_hash=False)


@unittest.skipUnless(_real_ssdeep is not None, "ssdeep not installed")
class TestSsdeepLtgroupPipeline(unittest.TestCase):
    """End-to-end: the manager pipeline with ltgroup_alg=ssdeep must run
    without crashing and assign/persist ltgids (online add + remake paths)."""

    @classmethod
    def setUpClass(cls):
        from amulog import config, testutil
        fd_log, cls._path_log = tempfile.mkstemp()
        os.close(fd_log)
        fd_db, cls._path_db = tempfile.mkstemp()
        os.close(fd_db)
        fd_dump, cls._path_dump = tempfile.mkstemp()
        os.close(fd_dump)

        cls._conf = config.open_config(verbose=False)
        cls._conf['general']['src_path'] = cls._path_log
        cls._conf['database']['sqlite3_filename'] = cls._path_db
        cls._conf['manager']['indata_filename'] = cls._path_dump
        cls._conf['log_template']['ltgroup_alg'] = 'ssdeep'

        tlg = testutil.TestLogGenerator(testutil.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_log)

    @classmethod
    def tearDownClass(cls):
        for path in (cls._path_log, cls._path_db, cls._path_dump):
            os.remove(path)

    def test_online_and_remake(self):
        from amulog import manager, log_db
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)

        manager.process_files_online(self._conf, targets, reset_db=True)
        ld = log_db.LogData(self._conf)
        ltgids = [ltobj.ltgid for ltobj in ld.iter_lt()]
        self.assertTrue(len(ltgids) > 0)
        # every template must get a concrete group id
        self.assertTrue(all(g is not None for g in ltgids))

        # remake path exercises make()/remake_all()
        manager.remake_ltgroup(self._conf)
        ld2 = log_db.LogData(self._conf)
        self.assertTrue(
            all(o.ltgid is not None for o in ld2.iter_lt()))


if __name__ == "__main__":
    unittest.main()
