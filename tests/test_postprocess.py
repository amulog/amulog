#!/usr/bin/env python
# coding: utf-8

import os
import tempfile
import unittest

from amulog import common
from amulog import config
from amulog import lt_common
from amulog import lt_regex


def _vre_path():
    return common.filepath_local(__file__, "test_re.conf")


class TestVariableLabelRegex(unittest.TestCase):

    def test_replace_word(self):
        rule = lt_common.VariableLabelRegex(lt_regex.VariableRegex(_vre_path()))
        self.assertEqual(rule.replace_word("192.168.1.1"), "ipaddr")
        self.assertEqual(rule.replace_word("eth0"), "ifname")
        # unknown words are not labeled (stay a plain wildcard upstream)
        self.assertIsNone(rule.replace_word("spam"))


class TestLTPostProcess(unittest.TestCase):
    """Example use of LTPostProcess: rule-based variable labeling that
    separates templates sharing the same word structure but whose variable
    is a different type."""

    def _conf(self):
        conf = config.open_config(verbose=False)
        conf['log_template_re']['variable_rule'] = _vre_path()
        conf['manager']['host_alias_filename'] = ""
        return conf

    def _make_pp(self, l_alg=("regex",)):
        return lt_common.LTPostProcess(self._conf(), None, None, list(l_alg))

    def test_variable_label_separation(self):
        pp = self._make_pp()
        tpl = ["connect", "to", lt_common.REPLACER]
        out_ip = pp.replace_variable(
            ["connect", "to", "192.168.1.1"], tpl, lt_common.REPLACER)
        out_if = pp.replace_variable(
            ["connect", "to", "eth0"], tpl, lt_common.REPLACER)
        out_unknown = pp.replace_variable(
            ["connect", "to", "spam"], tpl, lt_common.REPLACER)

        self.assertEqual(out_ip, ["connect", "to", "*ipaddr*"])
        self.assertEqual(out_if, ["connect", "to", "*ifname*"])
        # an unlabeled (unknown) variable stays a plain wildcard
        self.assertEqual(out_unknown, ["connect", "to", lt_common.REPLACER])
        # same structure, different variable type -> distinct templates
        self.assertNotEqual(out_ip, out_if)

    def test_dummy_alg_keeps_wildcard(self):
        # "dummy" rule labels nothing: every variable stays a plain wildcard
        pp = self._make_pp(["dummy"])
        tpl = ["connect", "to", lt_common.REPLACER]
        out = pp.replace_variable(
            ["connect", "to", "192.168.1.1"], tpl, lt_common.REPLACER)
        self.assertEqual(out, ["connect", "to", lt_common.REPLACER])

    def test_host_alg_constructs(self):
        # regression for the shared __init__ rule dispatch: the existing
        # "host" alg must still build; with no alias file it labels nothing.
        pp = self._make_pp(["host"])
        tpl = ["connect", "to", lt_common.REPLACER]
        out = pp.replace_variable(
            ["connect", "to", "192.168.1.1"], tpl, lt_common.REPLACER)
        self.assertEqual(out, ["connect", "to", lt_common.REPLACER])

    def test_rule_precedence_falls_through(self):
        # rules are tried in order; "dummy" (always None) falls through to
        # "regex", so the result matches regex-only labeling.
        pp = self._make_pp(["dummy", "regex"])
        tpl = ["connect", "to", lt_common.REPLACER]
        out = pp.replace_variable(
            ["connect", "to", "192.168.1.1"], tpl, lt_common.REPLACER)
        self.assertEqual(out, ["connect", "to", "*ipaddr*"])

    def test_unknown_alg_raises(self):
        with self.assertRaises(NotImplementedError):
            self._make_pp(["no-such-alg"])


class TestLTPostProcessE2E(unittest.TestCase):
    """End-to-end: run LTPostProcess over templates actually generated from
    a log, inspect the labeled output, and characterize how amulog's DB
    template machinery handles labeled variables."""

    @classmethod
    def setUpClass(cls):
        from amulog import manager, testutil
        from amulog import __main__ as amulog_main

        fd_log, cls._path_log = tempfile.mkstemp()
        os.close(fd_log)
        fd_db, cls._path_db = tempfile.mkstemp()
        os.close(fd_db)
        fd_dump, cls._path_dump = tempfile.mkstemp()
        os.close(fd_dump)

        cls._conf = config.open_config(verbose=False)
        cls._conf['general']['src_path'] = cls._path_log
        cls._conf['general']['logging'] = ""
        cls._conf['database']['sqlite3_filename'] = cls._path_db
        cls._conf['manager']['indata_filename'] = cls._path_dump
        cls._conf['manager']['host_alias_filename'] = ""
        cls._conf['log_template_re']['variable_rule'] = _vre_path()

        testutil.TestLogGenerator(
            testutil.DEFAULT_CONFIG, seed=3).dump_log(cls._path_log)
        manager.process_files_online(
            cls._conf, amulog_main.get_targets_conf(cls._conf), reset_db=True)

    @classmethod
    def tearDownClass(cls):
        for path in (cls._path_log, cls._path_db, cls._path_dump):
            os.remove(path)

    def test_postprocess_labels_real_template_and_db_gap(self):
        from amulog import log_db
        ld = log_db.LogData(self._conf)
        pp = lt_common.LTPostProcess(self._conf, None, None, ["regex"])

        # find a real template whose variable gets labeled by a rule
        target = None
        for ltobj in ld.iter_lt():
            if not ltobj.var_location():
                continue
            lm = next(ld.iter_lines(ltid=ltobj.ltid))
            labeled = pp.replace_variable(
                lm.l_w, ltobj.ltw, lt_common.REPLACER)
            if labeled != list(ltobj.ltw):
                target = (ltobj, labeled)
                break
        self.assertIsNotNone(
            target, "expected at least one template variable to be labeled")
        ltobj, labeled = target

        # at least one plain wildcard became a *label* token
        changed = [(o, n) for o, n in zip(ltobj.ltw, labeled) if o != n]
        self.assertTrue(changed)
        for old, new in changed:
            self.assertEqual(old, lt_common.REPLACER)
            self.assertTrue(new.startswith("*") and new.endswith("*"))

        # --- DB-integration gap (characterization) ---
        # amulog identifies variable positions by exact `== REPLACER`, so a
        # labeled variable (*label*) is NOT recognized as a variable by the
        # DB template machinery: its position drops out of var_location().
        # This documents why LTPostProcess output is not yet pipeline-ready.
        labeled_lt = lt_common.LogTemplate(
            999, None, labeled, ltobj.lts, 1)
        self.assertLess(len(labeled_lt.var_location()),
                        len(ltobj.var_location()))


if __name__ == "__main__":
    unittest.main()
