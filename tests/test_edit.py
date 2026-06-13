#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import unittest
import tempfile

from amulog import config
from amulog import manager
from amulog import testutil


class TestEditCli(unittest.TestCase):
    """Smoke tests for amulog.edit CLI handlers."""

    _path_testlog = None
    _path_testdb = None
    _path_ltgendump = None
    _path_conf = None

    @classmethod
    def setUpClass(cls):
        fd_testlog, cls._path_testlog = tempfile.mkstemp()
        os.close(fd_testlog)
        fd_testdb, cls._path_testdb = tempfile.mkstemp()
        os.close(fd_testdb)
        fd_ltgendump, cls._path_ltgendump = tempfile.mkstemp()
        os.close(fd_ltgendump)
        fd_conf, cls._path_conf = tempfile.mkstemp()
        os.close(fd_conf)

        conf = config.open_config(verbose=False)
        conf['general']['src_path'] = cls._path_testlog
        conf['database']['sqlite3_filename'] = cls._path_testdb
        conf['manager']['indata_filename'] = cls._path_ltgendump
        # CLI handlers call set_common_logging; "" -> stderr instead of
        # creating an auto.log file in the working directory
        conf['general']['logging'] = ""

        tlg = testutil.TestLogGenerator(testutil.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_testlog)

        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_online(conf, targets, reset_db=True)

        # persist conf so CLI handlers can reopen it from a path
        config.write(cls._path_conf, conf)

    @classmethod
    def tearDownClass(cls):
        for path in (cls._path_testlog, cls._path_testdb,
                     cls._path_ltgendump, cls._path_conf):
            os.remove(path)

    def _make_ns(self, **kwargs):
        ns = argparse.Namespace(conf_path=self._path_conf, debug=False)
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def test_show_lt_breakdown(self):
        from amulog.edit import __main__ as edit_main
        ns = self._make_ns(ltid=0, lines=5)
        # must not raise (regression: conf was passed instead of LogData)
        edit_main.show_lt_breakdown(ns)


if __name__ == "__main__":
    unittest.main()
