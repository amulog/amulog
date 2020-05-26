#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import config
from amulog import log_db

from . import testlog


class TestDB(unittest.TestCase):

    _path_testlog = None
    _path_testdb = None
    _path_ltgendump = None

    @classmethod
    def setUpClass(cls):
        fd_testlog, cls._path_testlog = tempfile.mkstemp()
        os.close(fd_testlog)
        fd_testdb, cls._path_testdb = tempfile.mkstemp()
        os.close(fd_testdb)
        fd_ltgendump, cls._path_ltgendump = tempfile.mkstemp()
        os.close(fd_ltgendump)

        cls._conf = config.open_config()
        cls._conf['general']['src_path'] = cls._path_testlog
        cls._conf['database']['sqlite3_filename'] = cls._path_testdb
        cls._conf['log_template']['indata_filename'] = cls._path_ltgendump

        tlg = testlog.TestLogGenerator(testlog.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_testlog)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._path_testlog)
        os.remove(cls._path_testdb)
        os.remove(cls._path_ltgendump)

    def test_makedb_online(self):
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)
        log_db.process_files_online(self._conf, targets, reset_db=True)

        ld = log_db.LogData(self._conf)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(3 < ltg_num < 20,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))

    def test_makedb_offline(self):
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)
        log_db.process_files_offline(self._conf, targets)

        ld = log_db.LogData(self._conf)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(3 < ltg_num < 20,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))


if __name__ == "__main__":
    unittest.main()

