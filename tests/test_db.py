#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import config
from amulog import log_db

from . import testlog


class TestDB(unittest.TestCase):

    def setUp(self):
        fd_testlog, self._path_testlog = tempfile.mkstemp()
        os.close(fd_testlog)
        fd_testdb, self._path_testdb = tempfile.mkstemp()
        os.close(fd_testdb)
        fd_ltgendump, self._path_ltgendump = tempfile.mkstemp()
        os.close(fd_ltgendump)

        self._conf = config.open_config()
        self._conf['general']['src_path'] = self._path_testlog
        self._conf['database']['sqlite3_filename'] = self._path_testdb
        self._conf['log_template']['indata_filename'] = self._path_ltgendump

        tlg = testlog.TestLogGenerator(testlog.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(self._path_testlog)

    def tearDown(self):
        os.remove(self._path_testlog)
        os.remove(self._path_testdb)
        os.remove(self._path_ltgendump)

    def test_000_makedb(self):
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


if __name__ == "__main__":
    unittest.main()

