#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import config
from amulog import log_db
from amulog import manager

from amulog import testutil


class TestMysql(unittest.TestCase):

    _path_testlog = None
    _path_testdb = None
    _path_ltgendump = None

    @classmethod
    def setUpClass(cls):
        fd_testlog, cls._path_testlog = tempfile.mkstemp()
        os.close(fd_testlog)
        fd_ltgendump, cls._path_ltgendump = tempfile.mkstemp()
        os.close(fd_ltgendump)

        tlg = testutil.TestLogGenerator(testutil.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_testlog)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._path_testlog)
        os.remove(cls._path_ltgendump)

    @classmethod
    def _conf_mysql(cls):
        conf = config.open_config()
        conf['general']['src_path'] = cls._path_testlog
        conf['database']['database'] = "mysql"
        conf['database']['mysql_host'] = "localhost"
        conf['database']['mysql_dbname'] = "test_amulog"
        conf['database']['mysql_user'] = "testamulog"
        conf['database']['mysql_passwd'] = "testamulog"
        conf['manager']['indata_filename'] = cls._path_ltgendump
        return conf

    def test_makedb_online(self):
        conf = self._conf_mysql()
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_online(conf, targets, reset_db=True)

        ld = log_db.LogData(conf)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(3 < ltg_num < 20,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))
        ld.drop_all()

    def test_makedb_offline(self):
        conf = self._conf_mysql()
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_offline(conf, targets, reset_db=True)

        ld = log_db.LogData(conf)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(3 < ltg_num < 20,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))
        ld.drop_all()
