#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import common
from amulog import config
from amulog import log_db
from amulog import manager

from amulog import testutil


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

        cls._conf = config.open_config(verbose=False)
        cls._conf['general']['src_path'] = cls._path_testlog
        cls._conf['database']['sqlite3_filename'] = cls._path_testdb
        cls._conf['manager']['indata_filename'] = cls._path_ltgendump

        tlg = testutil.TestLogGenerator(testutil.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_testlog)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._path_testlog)
        os.remove(cls._path_testdb)
        os.remove(cls._path_ltgendump)

    def test_makedb_online(self):
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)
        manager.process_files_online(self._conf, targets, reset_db=True)

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
        manager.process_files_offline(self._conf, targets, reset_db=True)

        ld = log_db.LogData(self._conf)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(3 < ltg_num < 20,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))

    def test_makedb_parallel(self):
        import copy
        conf = copy.copy(self._conf)
        conf["manager"]["n_process"] = "2"
        conf["log_template"]["lt_methods"] = "re"
        conf["log_template_re"]["variable_rule"] = \
            common.filepath_local(__file__, "test_re.conf")

        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_offline(conf, targets, reset_db=True, parallel=True)

        ld = log_db.LogData(self._conf)
        #for ltobj in ld.iter_lt():
        #    print(ltobj)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(3 < ltg_num < 20,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))

    def test_anonymize_overwrite(self):
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)
        manager.process_files_online(self._conf, targets, reset_db=True)

        from amulog import anonymize
        am = anonymize.AnonymizeMapper(self._conf)
        am.anonymize()

        import re
        reobj_message = re.compile(r"^[ *#]*$")
        reobj_host = re.compile(r"^host\d+$")
        ld = log_db.LogData(self._conf)
        for lm in ld.iter_lines(ltid=0):
            message = lm.restore_message()
            self.assertTrue(reobj_message.match(message))
            self.assertTrue(reobj_host.match(lm.host))


if __name__ == "__main__":
    unittest.main()
