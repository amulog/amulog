#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog import common
from amulog import config
from amulog import testlog
from amulog import log_db


class TestDB(unittest.TestCase):
    
    def test_db_sqlite3(self):
        path_testlog = "/tmp/amulog_testlog"
        path_db = "/tmp/amulog_db"

        conf = config.open_config()
        path_testlog = conf['general']['src_path']
        path_db = conf['database']['sqlite3_filename']

        tlg = testlog.TestLogGenerator(testlog.DEFAULT_CONFIG, seed = 3)
        tlg.dump_log(path_testlog)

        l_path = config.getlist(conf, "general", "src_path")
        if conf.getboolean("general", "src_recur"):
            targets = common.recur_dir(l_path)
        else:
            targets = common.rep_dir(l_path)
        log_db.process_files_online(conf, targets, True)

        ld = log_db.LogData(conf)
        num = ld.count_lines()
        self.assertEqual(num, 6539,
                         "not all logs added to database")
        ltg_num = len([gid for gid in ld.iter_ltgid()])
        self.assertTrue(ltg_num > 3 and ltg_num < 10,
                        ("log template generation fails? "
                         "(groups: {0})".format(ltg_num)))
        
        del ld
        common.rm(path_testlog)
        common.rm(path_db)


if __name__ == "__main__":
    unittest.main()

