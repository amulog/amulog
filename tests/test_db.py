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

    def test_makedb_parallel_incremental(self):
        # incremental parallel offline (reset_db=False) used to crash at
        # manager.process_offline: in parallel mode self._ltgen is None, so
        # the `self._ltgen.is_stateful()` guard raised AttributeError.
        import copy
        conf = copy.copy(self._conf)
        conf["manager"]["n_process"] = "2"
        conf["log_template"]["lt_methods"] = "re"
        conf["log_template_re"]["variable_rule"] = \
            common.filepath_local(__file__, "test_re.conf")

        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_offline(conf, targets,
                                      reset_db=True, parallel=True)
        n1 = log_db.LogData(self._conf).count_lines()

        # add the same lines again incrementally (re is stateless)
        manager.process_files_offline(conf, targets,
                                      reset_db=False, parallel=True)
        n2 = log_db.LogData(self._conf).count_lines()
        self.assertEqual(n2, 2 * n1)

    def test_load_clean_tolerate_none_ltgen(self):
        # In parallel mode the main manager keeps self._ltgen = None (the
        # ltgen lives in workers). load()/clean() must tolerate that like
        # dump() does, instead of dereferencing None.
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)
        manager.process_files_online(self._conf, targets, reset_db=True)

        ld = log_db.LogData(self._conf, edit=True, reset_db=False)
        ltm = manager.LTManager(self._conf, ld.db, ld.lttable, reset_db=False)
        ltm._ltgen = None  # simulate the parallel-mode main manager
        ltm.load()   # must not raise AttributeError
        ltm.clean()  # must not raise AttributeError

    def test_split_into_batches(self):
        # CR-37: offline_batchsize is the per-batch size, not the batch count.
        batches = manager.split_into_batches(list(range(10)), 3)
        self.assertEqual([len(b) for b in batches], [3, 3, 3, 1])
        # all items are covered exactly once, in order
        self.assertEqual([x for b in batches for x in b], list(range(10)))
        # batchsize >= len -> a single batch (no empty/strided buckets)
        self.assertEqual(manager.split_into_batches([1, 2, 3], 100), [[1, 2, 3]])
        # non-positive size falls back to 1 (no step-0 ValueError)
        self.assertEqual(manager.split_into_batches([1, 2], 0), [[1], [2]])

    def test_makedb_parallel_multibatch(self):
        # small batchsize -> many batches; chunking must cover all lines
        # exactly (regression for CR-37's batch partitioning).
        import copy
        conf = copy.copy(self._conf)
        conf["manager"]["n_process"] = "2"
        conf["manager"]["offline_batchsize"] = "500"
        conf["log_template"]["lt_methods"] = "re"
        conf["log_template_re"]["variable_rule"] = \
            common.filepath_local(__file__, "test_re.conf")

        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(conf)
        manager.process_files_offline(conf, targets,
                                      reset_db=True, parallel=True)
        self.assertEqual(log_db.LogData(self._conf).count_lines(), 6539)

    def test_fail_dump_separates_lines(self):
        # CR-13: fail_dump wrote msg verbatim, so a failing line without a
        # trailing newline (e.g. a file's last line) merged with the next.
        fd, fail_path = tempfile.mkstemp()
        os.close(fd)
        fd2, db_path = tempfile.mkstemp()
        os.close(fd2)
        conf = config.open_config(verbose=False)
        conf['general']['logging'] = ""
        conf['database']['sqlite3_filename'] = db_path
        conf['manager']['fail_output'] = fail_path
        try:
            ld = log_db.LogData(conf, edit=True, reset_db=True)
            ltm = manager.LTManager(conf, ld.db, ld.lttable, reset_db=True)
            ltm.fail_dump("no trailing newline")
            ltm.fail_dump("already has one\n")
            with open(fail_path, encoding="utf-8") as f:
                content = f.read()
            self.assertEqual(content, "no trailing newline\nalready has one\n")
        finally:
            os.remove(fail_path)
            os.remove(db_path)

    def test_data_from_data(self):
        # regression: data_from_data referenced [database].undefined_host
        # (NoOptionError, CR-38) and skipped add_esc before parse_line (CR-39).
        import shutil
        outdir = tempfile.mkdtemp()
        try:
            from amulog import __main__ as amulog_main
            targets = amulog_main.get_targets_conf(self._conf)
            manager.data_from_data(self._conf, targets, outdir,
                                   method="commit", reset=False)
            files = os.listdir(outdir)
            self.assertTrue(len(files) > 0, "no restored data written")
            total = sum(os.path.getsize(os.path.join(outdir, f))
                        for f in files)
            self.assertTrue(total > 0, "restored data is empty")
        finally:
            shutil.rmtree(outdir)

    def test_datetime_key_compat_and_info_term(self):
        # CR-14: dts/dte are canonical; top_dt/end_dt are accepted as legacy
        # aliases (compat shim now centralized). Both must select the same set.
        import datetime
        from amulog import __main__ as amulog_main
        targets = amulog_main.get_targets_conf(self._conf)
        manager.process_files_online(self._conf, targets, reset_db=True)

        ld = log_db.LogData(self._conf)
        dts = datetime.datetime(2000, 1, 1)
        dte = datetime.datetime(2200, 1, 1)
        n_canonical = sum(1 for _ in ld.iter_lines(dts=dts, dte=dte))
        n_legacy = sum(1 for _ in ld.iter_lines(top_dt=dts, end_dt=dte))
        self.assertEqual(n_canonical, n_legacy)
        self.assertEqual(n_canonical, 6539)

        # info_term (term-restricted DB status) runs without error
        log_db.info_term(self._conf, dts, dte)

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
