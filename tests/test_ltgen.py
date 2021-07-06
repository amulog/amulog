#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import common
from amulog import config
from amulog import manager
from amulog import lt_common

from amulog import testutil


class TestLTGen(unittest.TestCase):

    _path_testlog = None

    @classmethod
    def setUpClass(cls):
        fd_testlog, cls._path_testlog = tempfile.mkstemp()
        os.close(fd_testlog)
        tlg = testutil.TestLogGenerator(testutil.DEFAULT_CONFIG, seed=3)
        tlg.dump_log(cls._path_testlog)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._path_testlog)

    def _try_method(self, conf, online=True):
        table = lt_common.TemplateTable()
        ltgen = manager.init_ltgen_methods(conf, table)

        iterobj = manager.iter_plines(conf, [self._path_testlog])
        if online:
            for pline in iterobj:
                ltgen.process_line(pline)
        else:
            d_pline = {mid: pline for mid, pline in enumerate(iterobj)}
            ltgen.process_offline(d_pline)
        return table

    def test_import(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "import"
        tpl_path = common.filepath_local(__file__, "testlog_tpl.txt")
        conf['log_template_import']['def_path'] = tpl_path
        table = self._try_method(conf)

        n_tpl = len(table)
        self.assertTrue(n_tpl == 6)

    def test_drain(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "drain"
        table = self._try_method(conf)

        n_tpl = len(table)
        self.assertTrue(3 < n_tpl < 20)

    def test_lenma(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "lenma"
        table = self._try_method(conf)

        n_tpl = len(table)
        self.assertTrue(3 < n_tpl < 20)

    def test_dlog(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "dlog"
        table = self._try_method(conf, online=False)

        n_tpl = len(table)
        self.assertTrue(3 < n_tpl < 300)

    def test_fttree(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "fttree"
        table = self._try_method(conf)

        n_tpl = len(table)
        self.assertTrue(3 < n_tpl < 50)

    def test_va(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "va"
        table = self._try_method(conf)

        n_tpl = len(table)
        self.assertTrue(3 < n_tpl < 20)

    def test_shiso_first(self):
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "shiso"
        table = self._try_method(conf)

        n_tpl = len(table)
        self.assertTrue(3 < n_tpl < 20)


if __name__ == "__main__":
    unittest.main()


