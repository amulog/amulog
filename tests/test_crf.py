#!/usr/bin/env python
# coding: utf-8

import os
import unittest
import tempfile

from amulog import config
from amulog import lt_common


class TestCRF(unittest.TestCase):

    _path_trainfile = None
    _path_model = None

    @classmethod
    def setUpClass(cls):
        cls.data_train = [["ssh N D",
                           "login N D",
                           "failure N V",
                           "from N D",
                           "192.168.100.1 N V"],
                          ["su N D",
                           "user N D",
                           "sat N V",
                           "enabled N D"]]
        cls.data_test = ["ssh", "auth", "failure",
                         "from", "192.168.100.1", "user", "sat"]

        fd, cls._path_trainfile = tempfile.mkstemp()
        with os.fdopen(fd, 'w') as f:
            l_buf = []
            for lines in cls.data_train:
                l_buf.append("\n".join(lines))
            f.write("\n\n".join(l_buf))

        fd, cls._path_model = tempfile.mkstemp()
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls._path_trainfile)
        os.remove(cls._path_model)

    def test_tagging(self):
        from amulog.alg.crf import init_ltgen
        conf = config.open_config(verbose=False)
        conf['log_template']['lt_methods'] = "shiso"
        conf["log_template_crf"]["model_filename"] = self._path_model

        table = lt_common.TemplateTable()
        ltgen = init_ltgen(conf, table)
        ltgen.init_trainer()
        ltgen.train_from_file(self._path_trainfile)

        tmp_pline = {"words": self.data_test}
        tpl = ltgen.generate_tpl(tmp_pline)
        self.assertTrue("ssh" in tpl)
        self.assertTrue(lt_common.REPLACER in tpl)


if __name__ == "__main__":
    unittest.main()


