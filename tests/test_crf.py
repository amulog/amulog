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


class TestRestoreNormalizedTpl(unittest.TestCase):
    """_restore_normalized_tpl is a classmethod independent of the
    (optional) log_normalizer package, so it can be unit-tested directly."""

    def test_restore(self):
        from amulog.alg.crf.lt_crf import LTGenCRF
        D, V = LTGenCRF.LABEL_DESC, LTGenCRF.LABEL_VAR
        # "beta" normalized into 2 tokens, both labeled V -> wildcard;
        # "alpha"/"gamma" stay as description words.
        org_words = ["alpha", "beta", "gamma"]
        length_vec = [1, 2, 1]
        labels = [D, V, V, D]
        # regression (CR-06): labels[start, stop] (tuple index) -> TypeError
        tpl = LTGenCRF._restore_normalized_tpl(org_words, labels, length_vec)
        self.assertEqual(tpl, ["alpha", lt_common.REPLACER, "gamma"])

    def test_restore_dummy_label_raises(self):
        from amulog.alg.crf.lt_crf import LTGenCRF
        D, N = LTGenCRF.LABEL_DESC, LTGenCRF.LABEL_DUMMY
        with self.assertRaises(ValueError):
            LTGenCRF._restore_normalized_tpl(["a", "b"], [D, N], [1, 1])


if __name__ == "__main__":
    unittest.main()


