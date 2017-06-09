#!/usr/bin/env python
# coding: utf-8

import unittest

from amulog import common
from amulog import config
from amulog import lt_common
from amulog import lt_crf
from amulog.crf import convert


class TestCRF(unittest.TestCase):

    def setUp(self):
        self.data_train = [["ssh N D",
                      "login N D",
                      "failure N V",
                      "from N D",
                      "192.168.100.1 N V"],
                      ["su N D",
                       "user N D",
                       "sat N V",
                       "enabled N D"]]
        self.data_test = ["ssh", "auth", "failure",
                          "from", "192.168.100.1", "user", "sat"]


    def test_label_train(self):
        converter = convert.FeatureExtracter()
        for data_line in self.data_train:
            lineitem = [item.split() for item in data_line]
            fset = converter.feature(lineitem)
            self.assertEqual(len(fset), len(data_line))
            for fsubset in fset.items():
                self.assertTrue(len(fsubset) > 0)

    def test_tagging(self):
        conf = config.open_config()
        sym = conf.get("log_template", "variable_symbol")
        table = lt_common.TemplateTable()
        converter = convert.FeatureExtracter()
        ltgen = lt_crf.LTGenCRF(table, sym, conf)

        l_items = []
        for data_line in self.data_train:
            lineitem = [item.split() for item in data_line]
            l_items.append(lineitem) 
        ltgen.init_trainer()
        ltgen.train(l_items)

        tid, state = ltgen.process_line(self.data_test, None)
        tpl = ltgen._table.get_template(tid)
        self.assertTrue("ssh" in tpl)
        self.assertTrue("**" in tpl)
        common.rm(ltgen.model)


if __name__ == "__main__":
    unittest.main()


