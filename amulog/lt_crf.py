#!/usr/bin/env python
# coding: utf-8

import os
import re
import ipaddress
import logging

import pycrfsuite

from . import lt_common
from . import host_alias
from .crf import items
from .crf import convert

_logger = logging.getLogger(__package__)
DEFAULT_FEATURE_TEMPLATE = "/".join((os.path.dirname(__file__),
                                     "data/crf_template"))


class LTGenCRF(lt_common.LTGen):
    LABEL_DESC = "D"
    LABEL_VAR = "V"
    LABEL_DUMMY = "N"

    def __init__(self, table, sym, conf):
        super(LTGenCRF, self).__init__(table, sym)
        self.model = conf.get("log_template_crf", "model_filename")
        self.verbose = conf.getboolean("log_template_crf", "verbose")
        self._middle = conf.get("log_template_crf", "middle_label")
        self._template = conf.get("log_template_crf", "feature_template")

        self.trainer = None
        self.tagger = None
        self.converter = convert.FeatureExtracter(self._template)
        if self._middle == "re":
            self._lwobj = LabelWord(conf)
        elif len(self._middle) > 0 :
            raise NotImplementedError

    def _middle_label(self, w):
        if self._middle == "re":
            return self._lwobj.label(w)
        else:
            return w

    def init_trainer(self, alg = "lbfgs", verbose = False):
        self.trainer = pycrfsuite.Trainer(verbose = verbose)
        self.trainer.select(alg, "crf1d")
        d = {} # for parameter tuning, edit this
        if len(d) > 0:
            self.trainer.set_params(d)

    def train(self, iterable_items):
        for lineitem in iterable_items:
            xseq = self.converter.feature(lineitem)
            yseq = self.converter.label(lineitem)
            self.trainer.append(xseq, yseq)
        self.trainer.train(self.model)

    def train_from_file(self, fp):
        self.train(items.iter_items_from_file(fp))

    def init_tagger(self):
        if not os.path.exists(self.model):
            raise IOError("No trained CRF model for LTGenCRF")
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.model)

    def close_tagger(self):
        if not self.tagger is None:
            self.tagger.close()

    def process_line(self, l_w, l_s):
        if not self.tagger:
            self.init_tagger()
        lineitems = items.line2items(l_w, midlabel_func = self._middle_label,
                                      dummy_label = self.LABEL_DUMMY)
        fset = self.converter.feature(lineitems)
        l_label = self.tagger.tag(fset)        

        tpl = []
        for w, label in zip(l_w, l_label):
            if label == self.LABEL_DESC:
                tpl.append(w)
            elif label == self.LABEL_VAR:
                tpl.append(self._sym)
            elif label == self.LABEL_DUMMY:
                raise ValueError("Some word labeled as DUMMY in LTGenCRF")
            else:
                raise ValueError("Unknown labels in LTGenCRF")

        if self._table.exists(tpl):
            tid = self._table.get_tid(tpl)
            return tid, self.state_unchanged
        else:
            tid = self._table.add(tpl)
            return tid, self.state_added


class LabelWord():

    def __init__(self, conf):
        self._d_re = {}

        self._d_re["DIGIT"] = [re.compile(r"^\d+$")]
        self._d_re["DATE"] = [re.compile(r"^\d{2}/\d{2}$"),
                              re.compile(r"^\d{4}-\d{2}-\d{2}")]
        self._d_re["TIME"] = [re.compile(r"^\d{2}:\d{2}:\d{2}$")]

        self._other = "OTHER"
        self._ha = host_alias.HostAlias(conf)
        self._host = "HOST"

    def label(self, word):
        ret = self.isipaddr(word)
        if ret is not None:
            return ret

        if self._ha.isknown(word):
            return self._host
        
        for k, l_reobj in self._d_re.items():
            for reobj in l_reobj:
                if reobj.match(word):
                    return k
        
        return self._other

    @staticmethod
    def isipaddr(word):
        try:
            ret = ipaddress.ip_address(str(word))
            if isinstance(ret, ipaddress.IPv4Address):
                return "IPv4ADDR"
            elif isinstance(ret, ipaddress.IPv6Address):
                return "IPv6ADDR"
            else:
                raise TypeError("ip_address returns unknown type? {0}".format(
                        str(ret)))
        except ValueError:
            return None


#def test_label():
#    lwobj = LabelWord()
#    l_w = ["hoge", "hige", "1234", "[345]", "123.4.5.67", "8.8.8.8", "::2"]
#    for w in l_w:
#        print(" ".join(w, lwobj.label(w)))
#
#
#if __name__ == "__main__":
#    test_label()
#
