#!/usr/bin/env python
# coding: utf-8


import sys
import math
from collections import defaultdict

from . import common
from . import config
from . import lt_common
from . import lt_misc
from . import logparser


class LTGenVA(lt_common.LTGen):
    """
    Notice:
        In this log template generation algorithm,
        incremental process will fail in the beginning
        part of input data set.
        To avoid this, it is strongly recommended to use
        process_init_data (i.e., db-make-init)
        instead of process_line (i.e., db-make)
        with enough size of initial data.
    """

    def __init__(self, table, sym, method, threshold):
        super(LTGenVA, self).__init__(table, sym)
        self.method = method
        self.th = threshold
        self._d_wordcnt = defaultdict(int)

    def load(self, loadobj):
        self._d_wordcnt = loadobj

    def dumpobj(self):
        return self._d_wordcnt

    def _add_dict(self, l_w):
        self._d_wordcnt[None] += 1 # line count
        for w in l_w:
            self._d_wordcnt[w] += 1

    def _label(self, l_w):
        if self.method == "static":
            assert int(self.th) > 1, "lt_va.threshold is too small"
            l_label = []
            for w in l_w:
                if self._d_wordcnt[w] > self.th:
                    l_label.append(True)
                else:
                    l_label.append(False)
            return l_label
        elif self.method == "relative-line":
            assert self.th < 1, "lt_va.threshold is too large"
            l_label = []
            for w in l_w:
                if self._d_wordcnt[w] > self.th * self._d_wordcnt[None]:
                    l_label.append(True)
                else:
                    l_label.append(False)
            return l_label
        elif self.method == "relative-variable":
            # if th = 0.5, description words >= variable words
            assert self.th < 1, "lt_va.threshold is too large"
            l_cnt = [self._d_wordcnt[w] for w in l_w]
            temp_th = sorted(l_cnt)[int(self.th * len(l_w))]
            l_label = []
            for w in l_w:
                if self._d_wordcnt[w] >= temp_th:
                    l_label.append(True)
                else:
                    l_label.append(False)
            return l_label
        else:
            raise NotImplementedError

    def _label2tpl(self, l_w, l_label):
        ret = []
        for w, label in zip(l_w, l_label):
            if label == True:
                ret.append(w)
            elif label == False:
                ret.append(self._sym)
            else:
                raise ValueError
        return ret

    def process_init_data(self, lines):
        d = {}
        for line in lines:
            l_w, l_s = line
            self._add_dict(l_w)

        for mid, line in enumerate(lines):
            l_w, l_s = line
            l_label = self._label(l_w)
            tpl = self._label2tpl(l_w, l_label)

            if self._table.exists(tpl):
                tid = self._table.get_tid(tpl)
            else:
                tid = self._table.add(tpl)
            d[mid] = tid
        return d

    def process_line(self, l_w, l_s):
        self._add_dict(l_w)
        l_label = self._label(l_w)
        tpl = self._label2tpl(l_w, l_label)

        if self._table.exists(tpl):
            tid = self._table.get_tid(tpl)
            return tid, self.state_unchanged
        else:
            tid = self._table.add(tpl)
            return tid, self.state_added


def init_ltgen_va(conf, table, sym):
    method = conf.get("log_template_va", "method")
    threshold = conf.getfloat("log_template_va", "threshold")
    return LTGenVA(table, sym, method, threshold)


