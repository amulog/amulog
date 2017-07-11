#!/usr/bin/env python
# coding: utf-8

import os
import re
import ipaddress
import logging
import random
import datetime
import configparser
from collections import defaultdict

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

    POS_UNKNOWN = "unknown"

    def __init__(self, table, sym, model, verbose, template, ig_val, lwobj):
        super(LTGenCRF, self).__init__(table, sym)
        self.model = model
        self.verbose = verbose
        #self._middle = middle
        self._template = template

        self.trainer = None
        self.tagger = None
        self.converter = convert.FeatureExtracter(self._template,
                                                  [self.POS_UNKNOWN], ig_val)
        self.lwobj = lwobj
        #if self._middle == "re":
        #    self._lwobj = LabelWord(conf)
        #elif len(self._middle) > 0 :
        #    raise NotImplementedError

    def _middle_label(self, w):
        return self.lwobj.label(w)

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

    def label_line(self, lineitems):
        if self.tagger is None:
            self.init_tagger()
        fset = self.converter.feature(lineitems)
        l_label = self.tagger.tag(fset)        
        return l_label

    def process_line(self, l_w, l_s):
        lineitems = items.line2items(l_w, midlabel_func = self._middle_label,
                                     dummy_label = self.LABEL_DUMMY)
        l_label = self.label_line(lineitems)
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

    def __init__(self, conf, fn = None):
        self._ext = {}
        self._re = {}
        
        self.conf = conf
        self._ha = host_alias.HostAlias(conf)
        if fn is not None:
            self._load_rule(fn)

    def _load_rule(self, fn):

        def gettuple(conf, sec, opt):
            s = conf[sec][opt]
            return tuple([w.strip() for w in s.split(",")])

        conf = configparser.ConfigParser()
        loaded = conf.read(fn)
        if len(loaded) == 0:
            sys.exit("opening config {0} failed".format(fn))

        t_ext = gettuple(conf, "ext", "rules")
        for rule in t_ext:
            self._ext[rule] = getattr(self, conf["ext"][rule].strip())

        t_re_rules = gettuple(conf, "re", "rules")
        for rule in t_re_rules:
            temp = []
            t_re = gettuple(conf, "re", rule)
            for restr in t_re:
                temp.append(re.compile(restr))
            self._re[rule] = temp

    def label(self, word):
        for key, func in self._ext.items():
            if func(word):
                return key

        for key, l_reobj in self._re.items():
            for reobj in l_reobj:
                if reobj.match(word):
                    return key

        return self.POS_UNKNOWN


    def label_ipaddr(self, word):
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


    def label_host(self, word):
        if self._ha.isknown(word):
            return self._host
        else:
            return None


#class LabelWord():
#
#    def __init__(self, conf):
#        self._d_re = {}
#
#        self._d_re["DIGIT"] = [re.compile(r"^\d+$")]
#        self._d_re["DATE"] = [re.compile(r"^\d{2}/\d{2}$"),
#                              re.compile(r"^\d{4}-\d{2}-\d{2}")]
#        self._d_re["TIME"] = [re.compile(r"^\d{2}:\d{2}:\d{2}$")]
#
#        self._other = "OTHER"
#        self._ha = host_alias.HostAlias(conf)
#        self._host = "HOST"
#
#    def label(self, word):
#        ret = self.isipaddr(word)
#        if ret is not None:
#            return ret
#
#        if self._ha.isknown(word):
#            return self._host
#        
#        for k, l_reobj in self._d_re.items():
#            for reobj in l_reobj:
#                if reobj.match(word):
#                    return k
#        
#        return self._other
#
#    @staticmethod
#    def isipaddr(word):
#        try:
#            ret = ipaddress.ip_address(str(word))
#            if isinstance(ret, ipaddress.IPv4Address):
#                return "IPv4ADDR"
#            elif isinstance(ret, ipaddress.IPv6Address):
#                return "IPv6ADDR"
#            else:
#                raise TypeError("ip_address returns unknown type? {0}".format(
#                        str(ret)))
#        except ValueError:
#            return None


class MeasureAccuracy():

    def __init__(self, conf):
        from . import log_db
        self.ld = log_db.LogData(conf)
        self.ld.init_ltmanager()
        self.conf = conf
        self.measure_lt_method = conf.get("measure_lt", "lt_method")
        self.sample_from = conf.get("measure_lt", "sample_from")
        self.sample_rules = self._rules(
            conf.get("measure_lt", "sample_rules"))
        self.trials = conf.getint("measure_lt", "train_trials")
        self.results = []
        
        if self.sample_from == "cross":
            self.cross_k = conf.getint("measure_lt", "cross_k")
            assert self.trials <= self.cross_k, "trials is larger than k"
            self._eval_cross()
        elif self.sample_from == "diff":
            self.sample_train_rules = self._rules(
                conf.get("measure_lt", "sample_train_rules"))
            self.train_sample_method = conf.get("measure_lt",
                                                "train_sample_method")
            self.train_size = conf.getint("measure_lt", "train_size")
            self._eval_diff()
        elif self.sample_from == "file":
            self.fn = conf.get("measure_lt", "sample_from_file")
            self.train_sample_method = conf.get("measure_lt",
                                                "train_sample_method")
            self.train_size = conf.getint("measure_lt", "train_size")
            self._eval_file()
        else:
            raise NotImplementedError

    @staticmethod
    def _rules(rulestr):
        ret = {}
        l_rule = [rule.strip() for rule in rulestr.split(",")]
        for rule in l_rule:
            if rule == "":
                continue
            key, val = [v.strip() for v in rule.split("=")]
            if key in ("host", "area"):
                ret[key] = val
            elif key == "top_date":
                assert not "top_dt" in ret
                ret["top_dt"] = datetime.datetime.strptime(val, "%Y-%m-%d")
            elif key == "top_dt":
                assert not "top_dt" in ret
                ret["top_dt"] = datetime.datetime.strptime(val,
                                                           "%Y-%m-%d %H:%M:%S")
            elif key == "end_date":
                assert not "end_dt" in ret
                ret["end_dt"] = datetime.datetime.strptime(
                    val, "%Y-%m-%d") + datetime.timedelta(days = 1)
            elif key == "end_dt":
                assert not "end_dt" in ret
                ret["end_dt"] = datetime.datetime.strptime(val,
                                                           "%Y-%m-%d %H:%M:%S")
            else:
                raise NotImplementedError
        return ret

    @staticmethod
    def _make_ltidmap(l_ltid):
        ltidmap = defaultdict(int)
        for ltid in l_ltid:
            ltidmap[ltid] += 1
        return ltidmap

    def _eval_cross(self):

        def divide_size(size, groups):
            base = int(size // groups)
            ret = [base] * groups
            surplus = size - (base * groups)
            for i in range(surplus):
                ret[i] += 1
            assert sum(ret) == size
            return ret

        def agg_dict(l_d):
            keyset = set()
            for d in l_d:
                keyset.update(d.keys())

            new_d = defaultdict(int)
            for ltid in keyset:
                for d in l_d:
                    new_d[ltid] += d[ltid]
            return new_d

        l_labeled = [] # (ltid, train)
        for line in self.ld.iter_lines(**self.sample_rules):
            l_labeled.append((line.lt.ltid, items.line2train(line)))
        random.shuffle(l_labeled)

        l_group = []
        l_group_ltidmap = [] # {ltid: cnt}
        basenum = 0
        for size in divide_size(len(l_labeled), self.cross_k):
            l_ltid, l_lm = zip(*l_labeled[basenum:basenum+size])
            l_group.append(l_lm)
            l_group_ltidmap.append(self._make_ltidmap(l_ltid))
            basenum += size
        assert sum([len(g) for g in l_group]) == len(l_labeled)
        del l_labeled

        for trial in range(self.trials):
            l_train = None
            l_test = []
            l_test_ltidmap = []
            for gid, group, ltidmap in zip(range(self.cross_k),
                                           l_group, l_group_ltidmap):
                if gid == trial:
                    assert l_train is None
                    l_train = group
                    train_ltidmap = ltidmap
                else:
                    l_test += group
                    l_test_ltidmap.append(ltidmap)
            test_ltidmap = agg_dict(l_test_ltidmap)

            d_result = self._trial(l_train, l_test,
                                   train_ltidmap, test_ltidmap)
            self.results.append(d_result)

    def _eval_diff(self):

        l_test, test_ltidmap = self._load_test_diff()

        l_train_all = [] # (ltid, lineitems)
        for line in self.ld.iter_lines(**self.sample_train_rules):
            l_train_all.append(line)

        if self.train_sample_method == "all":
            if self.trials > 1:
                raise UserWarning(("measure_lt.trials is ignored "
                                   "because the results must be static"))
            l_ltid = [lm.lt.ltid for lm in l_train_all]
            train_ltidmap = self._make_ltidmap(l_ltid)
            l_train = [items.line2train(lm) for lm in l_train_all]
            d_result = self._trial(l_train, l_test,
                                   train_ltidmap, test_ltidmap)
            self.results.append(d_result)
        elif self.train_sample_method == "random":
            for i in range(self.trials):
                l_sampled = random.sample(l_train_all, self.train_size)
                l_ltid = [lm.lt.ltid for lm in l_sampled]
                train_ltidmap = self._make_ltidmap(l_ltid)
                l_train = [items.line2train(lm) for lm in l_sampled]
                d_result = self._trial(l_train, l_test,
                                       train_ltidmap, test_ltidmap)
                self.results.append(d_result)
        elif self.train_sample_method == "random-va":
            table = lt_common.TemplateTable()
            ltgen_va = lt_common.init_ltgen(self.conf, table, method = "va")
            d = ltgen_va.process_init_data([(lm.l_w, lm.lt.lts)
                                            for lm in l_train_all])

            for i in range(self.trials):
                d_group = defaultdict(list)
                for mid, lm in enumerate(l_train_all):
                    tid = d[mid]
                    d_group[tid].append(lm)
                for group in d_group.values():
                    random.shuffle(group)

                l_sampled = []
                while len(l_sampled) < self.train_size:
                    temp = self.train_size - len(l_sampled)
                    if temp >= len(d_group):
                        for group in d_group.values():
                            assert len(group) > 0
                            l_sampled.append(group.pop())
                    else:
                        for group in sorted(d_group.values(),
                                            key = lambda x: len(x),
                                            reverse = True)[:temp]:
                            assert len(group) > 0
                            l_sampled.append(group.pop())
                    for key in [key for key, val
                                in d_group.items() if len(val) == 0]:
                        d_group.pop(key)

                if not len(l_sampled) == self.train_size:
                    _logger.warning(
                        ("Train size is not equal to specified number, "
                         "it seems there is some bug"))
                    l_sampled = l_sampled[:self.train_size]
                l_ltid = [lm.lt.ltid for lm in l_sampled]
                train_ltidmap = self._make_ltidmap(l_ltid)
                l_train = [items.line2train(lm) for lm in l_sampled]
                d_result = self._trial(l_train, l_test,
                                       train_ltidmap, test_ltidmap)
                self.results.append(d_result)
        else:
            raise NotImplementedError

    def _eval_file(self):
        l_train, train_ltidmap = self._load_train_file()
        l_test, test_ltidmap = self._load_test_diff()
        d_result = self._trial(l_train, l_test,
                               train_ltidmap, test_ltidmap)
        self.results.append(d_result)

    def _load_test_diff(self):
        l_test = []
        test_ltidmap = defaultdict(int)
        for line in self.ld.iter_lines(**self.sample_rules):
            l_test.append(items.line2train(line))
            test_ltidmap[line.lt.ltid] += 1
        return l_test, test_ltidmap

    def _load_train_file(self):
        l_test = []
        for lineitems in items.iter_items_from_file(self.fn):
            l_test.append(lineitems)
        return l_test, defaultdict(int)

    def _trial(self, l_train, l_test, d_ltid_train, d_ltid_test):

        def form_template(ltgen, l_w, l_label):
            tpl = []
            for w, label in zip(l_w, l_label):
                if label == ltgen.LABEL_DESC:
                    tpl.append(w)
                elif label == ltgen.LABEL_VAR:
                    tpl.append(ltgen._sym)
                elif label == ltgen.LABEL_DUMMY:
                    raise ValueError("Some word labeled as DUMMY in LTGenCRF")
                else:
                    raise ValueError
            return tpl

        table = self.ld.ltm._table
        ltgen = lt_common.init_ltgen(self.conf, table, "crf")

        ltgen.init_trainer()
        ltgen.train(l_train)

        wa_numer = 0.0
        wa_denom = 0.0
        la_numer = 0.0
        la_denom = 0.0
        ta_numer = 0.0
        ta_denom = 0.0

        for lineitems in l_test:
            l_correct = items.items2label(lineitems)
            l_w = [item[0] for item in lineitems]
            l_label_correct = [item[-1] for item in lineitems]
            tpl = form_template(ltgen, l_w, l_label_correct)
            l_label = ltgen.label_line(lineitems)

            for w_correct, w_label in zip(l_correct, l_label):
                wa_denom += 1
                if w_correct == w_label:
                    wa_numer += 1
            assert ltgen._table.exists(tpl)
            ltid = ltgen._table.get_tid(tpl)
            cnt = d_ltid_test[ltid]
            la_denom += 1
            ta_denom += 1.0 / cnt
            if l_correct == l_label:
                la_numer += 1
                ta_numer += 1.0 / cnt

        d_result = {"word_acc": wa_numer / wa_denom,
                    "line_acc": la_numer / la_denom,
                    "tpl_acc": ta_numer / ta_denom,
                    "train_size": len(l_train),
                    "test_size": len(l_test),
                    "train_tpl_size": len(d_ltid_train),
                    "test_tpl_size": len(d_ltid_test)}
        return d_result

    def info(self):
        buf = []
        buf.append(("# Experiment for measuring "
                    "log template generation accuracy"))
        if self.sample_from == "cross":
            buf.append("# type: Cross-validation (k = {0})".format(
                self.cross_k))
            buf.append("# data-source: db({0})".format(self.sample_rules))
        elif self.sample_from in ("diff", "file"):
            buf.append("# type: Experiment with different data range / domain")
            if self.sample_from == "diff":
                buf.append("# train-source: db({0})".format(
                    self.sample_train_rules))
            elif self.sample_from == "file":
                buf.append("# train-source: file({0})".format(self.fn))
            buf.append("# test-source: db({0})".format(self.sample_rules))
        buf.append("# trials: {0}".format(self.trials))
        return "\n".join(buf)

    def result(self):
        import numpy as np
        buf = []
        for rid, result in enumerate(self.results):
            buf.append("Experiment {0}".format(rid))
            for key, val in result.items():
                buf.append("{0} {1}".format(key, val))
            buf.append("")

        buf.append("# General result")
        arr_wa = np.array([d["word_acc"] for d in self.results])
        wa = np.average(arr_wa)
        wa_err = np.std(arr_wa) / np.sqrt(arr_wa.size)
        buf.append("Average word accuracy: {0} (err: {1})".format(wa, wa_err))

        arr_la = np.array([d["line_acc"] for d in self.results])
        la = np.average(arr_la)
        la_err = np.std(arr_la) / np.sqrt(arr_la.size)
        buf.append("Average line accuracy: {0} (err: {1})".format(la, la_err))

        arr_ta = np.array([d["tpl_acc"] for d in self.results])
        ta = np.average(arr_ta)
        ta_err = np.std(arr_ta) / np.sqrt(arr_ta.size)
        buf.append("Average template accuracy: {0} (err: {1})".format(
            ta, ta_err))

        return "\n".join(buf)


def init_ltgen_crf(conf, table, sym):
    model = conf.get("log_template_crf", "model_filename")
    verbose = conf.getboolean("log_template_crf", "verbose")
    template = conf.get("log_template_crf", "feature_template")
    middle = conf.get("log_template_crf", "middle_label_rule")
    ig_val = conf.get("log_template_crf", "unknown_key_weight")
    if len(middle) > 0:
        lwobj = LabelWord(conf, middle)
    else:
        lwobj = None
    return LTGenCRF(table, sym, model, verbose, template, ig_val, lwobj)


def make_crf_train(conf, iterobj):
    buf = []
    table = lt_common.TemplateTable()
    ltgen = lt_common.init_ltgen(conf, table, method = "crf")
    for lm in iterobj:
        item = items.line2train(lm, midlabel_func = ltgen._middle_label)
        buf.append(items.items2str(item))
    return "\n\n".join(buf)

