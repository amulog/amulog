#!/usr/bin/env python
# coding: utf-8

import os
import logging
import random
import datetime
from collections import defaultdict

import pycrfsuite

from . import lt_common
from . import host_alias
from .crf import items
from .crf import convert
from .crf import midlabel

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
        _logger.info("crf: {0} lines learned".format(len(iterable_items)))
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

    def estimate_tpl(self, l_w, l_s):
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
        return tpl

    def process_line(self, l_w, l_s):
        tpl = self.estimate_tpl(l_w, l_s)
        if self._table.exists(tpl):
            tid = self._table.get_tid(tpl)
            return tid, self.state_unchanged
        else:
            tid = self._table.add(tpl)
            return tid, self.state_added


class MeasureAccuracy():

    def __init__(self, conf, s_ltid = None):
        """
        Args:
            conf
            s_ltid (set): if not None, use limited messages
                          of given ltid as train data"""
        from . import log_db
        self.ld = log_db.LogData(conf)
        self.ld.init_ltmanager()
        self.conf = conf
        self.s_ltid = s_ltid
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
        l_group_ltidlist = []
        basenum = 0
        for size in divide_size(len(l_labeled), self.cross_k):
            l_ltid, l_lm = zip(*l_labeled[basenum:basenum+size])
            l_group.append(l_lm)
            l_group_ltidlist.append(l_ltid)
            basenum += size
        assert sum([len(g) for g in l_group]) == len(l_labeled)
        del l_labeled

        for trial in range(self.trials):
            l_train = None
            l_test = []
            l_test_ltidlist = []
            for gid, group, ltidlist in zip(range(self.cross_k),
                                           l_group, l_group_ltidlist):
                if gid == trial:
                    assert l_train is None
                    l_train = group
                    train_ltidlist = ltidlist
                else:
                    l_test += group
                    test_ltidlist += ltidlist
            #test_ltidmap = agg_dict(l_test_ltidlist)

            l_train, train_ltidlist = self._filter_train(l_train,
                                                         train_ltidlist)
            d_result = self._trial(l_train, l_test,
                                   train_ltidlist, test_ltidlist)
            self.results.append(d_result)

    def _eval_diff(self):

        l_test, test_ltidlist = self._load_test_diff()

        l_train_all = [] # (ltid, lineitems)
        for line in self.ld.iter_lines(**self.sample_train_rules):
            l_train_all.append(line)

        if self.train_sample_method == "all":
            if self.trials > 1:
                raise UserWarning(("measure_lt.trials is ignored "
                                   "because the results must be static"))
            train_ltidlist = [lm.lt.ltid for lm in l_train_all]
            l_train = [items.line2train(lm) for lm in l_train_all]
            l_train, train_ltidlist = self._filter_train(l_train,
                                                         train_ltidlist)
            d_result = self._trial(l_train, l_test,
                                   train_ltidlist, test_ltidlist)
            self.results.append(d_result)
        elif self.train_sample_method == "random":
            for i in range(self.trials):
                l_train, train_ltidlist = train_sample_random(l_train_all,
                                                              self.train_size)
                l_train, train_ltidlist = self._filter_train(l_train,
                                                             train_ltidlist)
                d_result = self._trial(l_train, l_test,
                                       train_ltidlist, test_ltidlist)
                self.results.append(d_result)
        elif self.train_sample_method == "random-va":
            ltgen_va, ret_va = va_preprocess(self.conf, l_train_all)
            for i in range(self.trials):
                l_train, train_ltidlist = train_sample_random_va(
                    l_train_all, self.train_size, ltgen_va, ret_va)
                l_train, train_ltidlist = self._filter_train(l_train,
                                                             train_ltidlist)
                d_result = self._trial(l_train, l_test,
                                       train_ltidlist, test_ltidlist)
                self.results.append(d_result)
        else:
            raise NotImplementedError

    def _eval_file(self):
        l_train, train_ltidlist = self._load_train_file()
        l_test, test_ltidlist = self._load_test_diff()
        l_train, train_ltidlist = self._filter_train(l_train,
                                                     train_ltidlist)
        d_result = self._trial(l_train, l_test,
                               train_ltidlist, test_ltidlist)
        self.results.append(d_result)

    def _load_test_diff(self):
        l_test = []
        test_ltidlist = []
        test_ltidmap = defaultdict(int)
        for line in self.ld.iter_lines(**self.sample_rules):
            l_test.append(items.line2train(line))
            test_ltidlist.append(line.lt.ltid)
        return l_test, test_ltidlist

    def _load_train_file(self):
        l_test = []
        for lineitems in items.iter_items_from_file(self.fn):
            l_test.append(lineitems)
        return l_test, []

    def _filter_train(self, l_train, train_ltidlist):
        if self.s_ltid is None:
            return l_train, train_ltidlist
        ret_l_train = []
        ret_train_ltidlist = []
        for trainobj, ltid in zip(l_train, train_ltidlist):
            if ltid in self.s_ltid:
                ret_l_train.append(trainobj)
                ret_train_ltidlist.append(ltid)
        assert len(ret_l_train) > 0
        _logger.info("train data decreased with ltid-based filter "
                     "({0} -> {1})".format(len(l_train), len(ret_l_train)))
        return ret_l_train, ret_train_ltidlist

    def _trial(self, l_train, l_test, l_ltid_train, l_ltid_test):

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
        d_ltid_test = self._make_ltidmap(l_ltid_test)

        ltgen.init_trainer()
        ltgen.train(l_train)

        wa_numer = 0.0
        wa_denom = 0.0
        la_numer = 0.0
        la_denom = 0.0
        ta_numer = 0.0
        ta_denom = 0.0
        d_failure = defaultdict(int)

        for lineitems, ltid in zip(l_test, l_ltid_test):
            l_correct = items.items2label(lineitems)
            l_w = [item[0] for item in lineitems]
            l_label_correct = [item[-1] for item in lineitems]
            l_label = ltgen.label_line(lineitems)

            for wid, (w_correct, w_label) in enumerate(zip(l_correct,
                                                           l_label)):
                wa_denom += 1
                if w_correct == w_label:
                    wa_numer += 1
                else:
                    d_failure[(ltid, wid)] += 1
            if ltid is None:
                tpl = form_template(ltgen, l_w, l_label_correct)
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
                    "train_tpl_size": len(set(l_ltid_train)),
                    "test_tpl_size": len(set(l_ltid_test)),
                    "dict_ltid": d_ltid_test,
                    "failure": d_failure}
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
                if not type(val) in (list, tuple, dict, defaultdict):
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

    def failure_report(self, ld = None):

        def _failure_place(ld, ltid, wid):
            if ld is None:
                return None
            else:
                tpl = []
                for temp_wid, w in enumerate(ld.lt(ltid).ltw):
                    if temp_wid == wid:
                        tpl.append("<{0}>".format(w))
                    else:
                        tpl.append(w)
                return " ".join(tpl)

        buf = []
        buf2 = []

        whole_keys = set()
        for result in self.results:
            whole_keys = whole_keys | set(result["failure"].keys())

        d_average = defaultdict(float)
        for result in self.results:
            d_fail = result["failure"]
            for key in whole_keys:
                if key in d_fail:
                    d_average[key] += 1.0 * d_fail[key] / len(self.results)

        d_ltid = self.results[0]["dict_ltid"]
        for key, cnt in sorted(d_average.items(), key = lambda x: x[1],
                               reverse = True):
            ltid, wid = key
            ratio = 1.0 * cnt / d_ltid[ltid]
            buf.append([ltid, wid, ":", int(cnt), "({0})".format(ratio)])
            tpl_info = _failure_place(ld, ltid, wid)
            if tpl_info is not None:
                buf2.append(tpl_info)

        if len(buf2) == 0:
            return "\n".join(buf)
        else:
            from . import common
            ret = []
            for buf_line, buf2_line in zip(common.cli_table(buf).split("\n"),
                                           buf2):
                ret.append(buf_line)
                ret.append(buf2_line)
                ret.append("")
            return "\n".join(ret)


def train_sample_random(l_lm, size):
    l_sampled = random.sample(l_lm, size)
    ltidlist = [lm.lt.ltid for lm in l_sampled]
    l_train = [items.line2train(lm) for lm in l_sampled]
    return l_train, ltidlist


def va_preprocess(conf, l_lm):
    table = lt_common.TemplateTable()
    ltgen_va = lt_common.init_ltgen(conf, table, method = "va")
    ret_va = ltgen_va.process_init_data(
        [(lm.l_w, lm.lt.lts) for lm in l_lm])
    return ltgen_va, ret_va


def train_sample_random_va(l_lm, size, ltgen_va, ret_va):
    d_group = defaultdict(list)
    for mid, lm in enumerate(l_lm):
        tid = ret_va[mid]
        d_group[tid].append(lm)
    for group in d_group.values():
        random.shuffle(group)

    l_sampled = []
    while len(l_sampled) < size:
        temp = size - len(l_sampled)
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
        for key in [key for key, val in d_group.items() if len(val) == 0]:
            d_group.pop(key)
        if len(d_group) == 0:
            _logger.warning(
                "Less than {0} messages in given condition "
                "({1} messages from DB)".format(size, len(l_sampled)))
            break

    if not len(l_sampled) == size:
        _logger.warning("Train size is not equal to specified number,"
                        "maybe there is some bug")
        l_sampled = l_sampled[:size]
    ltidlist = [lm.lt.ltid for lm in l_sampled]
    l_train = [items.line2train(lm) for lm in l_sampled]
    return l_train, ltidlist


def init_ltgen_crf(conf, table, sym):
    model = conf.get("log_template_crf", "model_filename")
    verbose = conf.getboolean("log_template_crf", "verbose")
    template = conf.get("log_template_crf", "feature_template")
    middle = conf.get("log_template_crf", "middle_label_rule")
    ig_val = conf.getfloat("log_template_crf", "unknown_key_weight")
    if len(middle) > 0:
        lwobj = midlabel.LabelWord(conf, middle, LTGenCRF.POS_UNKNOWN)
    else:
        lwobj = None
    return LTGenCRF(table, sym, model, verbose, template, ig_val, lwobj)


def make_crf_train(conf, iterobj, size = 1000, method = "all",
                   return_ltidlist = False):
    l_train_all = iterobj
    if method == "all":
        l_train = [items.line2train(lm) for lm in l_train_all]
    elif method == "random":
        l_train, train_ltidlist = train_sample_random(l_train_all, size)
    elif method == "random-va":
        ltgen_va, ret_va = va_preprocess(conf, l_train_all)
        l_train, train_ltidlist = train_sample_random_va(
            l_train_all, size, ltgen_va, ret_va)
    else:
        raise NotImplementedError(
            "Invalid sampling method name {0}".format(method))

    if return_ltidlist:
        return l_train, train_ltidlist
    else:
        return l_train


def make_crf_model(conf, iterobj, size = 1000, method = "all"):
    l_train, train_ltidlist = make_crf_train(conf, iterobj, size, method,
                                             return_ltidlist = True)
    table = lt_common.TemplateTable()
    ltgen = lt_common.init_ltgen(conf, table, "crf")
    ltgen.init_trainer()
    ltgen.train(l_train)
    assert os.path.exists(ltgen.model)
    return ltgen.model


def make_crf_model_ideal(conf, ld, size = None):
    l_train = []
    train_ltidlist = []
    for lt in ld.iter_lt():
        iterobj = ld.iter_lines(ltid = lt.ltid)
        lm = iterobj.__next__()
        l_train.append(items.line2train(lm))
        train_ltidlist.append(lt.ltid)

    if size is not None:
        assert isinstance(size, int)
        temp = zip(l_train, train_ltidlist)
        random.shuffle(temp)
        l_train, train_ltidlist = zip(*temp[:size])

    table = lt_common.TemplateTable()
    ltgen = lt_common.init_ltgen(conf, table, "crf")
    ltgen.init_trainer()
    ltgen.train(l_train)
    assert os.path.exists(ltgen.model)
    return ltgen.model


def generate_lt_mprocess(conf, targets, check_import = False, pal = 1):
    """Generate log templates for all given log messages.
    This function does not generate DB,
    but instead of that this function can be processed in multiprocessing.
    This function is available only in CRF log template estimation
    (because other methods uses estimation results of other messages).
    """

    import multiprocessing
    from . import common
    timer = common.Timer("generate_lt task", output = _logger)
    timer.start()
    queue = multiprocessing.Queue()
    if check_import:
        target_func = generate_lt_import_file
    else:
        target_func = generate_lt_file
    l_args = generate_lt_args(conf, targets, queue)
    l_process = [multiprocessing.Process(name = args[2],
        target = target_func, args = args) for args in l_args]
    common.mprocess_queueing(l_process, pal)
    
    s_tpl = set()
    while not queue.empty():
        s_temp = queue.get_nowait()
        s_tpl = s_tpl | s_temp
    timer.stop()
    return s_tpl


def generate_lt_args(conf, targets, queue):
    return [(queue, conf, fp) for fp in targets]


def generate_lt_file(queue, conf, fp):
    _logger.info("processing {0} start".format(fp))
    from . import logparser
    lp = logparser.LogParser(conf)
    table = lt_common.TemplateTable()
    ltgen = lt_common.init_ltgen(conf, table, "crf")
    s_tpl = set()

    with open(fp, 'r') as f:
        for line in f:
            line = line.rstrip()
            dt, org_host, l_w, l_s = lp.process_line(line)
            tpl = ltgen.estimate_tpl(l_w, l_s)
            s_tpl.add(tuple(tpl))

    _logger.info("processing {0} done".format(fp))
    queue.put(s_tpl)


def generate_lt_import():
    pass




