#!/usr/bin/env python
# coding: utf-8

import os
import logging
import pycrfsuite

from amulog import lt_common
from amulog.alg.crf import _items, _convert

_logger = logging.getLogger(__package__)
DEFAULT_TEMPLATE = "/".join((os.path.dirname(os.path.abspath(__file__)),
                             "../../data/crf_template.default"))


class LTGenCRF(lt_common.LTGenStateless):
    LABEL_DESC = "D"
    LABEL_VAR = "V"
    LABEL_DUMMY = "N"

    POS_UNKNOWN = "unknown"

    def __init__(self, table, model, verbose, feature_template, unknown_weight,
                 vreobj, normalizer_conf=None, normalizer_rule="DEFAULT"):
        super().__init__(table)
        self._model = model
        self._verbose = verbose
        self._feature_template = feature_template

        self._trainer = None
        self._tagger = None
        self._converter = _convert.FeatureExtracter(self._feature_template,
                                                    [self.POS_UNKNOWN],
                                                    unknown_weight)
        self._vre = vreobj
        self._norm = normalizer_conf is not None  # use normalization or not
        if self._norm:
            self._norm_rule = normalizer_rule
            self._init_normalizer(normalizer_conf)

    def middle_label(self, w):
        if self._vre is None:
            return w
        else:
            return self._vre.label(w)

    def _init_normalizer(self, normalize_conf):
        from log_normalizer import LogNormalizer
        self._ln = LogNormalizer(normalize_conf)

    def _normalize_line(self, l_w, rule_name="DEFAULT"):
        words = []
        length_vec = []
        for w in l_w:
            try:
                if w == "" or w == lt_common.REPLACER:
                    replaced = w
                else:
                    replaced = self._ln.replace_word(w, rule_name)
            except IndexError:
                import pdb; pdb.set_trace()
            else:
                words.extend(replaced)
                length_vec.append(len(replaced))
        return words, length_vec

    @classmethod
    def _tpl_labels(cls, org_words, tpl):
        labels = []
        for org_w, tpl_w in zip(org_words, tpl):
            if org_w == tpl_w:
                labels.append(cls.LABEL_DESC)
            elif tpl_w == lt_common.REPLACER:
                labels.append(cls.LABEL_VAR)
            else:
                raise ValueError("template structure invalud")
        return labels

    @classmethod
    def _normalized_labels(cls, org_words, tpl, length_vec):
        labels = []
        for org_w, tpl_w, length in zip(org_words, tpl, length_vec):
            if org_w == tpl_w:
                labels.extend([cls.LABEL_DESC] * length)
            elif tpl_w == lt_common.REPLACER:
                labels.extend([cls.LABEL_VAR] * length)
            else:
                raise ValueError("template structure invalud")
        return labels

    @classmethod
    def _restore_tpl(cls, org_words, labels):
        tpl = []
        for w, label in zip(org_words, labels):
            if label == cls.LABEL_DESC:
                tpl.append(w)
            elif label == cls.LABEL_VAR:
                tpl.append(lt_common.REPLACER)
            elif label == cls.LABEL_DUMMY:
                raise ValueError("Some word labeled as DUMMY in LTGenCRF")
            else:
                raise ValueError("Unknown labels in LTGenCRF")
        return tpl

    @classmethod
    def _restore_normalized_tpl(cls, org_words, labels, length_vec):
        assert len(labels) == sum(length_vec)
        import numpy as np
        org_tpl = []
        # slicer: iterate slice range of labels
        slicer = zip(np.cumsum([0] + length_vec[:-1]), np.cumsum(length_vec))
        for org_wid, (start, stop) in enumerate(slicer):
            tmp_labels = labels[start, stop]
            if cls.LABEL_VAR in tmp_labels:
                org_tpl.append(lt_common.REPLACER)
            elif cls.LABEL_DUMMY in tmp_labels:
                raise ValueError("Some word labeled as DUMMY in LTGenCRF")
            else:
                org_tpl.append(org_words[org_wid])
        return org_tpl

    def trainitems(self, lm):
        if self._norm:
            words, length_vec = self._normalize_line(lm.l_w, self._norm_rule)
            labels = self._normalized_labels(lm.l_w, lm.lt.ltw, length_vec)
        else:
            words = lm.l_w
            labels = self._tpl_labels(lm.l_w, lm.lt.ltw)

        assert len(words) == len(labels)
        mlabels = [self.middle_label(w) for w in words]
        return list(zip(words, mlabels, labels))

    def init_trainer(self, alg="lbfgs", verbose=False):
        self._trainer = pycrfsuite.Trainer(verbose=verbose)
        self._trainer.select(alg, "crf1d")
        d = {}  # for parameter tuning, edit this
        if len(d) > 0:
            self._trainer.set_params(d)

    def _train_model(self, iterable_items, model):
        if model is None:
            model = self._model
        for lineitems in iterable_items:
            xseq = self._converter.feature(lineitems)
            yseq = self._converter.label(lineitems)
            self._trainer.append(xseq, yseq)
        self._trainer.train(model)

    def train(self, iterable_lm, output=None):
        def _iter_items():
            for lm in iterable_lm:
                yield self.trainitems(lm)

        if output is None:
            output = self._model
        self._train_model(_iter_items(), output)
        return output

    def train_from_file(self, fp, output=None):
        self._train_model(_items.load_trainitems(fp), output)
        return output

    def init_tagger(self):
        if not os.path.exists(self._model):
            raise IOError("No trained CRF model for LTGenCRF")
        self._tagger = pycrfsuite.Tagger()
        self._tagger.open(self._model)

    def close_tagger(self):
        if self._tagger is not None:
            self._tagger.close()

    def _tag_line(self, l_w):
        lineitems = _items.line2items(l_w, midlabel_func=self.middle_label,
                                      dummy_label=self.LABEL_DUMMY)
        if self._tagger is None:
            self.init_tagger()
        fset = self._converter.feature(lineitems)
        l_label = self._tagger.tag(fset)
        return l_label

    def generate_tpl(self, pline):
        org_words = pline["words"]
        if self._norm:
            l_w, length_vec = self._normalize_line(org_words, self._norm_rule)
            l_label = self._tag_line(l_w)
            tpl = self._restore_normalized_tpl(org_words, l_label, length_vec)
        else:
            l_label = self._tag_line(org_words)
            tpl = self._restore_tpl(org_words, l_label)
        return tpl


# class MeasureAccuracy():
#
#    def __init__(self, conf, s_ltid=None):
#        """
#        Args:
#            conf
#            s_ltid (set): if not None, use limited messages
#                          of given ltid as train data"""
#        from . import log_db
#        self.ld = log_db.LogData(conf)
#        self.ld.init_ltmanager()
#        self.conf = conf
#        self.s_ltid = s_ltid
#        self.measure_lt_method = conf.get("measure_lt", "lt_method")
#        self.sample_from = conf.get("measure_lt", "sample_from")
#        self.sample_rules = self._rules(
#            conf.get("measure_lt", "sample_rules"))
#        self.trials = conf.getint("measure_lt", "train_trials")
#        self.results = []
#
#        if self.sample_from == "cross":
#            self.cross_k = conf.getint("measure_lt", "cross_k")
#            assert self.trials <= self.cross_k, "trials is larger than k"
#            self._eval_cross()
#        elif self.sample_from == "diff":
#            self.sample_train_rules = self._rules(
#                conf.get("measure_lt", "sample_train_rules"))
#            self.train_sample_method = conf.get("measure_lt",
#                                                "train_sample_method")
#            self.train_size = conf.getint("measure_lt", "train_size")
#            self._eval_diff()
#        elif self.sample_from == "file":
#            self.fn = conf.get("measure_lt", "sample_from_file")
#            self.train_sample_method = conf.get("measure_lt",
#                                                "train_sample_method")
#            self.train_size = conf.getint("measure_lt", "train_size")
#            self._eval_file()
#        else:
#            raise NotImplementedError
#
#    @staticmethod
#    def _rules(rulestr):
#        ret = {}
#        l_rule = [rule.strip() for rule in rulestr.split(",")]
#        for rule in l_rule:
#            if rule == "":
#                continue
#            key, val = [v.strip() for v in rule.split("=")]
#            if key in ("host", "area"):
#                ret[key] = val
#            elif key == "top_date":
#                assert not "top_dt" in ret
#                ret["top_dt"] = datetime.datetime.strptime(val, "%Y-%m-%d")
#            elif key == "top_dt":
#                assert not "top_dt" in ret
#                ret["top_dt"] = datetime.datetime.strptime(val,
#                                                           "%Y-%m-%d %H:%M:%S")
#            elif key == "end_date":
#                assert not "end_dt" in ret
#                ret["end_dt"] = datetime.datetime.strptime(
#                    val, "%Y-%m-%d") + datetime.timedelta(days=1)
#            elif key == "end_dt":
#                assert not "end_dt" in ret
#                ret["end_dt"] = datetime.datetime.strptime(val,
#                                                           "%Y-%m-%d %H:%M:%S")
#            else:
#                raise NotImplementedError
#        return ret
#
#    @staticmethod
#    def _make_ltidmap(l_ltid):
#        ltidmap = defaultdict(int)
#        for ltid in l_ltid:
#            ltidmap[ltid] += 1
#        return ltidmap
#
#    def _eval_cross(self):
#
#        def divide_size(size, groups):
#            base = int(size // groups)
#            ret = [base] * groups
#            surplus = size - (base * groups)
#            for i in range(surplus):
#                ret[i] += 1
#            assert sum(ret) == size
#            return ret
#
#        def agg_dict(l_d):
#            keyset = set()
#            for d in l_d:
#                keyset.update(d.keys())
#
#            new_d = defaultdict(int)
#            for ltid in keyset:
#                for d in l_d:
#                    new_d[ltid] += d[ltid]
#            return new_d
#
#        l_labeled = []  # (ltid, train)
#        for line in self.ld.iter_lines(**self.sample_rules):
#            l_labeled.append((line.lt.ltid, items.lm2trainitems(line)))
#        random.shuffle(l_labeled)
#
#        l_group = []
#        l_group_ltidlist = []
#        basenum = 0
#        for size in divide_size(len(l_labeled), self.cross_k):
#            l_ltid, l_lm = zip(*l_labeled[basenum:basenum + size])
#            l_group.append(l_lm)
#            l_group_ltidlist.append(l_ltid)
#            basenum += size
#        assert sum([len(g) for g in l_group]) == len(l_labeled)
#        del l_labeled
#
#        for trial in range(self.trials):
#            l_train = None
#            l_test = []
#            l_test_ltidlist = []
#            for gid, group, ltidlist in zip(range(self.cross_k),
#                                            l_group, l_group_ltidlist):
#                if gid == trial:
#                    assert l_train is None
#                    l_train = group
#                    train_ltidlist = ltidlist
#                else:
#                    l_test += group
#                    test_ltidlist += ltidlist
#            # test_ltidmap = agg_dict(l_test_ltidlist)
#
#            l_train, train_ltidlist = self._filter_train(l_train,
#                                                         train_ltidlist)
#            d_result = self._trial(l_train, l_test,
#                                   train_ltidlist, test_ltidlist)
#            self.results.append(d_result)
#
#    def _eval_diff(self):
#
#        l_test, test_ltidlist = self._load_test_diff()
#
#        l_train_all = []  # (ltid, lineitems)
#        for line in self.ld.iter_lines(**self.sample_train_rules):
#            l_train_all.append(line)
#
#        if self.train_sample_method == "all":
#            if self.trials > 1:
#                raise UserWarning(("measure_lt.trials is ignored "
#                                   "because the results must be static"))
#            train_ltidlist = [lm.lt.ltid for lm in l_train_all]
#            l_train = [items.lm2trainitems(lm) for lm in l_train_all]
#            l_train, train_ltidlist = self._filter_train(l_train,
#                                                         train_ltidlist)
#            d_result = self._trial(l_train, l_test,
#                                   train_ltidlist, test_ltidlist)
#            self.results.append(d_result)
#        elif self.train_sample_method == "random":
#            for i in range(self.trials):
#                l_train, train_ltidlist = train_sample_random(l_train_all,
#                                                              self.train_size)
#                l_train, train_ltidlist = self._filter_train(l_train,
#                                                             train_ltidlist)
#                d_result = self._trial(l_train, l_test,
#                                       train_ltidlist, test_ltidlist)
#                self.results.append(d_result)
#        elif self.train_sample_method == "random-va":
#            ltgen_va, ret_va = va_preprocess(self.conf, l_train_all)
#            for i in range(self.trials):
#                l_train, train_ltidlist = train_sample_random_va(
#                    l_train_all, self.train_size, ltgen_va, ret_va)
#                l_train, train_ltidlist = self._filter_train(l_train,
#                                                             train_ltidlist)
#                d_result = self._trial(l_train, l_test,
#                                       train_ltidlist, test_ltidlist)
#                self.results.append(d_result)
#        else:
#            raise NotImplementedError
#
#    def _eval_file(self):
#        l_train, train_ltidlist = self._load_train_file()
#        l_test, test_ltidlist = self._load_test_diff()
#        l_train, train_ltidlist = self._filter_train(l_train,
#                                                     train_ltidlist)
#        d_result = self._trial(l_train, l_test,
#                               train_ltidlist, test_ltidlist)
#        self.results.append(d_result)
#
#    def _load_test_diff(self):
#        l_test = []
#        test_ltidlist = []
#        test_ltidmap = defaultdict(int)
#        for line in self.ld.iter_lines(**self.sample_rules):
#            l_test.append(items.lm2trainitems(line))
#            test_ltidlist.append(line.lt.ltid)
#        return l_test, test_ltidlist
#
#    def _load_train_file(self):
#        l_test = []
#        for lineitems in items.iter_items_from_file(self.fn):
#            l_test.append(lineitems)
#        return l_test, []
#
#    def _filter_train(self, l_train, train_ltidlist):
#        if self.s_ltid is None:
#            return l_train, train_ltidlist
#        ret_l_train = []
#        ret_train_ltidlist = []
#        for trainobj, ltid in zip(l_train, train_ltidlist):
#            if ltid in self.s_ltid:
#                ret_l_train.append(trainobj)
#                ret_train_ltidlist.append(ltid)
#        assert len(ret_l_train) > 0
#        _logger.info("train data decreased with ltid-based filter "
#                     "({0} -> {1})".format(len(l_train), len(ret_l_train)))
#        return ret_l_train, ret_train_ltidlist
#
#    def _trial(self, l_train, l_test, l_ltid_train, l_ltid_test):
#
#        def form_template(ltgen, l_w, l_label):
#            tpl = []
#            for w, label in zip(l_w, l_label):
#                if label == ltgen.LABEL_DESC:
#                    tpl.append(w)
#                elif label == ltgen.LABEL_VAR:
#                    tpl.append(ltgen._sym)
#                elif label == ltgen.LABEL_DUMMY:
#                    raise ValueError("Some word labeled as DUMMY in LTGenCRF")
#                else:
#                    raise ValueError
#            return tpl
#
#        table = self.ld.ltm._table
#        ltgen = lt_common.init_ltgen_methods(self.conf, table, "crf")
#        d_ltid_test = self._make_ltidmap(l_ltid_test)
#
#        ltgen.init_trainer()
#        ltgen.train(l_train)
#
#        wa_numer = 0.0
#        wa_denom = 0.0
#        la_numer = 0.0
#        la_denom = 0.0
#        ta_numer = 0.0
#        ta_denom = 0.0
#        d_failure = defaultdict(int)
#
#        for lineitems, ltid in zip(l_test, l_ltid_test):
#            l_correct = items.items2label(lineitems)
#            l_w = [item[0] for item in lineitems]
#            l_label_correct = [item[-1] for item in lineitems]
#            l_label = ltgen.label_line(lineitems)
#
#            for wid, (w_correct, w_label) in enumerate(zip(l_correct,
#                                                           l_label)):
#                wa_denom += 1
#                if w_correct == w_label:
#                    wa_numer += 1
#                else:
#                    d_failure[(ltid, wid)] += 1
#            if ltid is None:
#                tpl = form_template(ltgen, l_w, l_label_correct)
#                assert ltgen._table.exists(tpl)
#                ltid = ltgen._table.get_tid(tpl)
#            cnt = d_ltid_test[ltid]
#            la_denom += 1
#            ta_denom += 1.0 / cnt
#            if l_correct == l_label:
#                la_numer += 1
#                ta_numer += 1.0 / cnt
#
#        d_result = {"word_acc": wa_numer / wa_denom,
#                    "line_acc": la_numer / la_denom,
#                    "tpl_acc": ta_numer / ta_denom,
#                    "train_size": len(l_train),
#                    "test_size": len(l_test),
#                    "train_tpl_size": len(set(l_ltid_train)),
#                    "test_tpl_size": len(set(l_ltid_test)),
#                    "dict_ltid": d_ltid_test,
#                    "failure": d_failure}
#        return d_result
#
#    def info(self):
#        buf = []
#        buf.append(("# Experiment for measuring "
#                    "log template generation accuracy"))
#        if self.sample_from == "cross":
#            buf.append("# type: Cross-validation (k = {0})".format(
#                self.cross_k))
#            buf.append("# data-source: db({0})".format(self.sample_rules))
#        elif self.sample_from in ("diff", "file"):
#            buf.append("# type: Experiment with different data range / domain")
#            if self.sample_from == "diff":
#                buf.append("# train-source: db({0})".format(
#                    self.sample_train_rules))
#            elif self.sample_from == "file":
#                buf.append("# train-source: file({0})".format(self.fn))
#            buf.append("# test-source: db({0})".format(self.sample_rules))
#        buf.append("# trials: {0}".format(self.trials))
#        return "\n".join(buf)
#
#    def result(self):
#        import numpy as np
#        buf = []
#        for rid, result in enumerate(self.results):
#            buf.append("Experiment {0}".format(rid))
#            for key, val in result.items():
#                if not type(val) in (list, tuple, dict, defaultdict):
#                    buf.append("{0} {1}".format(key, val))
#            buf.append("")
#
#        buf.append("# General result")
#        arr_wa = np.array([d["word_acc"] for d in self.results])
#        wa = np.average(arr_wa)
#        wa_err = np.std(arr_wa) / np.sqrt(arr_wa.size)
#        buf.append("Average word accuracy: {0} (err: {1})".format(wa, wa_err))
#
#        arr_la = np.array([d["line_acc"] for d in self.results])
#        la = np.average(arr_la)
#        la_err = np.std(arr_la) / np.sqrt(arr_la.size)
#        buf.append("Average line accuracy: {0} (err: {1})".format(la, la_err))
#
#        arr_ta = np.array([d["tpl_acc"] for d in self.results])
#        ta = np.average(arr_ta)
#        ta_err = np.std(arr_ta) / np.sqrt(arr_ta.size)
#        buf.append("Average template accuracy: {0} (err: {1})".format(
#            ta, ta_err))
#
#        return "\n".join(buf)
#
#    def failure_report(self, ld=None):
#
#        def _failure_place(ld, ltid, wid):
#            if ld is None:
#                return None
#            else:
#                tpl = []
#                for temp_wid, w in enumerate(ld.lt(ltid).ltw):
#                    if temp_wid == wid:
#                        tpl.append("<{0}>".format(w))
#                    else:
#                        tpl.append(w)
#                return " ".join(tpl)
#
#        buf = []
#        buf2 = []
#
#        whole_keys = set()
#        for result in self.results:
#            whole_keys = whole_keys | set(result["failure"].keys())
#
#        d_average = defaultdict(float)
#        for result in self.results:
#            d_fail = result["failure"]
#            for key in whole_keys:
#                if key in d_fail:
#                    d_average[key] += 1.0 * d_fail[key] / len(self.results)
#
#        d_ltid = self.results[0]["dict_ltid"]
#        for key, cnt in sorted(d_average.items(), key=lambda x: x[1],
#                               reverse=True):
#            ltid, wid = key
#            ratio = 1.0 * cnt / d_ltid[ltid]
#            buf.append([ltid, wid, ":", int(cnt), "({0})".format(ratio)])
#            tpl_info = _failure_place(ld, ltid, wid)
#            if tpl_info is not None:
#                buf2.append(tpl_info)
#
#        if len(buf2) == 0:
#            return "\n".join(buf)
#        else:
#            from . import common
#            ret = []
#            for buf_line, buf2_line in zip(common.cli_table(buf).split("\n"),
#                                           buf2):
#                ret.append(buf_line)
#                ret.append(buf2_line)
#                ret.append("")
#            return "\n".join(ret)


def init_ltgen_crf(conf, table, **_):
    model = conf.get("log_template_crf", "model_filename")
    verbose = conf.getboolean("log_template_crf", "verbose")
    feature_template = conf.get("log_template_crf", "feature_template")
    if feature_template.strip() == "":
        feature_template = DEFAULT_TEMPLATE
    unknown_weight = conf.getfloat("log_template_crf", "unknown_key_weight")
    normalizer_conf = conf.get("log_template_crf", "normalizer_conf")
    if normalizer_conf.strip() == "":
        normalizer_conf = None
    normalizer_rule = conf.get("log_template_crf", "normalizer_rule")

    from amulog import lt_regex
    middle_fn = conf.get("log_template_crf", "middle_label_rule")
    if middle_fn.strip() == "":
        vreobj = None
    else:
        vreobj = lt_regex.VariableRegex(conf, middle_fn, LTGenCRF.POS_UNKNOWN)

    return LTGenCRF(table, model, verbose, feature_template, unknown_weight,
                    vreobj, normalizer_conf, normalizer_rule)
