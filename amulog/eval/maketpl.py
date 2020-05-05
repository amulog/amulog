#!/usr/bin/env python
# coding: utf-8

import logging
import json
import numpy as np
from collections import defaultdict

from amulog import common
from amulog import lt_common
from amulog.eval import cluster_metrics

_logger = logging.getLogger(__package__)


class MeasureLTGen:
    """Record and load measurement data of tempalte generation.
    This is implemented in memory-saving architecture;
    The calculated data for each line is appended into text file."""

    SPLITTER = "@@"
    LABEL_NONE = "N"
    LABEL_DESCRIPTION = "D"
    LABEL_VARIABLE = "V"

    def __init__(self, conf, n_trial):
        self.conf = conf
        self._number_of_trials = n_trial
        common.mkdir(self._output_dir_answer(conf))
        common.mkdir(self._output_dir_trial(conf))

        self._d_answer = {"l_tid": list(),
                          "n_lines": int(),
                          "d_n_lines": defaultdict(int),
                          "n_words": int(),
                          "d_n_words": defaultdict(int),
                          # "n_variables": int(),
                          # "d_n_variables": defaultdict(int),
                          # "n_descriptions": int(),
                          # "d_n_descriptions": defaultdict(int),
                          }

        if n_trial is None:
            return
        assert n_trial < 100
        self._d_trial = []
        for tid in range(n_trial):
            self._d_trial.append({"l_tid": list(),
                                  "n_c_lines": int(),
                                  "d_n_c_lines": defaultdict(int),
                                  "n_c_words": int(),
                                  "d_n_c_words": defaultdict(int),
                                  # "n_c_variables": int(),
                                  # "d_n_c_variables": defaultdict(int),
                                  # "n_c_descriptions": int(),
                                  # "d_n_c_descriptions": defaultdict(int),
                                  })

    # file IO methods
    @staticmethod
    def _output_dir_answer(conf):
        return conf["eval"]["ltgen_answer_dir"]

    @staticmethod
    def _output_dir_trial(conf):
        return conf["eval"]["ltgen_trial_dir"]

    def _org_word_path(self):
        return "{0}/word".format(self._output_dir_answer(self.conf))

    def _answer_label_path(self):
        return "{0}/label_answer".format(self._output_dir_answer(self.conf))

    def _trial_label_path(self, trial_id):
        return "{0}/label_trial{1}".format(self._output_dir_trial(self.conf),
                                           str(trial_id).zfill(2))

    def _answer_info_path(self):
        return "{0}/info_answer".format(self._output_dir_answer(self.conf))

    def _trial_info_path(self):
        return "{0}/info_trial".format(self._output_dir_trial(self.conf))

    def init_answer(self):
        common.rm(self._org_word_path())
        common.rm(self._answer_label_path())

    def init_trial(self, trial_id):
        common.rm(self._trial_label_path(trial_id))

    def load_answer_info(self):
        with open(self._answer_info_path(), 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self._d_answer = obj

    def load_trial_info(self):
        with open(self._trial_info_path(), 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self._d_trial = obj

    def dump_answer_info(self):
        obj = self._d_answer
        with open(self._answer_info_path(), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def dump_trial_info(self):
        obj = self._d_trial
        with open(self._trial_info_path(), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def load(self):
        self.load_answer_info()
        self.load_trial_info()

    # data APIs
    @classmethod
    def _tpl2dump(cls, tpl, l_w):
        if tpl is None:
            return cls.LABEL_NONE + "\n"
        else:
            l_label = [cls.LABEL_VARIABLE if w_tpl == lt_common.REPLACER
                       else cls.LABEL_DESCRIPTION
                       for w_tpl, w_org in zip(tpl, l_w)]
            return "".join(l_label) + "\n"

    @classmethod
    def _labels_isnone(cls, labels):
        return cls.LABEL_NONE in labels

    @classmethod
    def restore_tpl(cls, labels, l_w):
        if labels is None or labels == cls.LABEL_NONE:
            return None
        else:
            return [lt_common.REPLACER if label == cls.LABEL_VARIABLE
                    else w
                    for label, w in zip(labels, l_w)]

    @classmethod
    def restore_result(cls, labels, l_w):
        if labels is None or labels == cls.LABEL_NONE:
            return None
        else:
            return [lt_common.REPLACER_HEAD + w + lt_common.REPLACER_TAIL
                    if label == cls.LABEL_VARIABLE else w
                    for label, w in zip(labels, l_w)]

    def add_org(self, l_w):
        added_line = self.SPLITTER.join(l_w) + "\n"
        with open(self._org_word_path(), 'a') as f:
            f.write(added_line)

    def add_answer(self, tid, tpl, l_w):
        self._update_stat_answer(tid, tpl)
        added_line = self._tpl2dump(tpl, l_w)
        with open(self._answer_label_path(), 'a') as f:
            f.write(added_line)

    def add_trial(self, trial_id, tid_trial, tpl_trial,
                  tid_answer, tpl_answer, l_w):
        self._update_stat_trial(trial_id, tid_trial, tpl_trial,
                                tid_answer, tpl_answer)
        added_line = self._tpl2dump(tpl_trial, l_w)
        with open(self._trial_label_path(trial_id), 'a') as f:
            f.write(added_line)

    def _update_stat_answer(self, tid, tpl):
        self._d_answer["l_tid"].append(tid)
        if tid is not None:
            self._d_answer["n_lines"] += 1
            self._d_answer["d_n_lines"][str(tid)] += 1
            n_words = len(tpl)
            self._d_answer["n_words"] += n_words
            self._d_answer["d_n_words"][str(tid)] += n_words

    def _update_stat_trial(self, trial_id, tid_trial, tpl_trial,
                           tid_answer, tpl_answer):
        self._d_trial[trial_id]["l_tid"].append(tid_trial)
        if tid_answer is not None:
            if tpl_trial == tpl_answer:
                self._d_trial[trial_id]["n_c_lines"] += 1
                self._d_trial[trial_id]["d_n_c_lines"][str(tid_answer)] += 1

            assert len(tpl_trial) == len(tpl_answer)
            for w_trial, w_answer in zip(tpl_trial, tpl_answer):
                if w_trial == w_answer:
                    self._d_trial[trial_id]["n_c_words"] += 1
                    self._d_trial[trial_id]["d_n_c_words"][str(tid_answer)] += 1

    def iter_org(self):
        with open(self._org_word_path(), 'r') as f:
            for line in f:
                yield line.strip().split(self.SPLITTER)

    def iter_label_answer(self, pass_none=False):
        with open(self._answer_label_path(), 'r') as f:
            for line in f:
                labels_str = line.strip()
                if self._labels_isnone(labels_str):
                    if pass_none:
                        pass
                    else:
                        yield None
                else:
                    yield labels_str

    def iter_label_trial(self, trial_id, pass_none=False):
        with open(self._trial_label_path(trial_id), 'r') as f:
            for line in f:
                labels_str = line.strip()
                if self._labels_isnone(labels_str):
                    if pass_none:
                        pass
                    else:
                        yield None
                else:
                    yield labels_str

    def iter_tpl_answer(self, pass_none=False, fill_wildcard=False):
        for l_w, labels in zip(self.iter_org(), self.iter_label_answer()):
            if labels is None:
                if pass_none:
                    pass
                else:
                    yield None
            elif fill_wildcard:
                yield self.restore_result(labels, l_w)
            else:
                yield self.restore_tpl(labels, l_w)

    def iter_tpl_trial(self, trial_id, pass_none=False, fill_wildcard=False):
        for l_w, labels in zip(self.iter_org(), self.iter_label_trial(trial_id)):
            if labels is None:
                if pass_none:
                    pass
                else:
                    yield None
            elif fill_wildcard:
                yield self.restore_result(labels, l_w)
            else:
                yield self.restore_tpl(labels, l_w)

    def iter_tid_answer(self):
        return self._d_answer["l_tid"].__iter__()

    def iter_tid_trial(self, trial_id):
        return self._d_trial[trial_id]["l_tid"].__iter__()

    def tid_list_answer(self):
        return [tid for tid in self._d_answer["l_tid"]
                if tid is not None]

    def tid_list_trial(self, trial_id):
        return [tid for tid in self._d_trial[trial_id]["l_tid"]
                if tid is not None]

    def number_of_trials(self):
        return self._number_of_trials

    def number_of_messages(self):
        return self._d_answer["n_lines"]

    def number_of_answer_clusters(self):
        return len(self._d_answer["d_n_lines"])

    def number_of_answer_cluster_members(self):
        d = {}
        for key, val in self._d_answer["d_n_lines"].items():
            tid = int(key)
            d[tid] = val
        return d

    def number_of_trial_clusters(self, trial_id=None):
        if trial_id is None:
            l_cnt = [self.number_of_trial_clusters(i)
                     for i in range(self.number_of_trials())]
            if len(l_cnt) == 0:
                return l_cnt[0]
            else:
                return np.average(l_cnt)
        else:
            return np.unique(self.tid_list_trial(trial_id)).shape[0]

    # accuracy methods
    def _word_accuracy_trial(self, trial_id):
        # n_words: Number of all words in dataset
        n_words = self._d_answer["n_words"]
        # n_c_words: Number of words correctly labeled in dataset
        n_c_words = self._d_trial[trial_id]["n_c_words"]

        return 1.0 * n_c_words / n_words

    def word_accuracy(self, trial_id=None):
        if trial_id is None:
            l_acc = [self._word_accuracy_trial(i)
                     for i in range(self.number_of_trials())]
            return np.average(l_acc)
        else:
            return self._word_accuracy_trial(trial_id)

    def _line_accuracy_trial(self, trial_id):
        # n_lines: Number of all lines in dataset
        n_lines = self._d_answer["n_lines"]
        # n_c_lines: Number of lines correctly labeled in dataset
        n_c_lines = self._d_trial[trial_id]["n_c_lines"]

        return 1.0 * n_c_lines / n_lines

    def line_accuracy(self, trial_id=None):
        if trial_id is None:
            l_acc = [self._line_accuracy_trial(i)
                     for i in range(self.number_of_trials())]
            return np.average(l_acc)
        else:
            return self._line_accuracy_trial(trial_id)

    def _tpl_word_accuracy_trial(self, trial_id):
        # d_n_words: Number of words in a template cluster
        d_n_words = self._d_answer["d_n_words"]
        # d_n_c_words: Number of words correctly labeled in a template cluster
        d_n_c_words = self._d_trial[trial_id]["d_n_c_words"]

        l_acc = []
        for key in d_n_words:  # key: str(tid)
            l_acc.append(d_n_c_words.get(key, 0) / d_n_words.get(key, 0))
        return np.average(l_acc)

    def tpl_word_accuracy(self, trial_id=None):
        if trial_id is None:
            l_acc = [self._tpl_word_accuracy_trial(i)
                     for i in range(self.number_of_trials())]
            return np.average(l_acc)
        else:
            return self._tpl_word_accuracy_trial(trial_id)

    def _tpl_accuracy_trial(self, trial_id):
        # d_n_lines: Number of lines in a template cluster
        d_n_lines = self._d_answer["d_n_lines"]
        # d_n_c_lines: Number of lines correctly labeled in a template cluster
        d_n_c_lines = self._d_trial[trial_id]["d_n_c_lines"]

        l_acc = []
        for key in d_n_lines:
            l_acc.append(d_n_c_lines.get(key, 0) / d_n_lines.get(key, 0))
        return np.average(l_acc)

    def tpl_accuracy(self, trial_id=None):
        if trial_id is None:
            l_acc = [self._tpl_accuracy_trial(i)
                     for i in range(self.number_of_trials())]
            return np.average(l_acc)
        else:
            return self._tpl_accuracy_trial(trial_id)

    def tpl_word_accuracy_dist(self, trial_id):
        # d_n_words: Number of words in a template cluster
        d_n_words = self._d_answer["d_n_words"]
        # d_n_c_words: Number of words correctly labeled in a template cluster
        d_n_c_words = self._d_trial[trial_id]["d_n_c_words"]

        ret = {}
        for key in d_n_words:  # key: str(tid)
            tid = int(key)
            ret[tid] = d_n_c_words.get(key, 0) / d_n_words.get(key, 0)
        return ret

    def tpl_line_accuracy_dist(self, trial_id):
        # d_n_lines: Number of lines in a template cluster
        d_n_lines = self._d_answer["d_n_lines"]
        # d_n_c_lines: Number of lines correctly labeled in a template cluster
        d_n_c_lines = self._d_trial[trial_id]["d_n_c_lines"]

        ret = {}
        for key in d_n_lines:  # key: str(tid)
            tid = int(key)
            ret[tid] = d_n_c_lines.get(key, 0) / d_n_lines.get(key, 0)
        return ret

    # def word_accuracy_precision_recall_fmeasure(self, trial_id,
    #                                             standard_label="variable"):
    #     n_variables = self._d_answer["n_variables"]
    #     n_descriptions = self._d_answer["n_descriptions"]
    #     n_c_variables = self._d_trial[trial_id]["n_c_variables"]
    #     n_c_descriptions = self._d_trial[trial_id]["n_c_descriptions"]
    #     if standard_label == "description":
    #         tp = n_c_descriptions
    #         fp = n_variables - n_c_variables
    #         fn = n_descriptions - n_c_descriptions
    #         tn = n_c_variables
    #     elif standard_label == "variable":
    #         tp = n_c_variables
    #         fp = n_descriptions - n_c_descriptions
    #         fn = n_variables - n_c_variables
    #         tn = n_c_descriptions
    #     else:
    #         raise ValueError("standard_label is one of [description, variable]")
    #
    #     accuracy = (tp + tn) / (tp + fp + fn + tn)
    #     precision = tp / (tp + fp)
    #     recall = tp / (tp + fn)
    #     f1_score = 2. * recall * precision / (recall + precision)
    #     return accuracy, precision, recall, f1_score

    def rand_score(self, trial_id=None):
        if trial_id is None:
            l_score = [self.rand_score(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            l_tid_answer = self.tid_list_answer()
            l_tid_trial = self.tid_list_trial(trial_id)
            return cluster_metrics.rand_score(l_tid_answer, l_tid_trial)

    def adjusted_rand_score(self, trial_id=None):
        if trial_id is None:
            l_score = [self.adjusted_rand_score(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            from sklearn.metrics import adjusted_rand_score
            l_tid_answer = self.tid_list_answer()
            l_tid_trial = self.tid_list_trial(trial_id)
            return adjusted_rand_score(l_tid_answer, l_tid_trial)

    def f1_score(self, trial_id=None):
        if trial_id is None:
            l_score = [self.f1_score(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            l_tid_answer = self.tid_list_answer()
            l_tid_trial = self.tid_list_trial(trial_id)
            return cluster_metrics.precision_recall_fscore(
                l_tid_answer, l_tid_trial)[2]

    def parsing_accuracy(self, trial_id=None):
        if trial_id is None:
            l_score = [self.parsing_accuracy(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            l_tid_answer = self.tid_list_answer()
            l_tid_trial = self.tid_list_trial(trial_id)
            return cluster_metrics.parsing_accuracy(l_tid_answer, l_tid_trial)

    def cluster_accuracy(self, trial_id=None):
        if trial_id is None:
            l_score = [self.cluster_accuracy(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            l_tid_answer = self.tid_list_answer()
            l_tid_trial = self.tid_list_trial(trial_id)
            return cluster_metrics.cluster_accuracy(l_tid_answer, l_tid_trial)


def _iter_plines(conf, targets):
    from amulog import log_db
    from amulog import host_alias
    lp = log_db.load_log2seq(conf)
    ha = host_alias.init_hostalias(conf)
    drop_undefhost = conf.getboolean("database", "undefined_host")

    for f in log_db.iter_files(targets):
        for msg in f:
            pline = log_db.parse_line(msg, lp)
            if pline is None:
                continue
            pline = log_db.normalize_host(msg, pline, ha, None, drop_undefhost)
            if pline is None:
                pass
            else:
                log_db.log2seq_weight_save(pline)
                yield pline


def measure_accuracy_answer(conf, targets, n_trial=None):
    timer = common.Timer("measure-accuracy answer", output=_logger)
    timer.start()
    mlt = MeasureLTGen(conf, n_trial)
    mlt.init_answer()

    from amulog import lt_import
    table_answer = lt_common.TemplateTable()
    ltgen_answer = lt_import.init_ltgen_import(conf, table_answer)

    for pline in _iter_plines(conf, targets):
        tid, _ = ltgen_answer.process_line(pline)
        if tid is None:
            tpl = None
        else:
            tpl = ltgen_answer.get_tpl(tid)
        mlt.add_org(pline["words"])
        mlt.add_answer(tid, tpl, pline["words"])
    mlt.dump_answer_info()
    timer.stop()
    return mlt


def measure_accuracy_trial_offline(conf, targets, n_trial=None, mlt=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial"])
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load_answer_info()

    for trial_id in range(n_trial):
        timer = common.Timer("measure-accuracy-offline trial{0}".format(
            trial_id), output=_logger)
        timer.start()
        mlt.init_trial(trial_id)
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)

        input_lines = list(_iter_plines(conf, targets))
        d_tid = ltgen.process_offline(input_lines)
        iterobj = zip(input_lines,
                      mlt.iter_tid_answer(),
                      mlt.iter_tpl_answer(pass_none=False))
        for mid, (pline, tid_answer, tpl_answer) in enumerate(iterobj):
            if tid_answer is None:
                tid_trial = None
                tpl_trial = None
            else:
                tid_trial = d_tid[mid]
                if tid_trial is None:
                    tpl_trial = None
                else:
                    tpl_trial = ltgen.get_tpl(tid_trial)
            mlt.add_trial(trial_id, tid_trial, tpl_trial,
                          tid_answer, tpl_answer, pline["words"])
        timer.stop()

    mlt.dump_trial_info()
    return mlt


def measure_accuracy_trial_online(conf, targets_train, targets_test,
                                  n_trial=None, mlt=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial"])
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load_answer_info()

    for trial_id in range(n_trial):
        timer = common.Timer("measure-accuracy-online trial{0}".format(
            trial_id), output=_logger)
        timer.start()
        mlt.init_trial(trial_id)
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)

        if targets_train is not None:
            for pline in _iter_plines(conf, targets_train):
                ltgen.process_line(pline)

        iterobj = zip(_iter_plines(conf, targets_test),
                      mlt.iter_tid_answer(),
                      mlt.iter_tpl_answer(pass_none=False))
        for pline, tid_answer, tpl_answer in iterobj:
            if tid_answer is None:
                tid_trial = None
                tpl_trial = None
            else:
                tid_trial, _ = ltgen.process_line(pline)
                tpl_trial = ltgen.get_tpl(tid_trial)
            mlt.add_trial(trial_id, tid_trial, tpl_trial,
                          tid_answer, tpl_answer, pline["words"])
        timer.stop()

    mlt.dump_trial_info()
    return mlt


def print_metrics(conf, n_trial, mlt=None):
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load()

    print("number of trials: {0}".format(mlt.number_of_trials()))
    n_message = mlt.number_of_messages()
    print("number of messages: {0}".format(n_message))
    if n_message == 0:
        return

    print("number of clusters in answer: {0}".format(
        mlt.number_of_answer_clusters()))
    print("number of clusters in trial: {0}".format(
        mlt.number_of_trial_clusters()))
    print()

    print("word accuracy: {0}".format(mlt.word_accuracy()))
    print("line accuracy: {0}".format(mlt.line_accuracy()))
    print("tpl accuracy: {0}".format(mlt.tpl_accuracy()))
    print("tpl word accuracy: {0}".format(mlt.tpl_word_accuracy()))
    print("rand score: {0}".format(mlt.rand_score()))
    print("adjusted rand score: {0}".format(mlt.adjusted_rand_score()))
    print("f1 score: {0}".format(mlt.f1_score()))
    print("parsing accuracy: {0}".format(mlt.parsing_accuracy()))
    print("cluster accuracy: {0}".format(mlt.cluster_accuracy()))


def get_templates(conf, n_trial, trial_id=0, answer=False):
    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    if answer:
        iterobj = mlt.iter_tpl_answer(pass_none=True)
    else:
        iterobj = mlt.iter_tpl_trial(trial_id, pass_none=True)

    tpls = {tuple(tpl) for tpl in iterobj}
    for tpl in tpls:
        yield tpl


def search_fail_template(conf, n_trial, trial_id=0, pass_similar=True):
    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    s_pass = set()
    iterobj = zip(mlt.iter_org(), mlt.iter_tid_answer(),
                  mlt.iter_label_answer(), mlt.iter_label_trial(trial_id))
    for l_w, tid_answer, labels_answer, labels_trial in iterobj:
        if pass_similar and tid_answer in s_pass:
            continue
        if labels_answer == labels_trial:
            pass
        else:
            result_answer = mlt.restore_result(labels_answer, l_w)
            print("Answer: {0}".format(" ".join(result_answer)))
            result_trial = mlt.restore_result(labels_trial, l_w)
            print("Trial: {0}".format(" ".join(result_trial)))
            print("--------------------")
        pass
        s_pass.add(tid_answer)


def search_diff_template(conf1, conf2, n_trial,
                         trial_id1=0, trial_id2=0, pass_similar=True):
    mlt1 = MeasureLTGen(conf1, n_trial)
    mlt1.load()
    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load()

    s_pass = set()
    iterobj = zip(mlt1.iter_org(),
                  mlt1.iter_tid_answer(),
                  mlt1.iter_label_answer(),
                  mlt1.iter_label_trial(trial_id1),
                  mlt2.iter_label_trial(trial_id2))
    for l_w, tid_answer, labels_answer, labels_trial1, labels_trial2 in iterobj:
        if pass_similar and tid_answer in s_pass:
            continue
        if (not labels_trial1 == labels_answer) and \
                (labels_trial2 == labels_answer):
            tpl_answer = mlt1.restore_result(labels_answer, l_w)
            tpl_trial1 = mlt1.restore_result(labels_trial1, l_w)
            print("< Answer: {0}".format(" ".join(tpl_answer)))
            print("< Trial: {0}".format(" ".join(tpl_trial1)))
            print("--------------------")
        elif (labels_trial1 == labels_answer) and \
                (not labels_trial2 == labels_answer):
            tpl_answer = mlt1.restore_result(labels_answer, l_w)
            tpl_trial2 = mlt2.restore_result(labels_trial2, l_w)
            print("> Answer: {0}".format(" ".join(tpl_answer)))
            print("> Trial: {0}".format(" ".join(tpl_trial2)))
            print("--------------------")
        s_pass.add(tid_answer)


def _sample_partial_cluster(a_true, a_pred, n_samples):
    from sklearn.metrics.cluster import contingency_matrix
    cm = contingency_matrix(a_true, a_pred, sparse=True)
    # sklearn.metrics.cluster.contingency_matrix now uses
    # inverse output of np.unique(a_true, inverse=true)
    # as the input of contingency matrix.
    # Therefore, the unique output works as value mapping.
    a_true_map, a_true_inverse = np.unique(a_true, return_inverse=True)
    nz_true, _ = cm.nonzero()

    l_cluster = []
    for cls_true, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt > 1:
            tid_true = a_true_map[cls_true]
            div = []
            for tid_pred, cnt_pred in zip(*np.unique(
                    a_pred[a_true_inverse == cls_true], return_counts=True)):
                a_index = np.where((a_true == tid_true) &
                                   (a_pred == tid_pred))[0]
                tmp_n_samples = min(n_samples, a_index.shape[0])
                a_index_sample = a_index[:tmp_n_samples]
                div.append((tid_pred, cnt_pred, a_index_sample))
            l_cluster.append((tid_true, div))

    return l_cluster


def _get_complete_clusters(a_true, a_pred):
    from sklearn.metrics.cluster import contingency_matrix
    cm = contingency_matrix(a_true, a_pred, sparse=True)
    nz_true, _ = cm.nonzero()

    tids = []
    for tid_true, uniq_cnt in zip(*np.unique(nz_true, return_counts=True)):
        if uniq_cnt == 1:
            tids.append(tid_true)
    return tids


def search_fail_overdiv(conf, n_trial, trial_id=0, n_samples=1):
    """Search failed log clusters of over-division.
    e.g., 1 cls in answer ≡ 3 cls in trial"""

    timer = common.Timer("test fail_overdiv", output=_logger)
    timer.start()

    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    # overdiv cluster information
    a_tid_answer = np.array(mlt.tid_list_answer())
    a_tid_trial = np.array(mlt.tid_list_trial(trial_id))
    l_cluster = _sample_partial_cluster(a_tid_answer, a_tid_trial, n_samples)

    timer.lap("lap1")

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    timer.lap("lap2")

    # get templates for the indexes to show
    iterobj = mlt.iter_tpl_trial(trial_id, pass_none=True, fill_wildcard=True)
    d_result = {index: result for index, result in enumerate(iterobj)
                if index in s_index_to_show}

    timer.lap("lap3")

    # show
    for tid_answer, div in l_cluster:
        print("Template ID {0} (in answer)".format(tid_answer))
        iterobj = sorted(div, key=lambda x: x[1], reverse=True)
        for cls_id, (tid_trial, cnt_trial, a_index) in enumerate(iterobj):
            for index in a_index:
                print("{0} ({1}): {2}".format(cls_id, cnt_trial,
                                              " ".join(d_result[index])))
        print("--------------------")
    timer.stop()


def search_fail_overagg(conf, n_trial, trial_id=0, n_samples=1):
    """Search failed log clusters of over-division.
    e.g., 3 cls in answer ≡ 1 cls in trial"""

    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    # overagg cluster information
    a_tid_answer = np.array(mlt.tid_list_answer())
    a_tid_trial = np.array(mlt.tid_list_trial(trial_id))
    l_cluster = _sample_partial_cluster(a_tid_trial, a_tid_answer, n_samples)

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    # get templates for the indexes to show
    iterobj = mlt.iter_tpl_trial(trial_id, pass_none=True, fill_wildcard=True)
    d_result = {index: result for index, result in enumerate(iterobj)
                if index in s_index_to_show}

    # show
    for tid_trial, div in l_cluster:
        print("Cluster {0} (in trial)".format(tid_trial))
        iterobj = sorted(div, key=lambda x: x[1], reverse=True)
        for tid_answer, cnt_answer, a_index in iterobj:
            for index in a_index:
                print("ltid {0} ({1}): {2}".format(tid_answer, cnt_answer,
                                                   " ".join(d_result[index])))
        print("--------------------")


def search_diff_overdiv(conf1, conf2, n_trial, trial_id=0, n_samples=1):
    """Search log clusters that is accurate in conf1,
    but failed of over-division in conf2.
    e.g., 1 cls in answer ≡ 1 cls in trial-conf1 ≡ 3 cls in trial-conf2"""

    mlt1 = MeasureLTGen(conf1, n_trial)
    mlt1.load()
    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load()

    # clusters accurate in conf1
    a_tid_answer = np.array(mlt1.tid_list_answer())
    a_tid_trial1 = np.array(mlt1.tid_list_trial(trial_id))
    tids = _get_complete_clusters(a_tid_answer, a_tid_trial1)

    # cluster information that is overdiv in conf2
    a_tid_trial2 = np.array(mlt2.tid_list_trial(trial_id))
    l_cls_all = _sample_partial_cluster(a_tid_answer, a_tid_trial2, n_samples)
    l_cluster = [(tid_true, div) for tid_true, div
                 in l_cls_all if tid_true in tids]

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    # get templates for the indexes to show
    iterobj = mlt2.iter_tpl_trial(trial_id, pass_none=True, fill_wildcard=True)
    d_result = {index: result for index, result in enumerate(iterobj)
                if index in s_index_to_show}

    # show
    for tid_answer, div in l_cluster:
        print("Template ID {0} (in answer)".format(tid_answer))
        iterobj = sorted(div, key=lambda x: x[1], reverse=True)
        for cid, (tid_trial, cnt_trial, a_index) in enumerate(iterobj):
            for index in a_index:
                print("{0} ({1}): {2}".format(cid, cnt_trial,
                                              " ".join(d_result[index])))
        print("--------------------")


def search_diff_overagg(conf1, conf2, n_trial, trial_id=0, n_samples=1):
    """Search log clusters that is accurate in conf1,
    but failed of over-aggregation in conf2.
    e.g., 3 cls in answer ≡ 3 cls in trial-conf1 ≡ 1 cls in trial-conf2"""
    mlt1 = MeasureLTGen(conf1, n_trial)
    mlt1.load()
    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load()

    # clusters accurate in conf1
    a_tid_answer = np.array(mlt1.tid_list_answer())
    a_tid_trial1 = np.array(mlt1.tid_list_trial(trial_id))
    tids = _get_complete_clusters(a_tid_answer, a_tid_trial1)

    # cluster information that is overagg in conf2
    a_tid_trial2 = np.array(mlt2.tid_list_trial(trial_id))
    l_cls_all = _sample_partial_cluster(a_tid_trial2, a_tid_answer, n_samples)
    l_cluster = []
    for tid_trial2, div in l_cls_all:
        for tid_answer, a_index, cnt in div:
            if tid_answer not in tids:
                break
        else:
            l_cluster.append((tid_trial2, div))

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    # get templates for the indexes to show
    iterobj = mlt2.iter_tpl_trial(trial_id, pass_none=True, fill_wildcard=True)
    d_result = {index: result for index, result in enumerate(iterobj)
                if index in s_index_to_show}

    # show
    for tid_trial2, div in l_cluster:
        print("Cluster {0} (in trial)".format(tid_trial2))
        iterobj = sorted(div, key=lambda x: x[1], reverse=True)
        for tid_answer, cnt_answer, a_index in iterobj:
            for index in a_index:
                print("ltid {0} ({1}): {2}".format(tid_answer, cnt_answer,
                                                   " ".join(d_result[index])))
        print("--------------------")


def measure_time_online(conf, targets_train, targets_test, n_trial=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial_time"])

    d_time = {}
    for trial_id in range(n_trial):
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)
        if targets_train is not None:
            for pline in _iter_plines(conf, targets_train):
                ltgen.process_line(pline)

        timer = common.Timer("measure-time-online trial{0}".format(trial_id),
                             output=None)
        timer.start()
        for pline in _iter_plines(conf, targets_test):
            ltgen.process_line(pline)
        timer.stop()
        d_time[trial_id] = timer.total_time()

    return d_time


def measure_time_offline(conf, targets_test, n_trial=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial_time"])

    d_time = {}
    for trial_id in range(n_trial):
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)

        timer = common.Timer("measure-time-offline trial{0}".format(trial_id),
                             output=None)
        timer.start()
        input_lines = list(_iter_plines(conf, targets_test))
        ltgen.process_offline(input_lines)
        timer.stop()
        d_time[trial_id] = timer.total_time()

    return d_time
