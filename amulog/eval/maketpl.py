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
                          "n_variables": int(),
                          "d_n_variables": defaultdict(int),
                          "n_descriptions": int(),
                          "d_n_descriptions": defaultdict(int)}

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
                                  "n_c_variables": int(),
                                  "d_n_c_variables": defaultdict(int),
                                  "n_c_descriptions": int(),
                                  "d_n_c_descriptions": defaultdict(int)})

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

    @classmethod
    def _tpl2dump(cls, tpl, l_w):
        if tpl is None:
            return "\n"
        else:
            l_label = [cls.LABEL_VARIABLE if w_tpl == lt_common.REPLACER
                       else cls.LABEL_DESCRIPTION
                       for w_tpl, w_org in zip(tpl, l_w)]
            return "".join(l_label)

    @classmethod
    def restore_tpl(cls, labels, l_w):
        if labels is None or labels == "\n":
            return None
        else:
            return [lt_common.REPLACER if label == cls.LABEL_VARIABLE
                    else w
                    for label, w in zip(labels, l_w)]

    @classmethod
    def restore_result(cls, labels, l_w):
        if labels is None or labels == "\n":
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

    def iter_org(self):
        with open(self._org_word_path(), 'r') as f:
            for line in f:
                yield line.strip().split(" ")

    def iter_label_answer(self, pass_none=False):
        with open(self._answer_label_path(), 'r') as f:
            for line in f:
                if line == "\n" and pass_none:
                    pass
                else:
                    yield line.strip()

    def iter_label_trial(self, trial_id, pass_none=False):
        with open(self._trial_label_path(trial_id), 'r') as f:
            for line in f:
                if line == "\n" and pass_none:
                    pass
                else:
                    yield line.strip()

    def iter_tpl_answer(self, pass_none=True, fill_wildcard=False):
        for l_w, labels in zip(self.iter_org(), self.iter_label_answer()):
            if labels == "":
                if pass_none:
                    pass
                else:
                    return None
            elif fill_wildcard:
                yield self.restore_result(labels, l_w)
            else:
                yield self.restore_tpl(labels, l_w)

    def iter_tpl_trial(self, trial_id, pass_none=True, fill_wildcard=False):
        for l_w, labels in zip(self.iter_org(), self.iter_label_trial(trial_id)):
            if labels == "":
                if pass_none:
                    pass
                else:
                    return None
            elif fill_wildcard:
                yield self.restore_result(labels, l_w)
            else:
                yield self.restore_tpl(labels, l_w)

    def iter_tid_answer(self):
        return self._d_answer["l_tid"].__iter__()

#    def iter_answer_info(self):
#        for tid, tpl in zip(self._d_answer["l_tid"], self.iter_tpl_answer()):
#            yield tid, tpl

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

#    def _rand_score(self, trial_id):
#        # Referred R library "fossil::rand.index"
#        from scipy.special import comb
#        l_tid_answer = self._tid_list_answer()
#        l_tid_trial = self._tid_list_trial(trial_id)
#
#        x = np.array(l_tid_answer)
#        m_x_diff = np.abs(x - x[:, np.newaxis])
#        m_x_diff[m_x_diff > 1] = 1
#        y = np.array(l_tid_trial)
#        m_y_diff = np.abs(y - y[:, np.newaxis])
#        m_y_diff[m_y_diff > 1] = 1
#        diff_comb = np.sum(np.abs(m_x_diff - m_y_diff)) / 2
#        all_comb = comb(len(l_tid_answer), 2, exact=True)
#        return 1. - diff_comb / all_comb

    def rand_score(self, trial_id=None):
        if trial_id is None:
            l_score = [self.rand_score(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            l_tid_answer = self.tid_list_answer()
            l_tid_trial = self.tid_list_trial(trial_id)
            return cluster_metrics.rand_score(l_tid_answer, l_tid_trial)

#    def _adjusted_rand_score_trial(self, trial_id):
#        from sklearn.metrics import adjusted_rand_score
#        l_tid_answer = self._tid_list_answer()
#        l_tid_trial = self._tid_list_trial(trial_id)
#        return adjusted_rand_score(l_tid_answer, l_tid_trial)

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

#    def _f1_score_trial(self, trial_id):
#        from itertools import combinations
#        l_tid_answer = self._tid_list_answer()
#        pairs_answer = [n1 == n2
#                        for n1, n2 in combinations(l_tid_answer, 2)]
#        l_tid_trial = self._tid_list_trial(trial_id)
#        pairs_trial = [n1 == n2
#                       for n1, n2 in combinations(l_tid_trial, 2)]
#
#        from sklearn.metrics import f1_score
#        return f1_score(pairs_answer, pairs_trial)

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

#    def _parsing_accuracy_trial(self, trial_id):
#        # Referred https://github.com/logpai/logparser
#        from pandas import Series
#        sr_tid_answer = Series(self._tid_list_answer(), dtype=int)
#        sr_tid_trial = Series(self._tid_list_trial(trial_id), dtype=int)
#
#        n_correct = 0
#        for tid_answer in sr_tid_answer.unique():
#            logs_answer = sr_tid_answer[sr_tid_answer == tid_answer].index
#            division = sr_tid_trial[logs_answer].unique()
#            if division.size == 1:
#                logs_trial = sr_tid_trial[sr_tid_trial == division[0]].index
#                if logs_trial.size == logs_answer.size:
#                    n_correct += logs_answer.size
#        return 1.0 * n_correct / sr_tid_answer.size

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
                yield pline


def measure_accuracy_answer(conf, targets, n_trial=None):
    mlt = MeasureLTGen(conf, n_trial)
    mlt.init_answer()

    from amulog import lt_import
    table_answer = lt_common.TemplateTable()
    ltgen_answer = lt_import.init_ltgen_import(conf, table_answer)

    for pline in _iter_plines(conf, targets):
        tpl = ltgen_answer.generate_tpl(pline)
        tid, _ = ltgen_answer.match_table(tpl)
        mlt.add_org(pline["words"])
        mlt.add_answer(tid, tpl, pline["words"])
    mlt.dump_answer_info()
    return mlt


def measure_accuracy_trial(conf, targets, n_trial=None, mlt=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial"])
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load_answer_info()

    for trial_id in range(n_trial):
        mlt.init_trial(trial_id)
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)
        for pline, tid_answer, tpl_answer in zip(_iter_plines(conf, targets),
                                                 mlt.iter_tid_answer(),
                                                 mlt.iter_label_answer()):
            if tid_answer is None:
                tid_trial = None
                tpl_trial = None
            else:
                tpl_trial = ltgen.generate_tpl(pline)
                tid_trial, _ = ltgen.match_table(tpl_trial)
            mlt.add_trial(trial_id, tid_trial, tpl_trial,
                          tid_answer, tpl_answer, pline["words"])

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
    nz_true, _ = cm.nonzero()

    l_cluster = []
    for tid_true, cnt_true in zip(*np.unique(nz_true, return_counts=True)):
        if cnt_true <= 1:
            continue

        div = []
        for tid_pred, cnt_pred in zip(*np.unique(
                a_pred[a_true == tid_true], return_counts=True)):
            a_index = np.where((a_true == tid_true) &
                               (a_pred == tid_pred))[0]
            a_index_sample = a_index[:min(n_samples, a_index.shape[0])]
            div.append((tid_pred, cnt_pred, a_index_sample))
        l_cluster.append((tid_true, div))

    return l_cluster


def search_fail_overdiv(conf, n_trial, trial_id=0, n_samples=1):
    """Search failed log clusters of over-division.
    e.g., 1 cls in answer ≡ 3 cls in trial"""

    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    a_tid_answer = np.array(mlt.tid_list_answer())
    a_tid_trial = np.array(mlt.tid_list_trial(trial_id))
    l_cluster = _sample_partial_cluster(a_tid_answer, a_tid_trial, n_samples)
    s_index_to_show = set()
    for _, div in l_cluster:
        s_index_to_show = s_index_to_show | set(np.ravel(tuple(zip(*div))[-1]))

    # get templates for the indexes to show
    iterobj = mlt.iter_tpl_trial(trial_id, pass_none=True, fill_wildcard=True)
    d_result = {index: result for index, result in enumerate(iterobj)
                if index in s_index_to_show}

    # show
    for tid_answer, div in l_cluster:
        for cid, (tid_trial, a_index, cnt) in enumerate(div):
            print("Template ID {0} (in answer)".format(tid_answer))
            for index in a_index:
                tpl = d_result[index]
                print("{0} ({1}): {2}".format(cid, cnt, d_result[index]))
        print("--------------------")


def search_fail_overagg(conf, n_trial, trial_id=0, n_samples=1):
    """Search failed log clusters of over-division.
    e.g., 3 cls in answer ≡ 1 cls in trial"""

    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    a_tid_answer = np.array(mlt.tid_list_answer())
    a_tid_trial = np.array(mlt.tid_list_trial(trial_id))
    l_cluster = _sample_partial_cluster(a_tid_trial, a_tid_answer, n_samples)
    s_index_to_show = set()
    for _, div in l_cluster:
        s_index_to_show = s_index_to_show | set(np.ravel(tuple(zip(*div))[-1]))

    # get templates for the indexes to show
    iterobj = mlt.iter_tpl_trial(trial_id, pass_none=True, fill_wildcard=True)
    d_result = {index: result for index, result in enumerate(iterobj)
                if index in s_index_to_show}

    # show
    for tid_trial, div in l_cluster:
        for tid_answer, a_index, cnt in div:
            for index in a_index:
                print("{0} ({1}): {2}".format(tid_answer, cnt, d_result[index]))
        print("--------------------")


def measure_time(conf, targets, n_trial=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial_time"])

    timer = common.Timer("measure-time", output=None)
    timer.start()
    for trial_id in range(n_trial):
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)
        for pline in _iter_plines(conf, targets):
            tpl_trial = ltgen.generate_tpl(pline)
            tid_trial, _ = ltgen.match_table(tpl_trial)
        timer.lap_diff("trial{0}".format(trial_id))
    timer.stop()
    timer.stat()
