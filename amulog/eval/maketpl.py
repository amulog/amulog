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

    def __init__(self, conf, n_trial):
        self.conf = conf
        self._number_of_trials = n_trial
        common.mkdir(self._output_dir(conf))

        self._d_answer = {"l_tid": list(),
                          "n_lines": int(),
                          "d_n_lines": defaultdict(int),
                          "n_words": int(),
                          "d_n_words": defaultdict(int),
                          "n_variables": int(),
                          "d_n_variables": defaultdict(int),
                          "n_descriptions": int(),
                          "d_n_descriptions": defaultdict(int)}
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
    def _output_dir(conf):
        return conf["eval"]["measure_ltgen_dir"]

    def _trial_path(self, trial_id):
        return "{0}/trial{1}".format(self._output_dir(self.conf),
                                     str(trial_id).zfill(2))

    def _answer_path(self):
        return "{0}/answer".format(self._output_dir(self.conf))

    def init_answer(self):
        common.rm(self._answer_path())

    def init_trial(self, trial_id):
        common.rm(self._trial_path(trial_id))

    def _info_path(self):
        return "{0}/info".format(self._output_dir(self.conf))

    def load(self):
        with open(self._info_path(), 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self._d_answer, self._d_trial = obj

    def dump(self):
        obj = (self._d_answer, self._d_trial)
        with open(self._info_path(), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def add_trial(self, trial_id, tid_trial, tpl_trial,
                  tid_answer, tpl_answer):
        self._update_stat_trial(trial_id, tid_trial, tpl_trial,
                                tid_answer, tpl_answer)
        if tpl_trial is None:
            added_line = "\n"
        else:
            added_line = self.SPLITTER.join(tpl_trial) + "\n"
        with open(self._trial_path(trial_id), 'a') as f:
            f.write(added_line)

    def add_answer(self, tid, tpl):
        self._update_stat_answer(tid, tpl)
        if tpl is None:
            added_line = "\n"
        else:
            added_line = self.SPLITTER.join(tpl) + "\n"
        with open(self._answer_path(), 'a') as f:
            f.write(added_line)

    def iter_tpl_trial(self, trial_id, yield_none=True):
        with open(self._trial_path(trial_id), 'r') as f:
            for line in f:
                if line == "\n":
                    if yield_none:
                        yield None
                else:
                    yield line.rstrip().split(self.SPLITTER)

    def iter_tpl_answer(self, yield_none=True):
        with open(self._answer_path(), 'r') as f:
            for line in f:
                if line == "\n":
                    if yield_none:
                        yield None
                else:
                    yield line.rstrip().split(self.SPLITTER)

    def iter_answer_info(self, yield_none=True):
        for tid, tpl in zip(self._d_answer["l_tid"], self.iter_tpl_answer()):
            yield tid, tpl

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


def measure_accuracy(conf, targets, n_trial=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial"])
    mlt = MeasureLTGen(conf, n_trial)

    # generate answer
    # required (not enough with existing log_db.LogData)
    # because measurement numerators need to be initialized
    from amulog import lt_import
    table_answer = lt_common.TemplateTable()
    ltgen_answer = lt_import.init_ltgen_import(conf, table_answer)
    mlt.init_answer()
    for pline in _iter_plines(conf, targets):
        tpl = ltgen_answer.generate_tpl(pline)
        tid, _ = ltgen_answer.match_table(tpl)
        mlt.add_answer(tid, tpl)

    # estimate templates
    for trial_id in range(n_trial):
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)
        mlt.init_trial(trial_id)
        for pline, (tid_answer, tpl_answer) in zip(_iter_plines(conf, targets),
                                                   mlt.iter_answer_info()):
            if tid_answer is None:
                tid_trial = None
                tpl_trial = None
            else:
                tpl_trial = ltgen.generate_tpl(pline)
                tid_trial, _ = ltgen.match_table(tpl_trial)
            mlt.add_trial(trial_id, tid_trial, tpl_trial,
                          tid_answer, tpl_answer)

    mlt.dump()
    print_metrics(conf, n_trial, mlt=mlt)


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
    for (tid_answer, tpl_answer), tpl_trial in zip(
            mlt.iter_answer_info(), mlt.iter_tpl_trial(trial_id)):
        if pass_similar and tid_answer in s_pass:
            continue
        if tpl_answer == tpl_trial:
            pass
        else:
            print("Answer: {0}".format(" ".join(tpl_answer)))
            print("Trial: {0}".format(" ".join(tpl_trial)))
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
    iterobj = zip(mlt1.iter_answer_info(),
                  mlt2.iter_answer_info(),
                  mlt1.iter_tpl_trial(trial_id1),
                  mlt2.iter_tpl_trial(trial_id2))
    for elm in iterobj:
        tid_answer, tpl_answer1 = elm[0]
        _, tpl_answer2 = elm[1]
        if not tpl_answer1 == tpl_answer2:
            msg = ("answer data not matching: "
                   "comparing results of different input?")
            raise ValueError(msg)
        if pass_similar and tid_answer in s_pass:
            continue
        tpl_trial1, tpl_trial2 = elm[2:4]
        if (not tpl_trial1 == tpl_answer1) and (tpl_trial2 == tpl_answer2):
            print("< Answer: {0}".format(" ".join(tpl_answer1)))
            print("< Trial: {0}".format(" ".join(tpl_trial1)))
            print("--------------------")
        if (tpl_trial1 == tpl_answer1) and (not tpl_trial2 == tpl_answer2):
            print("> Answer: {0}".format(" ".join(tpl_answer2)))
            print("> Trial: {0}".format(" ".join(tpl_trial2)))
            print("--------------------")
        s_pass.add(tid_answer)


def _sample_cluster_diff(a_true, a_pred, n_samples):
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
    l_cluster = _sample_cluster_diff(a_tid_answer, a_tid_trial, n_samples)
    s_index_to_show = set()
    for _, div in l_cluster:
        s_index_to_show = s_index_to_show | set(np.ravel(tuple(zip(*div))[-1]))

    # get templates for the indexes to show
    d_tpl = {index: tpl for index, tpl
             in enumerate(mlt.iter_tpl_trial(trial_id, yield_none=False))
             if index in s_index_to_show}

    # show
    for tid_answer, div in l_cluster:
        for cid, (tid_trial, a_index, cnt) in enumerate(div):
            print("Template ID {0} (in answer)".format(tid_answer))
            for index in a_index:
                tpl = d_tpl[index]
                print("{0} ({1}): {2}".format(cid, cnt, " ".join(tpl)))
        print("--------------------")


def search_fail_overagg(conf, n_trial, trial_id=0, n_samples=1):
    """Search failed log clusters of over-division.
    e.g., 3 cls in answer ≡ 1 cls in trial"""

    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    a_tid_answer = np.array(mlt.tid_list_answer())
    a_tid_trial = np.array(mlt.tid_list_trial(trial_id))
    l_cluster = _sample_cluster_diff(a_tid_trial, a_tid_answer, n_samples)
    s_index_to_show = set()
    for _, div in l_cluster:
        s_index_to_show = s_index_to_show | set(np.ravel(tuple(zip(*div))[-1]))

    # get templates for the indexes to show
    d_tpl = {index: tpl for index, tpl
             in enumerate(mlt.iter_tpl_trial(trial_id, yield_none=False))
             if index in s_index_to_show}

    # show
    for tid_trial, div in l_cluster:
        for tid_answer, a_index, cnt in div:
            for index in a_index:
                tpl = d_tpl[index]
                print("{0} ({1}): {2}".format(tid_answer, cnt, " ".join(tpl)))
        print("--------------------")


def search_diff_overdiv(conf1, conf2, n_trial, trial_id=0, n_samples=1):
    from sklearn.metrics.cluster import contingency_matrix
    mlt1 = MeasureLTGen(conf1, n_trial)
    mlt1.load()
    a_tid_answer1 = np.array(mlt1.tid_list_answer())
    a_tid_trial1 = np.array(mlt1.tid_list_trial(trial_id))
    a_true1 = np.array(a_tid_answer1)
    a_pred1 = np.array(a_tid_trial1)
    cm1 = contingency_matrix(a_true1, a_pred1, sparse=True)
    nz_true1, _ = cm1.nonzero()

    s_tid_answer_correct = set()
    s_tid_answer = {tid_answer for tid_answer, cnt_answer
                    in zip(*np.unique(nz_true1, return_counts=True))
                    if cnt_answer > 1}
    del mlt1
    del cm1
    del nz_true1

    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load()
    a_tid_answer2 = np.array(mlt2.tid_list_answer())
    a_tid_trial2 = np.array(mlt2.tid_list_trial(trial_id))
    a_true2 = np.array(a_tid_answer2)
    a_pred2 = np.array(a_tid_trial2)
    cm2 = contingency_matrix(a_true2, a_pred2, sparse=True)
    nz_true2, _ = cm2.nonzero()

    for tid_answer in s_tid_answer:
        pass


    # determine list of indexes for failed clusters
    l_show = []
    s_pred = set()
    for tid_answer, cnt_answer in zip(*np.unique(nz_true, return_counts=True)):
        if cnt_answer <= 1:
            continue

        tmp = []
        for tid_trial, cnt_trial in zip(*np.unique(
                a_tid_trial[a_tid_answer == tid_answer], return_counts=True)):
            a_index = np.where((a_tid_answer == tid_answer) &
                               (a_tid_trial == tid_trial))[0]
            index_to_add = a_index[:min(n_samples, a_index.shape[0])]
            tmp.append((index_to_add, cnt_trial))
            s_pred = s_pred | set(index_to_add)

        assert len(tmp) > 1
        l_show.append(tmp)

    # get templates for the indexes to show
    d_tpl = {index: tpl
             for index, tpl in enumerate(mlt.iter_tpl_trial(trial_id))
             if index in s_pred}

    # show
    for result_indexes in l_show:
        for cid, (a_index, cnt) in enumerate(result_indexes):
            for index in a_index:
                print("{0} ({1}): {2}".format(cid, cnt, " ".join(d_tpl[index])))
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
