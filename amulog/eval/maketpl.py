#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from collections import defaultdict

from amulog import common
from amulog import lt_common


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
        self._d_trial = {}
        for tid in range(n_trial):
            self._d_trial[tid] = {"l_tid": list(),
                                  "n_c_lines": int(),
                                  "d_n_c_lines": defaultdict(int),
                                  "n_c_words": int(),
                                  "d_n_c_words": defaultdict(int),
                                  "n_c_variables": int(),
                                  "d_n_c_variables": defaultdict(int),
                                  "n_c_descriptions": int(),
                                  "d_n_c_descriptions": defaultdict(int)}

    @staticmethod
    def _output_dir(conf):
        return conf["eval"]["measure_ltgen_dir"]

    def _trial_path(self, trial_id):
        return "{0}/trial{1}".format(self._output_dir(self.conf),
                                     str(trial_id).zfill(2))

    def _answer_path(self):
        return "{0}/answer".format(self._output_dir(self.conf))

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
        if tid_answer is not None:
            self._update_stat_trial(trial_id, tid_trial, tpl_trial,
                                    tid_answer, tpl_answer)
            with open(self._trial_path(trial_id), 'a') as f:
                f.write(self.SPLITTER.join(tpl_answer))

    def add_answer(self, tid, tpl):
        if tid is not None:
            self._update_stat_answer(tid, tpl)
            with open(self._answer_path(), 'a') as f:
                f.write(self.SPLITTER.join(tpl))

    def iter_tpl_trial(self, trial_id):
        with open(self._trial_path(trial_id), 'r') as f:
            for line in f:
                yield line.rstrip().split(self.SPLITTER)

    def iter_tpl_answer(self):
        with open(self._answer_path(), 'r') as f:
            for line in f:
                yield line.rstrip().split(self.SPLITTER)

    def iter_answer_info(self):
        for tid, tpl in zip(self._d_answer["l_tid"], self.iter_tpl_answer()):
            yield tid, tpl

    def _update_stat_answer(self, tid, tpl):
        self._d_answer["l_tid"].append(tid)
        self._d_answer["n_lines"] += 1
        self._d_answer["d_n_lines"][tid] += 1
        n_words = len(tpl)
        self._d_answer["n_words"] += n_words
        self._d_answer["d_n_words"][tid] += n_words

    def _update_stat_trial(self, trial_id, tid_trial, tpl_trial,
                           tid_answer, tpl_answer):
        self._d_trial[trial_id]["l_tid"].append(tid_trial)

        if tpl_trial == tpl_answer:
            self._d_trial[trial_id]["n_c_lines"] += 1
            self._d_trial[trial_id]["d_n_c_lines"][tid_answer] += 1

        assert len(tpl_trial) == len(tpl_answer)
        for w_trial, w_answer in zip(tpl_trial, tpl_answer):
            if w_trial == w_answer:
                self._d_trial[trial_id]["n_c_words"] += 1
                self._d_trial[trial_id]["d_n_c_words"][tid_answer] += 1

    def number_of_trials(self):
        return self._number_of_trials

    def number_of_messages(self):
        return self._d_answer["n_lines"]

    def number_of_answer_clusters(self):
        return len(self._d_answer["d_n_lines"])

    def number_of_trial_clusters(self, trial_id=None):
        from pandas import Series
        if trial_id is None:
            l_n = [Series(self._d_trial[i]["l_tid"], dtype=int).unique()
                   for i in range(self.number_of_trials())]
            return np.average(l_n)
        else:
            return Series(self._d_trial[trial_id]["l_tid"], dtype=int).unique()

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
        for tid in d_n_words:
            l_acc.append(d_n_c_words[tid] / d_n_words[tid])
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
        for tid in d_n_lines:
            l_acc.append(d_n_c_lines[tid] / d_n_lines[tid])
        return np.average(l_acc)

    def tpl_accuracy(self, trial_id=None):
        if trial_id is None:
            l_acc = [self._tpl_accuracy_trial(i)
                     for i in range(self.number_of_trials())]
            return np.average(l_acc)
        else:
            return self._tpl_accuracy_trial(trial_id)

    def _adjusted_rand_score_trial(self, trial_id):
        from sklearn.metrics import adjusted_rand_score
        l_tid_answer = self._d_answer["l_tid"]
        l_tid_trial = self._d_trial[trial_id]["l_tid"]
        return adjusted_rand_score(l_tid_answer, l_tid_trial)

    def adjusted_rand_score(self, trial_id=None):
        if trial_id is None:
            l_score = [self._adjusted_rand_score_trial(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            return self._adjusted_rand_score_trial(trial_id)

    def _f1_score_trial(self, trial_id):
        from itertools import combinations
        l_tid_answer = self._d_answer["l_tid"]
        pairs_answer = [n1 == n2
                        for n1, n2 in combinations(l_tid_answer, 2)]
        l_tid_trial = self._d_trial[trial_id]["l_tid"]
        pairs_trial = [n1 == n2
                       for n1, n2 in combinations(l_tid_trial, 2)]

        from sklearn.metrics import f1_score
        return f1_score(pairs_answer, pairs_trial)

    def f1_score(self, trial_id=None):
        if trial_id is None:
            l_score = [self._f1_score_trial(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            return self._f1_score_trial(trial_id)

    def _parsing_accuracy_trial(self, trial_id):
        from pandas import Series
        sr_tid_answer = Series(self._d_answer["l_tid"], dtype=int)
        sr_tid_trial = Series(self._d_trial[trial_id]["l_tid"], dtype=int)

        n_correct = 0
        for tid_answer in sr_tid_answer.unique():
            logs_answer = sr_tid_answer[sr_tid_answer == tid_answer].index
            division = sr_tid_trial[logs_answer].unique()
            if division.size == 1:
                logs_trial = sr_tid_trial[sr_tid_trial == division[0]].index
                if logs_trial.size == logs_answer.size:
                    n_correct += logs_answer.size
        return 1.0 * n_correct / sr_tid_answer.size

    def parsing_accuracy(self, trial_id=None):
        if trial_id is None:
            l_score = [self._parsing_accuracy_trial(i)
                       for i in range(self.number_of_trials())]
            return np.average(l_score)
        else:
            return self._parsing_accuracy_trial(trial_id)


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


def measure(conf, targets, n_trial):
    mlt = MeasureLTGen(conf, n_trial)

    # generate answer
    # required (not enough with existing log_db.LogData)
    # because measurement numerators need to be initialized
    from amulog import lt_import
    table_answer = lt_common.TemplateTable()
    ltgen_answer = lt_import.init_ltgen_import(conf, table_answer)
    for pline in _iter_plines(conf, targets):
        tpl = ltgen_answer.generate_tpl(pline)
        tid, _ = ltgen_answer.match_table(tpl)
        mlt.add_answer(tid, tpl)

    # estimate templates
    for trial_id in range(n_trial):
        table = lt_common.TemplateTable()
        ltgen = lt_common.init_ltgen_methods(conf, table)
        for pline, (tid_answer, tpl_answer) in zip(_iter_plines(conf, targets),
                                                   mlt.iter_answer_info()):
            if tid_answer is None:
                continue
            tpl_trial = ltgen.generate_tpl(pline)
            tid_trial, _ = ltgen_answer.match_table(tpl_trial)
            mlt.add_trial(trial_id, tid_trial, tpl_trial,
                          tid_answer, tpl_answer)

    mlt.dump()
    print_metrics(conf, n_trial, mlt=mlt)


def print_metrics(conf, n_trial, mlt=None):
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load()

    print("number of trials: {0}".format(mlt.number_of_trials()))
    print("number of messages: {0}".format(mlt.number_of_messages()))
    print("number of clusters in answer: {0}".format(
        mlt.number_of_answer_clusters()))
    print("number of clusters in trial: {0}".format(
        mlt.number_of_trial_clusters()))
    print()

    print("word accuracy: {0}".format(mlt.word_accuracy()))
    print("line accuracy: {0}".format(mlt.line_accuracy()))
    print("tpl accuracy: {0}".format(mlt.tpl_accuracy()))
    print("tpl word accuracy: {0}".format(mlt.tpl_word_accuracy()))
    print("adjusted rand score: {0}".format(mlt.adjusted_rand_score()))
    print("f1 score: {0}".format(mlt.f1_score()))
    print("parsing accuracy: {0}".format(mlt.parsing_accuracy()))
