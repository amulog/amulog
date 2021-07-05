#!/usr/bin/env python
# coding: utf-8

import logging
import json
import numpy as np
from collections import defaultdict

import amulog.manager
from amulog import common
from amulog import lt_common
from amulog.eval import cluster_metrics
from amulog.eval import structure_metrics

_logger = logging.getLogger(__package__.partition(".")[0])


class MeasureLTGen:
    """Record and load measurement data of tempalte generation.
    This is implemented in memory-saving architecture;
    The calculated data for each line is appended into text file."""

    SPLITTER = "@@"
    LABEL_NONE = "N"
    LABEL_DESCRIPTION = "D"
    LABEL_VARIABLE = "V"
    FILEPATH_DIGIT_LENGTH = 2

    def __init__(self, conf, n_trial):
        self._conf = conf
        self._n_trial = n_trial

        self._current_trial = None
        self._d_answer = None
        self._d_trial = None

    def _init_answer_info(self):
        common.mkdir(self._output_dir_answer(self._conf))
        self._d_answer = {"l_tid": list(),
                          "n_lines": int(),
                          "d_n_lines": defaultdict(int),
                          "n_words": int(),
                          "d_n_words": defaultdict(int),
                          }

    def _init_trial_info(self):
        common.mkdir(self._output_dir_trial(self._conf))
        if self._n_trial is None:
            return
        assert self._n_trial < 10 ** self.FILEPATH_DIGIT_LENGTH
        self._d_trial = {"l_tid": list(),
                         "n_c_lines": int(),
                         "d_n_c_lines": defaultdict(int),
                         "n_c_words": int(),
                         "d_n_c_words": defaultdict(int),
                         }

    # file IO methods
    @staticmethod
    def _output_dir_answer(conf):
        return conf["eval"]["ltgen_answer_dir"]

    @staticmethod
    def _output_dir_trial(conf):
        return conf["eval"]["ltgen_trial_dir"]

    def _org_word_path(self):
        return "{0}/word".format(self._output_dir_answer(self._conf))

    def _answer_label_path(self):
        return "{0}/label_answer".format(self._output_dir_answer(self._conf))

    def _trial_label_path(self, trial_id):
        str_trial_id = str(trial_id).zfill(self.FILEPATH_DIGIT_LENGTH)
        return "{0}/label_trial{1}".format(self._output_dir_trial(self._conf),
                                           str_trial_id)

    def _answer_info_path(self):
        return "{0}/info_answer".format(self._output_dir_answer(self._conf))

    def _trial_info_path(self, trial_id):
        str_trial_id = str(trial_id).zfill(self.FILEPATH_DIGIT_LENGTH)
        return "{0}/info_trial{1}".format(self._output_dir_trial(self._conf),
                                          str_trial_id)

    def init_answer(self):
        common.rm(self._org_word_path())
        common.rm(self._answer_label_path())
        self._init_answer_info()

    def init_trial(self, trial_id):
        common.rm(self._trial_label_path(trial_id))
        self._current_trial = trial_id
        self._init_trial_info()

    def _load_answer_info(self):
        with open(self._answer_info_path(), 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self._d_answer = obj

    def _load_trial_info(self):
        trial_id = self._current_trial
        with open(self._trial_info_path(trial_id), 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self._d_trial = obj

    def _dump_answer_info(self):
        obj = self._d_answer
        with open(self._answer_info_path(), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def _dump_trial_info(self):
        trial_id = self._current_trial
        obj = self._d_trial
        with open(self._trial_info_path(trial_id), 'w', encoding='utf-8') as f:
            json.dump(obj, f)

    def load(self, trial_id=None):
        self._load_answer_info()
        if trial_id is not None:
            self.load_trial(trial_id)

    def load_trial(self, trial_id):
        self._current_trial = trial_id
        self._load_trial_info()

    def dump_answer(self):
        self._dump_answer_info()

    def dump_trial(self):
        self._dump_trial_info()

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

    def add_trial(self, tid_trial, tpl_trial,
                  tid_answer, tpl_answer, l_w):
        self._update_stat_trial(tid_trial, tpl_trial,
                                tid_answer, tpl_answer)
        added_line = self._tpl2dump(tpl_trial, l_w)
        with open(self._trial_label_path(self._current_trial), 'a') as f:
            f.write(added_line)

    def _update_stat_answer(self, tid, tpl):
        self._d_answer["l_tid"].append(tid)
        if tid is not None:
            self._d_answer["n_lines"] += 1
            self._d_answer["d_n_lines"][str(tid)] += 1
            n_words = len(tpl)
            self._d_answer["n_words"] += n_words
            self._d_answer["d_n_words"][str(tid)] += n_words

    def _update_stat_trial(self, tid_trial, tpl_trial,
                           tid_answer, tpl_answer):
        self._d_trial["l_tid"].append(tid_trial)
        if tid_answer is not None:
            if tpl_trial == tpl_answer:
                self._d_trial["n_c_lines"] += 1
                self._d_trial["d_n_c_lines"][str(tid_answer)] += 1

            assert len(tpl_trial) == len(tpl_answer)
            for w_trial, w_answer in zip(tpl_trial, tpl_answer):
                if w_trial == w_answer:
                    self._d_trial["n_c_words"] += 1
                    self._d_trial["d_n_c_words"][str(tid_answer)] += 1

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

    def iter_label_trial(self, pass_none=False):
        with open(self._trial_label_path(self._current_trial), 'r') as f:
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

    def iter_tpl_trial(self, pass_none=False, fill_wildcard=False):
        for l_w, labels in zip(self.iter_org(), self.iter_label_trial()):
            if labels is None:
                if pass_none:
                    pass
                else:
                    yield None
            elif fill_wildcard:
                yield self.restore_result(labels, l_w)
            else:
                yield self.restore_tpl(labels, l_w)

    def tid_list_answer(self, pass_none=False):
        if pass_none:
            return [tid for tid in self._d_answer["l_tid"]
                    if tid is not None]
        else:
            return self._d_answer["l_tid"]

    def tid_list_trial(self, pass_none=False):
        if pass_none:
            return [tid for tid in self._d_trial["l_tid"]
                    if tid is not None]
        else:
            return self._d_trial["l_tid"]

    def valid_tid_list_answer(self):
        return self.tid_list_answer(pass_none=True)

    def valid_tid_list_trial(self):
        return self.tid_list_trial(pass_none=True)

    def iter_cluster_answer(self):
        return self._d_answer["n_lines"].keys().__iter__()

    def iter_cluster_trial(self):
        return np.unique(self.tid_list_trial(pass_none=True))

    def number_of_trials(self):
        return self._n_trial

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

    def number_of_trial_clusters(self):
        return np.unique(self.valid_tid_list_trial()).shape[0]

    # accuracy methods
    def word_accuracy(self, recalculation=False):
        if recalculation:
            iterable_tpl_answer = self.iter_tpl_answer(pass_none=True)
            iterable_tpl_trial = self.iter_tpl_trial(pass_none=True)
            return structure_metrics.word_accuracy(
                iterable_tpl_answer, iterable_tpl_trial)
        else:
            # n_words: Number of all words in dataset
            n_words = self._d_answer["n_words"]
            # n_c_words: Number of words correctly labeled in dataset
            n_c_words = self._d_trial["n_c_words"]
            return 1.0 * n_c_words / n_words

    def line_accuracy(self, recalculation=False):
        if recalculation:
            iterable_tpl_answer = self.iter_tpl_answer(pass_none=True)
            iterable_tpl_trial = self.iter_tpl_trial(pass_none=True)
            return structure_metrics.line_accuracy(
                iterable_tpl_answer, iterable_tpl_trial)
        else:
            # n_lines: Number of all lines in dataset
            n_lines = self._d_answer["n_lines"]
            # n_c_lines: Number of lines correctly labeled in dataset
            n_c_lines = self._d_trial["n_c_lines"]
            return 1.0 * n_c_lines / n_lines

    def tpl_word_accuracy(self, recalculation=False):
        if recalculation:
            iterable_tpl_answer = self.iter_tpl_answer(pass_none=True)
            iterable_tpl_trial = self.iter_tpl_trial(pass_none=True)
            l_tid_answer = self.tid_list_answer(pass_none=True)
            return structure_metrics.tpl_word_accuracy(
                iterable_tpl_answer, iterable_tpl_trial, l_tid_answer)
        else:
            # d_n_words: Number of words in a template cluster
            d_n_words = self._d_answer["d_n_words"]
            # d_n_c_words: Number of words correctly labeled in a template cluster
            d_n_c_words = self._d_trial["d_n_c_words"]

            l_acc = []
            for key in d_n_words:  # key: str(tid)
                l_acc.append(d_n_c_words.get(key, 0) / d_n_words.get(key, 0))
            return np.average(l_acc)

    def tpl_accuracy(self, recalculation=False):
        if recalculation:
            iterable_tpl_answer = self.iter_tpl_answer(pass_none=True)
            iterable_tpl_trial = self.iter_tpl_trial(pass_none=True)
            l_tid_answer = self.tid_list_answer(pass_none=True)
            return structure_metrics.tpl_accuracy(
                iterable_tpl_answer, iterable_tpl_trial, l_tid_answer)
        else:
            # d_n_lines: Number of lines in a template cluster
            d_n_lines = self._d_answer["d_n_lines"]
            # d_n_c_lines: Number of lines correctly labeled in a template cluster
            d_n_c_lines = self._d_trial["d_n_c_lines"]

            l_acc = []
            for key in d_n_lines:
                l_acc.append(d_n_c_lines.get(key, 0) / d_n_lines.get(key, 0))
            return np.average(l_acc)

    def tpl_word_accuracy_dist(self):
        # d_n_words: Number of words in a template cluster
        d_n_words = self._d_answer["d_n_words"]
        # d_n_c_words: Number of words correctly labeled in a template cluster
        d_n_c_words = self._d_trial["d_n_c_words"]

        ret = {}
        for key in d_n_words:  # key: str(tid)
            tid = int(key)
            ret[tid] = d_n_c_words.get(key, 0) / d_n_words.get(key, 0)
        return ret

    def tpl_line_accuracy_dist(self):
        # d_n_lines: Number of lines in a template cluster
        d_n_lines = self._d_answer["d_n_lines"]
        # d_n_c_lines: Number of lines correctly labeled in a template cluster
        d_n_c_lines = self._d_trial["d_n_c_lines"]

        ret = {}
        for key in d_n_lines:  # key: str(tid)
            tid = int(key)
            ret[tid] = d_n_c_lines.get(key, 0) / d_n_lines.get(key, 0)
        return ret

    def tpl_description_accuracy(self):
        iterable_tpl_answer = self.iter_tpl_answer(pass_none=True)
        iterable_tpl_trial = self.iter_tpl_trial(pass_none=True)
        l_tid_answer = self.tid_list_answer(pass_none=True)
        return structure_metrics.tpl_desc_accuracy(
            iterable_tpl_answer, iterable_tpl_trial, l_tid_answer)

    def tpl_variable_accuracy(self):
        iterable_tpl_answer = self.iter_tpl_answer(pass_none=True)
        iterable_tpl_trial = self.iter_tpl_trial(pass_none=True)
        l_tid_answer = self.tid_list_answer(pass_none=True)
        return structure_metrics.tpl_var_accuracy(
            iterable_tpl_answer, iterable_tpl_trial, l_tid_answer)

    def rand_score(self):
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return cluster_metrics.rand_score(l_tid_answer, l_tid_trial)

    def adjusted_rand_score(self):
        from sklearn.metrics import adjusted_rand_score as score
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return score(l_tid_answer, l_tid_trial)

    def f1_score(self):
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return cluster_metrics.precision_recall_fscore(
            l_tid_answer, l_tid_trial)[2]

    def parsing_accuracy(self):
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return cluster_metrics.parsing_accuracy(l_tid_answer, l_tid_trial)

    def cluster_accuracy(self):
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return cluster_metrics.cluster_accuracy(l_tid_answer, l_tid_trial)

    def overdiv_ratio(self):
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return cluster_metrics.over_division_cluster_ratio(l_tid_answer,
                                                           l_tid_trial)

    def overagg_ratio(self):
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return cluster_metrics.over_aggregation_cluster_ratio(
            l_tid_answer, l_tid_trial)

    def homogeneity_score(self):
        from sklearn.metrics import homogeneity_score as score
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return score(l_tid_answer, l_tid_trial)

    def completeness_score(self):
        from sklearn.metrics import completeness_score as score
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return score(l_tid_answer, l_tid_trial)

    def v_measure_score(self, beta=1.0):
        from sklearn.metrics import v_measure_score as score
        l_tid_answer = self.valid_tid_list_answer()
        l_tid_trial = self.valid_tid_list_trial()
        return score(l_tid_answer, l_tid_trial, beta=beta)


def measure_accuracy_answer(conf, targets, n_trial=None):
    timer = common.Timer("measure-accuracy answer", output=_logger)
    timer.start()
    mlt = MeasureLTGen(conf, n_trial)
    mlt.init_answer()

    from amulog import lt_import
    table_answer = lt_common.TemplateTable()
    ltgen_answer = lt_import.init_ltgen_import(conf, table_answer)

    for pline in amulog.manager.iter_plines(conf, targets):
        tid, _ = ltgen_answer.process_line(pline)
        if tid is None:
            tpl = None
        else:
            tpl = ltgen_answer.get_tpl(tid)
        mlt.add_org(pline["words"])
        mlt.add_answer(tid, tpl, pline["words"])
    mlt.dump_answer()
    timer.stop()
    return mlt


def measure_accuracy_trial_offline(conf, targets, n_trial=None, mlt=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial"])
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load()

    for trial_id in range(n_trial):
        timer = common.Timer("measure-accuracy-offline trial{0}".format(
            trial_id), output=_logger)
        timer.start()
        mlt.init_trial(trial_id)
        table = lt_common.TemplateTable()
        ltgen = amulog.manager.init_ltgen_methods(conf, table)

        input_lines = list(amulog.manager.iter_plines(conf, targets))
        d_plines = {mid: pline for mid, pline in enumerate(input_lines)}
        d_tid = ltgen.process_offline(d_plines)

        iterobj = zip(input_lines,
                      mlt.tid_list_answer(),
                      mlt.iter_tpl_answer())
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
            mlt.add_trial(tid_trial, tpl_trial,
                          tid_answer, tpl_answer, pline["words"])
        mlt.dump_trial()
        timer.stop()

    return mlt


def measure_accuracy_trial_online(conf, targets_train, targets_test,
                                  n_trial=None, mlt=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial"])
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load()

    from amulog import log_db
    for trial_id in range(n_trial):
        timer = common.Timer("measure-accuracy-online trial{0}".format(
            trial_id), output=_logger)
        timer.start()
        mlt.init_trial(trial_id)
        table = lt_common.TemplateTable()
        ltgen = amulog.manager.init_ltgen_methods(conf, table)

        if targets_train is not None:
            iterobj = amulog.manager.iter_plines(conf, targets_train)
            d_plines = {mid: pline for mid, pline in enumerate(iterobj)}
            ltgen.process_offline(d_plines)

        iterobj = zip(amulog.manager.iter_plines(conf, targets_test),
                      mlt.tid_list_answer(),
                      mlt.iter_tpl_answer())
        for pline, tid_answer, tpl_answer in iterobj:
            if tid_answer is None:
                tid_trial = None
                tpl_trial = None
            else:
                tid_trial, _ = ltgen.process_line(pline)
                tpl_trial = ltgen.get_tpl(tid_trial)
            mlt.add_trial(tid_trial, tpl_trial,
                          tid_answer, tpl_answer, pline["words"])
        mlt.dump_trial()
        timer.stop()

    return mlt


def get_accuracy_average(conf, n_trial, functions):
    mlt = MeasureLTGen(conf, n_trial)
    mlt.load()

    results = []
    for trial_id in range(n_trial):
        mlt.load_trial(trial_id)
        d_values = {}
        for func_name in functions:
            d_values[func_name] = getattr(mlt, func_name)()
        results.append(d_values)

    d_average = {func_name: np.average([d_values[func_name] for d_values in results])
                 for func_name in functions}
    return d_average


def get_templates(conf, n_trial, trial_id=0, answer=False, mlt=None):
    """Get template list after all log parsing.
    In online algorithms, template structure can be changed while processing.
    This function pick up a result for the last message with each template.
    """
    if mlt is None:
        mlt = MeasureLTGen(conf, n_trial)
        mlt.load(trial_id)

    if answer:
        tids = np.array(mlt.tid_list_answer())
        iterobj = mlt.iter_tpl_answer()
    else:
        tids = np.array(mlt.tid_list_trial())
        iterobj = mlt.iter_tpl_trial()

    d_last_index = defaultdict(int)
    for mid, tid in enumerate(tids):
        if tid is not None:
            d_last_index[tid] = mid
    d_last_index_rev = {mid: tid for tid, mid in d_last_index.items()}

    d_tpl = {}
    for mid, tpl in enumerate(iterobj):
        if mid in d_last_index_rev:
            tid = d_last_index_rev[mid]
            d_tpl[tid] = tpl

    return d_tpl


def offline_structure_metrics(conf, n_trial, trial_id=0, partial=False):
    mlt = MeasureLTGen(conf, n_trial)
    mlt.load(trial_id)

    d_tpl = get_templates(conf, n_trial, trial_id, mlt=mlt)
    tids = mlt.tid_list_trial(pass_none=True)
    word_acc = structure_metrics.word_accuracy(
        mlt.iter_tpl_answer(pass_none=True),
        map(lambda x: d_tpl[x], tids))
    line_acc = structure_metrics.line_accuracy(
        mlt.iter_tpl_answer(pass_none=True),
        map(lambda x: d_tpl[x], tids))
    tpl_acc = structure_metrics.tpl_accuracy(
        mlt.iter_tpl_answer(pass_none=True),
        map(lambda x: d_tpl[x], tids), tids)
    tpl_word_acc = structure_metrics.tpl_word_accuracy(
        mlt.iter_tpl_answer(pass_none=True),
        map(lambda x: d_tpl[x], tids), tids)

    if partial:
        tpl_desc_fail = structure_metrics.tpl_desc_accuracy(
            mlt.iter_tpl_answer(pass_none=True),
            map(lambda x: d_tpl[x], tids), tids)
        tpl_var_fail = structure_metrics.tpl_var_accuracy(
            mlt.iter_tpl_answer(pass_none=True),
            map(lambda x: d_tpl[x], tids), tids)
        ret = (word_acc, line_acc, tpl_acc, tpl_word_acc,
               tpl_desc_fail, tpl_var_fail)
        return ret
    else:
        return word_acc, line_acc, tpl_acc, tpl_word_acc


def search_fail_template(conf, n_trial, trial_id=0, pass_similar=True):
    mlt = MeasureLTGen(conf, n_trial)
    mlt.load(trial_id)

    s_pass = set()
    iterobj = zip(mlt.iter_org(), mlt.tid_list_answer(),
                  mlt.iter_label_answer(), mlt.iter_label_trial())
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
    mlt1.load(trial_id1)
    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load(trial_id2)

    s_pass = set()
    iterobj = zip(mlt1.iter_org(),
                  mlt1.tid_list_answer(),
                  mlt1.iter_label_answer(),
                  mlt1.iter_label_trial(),
                  mlt2.iter_label_trial())
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
    mlt.load(trial_id)

    # overdiv cluster information
    a_tid_answer = np.array(mlt.valid_tid_list_answer())
    a_tid_trial = np.array(mlt.valid_tid_list_trial())
    l_cluster = _sample_partial_cluster(a_tid_answer, a_tid_trial, n_samples)

    timer.lap("lap1")

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    timer.lap("lap2")

    # get templates for the indexes to show
    iterobj = mlt.iter_tpl_trial(pass_none=True, fill_wildcard=True)
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
    mlt.load(trial_id)

    # overagg cluster information
    a_tid_answer = np.array(mlt.valid_tid_list_answer())
    a_tid_trial = np.array(mlt.valid_tid_list_trial())
    l_cluster = _sample_partial_cluster(a_tid_trial, a_tid_answer, n_samples)

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    # get templates for the indexes to show
    iterobj = mlt.iter_tpl_trial(pass_none=True, fill_wildcard=True)
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
    mlt1.load(trial_id)
    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load(trial_id)

    # clusters accurate in conf1
    a_tid_answer = np.array(mlt1.valid_tid_list_answer())
    a_tid_trial1 = np.array(mlt1.valid_tid_list_trial())
    tids = _get_complete_clusters(a_tid_answer, a_tid_trial1)

    # cluster information that is overdiv in conf2
    a_tid_trial2 = np.array(mlt2.valid_tid_list_trial())
    l_cls_all = _sample_partial_cluster(a_tid_answer, a_tid_trial2, n_samples)
    l_cluster = [(tid_true, div) for tid_true, div
                 in l_cls_all if tid_true in tids]

    # make sample tpl list to show
    s_index_to_show = set()
    for _, div in l_cluster:
        samples = [a_index_sample for _, _, a_index_sample in div]
        s_index_to_show = s_index_to_show | set(np.ravel(samples))

    # get templates for the indexes to show
    iterobj = mlt2.iter_tpl_trial(pass_none=True, fill_wildcard=True)
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
    mlt1.load(trial_id)
    mlt2 = MeasureLTGen(conf2, n_trial)
    mlt2.load(trial_id)

    # clusters accurate in conf1
    a_tid_answer = np.array(mlt1.valid_tid_list_answer())
    a_tid_trial1 = np.array(mlt1.valid_tid_list_trial())
    tids = _get_complete_clusters(a_tid_answer, a_tid_trial1)

    # cluster information that is overagg in conf2
    a_tid_trial2 = np.array(mlt2.valid_tid_list_trial())
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
    iterobj = mlt2.iter_tpl_trial(pass_none=True, fill_wildcard=True)
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
        ltgen = amulog.manager.init_ltgen_methods(conf, table)
        if targets_train is not None:
            for pline in amulog.manager.iter_plines(conf, targets_train):
                ltgen.process_line(pline)

        timer = common.Timer("measure-time-online trial{0}".format(trial_id),
                             output=None)
        timer.start()
        for pline in amulog.manager.iter_plines(conf, targets_test):
            ltgen.process_line(pline)
        timer.stop()
        d_time[trial_id] = timer.total_time().total_seconds()

    return d_time


def measure_time_offline(conf, targets_test, n_trial=None):
    if n_trial is None:
        n_trial = int(conf["eval"]["n_trial_time"])

    d_time = {}
    for trial_id in range(n_trial):
        table = lt_common.TemplateTable()
        ltgen = amulog.manager.init_ltgen_methods(conf, table)

        timer = common.Timer("measure-time-offline trial{0}".format(trial_id),
                             output=None)
        timer.start()
        input_lines = list(amulog.manager.iter_plines(conf, targets_test))
        d_plines = {mid: pline for mid, pline in enumerate(input_lines)}
        ltgen.process_offline(d_plines)
        timer.stop()
        d_time[trial_id] = timer.total_time().total_seconds()

    return d_time
