#!/usr/bin/env python
# coding: utf-8

"""
LenMa: Length Matters Clustering, proposed in [1].
K. Shima. Length Matters: Clustering System Log Messages using Length of Words. arXiv, 2016.

This code is based on https://github.com/keiichishima/templateminer .
"""

import numpy as np
from collections import defaultdict

from amulog import lt_common


class Cluster:

    def __init__(self, words):
        self._words = words
        self._nwords = words
        self._wordlens = np.array([len(w) for w in words])

    def _get_similarity_score_cosine(self, new_words):
        # cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        wordlens = self._wordlens.reshape(1, -1)
        new_wordlens = np.asarray([len(w) for w in new_words]).reshape(1, -1)
        cos_score = cosine_similarity(wordlens, new_wordlens)
        return cos_score

    def get_similarity_score(self, new_words, n_same_count=3, use_head_rule=True):
        # heuristic judge: the first word (process name) must be equal
        if use_head_rule and not self._words[0] == new_words[0]:
            return 0.0
        cnt_wildcard = sum(1 for w in self._words if w == lt_common.REPLACER)
        cnt_same = sum(1 for w1, w2 in zip(self._nwords, new_words)
                       if w1 == w2)
        # check exact match
        if cnt_same + cnt_wildcard == self._nwords:
            return 1.0
        # heuristics?
        if cnt_same < n_same_count:
            return 0.0

        return self._get_similarity_score_cosine(new_words)

    def update(self, new_words):
        self._wordlens = np.array([len(w) for w in new_words])
        self._words = [w1 if w1 == w2 else lt_common.REPLACER
                       for w1, w2 in zip(self._words, new_words)]


class LTGenLenMa(lt_common.LTGen):

    def __init__(self, table, threshold=0.9, n_same_count=3, use_head_rule=True):
        super().__init__(table)
        self._threshold = threshold
        self._n_same_count = n_same_count
        self._use_head_rule = use_head_rule
        self._clusters = {}  # key: tid, val: Cluster
        self._d_candidates = defaultdict(list)  # key: length, val: list of tid

    def load(self, loadobj):
        self._clusters, self._d_candidates = loadobj

    def dumpobj(self):
        return self._clusters, self._d_candidates

    def process_line(self, pline):
        words = pline["words"]
        nwords = len(words)

        candidates = []
        for tid in self._d_candidates[nwords]:
            cluster = self._clusters[tid]
            score = cluster.get_similarity_score(words, self._n_same_count,
                                                 self._use_head_rule)
            if score >= self._threshold:
                candidates.append((tid, score))

        if len(candidates) > 0:
            tid = max(candidates, key=lambda x: x[1])[0]
            state = self.merge_tpl(words, tid)
            return tid, state
        else:
            tid = self.add_tpl(words)
            self._clusters[tid] = Cluster(words)
            self._d_candidates[nwords].append(tid)
            return tid, self.state_added


def init_ltgen(conf, table, **_):
    threshold = conf.getfloat("log_template_lenma", "threshold")
    n_same_count = conf.getint("log_template_lenma", "n_same_count")
    use_head_rule = conf.getboolean("log_template_lenma", "use_head_rule")

    return LTGenLenMa(table, threshold=threshold, n_same_count=n_same_count,
                      use_head_rule=use_head_rule)


def get_param_candidates():
    from itertools import product
    params = []
    for th, n_cnt in product((0., 0.5, 0.7, 0.8, 0.9, 0.95, 0.99), (2, 3, 4)):
        params.append({"threshold": th,
                       "n_same_count": n_cnt})
    return params


def init_ltgen_with_params(conf, table, params, **_):
    threshold = params["threshold"]
    n_same_count = params["n_same_count"]
    use_head_rule = conf.getboolean("log_template_lenma", "use_head_rule")

    return LTGenLenMa(table, threshold=threshold, n_same_count=n_same_count,
                      use_head_rule=use_head_rule)
