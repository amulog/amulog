#!/usr/bin/env python
# coding: utf-8

"""
A log template generation algorithm proposed in [1].
[1] P. He, et al. Drain: An Online Log Parsing Approach with Fixed Depth Tree. ICWS 2017, pp.33–40, 2017.

After editing log templates manually (with lt_tool),
Drain do not work correctly.
Do NOT edit log templates manually if you still have unprocessed log data.
"""

import os
from amulog import lt_common

DEFAULT_REGEX_CONFIG = "/".join((os.path.dirname(os.path.abspath(__file__)),
                                 "../../data/drain_regex.conf"))


class Node:

    def __init__(self):
        self.child = {}
        self.clusters = None  # set of tid


class LTGenDrain(lt_common.LTGen):
    """
    Args:
        table (LogTemplateTable): see lt_common.LTGen
        threshold (int): Threshold value used in 2nd step (clustering).
        depth (int): Tree depth used in 1st part (preceding tokens).
            (depth - 2) top words except known variables
            are always considered NOT variable.
        vreobj (lt_regex.VariableRegex): Regex-based variable classifier.
            Mainly used in 1st step (preceding tokens).
    """

    def __init__(self, table, threshold, depth, vreobj,
                 count_replacer=False):
        super().__init__(table)
        self._threshold = threshold
        self._depth = depth
        self._vre = vreobj
        self._count_replacer = count_replacer
        self._root = Node()

    def load(self, loadobj):
        self._root = loadobj

    def dumpobj(self):
        return self._root

    def process_line(self, pline):
        # preprocess
        tokens = [lt_common.REPLACER if self._vre.match(w) else w
                  for w in pline["words"]]

        # search by length
        length = len(tokens)
        if length not in self._root.child:
            self._root.child[length] = Node()
        node = self._root.child[length]

        # search by preceding tokens
        ptokens = [w for w in tokens if w != lt_common.REPLACER]
        for i in range(self._depth - 2):
            try:
                key = ptokens[i]
            except IndexError:
                key = None
            if key not in node.child:
                node.child[key] = Node()
            node = node.child[key]

        # search by token similarlity
        max_sim_seq = 0
        tid = None
        if node.clusters is None:
            node.clusters = set()
        for tmp_tid in node.clusters:
            tpl = self._table[tmp_tid]
            sim_seq = sum(1 for t, w in zip(tokens, tpl) if t == w) / length
            if sim_seq >= self._threshold and sim_seq > max_sim_seq:
                max_sim_seq = sim_seq
                tid = tmp_tid

        # update clusters
        if tid is None:
            tid = self.add_tpl(tokens)
            node.clusters.add(tid)
            state = self.state_added
        else:
            state = self.merge_tpl(tokens, tid)

        return tid, state


def init_ltgen(conf, table, **_):
    threshold = conf.getfloat("log_template_drain", "threshold")
    depth = conf.getint("log_template_drain", "depth")

    from amulog.lt_regex import VariableRegex
    preprocess_fn = conf.get("log_template_drain", "preprocess_rule")
    if preprocess_fn.strip() == "":
        preprocess_fn = DEFAULT_REGEX_CONFIG
    vreobj = VariableRegex(conf, preprocess_fn)

    return LTGenDrain(table, threshold, depth, vreobj)


def get_param_candidates():
    import numpy as np
    from itertools import product
    params = []
    for th, depth in product(np.arange(0, 1.1, 0.1), (3, 4)):
        params.append({"threshold": th,
                       "depth": depth})
    return params


def init_ltgen_with_params(conf, table, params, **_):
    threshold = params["threshold"]
    depth = params["depth"]

    from amulog.lt_regex import VariableRegex
    preprocess_fn = conf.get("log_template_drain", "preprocess_rule")
    if preprocess_fn.strip() == "":
        preprocess_fn = DEFAULT_REGEX_CONFIG
    vreobj = VariableRegex(conf, preprocess_fn)

    return LTGenDrain(table, threshold, depth, vreobj)
