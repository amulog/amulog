#!/usr/bin/env python
# coding: utf-8

"""
FT-tree: a log template generation algorithm proposed in [1].
[1] S. Zhang, et al. Syslog Processing for Switch Failure Diagnosis and
    Prediction in Datacenter Networks. In IEEE/ACM 25th International
    Symposium on Quality of Service (IWQoS), pp. 1–10, 2017.

see also https://github.com/WeibinMeng/ft-tree
"""


import logging
from collections import defaultdict

from amulog import lt_common

_logger = logging.getLogger(__package__)


class Node:

    def __init__(self, word, depth):
        self.word = word
        self.depth = depth
        self.childs = {}

    def __len__(self):
        return len(self.childs)


class LTGenFTTree(lt_common.LTGen):

    def __init__(self, table, max_child=6, cut_depth=3, message_type_func=None):
        super().__init__(table)
        self._max_child = max_child
        self._cut_depth = cut_depth
        self._d_words = defaultdict(int)
        self._tree = {}

        if message_type_func is None:
            self._type_func = self.message_type_none
        else:
            self._type_func = message_type_func

    def load(self, loadobj):
        self._tree, self._d_words = loadobj

    def dumpobj(self):
        return self._tree, self._d_words

    @staticmethod
    def message_type_none(words):
        return None, words

    @staticmethod
    def message_type_top(words):
        return words[0], words[1:]

    @staticmethod
    def message_type_length(words):
        return str(len(words)), words

    def _add_line(self, pline):
        list_words = [(w, self._d_words[w]) for w in pline["words"]]
        message_type, list_words = self._type_func(list_words)
        list_words.sort(key=lambda x: (x[1], x[0]), reverse=True)

        if message_type not in self._tree:
            self._tree[message_type] = Node(message_type, 0)
        current = self._tree[message_type]

        s_tpl = set()
        for w, _ in list_words:
            if current.childs is None:
                break
            elif w not in current.childs:
                current.childs[w] = Node(w, current.depth + 1)
                # pruning
                if len(current.childs) >= self._max_child and \
                        current.depth >= self._cut_depth:
                    current.childs = None
                    break
            s_tpl.add(w)
            current = current.childs[w]

        tpl = [w if w in s_tpl else lt_common.REPLACER
               for w in pline["words"]]
        return tpl

    def process_offline(self, d_pline):
        # make word dictionary
        for pline in d_pline.values():
            for w in pline["words"]:
                self._d_words[w] += 1

        # make tree
        ret = {}
        for mid, pline in d_pline.items():
            tpl = self._add_line(pline)
            ret[mid] = self.update_table(tpl)
        return ret

    def process_line(self, pline):
        # update word dictionary
        for w in pline["words"]:
            self._d_words[w] += 1

        # update tree
        tpl = self._add_line(pline)
        return self.update_table(tpl)

    def generate_tpl(self, pline):
        tid, _ = self.process_line(pline)
        return self._table[tid]


def init_ltgen(conf, table, **_):
    max_child = conf.getint("log_template_fttree", "max_child")
    cut_depth = conf.getint("log_template_fttree", "cut_depth")
    type_func_name = conf.get("log_template_fttree", "type_func")
    if type_func_name == "top":
        message_type_func = LTGenFTTree.message_type_top
    elif type_func_name == "length":
        message_type_func = LTGenFTTree.message_type_length
    else:
        message_type_func = None
    return LTGenFTTree(table, max_child, cut_depth, message_type_func)


def get_param_candidates():
    from itertools import product
    params = []
    for th, n_cnt in product((2, 3, 6, 12, 24), (2, 3, 4)):
        params.append({"max_child": th,
                       "cut_depth": n_cnt})
    return params


def init_ltgen_with_params(conf, table, params, **_):
    max_child = params["max_child"]
    cut_depth = params["cut_depth"]
    type_func_name = conf.get("log_template_fttree", "type_func")
    if type_func_name == "top":
        message_type_func = LTGenFTTree.message_type_top
    elif type_func_name == "length":
        message_type_func = LTGenFTTree.message_type_length
    else:
        message_type_func = None
    return LTGenFTTree(table, max_child, cut_depth, message_type_func)

