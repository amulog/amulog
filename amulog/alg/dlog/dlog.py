#!/usr/bin/env python
# coding: utf-8

"""
Dlog: a log template generation algorithm proposed in [1].
[1] T. Li, et al. Dlog: diagnosing router events with syslogs
    for anomaly detection. The Journal of Supercomputing, pp. 1–23, 2017.

This algorithm does not work online, because it depends
on an assumption `appearance of template words are same`
and a bunch of log messages is required to check it.
"""

from collections import Counter
import logging

from amulog import lt_common

_logger = logging.getLogger(__package__)


class Node:

    def __init__(self, word, depth):
        self.word = word
        self.depth = depth
        self.count = 0
        self.childs = {}
        self.part_of_tpl = False

    def __len__(self):
        return len(self.childs)

    def __getitem__(self, key):
        if key not in self.childs:
            raise IndexError("index out of range")
        return self.childs[key]

    def add(self, num=1):
        self.count += num


class LTGenDlog(lt_common.LTGenOffline):

    def __init__(self, table, vreobj):
        super().__init__(table)
        self._vre = vreobj
        self._root = Node(None, 0)

    def _add_line(self, primary_tpl):
        words = [w for w in primary_tpl if not w == lt_common.REPLACER]
        current = self._root
        for w in words:
            if w not in current.childs:
                current.childs[w] = Node(w, current.depth + 1)
            current.childs[w].add()
            current = current.childs[w]

    @classmethod
    def _merge_subtree(cls, current_node, new_node):
        # replace a child node of the current_node into the new one
        if current_node.depth + 1 == new_node.depth:
            child = current_node.childs[new_node.word]
            for gchild_word in child.childs:
                # merge grandchild nodes
                if gchild_word in new_node.childs:
                    new_gchild = Node(gchild_word, new_node.depth + 1)
                    cls._merge_subtree(child, new_gchild)
                    cls._merge_subtree(new_node, new_gchild)
                else:
                    new_node.childs[gchild_word] = child.childs[gchild_word]
            current_node.childs[new_node.word] = new_node
        elif current_node.depth + 1 < new_node.depth:
            for child in current_node.childs.values():
                cls._merge_subtree(child, new_node)
        else:
            raise ValueError("subtree merge failure")

    @classmethod
    def _aggregate_tree(cls, node):
        d_cnt = Counter()
        # merge counts of all child nodes
        for child in node.childs.values():
            d_cnt += cls._aggregate_tree(child)

        # merge nodes of same count value
        for (word, depth), cnt in d_cnt.items():
            if cnt == node.count and cnt > 1:
                new_node = Node(word, depth)
                new_node.add(cnt)
                node.part_of_tpl = True
                new_node.part_of_tpl = True
                cls._merge_subtree(node, new_node)

        # add counts of current node
        d_cnt += Counter({(node.word, node.depth): node.count})
        return d_cnt

    def _get_primary_tpl(self, l_w):
        primary_tpl = []
        for w in l_w:
            if self._vre.match(w):
                primary_tpl.append(lt_common.REPLACER)
            else:
                primary_tpl.append(w)
        return primary_tpl

    def _restore_tpl(self, primary_tpl):
        node = self._root
        tpl = []
        for w in primary_tpl:
            if w == lt_common.REPLACER:
                tpl.append(w)
            else:
                # if w not in node.childs:
                #     import pdb; pdb.set_trace()
                node = node.childs[w]
                tpl.append(w if node.part_of_tpl else lt_common.REPLACER)
        return tpl

    def process_offline(self, d_pline):
        # make tree
        for mid, pline in d_pline.items():
            primary_tpl = self._get_primary_tpl(pline["words"])
            self._add_line(primary_tpl)

        # aggregate tree
        self._aggregate_tree(self._root)

        # restore tpl
        ret = {}
        for mid, pline in d_pline.items():
            primary_tpl = self._get_primary_tpl(pline["words"])
            tpl = self._restore_tpl(primary_tpl)
            tid, _ = self.update_table(tpl)
            ret[mid] = tid
        return ret


def init_ltgen(conf, table, **_):
    from amulog import host_alias
    from amulog.lt_regex import VariableRegex
    preprocess_fn = conf.get("log_template_dlog", "preprocess_rule")
    ha = host_alias.init_hostalias(conf)
    vreobj = VariableRegex(preprocess_fn, ha)

    return LTGenDlog(table, vreobj)
