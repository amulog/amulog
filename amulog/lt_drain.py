#!/usr/bin/env python
# coding: utf-8


"""
A log template generation algorithm proposed in [1].
[1] P. He, et al. Drain: An Online Log Parsing Approach with Fixed Depth Tree. ICWS 2017, pp.33–40, 2017.

After editing log templates manually (with lt_tool),
Drain do not work correctly.
Do NOT edit log templates manually if you still have unprocessed log data.
"""

from . import lt_common


class Node:

    def __init__(self):
        self.child = {}
        self.clusters = None  # set of tid


class LTGenDrain(lt_common.LTGen):

    def __init__(self, table, threshold, depth, vreobj):
        super(LTGenDrain, self).__init__(table)
        self._threshold = threshold
        self._depth = depth
        self._vre = vreobj
        self._root = Node()

    def generate_tpl(self, pline):
        tid, _ = self.process_line(pline)
        return self._table[tid]

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
        for i in range(self._depth - 2):
            key = tokens[i]
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
            tid = self._table.next_tid()
            state = self.update_table(tokens, tid, True)
            node.clusters.add(tid)
        else:
            state = self.update_table(tokens, tid, False)

        return tid, state


def init_ltgen_drain(conf, table, **kwargs):
    threshold = conf.getfloat("log_template_drain", "threshold")
    depth = conf.getint("log_template_drain", "depth")

    from . import lt_regex
    preprocess_fn = conf.get("log_template_drain", "preprocess_rule")
    vreobj = lt_regex.VariableRegex(conf, preprocess_fn)

    return LTGenDrain(table, threshold, depth, vreobj)
