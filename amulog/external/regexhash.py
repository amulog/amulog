#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict

from . import tpl_match


class RegexHashTable():

    def __init__(self, l_tpl, l_regex, headlen = 5):
        assert len(l_tpl) == len(l_regex)
        assert isinstance(headlen, int)
        self.headlen = headlen
        self.l_tpl = l_tpl
        self.table = self._make_table(l_tpl, l_regex)

    def _make_table(self, l_tpl, l_regex):
        table = defaultdict(list)
        for tplid, (tpl, reobj) in enumerate(zip(l_tpl, l_regex)):
            if self._head_isstable(tpl, self.headlen):
                key = tpl[:self.headlen]
            else:
                key = None
            table[key].append((tplid, reobj))
        return table

    @staticmethod
    def _head_isstable(tpl, headlen):
        replacer = tpl_match.REPLACER
        matchobj = replacer.search(tpl)
        if matchobj:
            return matchobj.start() >= headlen
        else:
            return True

    def search(self, mes):
        key = mes[:self.headlen]
        if not key in self.table:
            key = None

        for tplid, reobj in self.table[key]:
            matchobj = reobj.match(mes)
            if matchobj:
                return tplid, matchobj
        else:
            return None


