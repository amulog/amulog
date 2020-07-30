#!/usr/bin/env python
# coding: utf-8


from collections import defaultdict


class RegexTable(object):

    def __init__(self, l_tpl, l_regex):
        assert len(l_tpl) == len(l_regex)
        self._table = self._make_table(l_tpl, l_regex)

    def _make_table(self, l_tpl, l_regex):
        return [(tplid, reobj) for tplid, reobj in enumerate(l_regex)]

    def search(self, mes):
        for tplid, reobj in self._table:
            matchobj = reobj.match(mes)
            if matchobj:
                return tplid, matchobj
        else:
            return None

    def shuffle(self):
        import random
        random.shuffle(self._table)


class RegexHashTable(RegexTable):

    def __init__(self, l_tpl, l_regex, headlen=5):
        assert isinstance(headlen, int)
        assert headlen > 0
        self._headlen = headlen
        self._table = None
        super().__init__(l_tpl, l_regex)

    def _make_table(self, l_tpl, l_regex):
        table = defaultdict(list)
        for tplid, (tpl, reobj) in enumerate(zip(l_tpl, l_regex)):
            if self._head_isstable(tpl, self._headlen):
                key = tpl[:self._headlen]
            else:
                key = ""
            table[key].append((tplid, reobj))
        return table

    @staticmethod
    def _head_isstable(tpl, headlen):
        from amulog import lt_common
        replacer = lt_common.REPLACER_REGEX
        matchobj = replacer.search(tpl)
        if matchobj:
            return matchobj.start() >= headlen
        else:
            return True

    def search(self, mes):
        key = mes[:self._headlen]
        if key not in self._table:
            key = ""

        for tplid, reobj in self._table[key]:
            matchobj = reobj.match(mes)
            if matchobj:
                return tplid, matchobj
        for tplid, reobj in self._table[""]:
            matchobj = reobj.match(mes)
            if matchobj:
                return tplid, matchobj
        return None

    def shuffle(self):
        import random
        for key, subtable in self._table.items():
            random.shuffle(subtable)
