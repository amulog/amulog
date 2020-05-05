# !/usr/bin/env python
# coding: utf-8

import logging

from . import lt_common

_logger = logging.getLogger(__package__)


class LTGroupFuzzyHash(lt_common.LTGroup):
    """Classify templates based on ssdeep, an implementation of fuzzy hashing.
    Fuzzy hashing evaluate the similarity of string data. This class compares
    strings of templates including symbols and variable replacements, and
    make groups that have large scores of fuzzy hash comparizon each other.
    
    """

    def __init__(self, lttable, th=1, mem_hash=True):
        super(LTGroupFuzzyHash, self).__init__()
        self.th = th
        self._lttable = lttable
        self._mem_hash = mem_hash
        self._d_hash = {}

    def add(self, lt_new):
        l_score = self._calc_score(lt_new)
        if len(l_score) == 0:
            gid = self._next_groupid()
        else:
            ltid, score = max(l_score, key=lambda x: x[1])
            if score >= self.th:
                gid = self._d_rgroup[ltid]
            else:
                gid = self._next_groupid()
        self.add_ltid(gid, lt_new)
        return gid

    def _calc_score(self, lt_new):
        try:
            import ssdeep
        except ImportError:
            raise ImportError(
                "ltgroup algorithm <ssdeep> needs python package ssdeep")
        ret = []
        h1 = ssdeep.hash(str(lt_new))
        if self._mem_hash:
            if len(self._d_hash) == 0:
                # initialize d_hash
                for lt in self._lttable:
                    h = ssdeep.hash(str(lt))
                    self._d_hash[lt.ltid] = h
            for ltid, lt_temp in enumerate(self._lttable):
                h2 = self._d_hash[lt_temp.ltid]
                score = ssdeep.compare(h1, h2)
                ret.append((ltid, score))
            self._d_hash[lt_new.ltid] = h1
        else:
            for lt_temp in self._lttable:
                ltid = lt_temp.ltid
                score = ssdeep.hash_score(str(lt_new), str(lt_temp))
                ret.append((ltid, score))
        return ret


def edit_distance(m1, m2):
    """Calculate Levenshtein edit distance of 2 tokenized messages.
    This function considers wildcards."""

    table = [[0] * (len(m2) + 1) for _ in range(len(m1) + 1)]

    for i in range(len(m1) + 1):
        table[i][0] = i

    for j in range(len(m2) + 1):
        table[0][j] = j

    for i in range(1, len(m1) + 1):
        for j in range(1, len(m2) + 1):
            if m1[i - 1] == m2[j - 1] or \
                    m1[i - 1] == lt_common.REPLACER or \
                    m2[j - 1] == lt_common.REPLACER:
                cost = 0
            else:
                cost = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + cost)
    return table[-1][-1]


def norm_edit_distance(m1, m2):
    return edit_distance(m1, m2) / max(len(m1), len(m2))


#def generate_lt(conf, targets, check_import=False):
#    from . import strutil
#    from . import log_db
#    lp = log_db.load_log2seq(conf)
#    table = lt_common.TemplateTable()
#    ltgen_import = lt_common.init_ltgen_methods(conf, table, "import")
#    ltgen = lt_common.init_ltgen_methods(conf, table)
#    s_tpl = set()
#
#    def _is_known(pline, ltgen_import, flag):
#        if not flag:
#            return False
#        else:
#            tid, dummy = ltgen_import.process_line(pline)
#            return tid is not None
#
#    for fp in targets:
#        _logger.info("processing {0} start".format(fp))
#        with open(fp, 'r') as f:
#            for line in f:
#                pline = lp.process_line(strutil.add_esc(line))
#                if _is_known(pline, ltgen_import, check_import):
#                    # recorded in imported definition, ignore
#                    pass
#                else:
#                    tpl = ltgen.estimate_tpl(pline["words"], pline["symbols"])
#                    s_tpl.add(tuple(tpl))
#
#        _logger.info("processing {0} done".format(fp))
#    return s_tpl
