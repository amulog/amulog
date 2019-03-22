#!/usr/bin/env python
# coding: utf-8

import logging

from . import common
from . import lt_common

_logger = logging.getLogger(__package__)


class LTSearchTree():
    # Search tree for un-incremental lt generation algorithms

    def __init__(self, sym):
        self.sym = sym
        self.root = self._new_node()

    def __str__(self):
        l_buf = []

        def print_children(point, depth, l_sparent):
            word = point.word
            if word is None: word = "**"

            cnt = point.child_num()
            if cnt == 1:
                l_sparent.append(word)
            else:
                l_sparent.append(word)
                buf = "-" * (depth - len(l_sparent) + 1) + \
                        " {0}".format(" ".join(l_sparent))
                if point.end is not None:
                    buf += "  <-- ltid {0}".format(point.end)
                l_buf.append(buf)
                l_sparent = []

            for word in point.windex.keys():
                print_children(point.windex[word], depth + 1, l_sparent[:])
            if point.wild is not None:
                print_children(point.wild, depth + 1, l_sparent[:])


        point = self.root
        l_buf.append("<head of log template search tree>")
        for word in point.windex.keys():
            print_children(point.windex[word], 0, [])
        if point.wild is not None:
            print_children(point.wild, 0, [])
        return "\n".join(l_buf)

    @staticmethod
    def _new_node(parent = None, word = None):
        return LTSearchTreeNode(parent, word)

    def add(self, ltid, ltwords):
        point = self.root
        for w in ltwords:
            if w == self.sym:
                if point.wild is None:
                    point.wild = self._new_node(point, w)
                point = point.wild
            else:
                if not w in point.windex:
                    point.windex[w] = self._new_node(point, w)
                point = point.windex[w]
        else:
            point.set_ltid(ltid)

    def _trace(self, ltwords):
        _logger.debug("tracing message {0}".format(ltwords))
        point = self.root
        temp_ltwords = ltwords[:]
        check_points = []
            # points with 2 candidates to go, for example "word" and "**"
            # [(node, len(left_words)), ...]
            # use as stack
        while True:
            w = temp_ltwords.pop(0)
            if w == self.sym:
                # w is sym (only if input is template)
                # go wild node
                if point.wild is None:
                    return None
                else:
                    point = point.wild
            elif w in point.windex:
                # w is in windex
                # go the word node
                # also go to wild node, when check_points poped
                if point.wild is not None:
                    check_points.append((point, len(temp_ltwords)))
                    _logger.debug("# add checkpoint ({0} (+{1}))".format(point, len(temp_ltwords) + 1))
                _logger.debug("{0} (+{1}) -> goto {2}".format(point, len(temp_ltwords) + 1, w))
                point = point.windex[w]
            elif point.wild is not None:
                # w is not in windex, but have wild node
                _logger.debug("{0} (+{1}) -> goto {2}".format(point, len(temp_ltwords) + 1, self.sym))
                point = point.wild
            else:
                _logger.debug("{0} (+{1}) : no available children".format(point, len(temp_ltwords) + 1))
                # no template to process w, go back or end
                if len(check_points) == 0:
                    _logger.debug("template not found")
                    return None
                else:
                    p, left_wlen = check_points.pop(-1)
                    temp_ltwords = ltwords[-left_wlen:] if left_wlen > 0 else []
                        # +1 : for one **(wild) node
                    point = p.wild
                    _logger.debug("go back to a stacked branch : {0} (+{1})".format(point, len(temp_ltwords)))
                    _logger.debug("remaining words : {0}".format(temp_ltwords))

            while len(temp_ltwords) == 0:
                if point.end is None:
                    _logger.debug("{0} (+{1}) : no template in this node".format(point, len(temp_ltwords)))
                    if len(check_points) == 0:
                        _logger.debug("all ends are empty(no templates)")
                        return None
                    else:
                        p, left_wlen = check_points.pop(-1)
                        temp_ltwords = ltwords[-left_wlen:] if left_wlen > 0 else []
                        point = p.wild
                        _logger.debug("go back to a stacked branch : {0} (+{1})".format(point, len(temp_ltwords)))
                        _logger.debug("remaining words : {0}".format(temp_ltwords))
                else:
                    _logger.debug("done (tpl: {0}, id: {1})".format(point, point.end))
                    return point

    def remove(self, ltid, ltwords):
        node = self._trace(ltwords)
        if node is None:
            _logger.warning(
                    "LTSearchTree : Failed to remove ltid {0}".format(ltid))
        point.remove_ltid(ltid)
        while point.unnecessary():
            w = point.word
            point = point.parent
            if w is None:
                point.wild = None
            else:
                point.wdict.pop(w)
        else:
            if self.root is None:
                self.root = self._new_node()

    def search(self, ltwords):
        node = self._trace(ltwords)
        if node is None:
            return None
        else:
            return node.get_ltid()


class LTSearchTreeNode():

    def __init__(self, parent, word):
        self.windex = {}
        self.wild = None
        self.end = None
        self.parent = parent # for reverse search to remove
        self.word = word

    def __str__(self):
        ret = []
        p = self
        while p.parent is not None:
            if p.word is None:
                ret.append("<root>")
            else:
                ret.append(p.word)
            p = p.parent
        return " ".join(reversed(ret))

    def child(self, word = None):
        if word is None:
            # wildcard
            return self.wild
        elif word in self.windex:
            return self.windex[word]
        else:
            return None

    def child_num(self):
        cnt = len(self.windex.keys())
        if self.wild is not None:
            cnt += 1
        return cnt

    def current_point(self):
        buf = []
        point = self
        while point.parent is not None:
            buf = [point.word] + buf
            point = point.parent
        print(" ".join(buf))

    def set_ltid(self, ltid):
        self.end = ltid

    def remove_ltid(self, ltid):
        assert self.end == ltid
        self.end = None

    def get_ltid(self):
        return self.end

    def unnecessary(self):
        return (len(self.windex) == 0) and \
                (self.wild is None) and \
                (self.end is None)


class LTGroupFuzzyHash(lt_common.LTGroup):

    """Classify templates based on ssdeep, an implementation of fuzzy hashing.
    Fuzzy hashing evaluate the similarity of string data. This class compares
    strings of templates including symbols and variable replacements, and
    make groups that have large scores of fuzzy hash comparizon each other.
    
    """

    def __init__(self, lttable, th = 1, mem_hash = True):
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
            ltid, score = max(l_score, key = lambda x: x[1])
            if score >= self.th:
                gid = self.d_rgroup[ltid]
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
                score = hash_score(str(lt_new), str(lt_temp))
                ret.append((ltid, score))
        return ret


def generate_lt(conf, targets, check_import = False):
    from . import strutil
    from . import log_db
    lp = log_db._load_log2seq(conf)
    table = lt_common.TemplateTable()
    ltgen_import = lt_common.init_ltgen(conf, table, "import")
    ltgen = lt_common.init_ltgen(conf, table)
    s_tpl = set()

    def _is_known(pline, ltgen_import, flag):
        if not flag:
            return False
        else:
            tid, dummy = ltgen_import.process_line(pline)
            return tid is not None

    for fp in targets:
        _logger.info("processing {0} start".format(fp))
        with open(fp, 'r') as f:
            for line in f:
                pline = lp.process_line(strutil.add_esc(line))
                if _is_known(pline, ltgen_import, check_import):
                    # recorded in imported definition, ignore
                    pass
                else:
                    tpl = ltgen.estimate_tpl(pline["words"], pline["symbols"])
                    s_tpl.add(tuple(tpl))

        _logger.info("processing {0} done".format(fp))
    return s_tpl


