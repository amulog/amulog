"""
SHISO: a log template generation algorithm proposed in [1].
[1] Masayoshi Mizutani. Incremental Mining of System Log Format.
    in IEEE 10th International Conference on Services Computing, pp. 595–602, 2013.
"""

import logging
import numpy as np

from amulog import lt_common
from amulog import lt_misc

_logger = logging.getLogger(__package__)


class LTGenNode:

    def __init__(self, tid=None):
        self.l_child = []
        self.tid = tid

    def __len__(self):
        return len(self.l_child)

    def __iter__(self):
        return self.l_child.__iter__()

    def join(self, node):
        self.l_child.append(node)


class LTGenSHISO(lt_common.LTGen):

    def __init__(self, table, threshold, max_child, cfunc=None):
        super(LTGenSHISO, self).__init__(table)
        self._n_root = LTGenNode()
        self._threshold = threshold
        self._max_child = max_child
        if cfunc is None:
            self._cfunc = self.c_alphabet

    def load(self, loadobj):
        self._n_root = loadobj

    def dumpobj(self):
        return self._n_root

    def process_line(self, pline):
        l_w = pline["words"]
        n_parent = self._n_root
        while True:
            for n_child in n_parent:
                _logger.debug(
                    "comparing with tid {0}".format(n_child.tid))
                nc_tpl = self._table[n_child.tid]
                sr = self._seq_ratio(nc_tpl, l_w, self._cfunc)
                _logger.debug("seq_ratio : {0}".format(sr))

                if sr >= self._threshold:
                    # add input message into the node cluster
                    _logger.debug(
                        "merged with tid {0}".format(n_child.tid))
                    state = self.merge_tpl(l_w, n_child.tid)
                    return n_child.tid, state
            else:
                if len(n_parent) < self._max_child:
                    # add new node
                    _logger.debug("no node to be merged, add new node")
                    new_tid = self.add_tpl(l_w)
                    n_child = LTGenNode(new_tid)
                    n_parent.join(n_child)
                    return new_tid, self.state_added
                else:
                    # select child node
                    a_sim = np.array([self._sim(self.get_tpl(n_child.tid), l_w)
                                      for n_child in n_parent])
                    n_parent = n_parent.l_child[a_sim.argmin()]

    @staticmethod
    def c_original(w):
        a_cnt = np.zeros(4)  # upper_alphabet, lower_alphabet, digit, symbol
        for c in w:
            if c.isupper():
                a_cnt[0] += 1
            elif c.islower():
                a_cnt[1] += 1
            elif c.isdigit():
                a_cnt[2] += 1
            else:
                a_cnt[3] += 1
        return a_cnt / np.linalg.norm(a_cnt)

    @staticmethod
    def c_alphabet(w):
        a_cnt = np.zeros(26 + 26 + 2)  # A-Z, a-z, digit, symbol
        for c in w:
            if c.isupper():
                abc_index = ord(c) - 65
                a_cnt[abc_index] += 1.0
            elif c.islower():
                abc_index = ord(c) - 97
                a_cnt[26 + abc_index] += 1
            elif c.isdigit():
                a_cnt[-2] += 1.0
            else:
                a_cnt[-1] += 1.0
        return a_cnt / np.linalg.norm(a_cnt)

    @staticmethod
    def _seq_ratio(m1, m2, cfunc):
        if len(m1) == len(m2):
            length = len(m1)
            if length == 0:
                return 1.0

            sum_dist = 0.0
            for w1, w2 in zip(m1, m2):
                if lt_common.REPLACER in (w1, w2):
                    pass
                else:
                    sum_dist += sum(np.power(cfunc(w1) - cfunc(w2), 2))
            return 1.0 - (sum_dist / (2.0 * length))
        else:
            return 0.0

    @staticmethod
    def _sim(m1, m2):
        """Sim is different from SeqRatio, because Sim allows
        word sequence of different length. Original SHISO uses
        Levenshtein edit distance, and amulog follows that."""
        return lt_misc.edit_distance(m1, m2)

    @staticmethod
    def _equal(m1, m2):
        if not len(m1) == len(m2):
            return False
        for w1, w2 in zip(m1, m2):
            if w1 == w2 or w1 == lt_common.REPLACER or w2 == lt_common.REPLACER:
                pass
            else:
                return False
        else:
            return True


class LTGroupSHISO(lt_common.LTGroupOnline):

    def __init__(self, lttable, ngram_length=3,
                 th_lookup=0.3, th_distance=0.85, mem_ngram=True):
        super(LTGroupSHISO, self).__init__()
        self._lttable = lttable
        # self.d_group = {} # key : groupid, val : [ltline, ...]
        # self.d_rgroup = {} # key : ltid, val : groupid
        self.ngram_length = ngram_length
        self.th_lookup = th_lookup
        self.th_distance = th_distance
        self.mem_ngram = mem_ngram
        if self.mem_ngram:
            self.d_ngram = {}  # key : ltid, val : [str, ...]

    def add(self, lt_new):
        _logger.debug("group search for ltid {0}".format(lt_new.ltid))

        r_max = 0.0
        lt_max = None
        l_cnt = [0] * len(self._lttable)
        l_ng1 = self._get_ngram(lt_new)
        for ng in l_ng1:
            for lt_temp, l_ng2 in self._lookup_ngram(ng):
                l_cnt[lt_temp.ltid] += 1
                r = 2.0 * l_cnt[lt_temp.ltid] / (len(l_ng1) + len(l_ng2))
                if r > r_max:
                    r_max = r
                    lt_max = lt_temp
        _logger.debug("ngram co-occurrance map : {0}".format(l_cnt))
        _logger.debug("r_max : {0}".format(r_max))
        if r_max > self.th_lookup:
            assert lt_max is not None, "bad threshold for lt group lookup"
            _logger.debug("lt_max ltid : {0}".format(lt_max.ltid))
            d = 2.0 * lt_misc.edit_distance(lt_new.ltw, lt_max.ltw) / \
                (len(lt_new.ltw) + len(lt_max.ltw))
            _logger.debug("edit distance ratio : {0}".format(d))
            if d < self.th_distance:
                gid = self._mk_group(lt_new, lt_max)
                _logger.debug("smaller than threshold")
                _logger.debug("merge it (gid {0})".format(gid))
                _logger.debug("gid {0} : {1}".format(gid,
                                                     [ltline.ltid for ltline in self._d_group[gid]]))
                return gid
        # No merge, make group with single lt
        gid = self._mk_group_single(lt_new)
        _logger.debug("No similar format found, make group with single lt")
        _logger.debug("gid {0} : {1}".format(gid,
                                             [ltline.ltid for ltline in self._d_group[gid]]))
        return gid

    def _mk_group(self, lt_new, lt_old):
        assert lt_new.ltid not in self._d_rgroup
        if lt_old.ltid in self._d_rgroup:
            groupid = self._d_rgroup[lt_old.ltid]
            self.add_ltid(groupid, lt_new)
        else:
            groupid = self._next_groupid()
            self.add_ltid(groupid, lt_old)
            self.add_ltid(groupid, lt_new)
        return groupid

    def _mk_group_single(self, lt_new):
        assert lt_new.ltid not in self._d_rgroup
        groupid = self._next_groupid()
        self.add_ltid(groupid, lt_new)
        return groupid

    def _get_ngram(self, ltline):

        def ngram(ltw, length):
            return [ltw[i:i + length] for i in range(len(ltw) - length)]

        if self.mem_ngram:
            if ltline.ltid not in self.d_ngram:
                self.d_ngram[ltline.ltid] = \
                    ngram(ltline.ltw, self.ngram_length)
            return self.d_ngram[ltline.ltid]
        else:
            return ngram(ltline.ltw, self.ngram_length)

    def _lookup_ngram(self, ng):
        ret = []
        for ltline in self._lttable:
            if ng in self._get_ngram(ltline):
                ret.append((ltline, ng))
        return ret

    def load(self, obj):
        pass

    def dumpobj(self):
        return None


def init_ltgen(conf, table, **_):
    threshold = conf.getfloat("log_template_shiso", "ltgen_threshold")
    max_child = conf.getint("log_template_shiso", "ltgen_max_child")
    return LTGenSHISO(table, threshold, max_child)
