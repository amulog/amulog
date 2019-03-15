#!/usr/bin/env python
# coding: utf-8

"""
A log template generation algorithm proposed in [1].
[1] Masayoshi Mizutani. Incremental Mining of System Log Format.
in IEEE 10th International Conference on Services Computing, pp. 595–602, 2013.

After editing log templates manually (with lt_tool),
SHISO do not work correctly.
Do NOT edit log templates manually if you still have unprocessed log data.
"""

import sys
import logging
import numpy

from . import config
from . import lt_common

_logger = logging.getLogger(__package__)


class LTGenNode():

    def __init__(self, tid = None):
        self.l_child = []
        self.tid = tid

    def __len__(self):
        return len(self.l_child)

    def __iter__(self):
        return self.l_child.__iter__()

    def join(self, node):
        self.l_child.append(node)


class LTGenSHISO(lt_common.LTGen):

    def __init__(self, table, sym, threshold, max_child):
        super(LTGenSHISO, self).__init__(table, sym)
        self._n_root = LTGenNode()
        self.threshold = threshold
        self.max_child = max_child

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
                sr = self.seq_ratio(nc_tpl, l_w)
                _logger.debug("seq_ratio : {0}".format(sr))
                if sr >= self.threshold:
                    _logger.debug(
                            "merged with tid {0}".format(n_child.tid))
                    state = self.update_table(l_w, n_child.tid, False)
                    return n_child.tid, state
                else:
                    if self.equal(nc_tpl, l_w):
                        _logger.warning(
                            "comparing same line, but seqratio is small...")
            else:
                if len(n_parent) < self.max_child:
                    _logger.debug("no node to be merged, add new node")
                    n_child = LTGenNode(self._table.next_tid())
                    n_parent.join(n_child)
                    state = self.update_table(l_w, n_child.tid, True)
                    return n_child.tid, state
                else:
                    _logger.debug("children : {0}".format(
                            [e.tid for e in n_parent.l_child]))
                    l_sim = [(edit_distance(self._table[n_child.tid], l_w,
                            self._sym), n_child) for n_child in n_parent]
                    n_parent = max(l_sim, key=lambda x: x[0])[1]
                    _logger.debug("go down to node(tid {0})".format(
                            n_parent.tid))

    def seq_ratio(self, m1, m2):

        #def c_coordinate(w):
        #    # retrun vector of characters of word
        #    # 4 demension : upper-case, lower-case, digit, symbol(others)
        #    l_cnt = [0.0 for i in range(4)]
        #    for c in w:
        #        if c.islower():
        #            l_cnt[0] += 1.0
        #        elif c.isupper():
        #            l_cnt[1] += 1.0
        #        elif c.isdigit():
        #            l_cnt[2] += 1.0
        #        else:
        #            l_cnt[3] += 1.0
        #    deno = numpy.linalg.norm(l_cnt)
        #    return [1.0 * e / deno for e in l_cnt]

        def c_coordinate(w):
            l_cnt = [0.0 for i in range(26 + 26 + 2)] # A-Z, a-z, digit, symbol
            for c in w:
                if c.isupper():
                    ind = ord(c) - 65
                    l_cnt[ind] += 1.0
                elif c.islower():
                    ind = ord(c) - 97
                    l_cnt[ind + 26] += 1.0
                elif c.isdigit():
                    l_cnt[-2] += 1.0
                else:
                    l_cnt[-1] += 1.0
            deno = numpy.linalg.norm(l_cnt)
            return [1.0 * e / deno for e in l_cnt]

        if len(m1) == len(m2):
            length = len(m1)
            if length == 0:
                return 1.0

            sum_dist = 0.0
            for w1, w2 in zip(m1, m2):
                if w1 == self._sym or w2 == self._sym:
                    pass
                else:
                    c_w1 = c_coordinate(w1)
                    c_w2 = c_coordinate(w2)
                    dist = sum([numpy.power(e1 - e2, 2)
                            for e1, e2 in zip(c_w1, c_w2)])
                    sum_dist += dist
            return 1.0 - (sum_dist / (2.0 * length))
        else:
            return 0.0

    def equal(self, m1, m2):
        if len(m1) == len(m2):
            for w1, w2 in zip(m1, m2):
                if w1 == w2 or w1 == self._sym or w2 == self._sym:
                    pass
                else:
                    return False
            else:
                return True
        else:
            return False


class LTGroupSHISO(lt_common.LTGroup):

    def __init__(self, lttable, ngram_length = 3,
            th_lookup = 0.3, th_distance = 0.85, mem_ngram = True):
        super(LTGroupSHISO, self).__init__()
        self._lttable = lttable
        self._sym = lttable.sym
        #self.d_group = {} # key : groupid, val : [ltline, ...]
        #self.d_rgroup = {} # key : ltid, val : groupid
        self.ngram_length = ngram_length
        self.th_lookup = th_lookup
        self.th_distance = th_distance
        self.mem_ngram = mem_ngram
        if self.mem_ngram:
            self.d_ngram = {} # key : ltid, val : [str, ...]

    def add(self, lt_new):
        _logger.debug("group search for ltid {0}".format(lt_new.ltid))

        r_max = 0.0
        lt_max = None
        l_cnt = [0 for i in self._lttable]
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
            ltw2 = lt_max.ltw
            d = 2.0 * edit_distance(lt_new.ltw, lt_max.ltw, self._sym) / \
                    (len(lt_new.ltw) + len(lt_max.ltw))
            _logger.debug("edit distance ratio : {0}".format(d))
            if d < self.th_distance:
                gid = self._mk_group(lt_new, lt_max)
                _logger.debug("smaller than threshold")
                _logger.debug("merge it (gid {0})".format(gid))
                _logger.debug("gid {0} : {1}".format(gid, \
                        [ltline.ltid for ltline in self.d_group[gid]]))
                return gid
        # No merge, make group with single lt
        gid = self._mk_group_single(lt_new)
        _logger.debug("No similar format found, make group with single lt")
        _logger.debug("gid {0} : {1}".format(gid, \
                [ltline.ltid for ltline in self.d_group[gid]]))
        return gid

    def _mk_group(self, lt_new, lt_old):
        assert not lt_new.ltid in self.d_rgroup
        if lt_old.ltid in self.d_rgroup:
            groupid = self.d_rgroup[lt_old.ltid]
            self.add_ltid(groupid, lt_new)
        else:
            groupid = self._next_groupid()
            self.add_ltid(groupid, lt_old)
            self.add_ltid(groupid, lt_new)
        return groupid

    def _mk_group_single(self, lt_new):
        assert not lt_new.ltid in self.d_rgroup
        groupid = self._next_groupid()
        self.add_ltid(groupid, lt_new)
        return groupid

    def _get_ngram(self, ltline):

        def ngram(ltw, length):
            return [ltw[i:i+length] for i in range(len(ltw) - length)]
            
        if self.mem_ngram:
            if not ltline.ltid in self.d_ngram:
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


def init_ltgen_shiso(conf, table, sym):
    threshold = conf.getfloat("log_template_shiso", "ltgen_threshold")
    max_child = conf.getint("log_template_shiso", "ltgen_max_child")
    return LTGenSHISO(table, sym, threshold, max_child)


def edit_distance(m1, m2, sym):
    # return levenshtein distance that allows wildcard

    table = [ [0] * (len(m2) + 1) for i in range(len(m1) + 1) ]

    for i in range(len(m1) + 1):
        table[i][0] = i

    for j in range(len(m2) + 1):
        table[0][j] = j

    for i in range(1, len(m1) + 1):
        for j in range(1, len(m2) + 1):
            if (m1[i - 1] == m2[j - 1]) or \
                    m1[i - 1] == sym or m2[j - 1] == sym:
                cost = 0
            else:
                cost = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][ j - 1] + 1, \
                    table[i - 1][j - 1] + cost)
    return table[-1][-1]


#def test_ltgen(conf):
#    ltm = LTManager(conf)
#    if conf.getboolean("general", "src_recur"):
#        l_fp = common.recur_dir(config.getlist(conf, "general", "src_path"))
#    else:
#        l_fp = common.rep_dir(config.getlist(conf, "general", "src_path"))
#    ltm.process_dataset(conf, l_fp)
#
#
#if __name__ == "__main__":
#    #logger_super = logging.getLogger("lt_common")
#    #ch = logging.StreamHandler()
#    #ch.setLevel(logging.DEBUG)
#    #logger_super.setLevel(logging.DEBUG)
#    #logger_super.addHandler(ch)
#    #_logger.setLevel(logging.DEBUG)
#    #_logger.addHandler(ch)
#    #test_make()
#
#    usage = "usage: {0} [options]".format(sys.argv[0])
#    op = optparse.OptionParser(usage)
#    op.add_option("-c", "--config", action="store",
#            dest="conf", type="string", default=config.DEFAULT_CONFIG_NAME,
#            help="configuration file path")
#    options, args = op.parse_args()
#
#    conf = config.open_config(options.conf)
#    test_ltgen(conf)

