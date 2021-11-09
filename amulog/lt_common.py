#!/usr/bin/env python
# coding: utf-8

import re
import logging
from collections import defaultdict
from abc import ABC, abstractmethod

from . import strutil

REPLACER = "**"
REPLACER_HEAD = "*"
REPLACER_TAIL = "*"
REPLACER_REGEX = re.compile(r"\*[A-Z]*?\*")  # shortest match
ANONYMIZED_DESC = "##"

_logger = logging.getLogger(__package__)


class LTTable:

    def __init__(self):
        self._ltdict = {}

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for ltid in self._ltdict:
            yield self._ltdict[ltid]

    def __len__(self):
        return len(self._ltdict)

    def __getitem__(self, key):
        assert isinstance(key, int)
        if key not in self._ltdict:
            raise IndexError("index out of range")
        return self._ltdict[key]

    def next_ltid(self):
        cnt = 0
        while cnt in self._ltdict:
            cnt += 1
        else:
            return cnt

    def restore_lt(self, ltid, ltgid, ltw, lts, count):
        assert ltid not in self._ltdict
        self._ltdict[ltid] = LogTemplate(ltid, ltgid, ltw, lts, count)

    def add_lt(self, ltline):
        assert ltline.ltid not in self._ltdict
        self._ltdict[ltline.ltid] = ltline

    def update_lt(self, ltobj):
        assert ltobj.ltid in self._ltdict
        self._ltdict[ltobj.ltid] = ltobj

    def remove_lt(self, ltid):
        self._ltdict.pop(ltid)


class LogTemplate:

    def __init__(self, ltid, ltgid, ltw, lts, count):
        if len(ltw) == 0:
            raise ValueError("empty ltw, failed to generate LogTemplate")
        self.ltid = ltid
        self.ltgid = ltgid
        self.ltw = ltw
        self.lts = lts
        self.count = count

    def __iter__(self):
        return self.ltw

    def __str__(self):
        return self.restore_message(self.ltw)

    def get(self, key):
        if key == "ltid":
            return self.ltid
        elif key == "ltgid":
            return self.ltgid
        else:
            raise KeyError

    def desc(self):
        return [w for w in self.ltw if w != REPLACER]

    def var(self, l_w):
        if len(l_w) == 0:
            return [REPLACER for w in self.ltw if w == REPLACER]
        else:
            return [w_org for w_org, w_lt in zip(l_w, self.ltw)
                    if w_lt == REPLACER]

    def var_location(self):
        return [i for i, w_lt in enumerate(self.ltw) if w_lt == REPLACER]

    def restore_message(self, l_w, esc=False):
        if l_w is None or len(l_w) == 0:
            l_w = self.ltw
        if esc:
            l_w = [w for w in l_w]
        else:
            l_w = [strutil.restore_esc(w) for w in l_w]

        if self.lts is None:
            return "".join(l_w)
        else:
            return "".join([s + w for w, s in zip(l_w + [""], self.lts)])

    def increment(self):
        self.count += 1
        return self.count

    def replace(self, l_w, l_s=None, count=None):
        self.ltw = l_w
        if l_s is not None:
            self.lts = l_s
        if count is not None:
            self.count = count


class TemplateTable:
    """Temporal template table for log template generator."""

    def __init__(self):
        self._d_tpl = {}  # key = tid, val = template
        self._d_rtpl = {}  # key = key_template, val = tid
        self._d_ltid = {}  # key = tid, val = ltid
        self._d_cand = defaultdict(list)  # key = tid, val = List[ltid]
        self._last_modified = None  # used for LTGenJoint

    def __str__(self):
        ret = []
        for tid, tpl in self._d_tpl.items():
            ret.append(" ".join([str(tid)] + tpl))
        return "\n".join(ret)

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for tid in self._d_tpl:
            yield self._d_tpl[tid]

    def __getitem__(self, key):
        assert isinstance(key, int)
        if key not in self._d_tpl:
            raise IndexError("index out of range")
        return self._d_tpl[key]

    def __len__(self):
        return len(self._d_tpl)

    def next_tid(self):
        cnt = 0
        while cnt in self._d_tpl:
            cnt += 1
        else:
            return cnt

    def tids(self):
        return self._d_tpl.keys()

    @staticmethod
    def _key_template(template):
        # l_word = [strutil.add_esc(w) for w in template]
        # return "@".join(l_word)
        return tuple(template)

    def exists(self, template):
        key = self._key_template(template)
        return key in self._d_rtpl

    def get_tid(self, template):
        key = self._key_template(template)
        return self._d_rtpl[key]

    def get_template(self, tid):
        return self._d_tpl[tid]

    def add(self, template):
        tid = self.next_tid()
        self._d_tpl[tid] = template
        self._d_rtpl[self._key_template(template)] = tid
        return tid

    def replace(self, tid, template):
        self._last_modified = self._d_tpl[tid]
        self._d_tpl[tid] = template
        self._d_rtpl[self._key_template(template)] = tid

    def get_updated(self):
        return self._last_modified

    def add_ltid(self, tid, ltid):
        self._d_ltid[tid] = ltid

    def get_ltid(self, tid):
        return self._d_ltid[tid]

#    def getcand(self, tid):
#        return self._d_cand[tid]
#
#    def addcand(self, tid, ltid):
#        self._d_cand[tid].append(ltid)

    def load(self, obj):
        self._d_tpl, self._d_cand = obj
        for tid, tpl in self._d_tpl.items():
            self._d_rtpl[self._key_template(tpl)] = tid

    def dumpobj(self):
        return self._d_tpl, self._d_cand


class LTGen(ABC):
    state_added = 0
    state_changed = 1
    state_unchanged = 2

    def __init__(self, table):
        if table is None:
            self._table = TemplateTable()
        else:
            self._table = table

    def is_stateful(self):
        return True

    def get_tpl(self, tid):
        return self._table[tid]

    def add_tpl(self, ltw, tid=None):
        if tid is None:
            tid = self._table.add(ltw)
        return tid

    def update_tpl(self, ltw, tid):
        self._table.replace(tid, ltw)

    def merge_tpl(self, l_w, tid):
        old_tpl = self._table[tid]
        new_tpl = merged_template(old_tpl, l_w)
        if old_tpl == new_tpl:
            return self.state_unchanged
        else:
            self.update_tpl(new_tpl, tid)
            return self.state_changed

    def update_table(self, tpl):
        if tpl is None:
            return None, None
        elif self._table.exists(tpl):
            tid = self._table.get_tid(tpl)
            return tid, self.state_unchanged
        else:
            tid = self._table.add(tpl)
            return tid, self.state_added

    def preprocess(self, plines):
        """This function do the pre-process for given data if needed.
        This function is called by process_init_data."""
        pass

    def process_offline(self, d_pline):
        """If there is no need of special process for init phase,
        this function simply call process_line multiple times.
        """
        self.preprocess(d_pline.values())
        d = {}
        for mid, pline in d_pline.items():
            tid, state = self.process_line(pline)
            d[mid] = tid
        return d

    @abstractmethod
    def process_line(self, pline):
        """Estimate log template for given message.
        This method works in incremental processing phase.

        Args:
            pline (dict): parsed log message with log2seq

        Returns:
            tid (int): A template id in TemplateTable.
            state (int)
        """
        raise NotImplementedError

    def postprocess(self, plines):
        """This function do the post-process for given data if needed.
        This function is called by process_init_data."""
        pass

    @abstractmethod
    def load(self, loadobj):
        raise NotImplementedError

    @abstractmethod
    def dumpobj(self):
        raise NotImplementedError


class LTGenOffline(LTGen, ABC):

    @abstractmethod
    def process_offline(self, plines):
        raise NotImplementedError

    def process_line(self, pline):
        msg = "offline LTGen does not support incremental processing"
        raise RuntimeError(msg)

    def load(self, _):
        # offline LTGen does not support suspension and restart
        pass

    def dumpobj(self):
        # offline LTGen does not support suspension and restart
        return None


class LTGenStateless(LTGen, ABC):

    """Subclasses of LTGenStateless is acceptable
    for multiprocessing in offline template generation."""

    @abstractmethod
    def generate_tpl(self, pline):
        raise NotImplementedError

    def is_stateful(self):
        return False

    def process_line(self, pline):
        tpl = self.generate_tpl(pline)
        return self.update_table(tpl)

    def load(self, loadobj):
        # stateless
        pass

    def dumpobj(self):
        # stateless
        return None


class LTGenJoint(LTGen):

    def __init__(self, table: TemplateTable, l_ltgen, ltgen_import_index=None):
        super().__init__(table)
        self._l_ltgen = l_ltgen
        self._import_index = ltgen_import_index

        class_names = [c.__class__.__name__ for c in self._l_ltgen]
        _logger.warning("LTGenJoint init with {0}".format(class_names))

    def is_stateful(self):
        return any([ltgen.is_stateful() for ltgen in self._l_ltgen])

    def _update_ltmap(self, pline, index, tid, state):
        from .lt_import import LTGenImport
        if (self._import_index is not None) and \
                (index != self._import_index):
            ltgen_import: LTGenImport = self._l_ltgen[self._import_index]
            new_tpl = self._table.get_template(tid)
            if state == self.state_added:
                ltgen_import.add_definition(new_tpl)
                msg = "LTGenJoint new template: {0}".format(new_tpl)
                _logger.debug(msg)
            elif state == self.state_changed:
                old_tpl = self._table.get_updated()
                ltgen_import.update_definition(old_tpl, new_tpl)
                msg = "LTGenJoint template update: {0} -> {1}".format(
                    old_tpl, new_tpl)
                _logger.debug(msg)

    def process_line(self, pline):
        _logger.debug("LTGenJoint input: {0}".format(pline["words"]))
        for index, ltgen in enumerate(self._l_ltgen):
            tid, state = ltgen.process_line(pline)
            if tid is not None:
                msg = ("LTGenJoint: method {0} ".format(index) +
                       "successfully generate a template")
                _logger.debug(msg)
                self._update_ltmap(pline, index, tid, state)
                return tid, state
        else:
            msg = "Template not matched/generated: {0}".format(pline["words"])
            _logger.debug(msg)
            return None, None

    def load(self, loadobj):
        for ltgen, ltgen_data in zip(self._l_ltgen, loadobj):
            ltgen.load(ltgen_data)

    def dumpobj(self):
        return [ltgen.dumpobj() for ltgen in self._l_ltgen]


#class LTGenMultiProcess(LTGenOffline):
#
#    _ltgen: LTGenStateless
#
#    def __init__(self, table: TemplateTable, n_proc, kwargs):
#        super().__init__(table)
#        self._n_proc = n_proc
##        self._ltgen = init_ltgen(**kwargs)
##        assert not self._ltgen.is_stateful(), \
##            "multiprocessing is limited to stateless methods"
#        assert(n_proc < 20)
#
#        from multiprocessing import Pool
#        self._pool = Pool(processes=self._n_proc,
#                          initializer=self._pool_init, initargs=(kwargs,))
#
#    @staticmethod
#    def _pool_init(ltgen_kwargs):
#        global _LTGEN_MP_LOCAL
#        _LTGEN_MP_LOCAL = init_ltgen(**ltgen_kwargs)
#        assert not _LTGEN_MP_LOCAL.is_stateful(), \
#            "multiprocessing is limited to stateless methods"
#
#    @staticmethod
#    def _pool_task(args):
#        ret = []
#        for message_id, pline in args:
#            template = _LTGEN_MP_LOCAL.generate_tpl(pline)
#            ret.append((message_id, template))
#        return ret
#
#        #message_id, pline = args
#        #template = _LTGEN_MP_LOCAL.generate_tpl(pline)
#        #return message_id, template
#
#    def _process_offline_pool(self, plines):
#        l_tmp_args = [(mid, pline)
#                      for mid, pline in enumerate(plines)]
#        l_args = np.array_split(l_tmp_args, self._n_proc * 10)
#        try:
#            d_tpl = {}
#            for ret in self._pool.imap_unordered(self._pool_task, l_args):
#                for mid, tpl in ret:
#                    d_tpl[mid] = tpl
#            self._pool.close()
#        except KeyboardInterrupt:
#            self._pool.terminate()
#            exit()
#        else:
#            ret = {}
#            for mid, tpl in d_tpl.items():
#                tid, _ = self.update_table(tpl)
#                ret[mid] = tid
#            return ret
#
#    def process_offline(self, plines):
#        return self._process_offline_pool(plines)


class LTGroup(ABC):

    def __init__(self):
        self._init_dict()

    def _init_dict(self):
        self._d_group = {}  # key : groupid, val : [ltline, ...]
        self._d_rgroup = {}  # key : ltid, val : groupid

    def _next_groupid(self):
        cnt = 0
        while cnt in self._d_group:
            cnt += 1
        else:
            return cnt

    @abstractmethod
    def make(self) -> LTTable:
        raise NotImplementedError

    def add_lt(self, gid, ltline):
        self._d_group.setdefault(gid, []).append(ltline)
        self._d_rgroup[ltline.ltid] = gid

    def restore_ltg(self, db, table):
        for ltid, ltgid in db.iter_ltg_def():
            self._d_group.setdefault(ltgid, []).append(table[ltid])
            self._d_rgroup[ltid] = ltgid

    def update_lttable(self, lttable):
        for ltid, ltgid in self._d_rgroup.items():
            lttable[ltid].ltgid = ltgid
        return lttable

    def load(self, loadobj):
        pass

    def dumpobj(self):
        return None


class LTGroupOnline(LTGroup, ABC):

    def __init__(self, lttable):
        super().__init__()
        self.lttable = lttable

    @abstractmethod
    def add(self, ltline):
        """Returns ltgid"""
        raise NotImplementedError

    def make(self):
        self.remake_all()
        return self.lttable

    def remake_all(self):
        self._init_dict()
        for ltline in self.lttable:
            ltgid = self.add(ltline)
            ltline.ltgid = ltgid


class LTGroupDummy(LTGroupOnline):
    # This class will work as a dummy
    # (ltgid is always same as ltid)

    def add(self, ltline):
        gid = ltline.ltid
        self.add_lt(gid, ltline)
        return gid


class LTGroupOffline(LTGroup, ABC):

    def __init__(self, lttable):
        super().__init__()
        self.lttable = lttable
        self._n_groups = 0

    @abstractmethod
    def make(self) -> LTTable:
        raise NotImplementedError

    @property
    def n_groups(self):
        return self._n_groups


class LTPostProcess(object):

    def __init__(self, conf, table, lttable, l_alg):
        self._table = table
        self._lttable = lttable
        self._rules = []
        for alg in l_alg:
            if alg == "dummy":
                self._rules.append(VariableLabelRule())
            elif alg == "host":
                self._rules.append(VariableLabelHost(conf))
            else:
                raise NotImplementedError

        self.sym_header = REPLACER_HEAD
        self.sym_footer = REPLACER_TAIL

    def _labeled_variable(self, w):
        return "".join((self.sym_header, w, self.sym_footer))

    def replace_variable(self, l_w, tpl, sym):
        ret = []
        for org_w, tpl_w in zip(l_w, tpl):
            if tpl_w == sym:
                for r in self._rules:
                    ww = r.replace_word(org_w)
                    if ww is not None:
                        ret.append(self._labeled_variable(ww))
                        break
                else:
                    ret.append(tpl_w)
            else:
                ret.append(tpl_w)
        return ret

    def search(self, tid, ltw):
        """Search existing candidates of template derivation. Return None
        if no possible candidates found."""
        l_ltid = self._table.getcand(tid)
        for ltid in l_ltid:
            if self._lttable[ltid].ltw == ltw:
                return ltid
        else:
            return None

        # if len(self._table.getcand(tid)) == 0:
        #    return None
        # else:
        #    return self._table.getcand(tid)[0]


class VariableLabelRule(object):

    def __init__(self):
        pass

    def replace_word(self, w):
        """str: If the given word is a member of some host group,
        return the group name. Otherwise, return None."""
        return None


class VariableLabelHost(VariableLabelRule):

    def __init__(self, conf):
        super().__init__()
        from . import host_alias
        self.ha = host_alias.init_hostalias(conf)

    def replace_word(self, w):
        return self.ha.get_group(w)


def merged_template(m1, m2):
    """Return common area of log message (to be log template)"""
    ret = []
    for w1, w2 in zip(m1, m2):
        if w1 == w2:
            ret.append(w1)
        else:
            ret.append(REPLACER)
    return ret


def template_from_messages(l_lm):
    """Generate a log template as the common part of given instances.

    Args:
        l_lm (List[log_db.LogMessage]): Log instances

    Returns:
        tpl (List[str])
    """
    tpl = []
    for words in zip(*[lm.l_w for lm in l_lm]):
        s_words = set(words)
        s_words.discard(REPLACER)
        if len(s_words) == 1:
            tpl.append(words[0])
        else:
            tpl.append(REPLACER)
    return tpl
