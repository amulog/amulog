#!/usr/bin/env python
# coding: utf-8

import os
import re
import pickle
import logging
from collections import defaultdict
from importlib import import_module

from . import config
from . import strutil

REPLACER = "**"
REPLACER_HEAD = "*"
REPLACER_TAIL = "*"
REPLACER_REGEX = re.compile(r"\*[A-Z]*?\*")  # shortest match

_logger = logging.getLogger(__package__)


class LTManager(object):
    """
    A log template manager. This class define log templates from messages.
    In addition, this class update log template table on memory and DB.

    Log template generation process can be classified to following 2 types.
        Grouping : Classify messages into groups, and generate template from
                the common parts of group members.
        Templating : Estimate log template from messages by classifying words,
                and make groups that have common template.

    Attributes:
        #TODO
    
    """

    # adding lt to db (ltgen do not add)

    def __init__(self, conf, db, lttable, reset_db):
        self._reset_db = reset_db
        self.filename = conf.get("log_template", "indata_filename")

        self._db = db
        self._lttable = lttable
        self._table = TemplateTable()
        self._ltgen = None
        self._ltspl = None
        self._ltgroup = None

    def set_ltgen(self, ltgen):
        self._ltgen = ltgen

    def set_ltgroup(self, ltgroup):
        self._ltgroup = ltgroup
        if not self._reset_db:
            self._ltgroup.restore_ltg(self._db, self._lttable)

    def set_ltspl(self, ltspl):
        self._ltspl = ltspl

    def process_offline(self, plines):
        d_tid = self._ltgen.process_offline(plines)

        for mid, pline in enumerate(plines):
            l_w = pline["words"]
            l_s = pline["symbols"]
            tid = d_tid[mid]
            tpl = self._table[tid]
            ltw = self._ltspl.replace_variable(l_w, tpl, REPLACER)
            ltid = self._ltspl.search(tid, ltw)
            if ltid is None:
                ltline = self.add_lt(ltw, l_s)
                self._table.addcand(tid, ltline.ltid)
            else:
                self.count_lt(ltid)
                ltline = self._lttable[ltid]
            yield ltline

    def process_line(self, pline):

        def _lt_diff(ltid, ltw):
            d_diff = {}
            for wid, new_w, old_w in zip(range(len(ltw)), ltw,
                                         self._lttable[ltid].ltw):
                if new_w == old_w:
                    pass
                else:
                    d_diff[wid] = new_w
            return d_diff

        def _lt_repl(ltw, d_diff):
            ret = []
            for wid, w in enumerate(ltw):
                if wid in d_diff:
                    ret.append(d_diff[wid])
                else:
                    ret.append(w)
            return ret

        l_w = pline["words"]
        l_s = pline["symbols"]
        tid, state = self._ltgen.process_line(pline)
        if tid is None:
            return None

        tpl = self._table[tid]
        ltw = self._ltspl.replace_variable(l_w, tpl, REPLACER)
        if state == LTGen.state_added:
            ltline = self.add_lt(ltw, l_s)
            self._table.addcand(tid, ltline.ltid)
        else:
            ltid = self._ltspl.search(tid, ltw)
            if ltid is None:
                # tpl exists, but no lt matches
                ltline = self.add_lt(ltw, l_s)
                self._table.addcand(tid, ltline.ltid)
            else:
                if state == LTGen.state_changed:
                    # update all lt that belong to the edited tpl
                    d_diff = _lt_diff(ltid, ltw)
                    for temp_ltid in self._table.getcand(tid):
                        if temp_ltid == ltid:
                            self.replace_and_count_lt(ltid, ltw)
                        else:
                            old_ltw = self._lttable[temp_ltid]
                            new_ltw = _lt_repl(old_ltw, d_diff)
                            self.replace_lt(ltid, new_ltw)
                elif state == LTGen.state_unchanged:
                    self.count_lt(ltid)
                else:
                    raise AssertionError
                ltline = self._lttable[ltid]

        return ltline

    def add_lt(self, l_w, l_s, cnt=1):
        # add new lt to db and table
        ltid = self._lttable.next_ltid()
        ltline = LogTemplate(ltid, None, l_w, l_s, cnt)
        ltgid = self._ltgroup.add(ltline)
        ltline.ltgid = ltgid
        self._lttable.add_lt(ltline)
        self._db.add_lt(ltline)
        return ltline

    def replace_lt(self, ltid, l_w, l_s=None, cnt=None):
        self._lttable[ltid].replace(l_w, l_s, cnt)
        self._db.update_lt(ltid, l_w, l_s, cnt)

    def replace_and_count_lt(self, ltid, l_w, l_s=None):
        cnt = self._lttable[ltid].add()
        self._lttable[ltid].replace(l_w, l_s, None)
        self._db.update_lt(ltid, l_w, l_s, cnt)

    def count_lt(self, ltid):
        cnt = self._lttable[ltid].count()
        self._db.update_lt(ltid, None, None, cnt)

    def remove_lt(self, ltid):
        self._lttable.remove_lt(ltid)
        self._db.remove_lt(ltid)

    def remake_ltg(self):
        self._db.reset_ltg()
        self._ltgroup._init_dict()
        temp_lttable = self._lttable
        self._ltgroup._lttable = LTTable()

        for ltline in temp_lttable:
            ltgid = self._ltgroup.add(ltline)
            ltline.ltgid = ltgid
            self._ltgroup._lttable.add_lt(ltline)
            self._db.add_ltg(ltline.ltid, ltgid)
        assert self._ltgroup._lttable.ltdict == temp_lttable.ltdict

    def load(self):
        with open(self.filename, 'rb') as f:
            obj = pickle.load(f)
        table_data, ltgen_data, ltgroup_data = obj
        self._table.load(table_data)
        self._ltgen.load(ltgen_data)
        self._ltgroup.load(ltgroup_data)

    def dump(self):
        table_data = self._table.dumpobj()
        ltgen_data = self._ltgen.dumpobj()
        ltgroup_data = self._ltgroup.dumpobj()
        obj = (table_data, ltgen_data, ltgroup_data)
        with open(self.filename, 'wb') as f:
            pickle.dump(obj, f)


class LTTable:

    def __init__(self):
        self.ltdict = {}

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for ltid in self.ltdict:
            yield self.ltdict[ltid]

    def __len__(self):
        return len(self.ltdict)

    def __getitem__(self, key):
        assert isinstance(key, int)
        if key not in self.ltdict:
            raise IndexError("index out of range")
        return self.ltdict[key]

    def next_ltid(self):
        cnt = 0
        while cnt in self.ltdict:
            cnt += 1
        else:
            return cnt

    def restore_lt(self, ltid, ltgid, ltw, lts, count):
        assert ltid not in self.ltdict
        self.ltdict[ltid] = LogTemplate(ltid, ltgid, ltw, lts, count)

    def add_lt(self, ltline):
        assert not ltline.ltid in self.ltdict
        self.ltdict[ltline.ltid] = ltline

    def remove_lt(self, ltid):
        self.ltdict.pop(ltid)


class LogTemplate:

    def __init__(self, ltid, ltgid, ltw, lts, count):
        if len(ltw) == 0:
            raise ValueError("empty ltw, failed to generate LogTemplate")
        self.ltid = ltid
        self.ltgid = ltgid
        self.ltw = ltw
        self.lts = lts
        self.cnt = count

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

    def count(self):
        self.cnt += 1
        return self.cnt

    def replace(self, l_w, l_s=None, count=None):
        self.ltw = l_w
        if l_s is not None:
            self.lts = l_s
        if count is not None:
            self.cnt = count


class TemplateTable:
    """Temporal template table for log template generator."""

    def __init__(self):
        self._d_tpl = {}  # key = tid, val = template
        self._d_rtpl = {}  # key = key_template, val = tid
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
        l_word = [strutil.add_esc(w) for w in template]
        return "@".join(l_word)

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

    def getcand(self, tid):
        return self._d_cand[tid]

    def addcand(self, tid, ltid):
        self._d_cand[tid].append(ltid)

    def load(self, obj):
        self._d_tpl, self._d_cand = obj
        for tid, tpl in self._d_tpl.items():
            self._d_rtpl[self._key_template(tpl)] = tid

    def dumpobj(self):
        return self._d_tpl, self._d_cand


class LTGen(object):
    stateful = True
    state_added = 0
    state_changed = 1
    state_unchanged = 2

    def __init__(self, table: TemplateTable):
        self._table = table

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

    def process_offline(self, plines):
        """If there is no need of special process for init phase,
        this function simply call process_line multiple times.
        """
        self.preprocess(plines)
        d = {}
        for mid, pline in enumerate(plines):
            tid, state = self.process_line(pline)
            d[mid] = tid
        return d

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

    def load(self, loadobj):
        raise NotImplementedError

    def dumpobj(self):
        raise NotImplementedError


class LTGenStateless(LTGen):

    """Subclasses of LTGenStateless is acceptable
    for multiprocessing in offline template generation."""

    stateful = False

    def generate_tpl(self, pline):
        raise NotImplementedError

    def process_line(self, pline):
        tpl = self.generate_tpl(pline)
        return self.update_table(tpl)

    def load(self, loadobj):
        pass

    def dumpobj(self):
        return None


class LTGenJoint(LTGen):

    def __init__(self, table: TemplateTable, l_ltgen, ltgen_import_index=None):
        super().__init__(table)
        self._l_ltgen = l_ltgen
        self._import_index = ltgen_import_index

    def _update_ltmap(self, pline, index, tid, state):
        from .lt_import import LTGenImport
        if (self._import_index is not None) and \
                (not index == self._import_index):
            ltgen_import: LTGenImport = self._l_ltgen[self._import_index]
            if state == self.state_added:
                ltgen_import.add_definition(pline["words"])
            elif state == self.state_changed:
                old_tpl = self._table.get_updated()
                new_tpl = self._table.get_template(tid)
                ltgen_import.update_definition(old_tpl, new_tpl)

    def process_line(self, pline):
        for index, ltgen in enumerate(self._l_ltgen):
            tid, state = ltgen.process_line(pline)
            if tid is not None:
                self._update_ltmap(pline, index, tid, state)
                return tid, state
        else:
            msg = "Template for a message not matched/generated: {0}".format(
                pline["message"])
            _logger.debug(msg)
            return None, None

    def load(self, loadobj):
        for ltgen, ltgen_data in zip(self._l_ltgen, loadobj):
            ltgen.load(ltgen_data)

    def dumpobj(self):
        return [ltgen.dumpobj() for ltgen in self._l_ltgen]


class LTGenMultiProcess(LTGen):

    _ltgen: LTGenStateless

    def __init__(self, table: TemplateTable, n_proc, kwargs):
        super().__init__(table)
        self._n_proc = n_proc
        self._ltgen = init_ltgen(**kwargs)
        assert not self._ltgen.stateful, \
            "multiprocessing is limited to stateless methods"

        from multiprocessing import Pool
        self._pool = Pool(processes=self._n_proc)

    def load(self, _):
        pass

    def dumpobj(self):
        return None

    @staticmethod
    def _task(args):
        ltgen, message_id, pline = args
        template = ltgen.generate_tpl(pline)
        return message_id, template

    def process_line(self, _):
        raise ValueError("use process_offline")

    def process_offline(self, plines):
        l_args = [(self._ltgen, mid, pline)
                  for mid, pline in enumerate(plines)]
        try:
            d_tpl = {}
            for mid, tpl in self._pool.imap_unordered(self._task, l_args):
                d_tpl[mid] = tpl
            self._pool.close()
        except KeyboardInterrupt:
            self._pool.terminate()
            exit()
        else:
            ret = {}
            for mid, tpl in d_tpl.items():
                tid, _ = self.update_table(tpl)
                ret[mid] = tid
            return ret


class LTGroup(object):

    # usually used as super class of other ltgroup
    # If used directly, this class will work as a dummy
    # (ltgid is always same as ltid)

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

    def add(self, ltline):
        gid = ltline.ltid
        self.add_ltid(gid, ltline)
        return gid

    def add_ltid(self, gid, ltline):
        self._d_group.setdefault(gid, []).append(ltline)
        self._d_rgroup[ltline.ltid] = gid

    def restore_ltg(self, db, table):
        for ltid, ltgid in db.iter_ltg_def():
            self._d_group.setdefault(ltgid, []).append(table[ltid])
            self._d_rgroup[ltid] = ltgid

    def load(self, loadobj):
        pass

    def dumpobj(self):
        return None


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
        #self.sym_header = conf.get("log_template",
        #                           "labeled_variable_symbol_header")
        #self.sym_footer = conf.get("log_template",
        #                           "labeled_variable_symbol_footer")

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


def init_ltgen(conf, table, method, shuffle=False):
    kwargs = {"conf": conf,
              "table": table,
              "shuffle": shuffle}
    if method == "import":
        from . import lt_import
        return lt_import.init_ltgen_import(**kwargs)
    elif method == "import-ext":
        from . import lt_import_ext
        return lt_import_ext.init_ltgen_import_ext(**kwargs)
    elif method == "crf":
        from amulog.alg.crf import lt_crf
        return lt_crf.init_ltgen_crf(**kwargs)
    elif method == "re":
        from amulog import lt_regex
        return lt_regex.init_ltgen_regex(**kwargs)
    elif method == "va":
        from . import lt_va
        return lt_va.init_ltgen_va(**kwargs)
    else:
        modname = "amulog.alg." + method
        alg_module = import_module(modname)
        return alg_module.init_ltgen(**kwargs)


def init_ltgen_methods(conf, table, lt_methods=None, shuffle=False,
                       multiprocess=None):
    if lt_methods is None:
        lt_methods = config.getlist(conf, "log_template", "lt_methods")
    if multiprocess is None:
        multiprocess = conf.getboolean("log_template", "ltgen_multiprocess")
    ltgen_kwargs = {"conf": conf,
                    "table": table,
                    "shuffle": shuffle}

    if multiprocess:
        n_proc = conf["log_template"]["n_proc"].strip()
        if n_proc.isdigit():
            n_proc = int(n_proc)
        else:
            n_proc = None
        assert len(lt_methods) == 1, \
            "for multiprocessing, lt_methods should be single"
        ltgen_kwargs["method"] = lt_methods[0]
        return LTGenMultiProcess(table, n_proc, ltgen_kwargs)
    elif len(lt_methods) > 1:
        l_ltgen = []
        import_index = None
        for mid, method_name in enumerate(lt_methods):
            l_ltgen.append(init_ltgen(conf, table, method_name, shuffle))
            if method_name == "import":
                import_index = mid
        return LTGenJoint(table, l_ltgen, import_index)
    elif len(lt_methods) == 1:
        return init_ltgen(conf, table, lt_methods[0], shuffle)
    else:
        raise ValueError


def init_ltmanager(conf, db, table, reset_db):
    """Initializing ltmanager by loading argument parameters."""
    # sym = conf.get("log_template", "variable_symbol")
    ltm = LTManager(conf, db, table, reset_db)

    ltm.set_ltgen(init_ltgen_methods(conf, ltm._table))

    ltg_alg = conf.get("log_template", "ltgroup_alg")
    if ltg_alg == "shiso":
        from amulog.alg.shiso import shiso
        ltgroup = shiso.LTGroupSHISO(table,
                                     ngram_length=conf.getint(
                                            "log_template_shiso", "ltgroup_ngram_length"),
                                     th_lookup=conf.getfloat(
                                            "log_template_shiso", "ltgroup_th_lookup"),
                                     th_distance=conf.getfloat(
                                            "log_template_shiso", "ltgroup_th_distance"),
                                     mem_ngram=conf.getboolean(
                                            "log_template_shiso", "ltgroup_mem_ngram")
                                     )
    elif ltg_alg == "ssdeep":
        from . import lt_misc
        ltgroup = lt_misc.LTGroupFuzzyHash(table)
    elif ltg_alg == "none":
        ltgroup = LTGroup()
    else:
        raise ValueError("ltgroup_alg({0}) invalid".format(ltg_alg))
    ltm.set_ltgroup(ltgroup)

    post_alg = config.gettuple(conf, "log_template", "post_alg")
    ltspl = LTPostProcess(conf, ltm._table, ltm._lttable, post_alg)
    ltm.set_ltspl(ltspl)

    if os.path.exists(ltm.filename) and not reset_db:
        ltm.load()

    return ltm


def merged_template(m1, m2):
    """Return common area of log message (to be log template)"""
    ret = []
    for w1, w2 in zip(m1, m2):
        if w1 == w2:
            ret.append(w1)
        else:
            ret.append(REPLACER)
    return ret
