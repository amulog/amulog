#!/usr/bin/env python
# coding: utf-8

import os
import re
import pickle
from collections import defaultdict

from . import common
from . import config
from . import strutil

REPLACER = "**"
REPLACER_REGEX = re.compile(r"\*[A-Z]*?\*") # shortest match


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

    def __init__(self, conf, db, lttable, reset_db, lt_alg, ltg_alg, post_alg):
        self._reset_db = reset_db
        self.sym = conf.get("log_template", "variable_symbol")
        self.filename = conf.get("log_template", "indata_filename")
        self._fail_fn = conf.get("log_template", "fail_output")
        self.pickle_comp = conf.get("general", "pickle_compatible")

        self._db = db
        self._lttable = lttable
        self._table = TemplateTable()
        self.ltgen = None
        self.ltspl = None
        self.ltgroup = None

    def _set_ltgen(self, ltgen):
        self.ltgen = ltgen

    def _set_ltgroup(self, ltgroup):
        self.ltgroup = ltgroup
        if not self._reset_db:
            self.ltgroup.restore_ltg(self._db, self._lttable)

    def _set_ltspl(self, ltspl):
        self.ltspl = ltspl

    def process_init_data(self, plines):
        d = self.ltgen.process_init_data(plines)
        for mid, pline in enumerate(plines):
            l_w = pline["words"]
            l_s = pline["symbols"]
            tid = d[mid]
            tpl = self._table[tid]
            ltw = self.ltspl.replace_variable(l_w, tpl, self.sym)
            ltid = self.ltspl.search(tid, ltw)
            if ltid is None:
                ltline = self.add_lt(ltw, l_s)
                self._table.addcand(tid, ltline.ltid)
            else:
                self.count_lt(ltid)
                ltline = self._lttable[ltid]
            yield ltline

    def process_line(self, pline):

        def lt_diff(ltid, ltw):
            d_diff = {}
            for wid, new_w, old_w in zip(range(len(ltw)), ltw,
                    self._lttable[ltid].ltw):
                if new_w == old_w:
                    pass
                else:
                    d_diff[wid] = new_w
            return d_diff

        def lt_repl(ltw, d_diff):
            ret = []
            for wid, w in enumerate(ltw):
                if wid in d_diff:
                    ret.append(d_diff[wid])
                else:
                    ret.append(w)
            return ret

        l_w = pline["words"]
        l_s = pline["symbols"]
        tid, state = self.ltgen.process_line(pline)
        if tid is None:
            return None

        tpl = self._table[tid]
        ltw = self.ltspl.replace_variable(l_w, tpl, self.sym)
        if state == LTGen.state_added:
            ltline = self.add_lt(ltw, l_s)
            self._table.addcand(tid, ltline.ltid)
        else:
            ltid = self.ltspl.search(tid, ltw)
            if ltid is None:
                # tpl exists, but no lt matches
                ltline = self.add_lt(ltw, l_s)
                self._table.addcand(tid, ltline.ltid)
            else:
                if state == LTGen.state_changed:
                    # update all lt that belong to the edited tpl
                    d_diff = lt_diff(ltid, ltw)
                    for temp_ltid in self._table.getcand(tid):
                        if temp_ltid == ltid:
                            self.replace_and_count_lt(ltid, ltw)
                        else:
                            old_ltw = self._lttable[temp_ltid]
                            new_ltw = lt_repl(old_ltw, d_diff)
                            self.replace_lt(ltid, new_ltw)
                elif state == LTGen.state_unchanged:
                    self.count_lt(ltid)
                else:
                    raise AssertionError
                ltline = self._lttable[ltid]
    
        return ltline

    def add_lt(self, l_w, l_s, cnt = 1):
        # add new lt to db and table
        ltid = self._lttable.next_ltid()
        ltline = LogTemplate(ltid, None, l_w, l_s, cnt, self.sym)
        ltgid = self.ltgroup.add(ltline)
        ltline.ltgid = ltgid
        self._lttable.add_lt(ltline)
        self._db.add_lt(ltline)
        return ltline

    def replace_lt(self, ltid, l_w, l_s = None, cnt = None):
        self._lttable[ltid].replace(l_w, l_s, cnt)
        self._db.update_lt(ltid, l_w, l_s, cnt)
    
    def replace_and_count_lt(self, ltid, l_w, l_s = None):
        cnt = self._lttable[ltid].count()
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
        self.ltgroup.init_dict()
        temp_lttable = self._lttable
        self.ltgroup._lttable = LTTable(self.sym)

        for ltline in temp_lttable:
            ltgid = self.ltgroup.add(ltline)
            ltline.ltgid = ltgid
            self.ltgroup._lttable.add_lt(ltline)
            self._db.add_ltg(ltline.ltid, ltgid)
        assert self.ltgroup._lttable.ltdict == temp_lttable.ltdict

    def failure_output(self, line):
        with open(self._fail_fn, "a") as f:
            line = line.rstrip("\n")
            f.write(line + "\n")

    def load(self):
        kwargs = common.pickle_comp_args(self.pickle_comp)
        with open(self.filename, 'rb') as f:
            obj = pickle.load(f, **kwargs)
        table_data, ltgen_data, ltgroup_data = obj
        self._table.load(table_data)
        self.ltgen.load(ltgen_data)
        self.ltgroup.load(ltgroup_data)

    def dump(self):
        #kwargs = common.pickle_comp_args(self.pickle_comp)
        table_data = self._table.dumpobj()
        ltgen_data = self.ltgen.dumpobj()
        ltgroup_data = self.ltgroup.dumpobj()
        obj = (table_data, ltgen_data, ltgroup_data)
        with open(self.filename, 'wb') as f:
            pickle.dump(obj, f)
            #pickle.dump(obj, f, **kwargs)


class LTTable():

    def __init__(self, sym):
        self.ltdict = {}
        self.sym = sym

    def __iter__(self):
        return self._generator()

    def _generator(self):
        for ltid in self.ltdict:
            yield self.ltdict[ltid]

    def __len__(self):
        return len(self.ltdict)

    def __getitem__(self, key):
        assert isinstance(key, int)
        if not key in self.ltdict:
            raise IndexError("index out of range")
        return self.ltdict[key]
    
    def next_ltid(self):
        cnt = 0
        while cnt in self.ltdict:
            cnt += 1 
        else:
            return cnt
    
    def restore_lt(self, ltid, ltgid, ltw, lts, count):
        assert not ltid in self.ltdict
        self.ltdict[ltid] = LogTemplate(ltid, ltgid, ltw, lts, count, self.sym)

    def add_lt(self, ltline):
        assert not ltline.ltid in self.ltdict
        self.ltdict[ltline.ltid] = ltline

    def remove_lt(self, ltid):
        self.ltdict.pop(ltid)


class LogTemplate():

    def __init__(self, ltid, ltgid, ltw, lts, count, sym):
        if len(ltw) == 0:
            raise ValueError("empty ltw, failed to generate LogTemplate")
        self.ltid = ltid
        self.ltgid = ltgid
        self.ltw = ltw
        self.lts = lts
        self.cnt = count
        self.sym = sym

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
            return [self.sym for w in self.ltw if w == self.sym]
        else:
            return [w_org for w_org, w_lt in zip(l_w, self.ltw)
                    if w_lt == self.sym]

    def var_location(self):
        return [i for i, w_lt in enumerate(self.ltw) if w_lt == self.sym]

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

    def replace(self, l_w, l_s = None, count = None):
        self.ltw = l_w
        if l_s is not None:
            self.lts = l_s
        if count is not None:
            self.cnt = count


class TemplateTable():
    """Temporal template table for log template generator."""

    def __init__(self):
        self._d_tpl = {} # key = tid, val = template
        self._d_rtpl = {} # key = key_template, val = tid
        self._d_cand = defaultdict(list) # key = tid, val = List[ltid]

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
        if not key in self._d_tpl:
            raise IndexError("index out of range")
        return self._d_tpl[key]

    def next_tid(self):
        cnt = 0
        while cnt in self._d_tpl:
            cnt += 1 
        else:
            return cnt

    def tids(self):
        return self._d_tpl.keys()

    def _key_template(self, template):
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
        self._d_tpl[tid] = template
        self._d_rtpl[self._key_template(template)] = tid

    def getcand(self, tid):
        return self._d_cand[tid]

    def addcand(self, tid, ltid):
        self._d_cand[tid].append(ltid)

    def load(self, obj):
        self._d_tpl, self._d_cand = obj
        for tid, tpl in self._d_tpl.items():
            self._d_rtpl[self._key_template(tpl)] = tid

    def dumpobj(self):
        return (self._d_tpl, self._d_cand)


class LTGen(object):

    state_added = 0
    state_changed = 1
    state_unchanged = 2

    def __init__(self, table, sym):
        self._table = table
        self._sym = sym

    def update_table(self, l_w, tid, added_flag):
        if added_flag:
            new_tid = self._table.add(l_w)
            assert new_tid == tid
            return self.state_added
        else:
            old_tpl = self._table[tid]
            new_tpl = merge_lt(old_tpl, l_w, self._sym)
            if old_tpl == new_tpl:
                return self.state_unchanged
            else:
                self._table.replace(tid, new_tpl)
                return self.state_changed

    def process_init_data(self, plines):
        """If there is no need of special process for init phase,
        this function simply call process_line multiple times.
        """
        d = {}
        for mid, pline in enumerate(plines):
            tid, state = self.process_line(pline)
            d[mid] = tid
        return d

    def process_line(self, pline):
        """Estimate log template for given message.
        This method works in incremental processing phase.

        Args:
            l_w (List[str])
            l_s (List[str])

        Returns:
            tid (int): A template id in TemplateTable.
            state (int)
        """
        raise NotImplementedError

    def load(self, loadobj):
        pass

    def dumpobj(self):
        return None


class LTGroup(object):

    # usually used as super class of other ltgroup
    # If used directly, this class will work as a dummy
    # (ltgid is always same as ltid)

    def __init__(self):
        self.init_dict()

    def init_dict(self):
        self.d_group = {} # key : groupid, val : [ltline, ...]
        self.d_rgroup = {} # key : ltid, val : groupid

    def _next_groupid(self):
        cnt = 0
        while cnt in self.d_group:
            cnt += 1
        else:
            return cnt

    def add(self, ltline):
        gid = ltline.ltid
        self.add_ltid(gid, ltline)
        return gid

    def add_ltid(self, gid, ltline):
        self.d_group.setdefault(gid, []).append(ltline)
        self.d_rgroup[ltline.ltid] = gid

    def restore_ltg(self, db, table):
        for ltid, ltgid in db.iter_ltg_def():
            self.d_group.setdefault(ltgid, []).append(table[ltid])
            self.d_rgroup[ltid] = ltgid

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
        self.sym_header = conf.get("log_template",
                "labeled_variable_symbol_header")
        self.sym_footer = conf.get("log_template",
                "labeled_variable_symbol_footer")

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

        #if len(self._table.getcand(tid)) == 0:
        #    return None
        #else:
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
        from . import host_alias
        self.ha = host_alias.init_hostalias(conf)

    def replace_word(self, w):
        return self.ha.get_group(w)


def init_ltgen(conf, table, method = None):
    if method is None:
        lt_alg = conf.get("log_template", "lt_alg")
    else:
        lt_alg = method
    sym = conf.get("log_template", "variable_symbol")
    args = [conf, table, sym]

    if lt_alg == "shiso":
        from . import lt_shiso
        ltgen = lt_shiso.init_ltgen_shiso(*args)
        pass
    elif lt_alg == "import":
        from . import lt_import
        ltgen = lt_import.init_ltgen_import(*args)
    elif lt_alg == "import-ext":
        from . import lt_import_ext
        ltgen = lt_import_ext.init_ltgen_import_ext(*args)
    elif lt_alg == "crf":
        from . import lt_crf
        ltgen = lt_crf.init_ltgen_crf(*args)
    elif lt_alg == "va":
        from . import lt_va
        ltgen = lt_va.init_ltgen_va(*args)
    else:
        raise ValueError("lt_alg({0}) invalid".format(lt_alg))
    return ltgen


def init_ltmanager(conf, db, table, reset_db):
    """Initializing ltmanager by loading argument parameters."""
    lt_alg = conf.get("log_template", "lt_alg")
    ltg_alg = conf.get("log_template", "ltgroup_alg")
    post_alg = config.gettuple(conf, "log_template", "post_alg")
    sym = conf.get("log_template", "variable_symbol")
    ltm = LTManager(conf, db, table, reset_db,
            lt_alg, ltg_alg, post_alg)

    ltgen = init_ltgen(conf, ltm._table)
    ltm._set_ltgen(ltgen)

    if ltg_alg == "shiso":
        from . import lt_shiso
        ltgroup = lt_shiso.LTGroupSHISO(table,
                ngram_length = conf.getint(
                    "log_template_shiso", "ltgroup_ngram_length"),
                th_lookup = conf.getfloat(
                    "log_template_shiso", "ltgroup_th_lookup"),
                th_distance = conf.getfloat(
                    "log_template_shiso", "ltgroup_th_distance"),
                mem_ngram = conf.getboolean(
                    "log_template_shiso", "ltgroup_mem_ngram")
                )
    elif ltg_alg == "ssdeep":
        from . import lt_misc
        ltgroup = lt_misc.LTGroupFuzzyHash(table)
    elif ltg_alg == "none":
        ltgroup = LTGroup()
    else:
        raise ValueError("ltgroup_alg({0}) invalid".format(ltg_alg))
    ltm._set_ltgroup(ltgroup)

    ltspl = LTPostProcess(conf, ltm._table, ltm._lttable, post_alg)
    ltm._set_ltspl(ltspl)

    if os.path.exists(ltm.filename) and not reset_db:
        ltm.load()

    return ltm


def merge_lt(m1, m2, sym):
    """Return common area of log message (to be log template)"""
    ret = []
    for w1, w2 in zip(m1, m2):
        if w1 == w2:
            ret.append(w1)
        else:
            ret.append(sym)
    return ret


