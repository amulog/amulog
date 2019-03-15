#!/usr/bin/env python
# coding: utf-8

import os
import logging

from . import common
from . import lt_common
from . import lt_misc
from .external import tpl_match
from .external import mod_tplseq
from .external import regexhash

_logger = logging.getLogger(__package__)


class LTGenImportExternal(lt_common.LTGen):

    def __init__(self, table, sym, filename, mode, lp, head):
        super(LTGenImportExternal, self).__init__(table, sym)
        self._table = table
        self._fp = filename
        self._l_tpl = self._load_tpl(self._fp, mode)
        self._l_regex = [tpl_match.generate_regex(tpl)
                         for tpl in self._l_tpl]
        self._rtable = regexhash.RegexHashTable(self._l_tpl, self._l_regex,
                                                head)

    @staticmethod
    def _load_tpl(fp, mode):
        l_tpl = []
        if not os.path.exists(fp):
            errmes = ("log_template_import.def_path"
                      " {0} is invalid".format(fp))
            raise ValueError(errmes)
        with open(fp, 'r') as f:
            for line in f:
                if mode == "plain":
                    mes = line.rstrip("\n")
                elif mode == "ids":
                    line = line.rstrip("\n")
                    mes = line.partition(" ")[2].strip()
                else:
                    raise ValueError("invalid import_mode {0}".format(mode))
                l_tpl.append(mes)
        return l_tpl

    def process_line(self, pline):
        mes = pline["message"]
        ret = self._rtable.search(mes)
        if ret is None:
            _logger.debug(
                    "No log template found for message : {0}".format(mes))
            return None, None
        else:
            tplid, matchobj = ret
            tpl = self._rtable.l_tpl[tplid]
            new_tpl = mod_tplseq.redefine_tpl(tpl, pline, self.sym,
                                              matchobj = matchobj)

            if self._table.exists(new_tpl):
                tid = self._table.get_tid(new_tpl)
                return tid, self.state_unchanged
            else:
                tid = self._table.add(new_tpl)
                return tid, self.state_added


def init_ltgen_import_ext(conf, table, sym):
    fn = conf.get("log_template_import", "def_path")
    mode = conf.get("log_template_import", "mode")
    head = conf.getint("log_template_import", "hash_strlen")

    from . import log_db
    lp = log_db._load_log2seq(conf)
    return LTGenImportExternal(table, sym, fn, mode, lp, head)


