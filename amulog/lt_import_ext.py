#!/usr/bin/env python
# coding: utf-8

import os
import logging

from . import lt_common
from .external import tpl_match
from .external import mod_tplseq
from .external import regexhash

_logger = logging.getLogger(__package__)


class LTGenImportExternal(lt_common.LTGenStateless):

    def __init__(self, table, filename, mode, mode_esc, ltmap, head, shuffle=False):
        super(LTGenImportExternal, self).__init__(table)
        self._table = table
        self._fp = filename
        self._l_tpl = self._load_tpl(self._fp, mode, mode_esc)
        self._l_regex = [tpl_match.generate_regex(tpl)
                         for tpl in self._l_tpl]
        if ltmap == "hash":
            self._rtable = regexhash.RegexHashTable(self._l_tpl, self._l_regex,
                                                    head)
        elif ltmap == "table":
            self._rtable = regexhash.RegexTable(self._l_tpl, self._l_regex)
        else:
            raise NotImplementedError

        if shuffle:
            self._rtable.shuffle()

    @staticmethod
    def _load_tpl(fp, mode, mode_esc):
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
                if len(mes) == 0:
                    continue
                if mode_esc:
                    l_tpl.append(mes)
                else:
                    l_tpl.append(tpl_match.add_esc_external(mes))
        return l_tpl

    def generate_tpl(self, pline):
        mes = pline["message"]
        ret = self._rtable.search(mes)
        if ret is None:
            _logger.debug(
                "No log template found for message : {0}".format(mes))
            return None
        else:
            tplid, matchobj = ret
            tpl = self._l_tpl[tplid]
            new_tpl = mod_tplseq.redefine_tpl(tpl, pline, matchobj=matchobj)
            return new_tpl

    #def process_line(self, pline):
    #    mes = pline["message"]
    #    ret = self._rtable.search(mes)
    #    if ret is None:
    #        _logger.debug(
    #            "No log template found for message : {0}".format(mes))
    #        return None, None
    #    else:
    #        tplid, matchobj = ret
    #        tpl = self._rtable.l_tpl[tplid]
    #        new_tpl = mod_tplseq.redefine_tpl(tpl, pline, self.sym,
    #                                          matchobj=matchobj)

    #        if self._table.exists(new_tpl):
    #            tid = self._table.get_tid(new_tpl)
    #            return tid, self.state_unchanged
    #        else:
    #            tid = self._table.add(new_tpl)
    #            return tid, self.state_added


def init_ltgen_import_ext(conf, table, shuffle, **kwargs):
    fn = conf.get("log_template_import", "def_path_ext")
    mode = conf.get("log_template_import", "import_format_ext")
    if fn == "":
        fn = conf.get("log_template_import", "def_path")
    mode_esc = conf.getboolean("log_template_import", "import_format_ext_esc")
    ltmap = conf.get("log_template_import", "ext_search_method")
    head = conf.getint("log_template_import", "hash_strlen")

    return LTGenImportExternal(table, fn, mode, mode_esc, ltmap, head, shuffle)
