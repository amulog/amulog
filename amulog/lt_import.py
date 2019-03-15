#!/usr/bin/env python
# coding: utf-8

import os
import logging

from . import common
from . import lt_common
from . import lt_misc

_logger = logging.getLogger(__package__)


class LTGenImport(lt_common.LTGen):

    def __init__(self, table, sym, filename, mode, lp):
        super(LTGenImport, self).__init__(table, sym)
        self._table = table
        self._d_def = common.IDDict(lambda x: tuple(x))
        self.searchtree = lt_misc.LTSearchTree(sym)
        self._lp = lp
        self._open_def(filename, mode)

    def _open_def(self, filename, mode):
        cnt = 0
        if not os.path.exists(filename):
            errmes = ("log_template_import.def_path"
                      " {0} is invalid".format(filename))
            raise ValueError(errmes)
        with open(filename, 'r') as f:
            for line in f:
                if mode == "plain":
                    mes = line.rstrip("\n")
                elif mode == "ids":
                    line = line.rstrip("\n")
                    mes = line.partition(" ")[2].strip()
                else:
                    raise ValueError("invalid import_mode {0}".format(
                            self.mode))
                ltw, lts = self._lp.process_message(mes)
                defid = self._d_def.add(ltw)
                self.searchtree.add(defid, ltw)
                cnt += 1
        _logger.info("{0} template imported".format(cnt))

    def process_line(self, pline):
        l_w = pline["words"]
        defid = self.searchtree.search(l_w)
        if defid is None:
            _logger.debug(
                    "No log template found for message : {0}".format(l_w))
            return None, None
        else:
            tpl = self._d_def.get(defid)
            if self._table.exists(tpl):
                tid = self._table.get_tid(tpl)
                return tid, self.state_unchanged
            else:
                tid = self._table.add(tpl)
                return tid, self.state_added


def init_ltgen_import(conf, table, sym):
    fn = conf.get("log_template_import", "def_path")
    mode = conf.get("log_template_import", "mode")
    from . import log_db
    lp = log_db._load_log2seq(conf)
    return LTGenImport(table, sym, fn, mode, lp)


def search_exception(conf, targets, form):
    table = lt_common.TemplateTable()
    sym = conf.get("log_template", "variable_symbol")
    ltgen = init_ltgen_import(conf, table, sym)
    from . import log_db
    lp = log_db._load_log2seq(conf)

    for fp in targets:
        _logger.info("lt_import job for ({0}) start".format(fn))
        with open(fp, "r") as f:
            for line in f:
                pline = lp.process_line(line)
                tid, dummy = ltgen.process_line(pline)
                if tid is None:
                    print(pline["message"])
        _logger.info("lt_import job for ({0}) done".format(fn))


def filter_org(conf, targets, dirname, style = "date", method = "commit"):
    from . import logparser
    from . import log_db
    from . import strutil
    table = lt_common.TemplateTable()
    sym = conf.get("log_template", "variable_symbol")
    ltgen = init_ltgen_import(conf, table, sym)
    from . import log_db
    lp = log_db._load_log2seq(conf)
    rod = log_db.RestoreOriginalData(dirname, style = style, method = method)

    for fp in targets:
        _logger.info("lt_import job for ({0}) start".format(fn))
        with open(fp, "r") as f:
            for line in f:
                pline = lp.process_line(strutil.add_esc(line))
                l_w = pline["words"]
                if l_w is None or len(l_w) == 0:
                    continue
                tid, dummy = ltgen.process_line(pline)
                if tid is not None:
                    rod.add_str(pline["timestamp"], line.rstrip("\n"))
        rod.commit()
        _logger.info("lt_import job for ({0}) done".format(fn))


