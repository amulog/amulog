#!/usr/bin/env python
# coding: utf-8

import os
import logging

from . import common
from . import lt_common
from . import lt_misc

_logger = logging.getLogger(__package__)


class LTGenImport(lt_common.LTGen):

    def __init__(self, table, filename, mode, ltmap, lp, shuffle=False):
        super(LTGenImport, self).__init__(table)
        self._table = table
        self._d_def = common.IDDict(lambda x: tuple(x))
        self._ltmap = lt_misc.init_ltsearcher(ltmap)
        self._lp = lp
        self._open_def(filename, mode, shuffle)

    def get_ltmap(self):
        return self._ltmap

    def _open_def(self, filename, mode, shuffle):
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
                        mode))
                if len(mes) == 0:
                    continue
                ltw, lts = self._lp.process_message(mes)
                self.add_tpl(ltw)
                cnt += 1
        _logger.info("{0} template imported".format(cnt))
        if shuffle:
            self._ltmap.shuffle()

    def add_tpl(self, ltw):
        defid = self._d_def.add(ltw)
        self._ltmap.add(defid, ltw)

    def update_tpl(self, old_ltw, new_ltw):
        defid = self._ltmap.search(old_ltw)
        self._ltmap.remove(defid, old_ltw)
        self._d_def.set_item(defid, new_ltw)
        self._ltmap.add(defid, new_ltw)

    def generate_tpl(self, pline):
        defid = self._ltmap.search(pline["words"])
        if defid is None:
            return None
        else:
            return self._d_def.get(defid)


def init_ltgen_import(conf, table):
    fn = conf.get("log_template_import", "def_path")
    mode = conf.get("log_template_import", "import_format")
    ltmap = conf.get("log_template_import", "search_method")
    from . import log_db
    lp = log_db.load_log2seq(conf)
    return LTGenImport(table, fn, mode, ltmap, lp)


#def search_exception(conf, targets, form):
#    table = lt_common.TemplateTable()
#    ltgen = init_ltgen_import(conf, table)
#    from . import log_db
#    lp = log_db.load_log2seq(conf)
#
#    for fp in targets:
#        _logger.info("lt_import job for ({0}) start".format(fp))
#        with open(fp, "r") as f:
#            for line in f:
#                pline = lp.process_line(line)
#                tid, dummy = ltgen.process_line(pline)
#                if tid is None:
#                    print(pline["message"])
#        _logger.info("lt_import job for ({0}) done".format(fp))
#
#
#def filter_org(conf, targets, dirname, style="date", method="commit"):
#    from . import strutil
#    table = lt_common.TemplateTable()
#    ltgen = init_ltgen_import(conf, table)
#    from . import log_db
#    lp = log_db.load_log2seq(conf)
#    rod = log_db.RestoreOriginalData(dirname, style=style, method=method)
#
#    for fp in targets:
#        _logger.info("lt_import job for ({0}) start".format(fp))
#        with open(fp, "r") as f:
#            for line in f:
#                pline = lp.process_line(strutil.add_esc(line))
#                l_w = pline["words"]
#                if l_w is None or len(l_w) == 0:
#                    continue
#                tid, dummy = ltgen.process_line(pline)
#                if tid is not None:
#                    rod.add_str(pline["timestamp"], line.rstrip("\n"))
#        rod.commit()
#        _logger.info("lt_import job for ({0}) done".format(fp))
