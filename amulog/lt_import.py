#!/usr/bin/env python
# coding: utf-8

import os
import logging

import amulog.manager
from amulog import common
from amulog import strutil
from amulog import lt_common
from amulog import lt_search

_logger = logging.getLogger(__package__)


class LTGenImport(lt_common.LTGenStateless):

    def __init__(self, table, filename, mode, ltmap, lp, shuffle=False):
        super(LTGenImport, self).__init__(table)
        self._table = table
        self._d_def = common.IDDict(self._keyfunc)
        self._ltmap = lt_search.init_searcher(ltmap)
        self._lp = lp
        self._open_def(filename, mode, shuffle)

    @staticmethod
    def _keyfunc(x):
        return tuple(x)

    def _open_def(self, filename, mode, shuffle):
        if filename.strip() == "":
            errmes = "lt_import initialized with empty template set"
            _logger.info(errmes)
            return
        elif not os.path.exists(filename):
            errmes = ("log_template_import.def_path {0} is invalid".format(
                filename))
            raise IOError(errmes)

        cnt = 0
        with open(filename, 'r') as f:
            for line in f:
                if mode == "plain":
                    mes = line.rstrip("\n")
                elif mode == "manual":
                    pline = self._lp.process_message(strutil.add_esc(line))
                    mes = pline["words"]
                else:
                    raise ValueError("invalid import_mode {0}".format(
                        mode))
                if len(mes) == 0:
                    continue
                ltw, lts = self._lp.process_statement(mes)
                self.add_definition(ltw)
                cnt += 1
        _logger.info("LTGenImport: {0} template imported".format(cnt))
        if shuffle:
            self._ltmap.shuffle()

    def add_definition(self, ltw):
        defid = self._d_def.add(ltw)
        self._ltmap.add(defid, ltw)
        _logger.debug("defid {0}: {1}".format(defid, ltw))

    def update_definition(self, old_ltw, new_ltw):
        defid = self._ltmap.remove(old_ltw)
        _logger.debug("defid remove {0}: {1}".format(defid, old_ltw))
        self._d_def.set_item(defid, new_ltw)
        self._ltmap.add(defid, new_ltw)
        _logger.debug("defid {0}: {1}".format(defid, new_ltw))

    def generate_tpl(self, pline):
        defid = self._ltmap.search(pline["words"])
        if defid is None:
            return None
        else:
            tpl = self._d_def.get(defid)
            return tpl


def init_ltgen_import(conf, table, shuffle=False, **_):
    fn = conf.get("log_template_import", "def_path")
    mode = conf.get("log_template_import", "import_format")
    ltmap = conf.get("log_template_import", "search_method")
    from amulog import log_db
    lp = amulog.manager.load_log2seq(conf)
    return LTGenImport(table, fn, mode, ltmap, lp, shuffle)
