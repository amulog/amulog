#!/usr/bin/env python
# coding: utf-8

"""
Interface to use some functions about Log DB from CLI.
"""

import sys
import logging
import argparse
from collections import namedtuple

from . import config

_logger = logging.getLogger(__package__)


def get_targets(ns, conf):
    if ns.recur:
        targets = common.recur_dir(ns.files)
    else:
        targets = common.rep_dir(ns.files)
    return targets


def get_targets_opt(ns, conf):
    if len(ns.files) == 0:
        l_path = config.getlist(conf, "database", "src_path")
        if conf.getboolean("database", "src_recur"):
            targets = common.recur_dir(l_path)
        else:
            targets = common.rep_dir(l_path)
    else:
        targets = get_targets(ns, conf)
    return targets


def generate_testdata(ns):
    pass


def db_make(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    targets = get_targets_opt(ns, conf)
    import log_db

    timer = common.Timer("db-make", output = _logger)
    timer.start()
    log_db.process_files(conf, targets, True)
    timer.stop()


def db_make_init(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    targets = get_targets_opt(ns, conf)
    import log_db

    timer = common.Timer("db-make-init", output = _logger)
    timer.start()
    log_db.process_files(conf, targets, True)
    timer.stop()


def db_add(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    targets = get_targets(ns, conf)
    import log_db

    timer = common.Timer("db-add", output = _logger)
    timer.start()
    log_db.process_files(conf, targets, False)
    timer.stop()


def db_update(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    targets = get_targets(ns, conf)
    import log_db

    timer = common.Timer("db-update", output = _logger)
    timer.start()
    log_db.process_files(conf, targets, False, diff = True)
    timer.stop()


def db_anonymize(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db
    
    timer = common.Timer("db-anonymize", output = _logger)
    timer.start()
    log_db.anonymize(conf)
    timer.stop()


def show_db_info(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db

    log_db.info(conf)


def show_lt(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db

    log_db.dump_lt(conf)


def show_ltg(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db

    log_db.show_lt(conf)


def show_lt_import(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db

    log_db.show_lt_import(conf)


def show_host(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db

    log_db.show_all_host(conf)


def show_log(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    import log_db

    ltid = None; gid = None; top_dt = None; end_dt = None;
    host = None; area = None
    for arg in ns.conditions:
        if not "=" in key:
            raise SyntaxError
        key = arg.partition("=")[0]
        if key == "ltid":
            ltid = int(arg.partition("=")[-1])
        elif key == "gid":
            gid = int(arg.partition("=")[-1])
        elif key == "top_date":
            date_string = arg.partition("=")[-1]
            top_dt = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        elif key == "end_date":
            date_string = arg.partition("=")[-1]
            end_dt = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        elif key == "date":
            date_string = arg.partition("=")[-1]
            top_dt = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            end_dt = top_dt + datetime.timedelta(days = 1)
        elif key == "host":
            host = arg.partition("=")[-1]
        elif key == "area":
            area = arg.partition("=")[-1]
    ld = log_db.LogData(conf)
    for e in ld.iter_lines(ltid = ltid, ltgid = gid, top_dt = top_dt,
                           end_dt = end_dt, host = host, area = area):
        print(e.restore__line())


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "nargs": 1,
               "default": config.DEFAULT_CONFIG,
               "help": "configuration file path for amulog"}]
OPT_RECUR = [["-r", "--recur"],
             {"dest": "recur", "action": "store_true",
              "help": "recursively search files to process"}]
OPT_TERM = [["-t", "--term"],
            {"dest": "dt_range",
             "metavar": "DATE1 DATE2", "nargs": 2,
             "help": ("datetime range, start and end in %Y-%M-%d style."
                      "(optional; defaultly use all data in database)")}]
ARG_FILE = [["file"],
             {"metavar": "PATH", "nargs": 1,
              "help": "filepath to output"}]
ARG_FILES = [["files"],
             {"metavar": "PATH", "nargs": "+",
              "help": "files or directories as input"}]
ARG_FILES_OPT = [["files"],
                 {"metavar": "PATH", "nargs": "*",
                  "help": ("files or directories as input "
                           "(optional; defaultly read from config")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "testdata": ["Generate test log data.",
                 [OPT_CONFIG, OPT_DEBUG, ARG_FILE,
                  [["-s", "--seed"],
                   {"dest": "seed", "metavar": "INT", "nargs": 1,
                    "default": 0,
                    "help": "seed value to generate random values"}]],
                 generate_testdata],
    "db-make": [("Initialize database and add log data. "
                 "This fuction works incrementaly."),
                [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES_OPT],
                db_make],
    "db-make-init": [("Initialize database and add log data "
                      "for given dataset. "
                      "This function does not consider "
                      "to add other data afterwards."),
                     [OPT_CONFIG, OPT_DEBUG, OPT_RECUR,
                      ARG_FILES_OPT],
                     db_make_init],
    "db-add": ["Add log data to existing database.",
               [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES],
               db_add],
    "db-update": [("Add newer log data (seeing timestamp range) "
                   "to existing database."),
                  [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES],
                  db_update],
    "db-anonymize": [("Remove variables in log messages. "
                      "(Not anonymize hostnames; to be added)"),
                     [OPT_CONFIG, OPT_DEBUG],
                     db_anonymize],
    "show-db-info": ["Show abstruction of database status.",
                     [OPT_CONFIG, OPT_DEBUG, OPT_TERM],
                     show_db_info],
    "show-lt": ["Show all log templates in database.",
                [OPT_CONFIG, OPT_DEBUG],
                show_lt],
    "show-ltg": ["Show all log template groups and their members in database.",
                 [OPT_CONFIG, OPT_DEBUG],
                 show_ltg],
    "show-lt-import": ["Output log template definitions in lt_import format.",
                       [OPT_CONFIG, OPT_DEBUG],
                       show_lt_import],
    "show-host": ["Show all hostnames in database.",
                  [OPT_CONFIG, OPT_DEBUG],
                  show_host],
    "show-log": ["Show log messages that satisfy given conditions in args.",
                 [OPT_CONFIG, OPT_DEBUG,
                  [["conditions"],
                   {"metavar": "CONDITION", "nargs": "+",
                    "help": ("Conditions to search log messages. "
                             "Example: show-log gid=24 date=2012-10-10 ..., "
                             "Keys: ltid, gid, date, top_date, end_date, "
                             "host, area")}]],
                 show_log],
}

USAGE_COMMANDS = "\n".join(["  {0}: {1}".format(key, val[0])
                            for key, val in DICT_ARGSET.items()])
USAGE = ("usage: {0} MODE [options and arguments] ...\n\n"
         "mode:\n".format(
        sys.argv[0])) + USAGE_COMMANDS


if __name__ == "__main__":
    if len(sys.argv) < 1:
        sys.exit(USAGE)
    mode = sys.argv[1]
    if mode in ("-h", "--help"):
        sys.exit(USAGE)
    commandline = sys.argv[2:]

    desc, l_argset, func = DICT_ARGSET[mode]
    ap = argparse.ArgumentParser(prog = " ".join(sys.argv[0:2]),
                                 description = desc)
    for args, kwargs in l_argset:
        ap.add_argument(*args, **kwargs)
    ns = ap.parse_args()
    func(ns)

