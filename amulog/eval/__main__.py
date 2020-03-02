#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse

from amulog import common
from amulog import config

_logger = logging.getLogger(__package__)


def get_targets_opt(ns, conf):
    from amulog import __main__ as amulog_main
    return amulog_main.get_targets_opt(ns, conf)


def measure_accuracy(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    timer = common.Timer("measure-accuracy", output=_logger)
    timer.start()
    targets = get_targets_opt(ns, conf)
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    maketpl.measure_accuracy(conf, targets, n_trial)
    timer.stop()


def show_accuracy(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    maketpl.print_metrics(conf, n_trial)


def measure_time(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    targets = get_targets_opt(ns, conf)
    maketpl.measure_time(conf, targets)


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               # "default": config.DEFAULT_CONFIG,
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_RECUR = [["-r", "--recur"],
             {"dest": "recur", "action": "store_true",
              "help": "recursively search files to process"}]
ARG_FILES_OPT = [["files"],
                 {"metavar": "PATH", "nargs": "*",
                  "help": ("files or directories as input "
                           "(optional; defaultly read from config")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "measure-accuracy": [("Measure accuracy of log template generation."
                          "This works in online mode."),
                         [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES_OPT],
                         measure_accuracy],
    "measure-time": [("Measure processing of log template generation."
                      "This works in online mode."),
                     [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES_OPT],
                     measure_time],
    "show-accuracy": ["Load and show results of measuring accuracy",
                      [OPT_CONFIG, OPT_DEBUG],
                      show_accuracy],
}

USAGE_COMMANDS = "\n".join(["  {0}: {1}".format(key, val[0])
                            for key, val in sorted(DICT_ARGSET.items())])
USAGE = ("usage: {0} MODE [options and arguments] ...\n\n"
         "mode:\n".format(sys.argv[0])) + USAGE_COMMANDS


def _main():
    if len(sys.argv) < 1:
        sys.exit(USAGE)
    mode = sys.argv[1]
    if mode in ("-h", "--help"):
        sys.exit(USAGE)
    commandline = sys.argv[2:]

    desc, l_argset, func = DICT_ARGSET[mode]
    ap = argparse.ArgumentParser(prog=" ".join(sys.argv[0:2]),
                                 description=desc)
    for args, kwargs in l_argset:
        ap.add_argument(*args, **kwargs)
    ns = ap.parse_args(commandline)
    func(ns)


if __name__ == "__main__":
    _main()
