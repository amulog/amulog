#!/usr/bin/env python
# coding: utf-8

import logging

from amulog import cli
from amulog import common
from amulog import config

_logger = logging.getLogger(__package__)


def show_lt_words(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from amulog import log_db
    ld = log_db.LogData(conf)

    from . import search
    d = search.agg_words(ld, target="all")
    print(common.cli_table(sorted(d.items(), key=lambda x: x[1],
                                  reverse=True)))


def show_lt_descriptions(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from amulog import log_db
    ld = log_db.LogData(conf)

    from . import search
    d = search.agg_words(ld, target="description")
    print(common.cli_table(sorted(d.items(), key=lambda x: x[1],
                                  reverse=True)))


def show_lt_variables(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from amulog import log_db
    ld = log_db.LogData(conf)

    from . import search
    d = search.agg_words(ld, target="variable")
    if ns.repld:
        import re
        reobj = re.compile(r"[0-9]+")
        keys = list(d.keys())
        for k in keys:
            new_k = reobj.sub(r"\d", k)
            if k == new_k:
                pass
            else:
                d[new_k] += d.pop(k)

    print(common.cli_table(sorted(d.items(), key=lambda x: x[1],
                                  reverse=True)))


def show_lt_breakdown(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import search
    ltid = ns.ltid
    limit = ns.lines
    print(search.breakdown_lt(conf, ltid, limit))


def show_lt_vstable(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from amulog import log_db
    ld = log_db.LogData(conf)

    from . import search
    for ret in search.stable_variables(ld, ltid=ns.ltid, th=1):
        print(ld.show_lt_info(ret["ltid"]))
        msg = "variable {0.vid} (word location {0.vloc}): {0.dict}".format(ret)
        print("  " + msg)


def lttool_merge(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import lt_tool

    lt_tool.merge_ltid(conf, ns.ltid1, ns.ltid2)


def lttool_separate(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import lt_tool
    lt_tool.separate_ltid(conf, ns.ltid, ns.vid, ns.word)


def lttool_split(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import lt_tool
    lt_tool.split_ltid(conf, ns.ltid, ns.vid)


def lttool_fix(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import lt_tool
    lt_tool.fix_ltid(conf, ns.ltid, ns.vids)


def lttool_free(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import lt_tool
    lt_tool.free_ltid(conf, ns.ltid, ns.wids)


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               "default": None,
               "help": "configuration file path for amulog"}]


# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "show-lt-words": ["Show words and their counts in all messages",
                      [OPT_CONFIG, OPT_DEBUG],
                      show_lt_words],
    "show-lt-description": ["Show description words and their counts",
                            [OPT_CONFIG, OPT_DEBUG],
                            show_lt_descriptions],
    "show-lt-variable": ["Show variable words and their counts",
                         [OPT_CONFIG, OPT_DEBUG,
                          [["-d", "--digit"],
                           {"dest": "repld", "action": "store_true",
                            "help": "replace digit to \\d"}]],
                         show_lt_variables],
    "show-lt-breakdown": ["Show variable samples in a log template.",
                          [OPT_CONFIG, OPT_DEBUG,
                           [["-l", "--ltid"],
                            {"dest": "ltid", "metavar": "LTID",
                             "action": "store", "type": int, "default": None,
                             "help": "log template identifier to investigate"}],
                           [["-n", "--number"],
                            {"dest": "lines", "metavar": "LINES",
                             "action": "store", "type": int, "default": 5,
                             "help": "number of variable candidates to show"}]],
                          show_lt_breakdown],
    "show-lt-vstable": ["Show stable variables in the template.",
                        [OPT_CONFIG, OPT_DEBUG,
                         [["-l", "--ltid"],
                          {"dest": "ltid", "metavar": "LTID",
                           "action": "store", "type": int, "default": None,
                           "help": "log template identifier to investigate"}],
                         [["-n", "--number"],
                          {"dest": "number", "metavar": "NUMBER",
                           "action": "store", "type": int, "default": 1,
                           "help": "thureshold number to be stable"}],
                         ],
                        show_lt_vstable],
    "lt-merge": ["Merge 2 templates and generate a new template.",
                 [OPT_CONFIG, OPT_DEBUG,
                  [["ltid1"],
                   {"metavar": "LTID1", "action": "store", "type": int,
                    "help": "first log template to merge"}],
                  [["ltid2"],
                   {"metavar": "LTID2", "action": "store", "type": int,
                    "help": "second log template to merge"}], ],
                 lttool_merge],
    "lt-separate": [("Separate messages satisfying the given condition "
                     "and make it a new log template."),
                    [OPT_CONFIG, OPT_DEBUG,
                     [["ltid"],
                      {"metavar": "LTID", "action": "store", "type": int,
                       "help": "log template indentifier"}],
                     [["vid"],
                      {"metavar": "VARIABLE-ID", "action": "store",
                       "type": int,
                       "help": "variable identifier to fix"}],
                     [["word"],
                      {"metavar": "WORD", "action": "store",
                       "help": "a word to fix in the location of vid"}]],
                    lttool_separate],
    "lt-split": [("Fix variables as descriptions to split template clusters.@ "
                  "Use carefully because this function may cause "
                  "generating enormous unexpected templates."),
                 [OPT_CONFIG, OPT_DEBUG,
                  [["ltid"],
                   {"metavar": "LTID", "action": "store", "type": int,
                    "help": "log template indentifier"}],
                  [["vid"],
                   {"metavar": "VARIABLE-ID", "action": "store",
                    "help": "variable identifier to fix"}]],
                 lttool_split],
    "lt-fix": [("Fix specified stable variable as a description.@ "
                "Usually this command add one new template."
                "For unstable variables, use lt-split or lt-separate."),
               [OPT_CONFIG, OPT_DEBUG,
                [["ltid"],
                 {"metavar": "LTID", "action": "store", "type": int,
                  "help": "log template to fix"}],
                [["vids"],
                 {"metavar": "VARIABLE-IDs", "nargs": "+", "type": int,
                  "help": "variable identifiers to fix"}]],
               lttool_fix],
    "lt-free": ["Make a description word into a variable.",
                [OPT_CONFIG, OPT_DEBUG,
                 [["ltid"],
                  {"metavar": "LTID", "action": "store", "type": int,
                   "help": "log template indentifier"}],
                 [["wids"],
                  {"metavar": "WORD-IDs", "nargs": "+", "type": int,
                   "help": "word locations to fix"}]],
                lttool_free],
}


def main():
    cli.main(DICT_ARGSET)


if __name__ == "__main__":
    main()
