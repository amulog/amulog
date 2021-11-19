#!/usr/bin/env python
# coding: utf-8

"""
Interface to use some functions about Log DB from CLI.
"""

import datetime
import logging

from . import cli
from . import common
from . import config

_logger = logging.getLogger(__package__)
SUBLIB = ["edit", "eval"]


def get_targets_arg(ns):
    if ns.recur:
        targets = common.recur_dir(ns.files)
    else:
        targets = common.rep_dir(ns.files)
    return targets


def get_targets_conf(conf):
    l_path = config.getlist(conf, "general", "src_path")
    if conf.getboolean("general", "src_recur"):
        return common.recur_dir(l_path)
    else:
        return common.rep_dir(l_path)


def get_targets(ns, conf):
    if ns is None or len(ns.files) == 0:
        return get_targets_conf(conf)
    else:
        return get_targets_arg(ns)


def is_online(conf, parallel):
    from .alg import is_online as alg_is_online
    mode = conf["log_template"]["processing_mode"]
    lt_methods = config.getlist(conf, "log_template", "lt_methods")
    return alg_is_online(mode, lt_methods, parallel)


def data_from_db(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    dirname = ns.dirname
    if ns.incr:
        method = "incremental"
    else:
        method = "commit"
    reset = ns.reset

    from . import log_db
    log_db.data_from_db(conf, dirname, method, reset)


def data_from_data(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    dirname = ns.dirname
    targets = get_targets(ns, conf)
    if ns.incr:
        method = "incremental"
    else:
        method = "commit"
    reset = ns.reset

    from . import manager
    manager.data_from_data(conf, targets, dirname, method, reset)


def data_parse(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    targets = get_targets(ns, conf)

    from . import strutil
    from . import manager
    lp = manager.load_log2seq(conf)
    for line in manager.iter_lines(targets):
        pline = manager.parse_line(strutil.add_esc(line), lp)
        if pline is None:
            pass
        elif ns.words:
            print(" ".join(pline["words"]))
        else:
            print(pline)


def db_make(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    targets = get_targets(ns, conf)

    from . import manager
    timer = common.Timer("db-make", output=_logger)
    timer.start()
    if is_online(conf, ns.parallel):
        manager.process_files_online(conf, targets, True)
    else:
        manager.process_files_offline(conf, targets, True, ns.parallel)
    timer.stop()


def db_add(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    targets = get_targets_arg(ns)

    from . import manager

    timer = common.Timer("db-add", output=_logger)
    timer.start()
    if is_online(conf, ns.parallel):
        manager.process_files_online(conf, targets, False)
    else:
        manager.process_files_offline(conf, targets, False, ns.parallel)
    timer.stop()


def db_remake_group(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import manager
    timer = common.Timer("db-remake-group", output=_logger)
    timer.start()
    manager.remake_ltgroup(conf)
    timer.stop()


def db_tag(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import lt_label
    timer = common.Timer("db-tag", output=_logger)
    timer.start()
    lt_label.generate_all_tags(conf)
    timer.stop()


def db_repair(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import log_db
    db = log_db.LogDB(conf, edit=True, reset_db=False)
    db.repair_tables()


def db_anonymize(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    timer = common.Timer("db-anonymize", output=_logger)
    timer.start()
    if ns.conf_export:
        conf2 = config.open_config(ns.conf_export)
    else:
        conf2 = None
    from . import anonymize
    am = anonymize.AnonymizeMapper(conf)
    am.anonymize(conf2)
    fp = am.dump()
    timer.stop()

    print("> " + fp)


def db_anonymize_mapping(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import anonymize
    am = anonymize.AnonymizeMapper(conf)
    am.mapping()
    fp = am.dump()

    print("> " + fp)


def show_db_info(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    log_db.info(conf)


def show_lt(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    simple = ns.simple
    print(log_db.show_all_lt(conf, simple=simple))


def show_ltg(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    print(log_db.show_all_ltg(conf))


def show_tag(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    print(log_db.show_tag(conf, tag=ns.tag))


def show_tag_stats(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    print(log_db.show_tag_stats(conf))


def show_lt_import(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    external = ns.external
    ld = log_db.LogData(conf)
    for ltobj in ld.iter_lt():
        if external:
            # unsegmented, with escape
            print(ltobj.restore_message(None, esc=True))
        else:
            # segmented, with escape
            print(" ".join(ltobj.ltw))


def show_host(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db

    log_db.show_all_host(conf)


def show_log(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)
    from . import log_db
    lidflag = ns.lid

    d = parse_condition(ns.conditions)
    ld = log_db.LogData(conf)
    for e in ld.iter_lines(**d):
        if lidflag:
            print("{0} {1}".format(e.lid, e.restore_line()))
        else:
            print(e.restore_line())


def parse_condition(conditions):
    """
    Args:
        conditions (list)
    """
    from dateutil import parser

    d = {}
    for arg in conditions:
        if "=" not in arg:
            raise SyntaxError
        key = arg.partition("=")[0]
        if key == "ltid":
            d["ltid"] = int(arg.partition("=")[-1])
        elif key in ("gid", "ltgid"):
            d["ltgid"] = int(arg.partition("=")[-1])
        elif key in ("time_from", "dts"):
            time_string = arg.partition("=")[-1]
            time_string = time_string.strip('"' + "'")
            d["dts"] = parser.parse(time_string)
        elif key in ("time_to", "dte"):
            time_string = arg.partition("=")[-1]
            time_string = time_string.strip('"' + "'")
            d["dte"] = parser.parse(time_string)
        # elif key in ("date_from", "top_date"):
        #     date_string = arg.partition("=")[-1]
        #     d["dts"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        # elif key in ("date_to", "end_date"):
        #     date_string = arg.partition("=")[-1]
        #     d["dte"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        elif key == "date":
            date_string = arg.partition("=")[-1]
            d["dts"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            d["dte"] = d["dts"] + datetime.timedelta(days=1)
        elif key == "host":
            d["host"] = arg.partition("=")[-1]
        elif key == "host_like":
            d["host_like"] = arg.partition("=")[-1]
        elif key == "host_regexp":
            d["host_regexp"] = arg.partition("=")[-1]
        else:
            raise ValueError
    return d


def conf_defaults(_):
    config.show_default_config()


def conf_diff(ns):
    files = ns.files[:]
    if ns.configset:
        files += config.read_config_group(ns.configset)
    config.show_config_diff(files)


def conf_minimum(ns):
    conf = config.open_config(ns.conf_path, base_default=False, ignore_import=True)
    conf = config.minimize(conf)
    config.write(ns.conf_path, conf)
    print("rewrite {0}".format(ns.conf_path))


def conf_set_edit(ns):
    l_conf_name = config.read_config_group(ns.configset)
    key = ns.key
    rulestr = ns.rule

    if "(" in rulestr:
        temp = rulestr.rstrip(")").split("(")
        if not len(temp) == 2:
            raise ValueError("bad format for value specification")
        rulename, argstr = temp
        args = argstr.split(",")

        if rulename == "list":
            assert len(args) == len(l_conf_name)
            d = {key: args}
        elif rulename == "range":
            assert len(args) == 2
            start, step = [int(v) for v in args]
            l_val = [start + i * step for i in range(len(l_conf_name))]
            d = {key: l_val}
        elif rulename == "power":
            assert len(args) == 2
            start, step = [int(v) for v in args]
            l_val = [start * (i ** step) for i in range(len(l_conf_name))]
            d = {key: l_val}
        elif rulename == "withconf":
            assert len(args) == 1
            l_val = [args[0] + name for i, name in enumerate(l_conf_name)]
            d = {key: l_val}
        elif rulename == "namerange":
            assert len(args) == 1
            l_val = [args[0] + str(i) for i in range(len(l_conf_name))]
            d = {key: l_val}
        else:
            raise NotImplementedError("invalid rule name")
    else:
        l_val = [rulestr] * len(l_conf_name)
        d = {key: l_val}

    config.config_group_edit(l_conf_name, d)


def conf_shadow(ns):
    cond = {}
    incr = []
    for rule in ns.rules:
        if "=" in rule:
            key, val = rule.split("=")
            cond[key] = val
        else:
            incr.append(rule)
    l_conf_name = config.config_shadow(n=ns.number, cond=cond, incr=incr,
                                       fn=ns.conf_path, output=ns.output,
                                       ignore_overwrite=ns.force)

    if ns.configset is not None:
        config.dump_config_group(ns.configset, l_conf_name)
        print(ns.configset)


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
OPT_CONFIG = [["-c", "--config"],
              {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
               # "default": config.DEFAULT_CONFIG,
               "default": None,
               "help": "configuration file path for amulog"}]
OPT_CONFIG_SET = [["-s", "--configset"],
                  {"dest": "configset", "metavar": "CONFIG_SET",
                   "default": None,
                   "help": "use config group definition file"}]
OPT_PARALLEL = [["-p", "--parallel"],
                {"dest": "parallel", "action": "store_true",
                 "help": "parallel processing in offline mode"}]
OPT_RECUR = [["-r", "--recur"],
             {"dest": "recur", "action": "store_true",
              "help": "recursively search files to process"}]
OPT_DRY = [["-d", "--dry"],
           {"dest": "dry", "action": "store_true",
            "help": "do not store data into db"}]
# OPT_TERM = [["-t", "--term"],
#             {"dest": "dt_range",
#              "metavar": "DATE1 DATE2", "nargs": 2,
#              "help": ("datetime range, start and end in YY-MM-dd style. "
#                       "(optional; defaultly use all data)")}]
ARG_FILE = [["file"],
            {"metavar": "PATH", "action": "store",
             "help": "filepath to output"}]
ARG_FILES = [["files"],
             {"metavar": "PATH", "nargs": "+",
              "help": "files or directories as input"}]
ARG_FILES_OPT = [["files"],
                 {"metavar": "PATH", "nargs": "*",
                  "help": ("files or directories as input "
                           "(optional; defaultly read from config")}]
ARG_DBSEARCH = [["conditions"],
                {"metavar": "CONDITION", "nargs": "+",
                 "help": ("Conditions to search log messages. "
                          "Example: MODE gid=24 date=2012-10-10 ..., "
                          "Keys: ltid, gid, date, time_from, time_to, host, "
                          "host_like, host_regexp")}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "data-from-db": ["Generate log data from DB.",
                     [OPT_CONFIG, OPT_DEBUG,
                      [["-d", "--dirname"],
                       {"dest": "dirname", "metavar": "DIRNAME",
                        "action": "store",
                        "help": "directory name to output"}],
                      [["-i", "--incr"],
                       {"dest": "incr", "action": "store_true",
                        "help": "output incrementally, use with small memory"}],
                      [["--reset"],
                       {"dest": "reset", "action": "store_true",
                        "help": "reset log file directory before processing"}],
                      ],
                     data_from_db],
    "data-from-data": ["Re-arrange log file, splitting messages by date.",
                       [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES_OPT,
                        [["-d", "--dirname"],
                         {"dest": "dirname", "metavar": "DIRNAME",
                          "action": "store",
                          "help": "directory name to output"}],
                        [["-i", "--incr"],
                         {"dest": "incr", "action": "store_true",
                          "help": "output incrementally, use with small memory"}],
                        [["--reset"],
                         {"dest": "reset", "action": "store_true",
                          "help": "reset log file directory before processing"}],
                        ],
                       data_from_data],
    "data-parse": ["Check log data parsing with log2seq.",
                   [OPT_CONFIG, OPT_DEBUG, OPT_RECUR,
                    [["-w", "--words"],
                     {"dest": "words", "action": "store_true",
                      "help": "only show parsed words"}],
                    ARG_FILES_OPT],
                   data_parse],
    "db-make": ["Initialize database and add log data. ",
                [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, OPT_PARALLEL, ARG_FILES_OPT],
                db_make],
    "db-add": ["Add log data to existing database.",
               [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, OPT_PARALLEL, ARG_FILES],
               db_add],
    "db-remake-group": ["Remake log template groups",
                        [OPT_CONFIG, OPT_DEBUG],
                        db_remake_group],
    "db-tag": ["Make log template tags.",
               [OPT_CONFIG, OPT_DEBUG],
               db_tag],
    "db-repair": ["Repair db schema after version updates.",
                  [OPT_CONFIG, OPT_DEBUG],
                  db_repair],
    "db-anonymize": ["Anonymize templates and hostnames.",
                     [OPT_CONFIG, OPT_DEBUG,
                      [["--config-export"],
                       {"dest": "conf_export", "metavar": "CONFIG_EXPORT",
                        "action": "store", "default": None,
                        "help": "generate another DB with given config"}],
                      ],
                     db_anonymize],
    "db-anonymize-mapping": ["Generate anonymization mapping json.",
                             [OPT_CONFIG, OPT_DEBUG],
                             db_anonymize_mapping],
    "show-db-info": ["Show abstruction of database status.",
                     [OPT_CONFIG, OPT_DEBUG],
                     show_db_info],
    "show-lt": ["Show all log templates in database.",
                [OPT_CONFIG, OPT_DEBUG,
                 [["-s", "--simple"],
                  {"dest": "simple", "action": "store_true",
                   "help": "only show log templates"}]],
                show_lt],
    "show-ltg": ["Show all log template groups and their members.",
                 [OPT_CONFIG, OPT_DEBUG,
                  [["-g", "--group"],
                   {"dest": "group", "action": "store", "default": None,
                    "help": "show members of given labeling group"}]],
                 show_ltg],
    "show-tag": ["Show all tags and their templates.",
                 [OPT_CONFIG, OPT_DEBUG,
                  [["tag"],
                   {"metavar": "TAG", "action": "store",
                    "nargs": "?", "default": None,
                    "help": "tag to search (if not given, show all tags)"}]],
                 show_tag],
    "show-tag-stats": ["Show all tags and their templates.",
                       [OPT_CONFIG, OPT_DEBUG],
                       show_tag_stats],
    "show-lt-import": ["Output log template definitions in importable format.",
                       [OPT_CONFIG, OPT_DEBUG,
                        [["-e", "--external"],
                         {"dest": "external", "action": "store_true",
                          "help": "output in non-amulog RE parser format"}]],
                       show_lt_import],
    "show-host": ["Show all hostnames in database.",
                  [OPT_CONFIG, OPT_DEBUG],
                  show_host],
    "show-log": ["Search and show log messages.",
                 [OPT_CONFIG, OPT_DEBUG,
                  [["--lid"],
                   {"dest": "lid", "action": "store_true",
                    "help": "show lid"}],
                  ARG_DBSEARCH],
                 show_log],
    "conf-defaults": ["Show default configurations.",
                      [],
                      conf_defaults],
    "conf-diff": ["Show differences of 2 configuration files.",
                  [OPT_CONFIG_SET,
                   [["files"],
                    {"metavar": "FILENAME", "nargs": "*",
                     "help": "configuration file"}]],
                  conf_diff],
    "conf-minimum": ["Remove default options and comments.",
                     [[["-o", "--overwrite"],
                       {"dest": "overwrite", "action": "store_true",
                        "help": "overwrite file instead of stdout dumping"}],
                      [["conf_path"],
                       {"metavar": "PATH",
                        "help": "config filepath to load"}]],
                     conf_minimum],
    "conf-group-edit": ["Edit configuration files in a config group.",
                        [[["configset"],
                          {"metavar": "CONFIG_SET",
                           "help": "config group definition file to use"}],
                         [["key"],
                          {"metavar": "KEY",
                           "help": "\"SECTION.OPTION\" to edit"}],
                         [["rule"],
                          {"metavar": "RULE",
                           "help": ("Value specification rule "
                                    "defined in function-like format. "
                                    "Example: \"List(1,10,100,1000)\" "
                                    "Available format: "
                                    "list(values for each config), "
                                    "range(START,STEP), "
                                    "power(START,STEP), "
                                    "withconf(NAME),"
                                    "namerange(NAME)."
                                    "Note that 1 rule for 1 execution.")}]],
                        conf_set_edit],
    "conf-shadow": ["Copy configuration files.",
                    [OPT_CONFIG,
                     [["-f", "--force"],
                      {"dest": "force", "action": "store_true",
                       "help": "Ignore overwrite of output file"}],
                     [["-n", "--number"],
                      {"dest": "number", "metavar": "INT",
                       "action": "store", "type": int, "default": 1,
                       "help": "number of files to generate"}],
                     [["-o", "--output"],
                      {"dest": "output", "metavar": "FILENAME",
                       "action": "store", "type": str, "default": None,
                       "help": "basic output filename"}],
                     [["-s", "--configset"],
                      {"dest": "configset", "metavar": "CONFIG_SET",
                       "default": None,
                       "help": ("define config group "
                                "and dump it in given filename")}],
                     [["rules"],
                      {"metavar": "RULES", "nargs": "*",
                       "help": ("Rules to replace options. You can indicate "
                                "option, or option and its value with =. "
                                "You can use both of them together. "
                                "For example: \"general.import=hoge.conf "
                                "general.logging\"")}]],
                    conf_shadow],
}

ALIASES = {
    "dump-lt": "show-lt-import",
}


def main():
    cli.main(DICT_ARGSET, ALIASES, SUBLIB)


if __name__ == "__main__":
    main()

    # import cProfile
    # cProfile.run('_main()', filename='main.prof')
