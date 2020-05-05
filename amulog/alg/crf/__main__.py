#!/usr/bin/env python
# coding: utf-8

import sys
import datetime
import logging
import argparse

from amulog import config


_logger = logging.getLogger(__package__)


def make_crf_train(ns):
    conf_load = config.open_config(ns.config_load)
    conf_dump = config.open_config(ns.config_dump)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf_dump, logger=_logger, lv=lv)

    from . import train
    from amulog import log_db
    d = parse_condition(ns.conditions)
    ld = log_db.LogData(conf_load)
    iterobj = ld.iter_lines(**d)
    print(train.crf_trainfile(conf_dump, iterobj))


def make_crf_model(ns):
    conf_load = config.open_config(ns.config_load)
    conf_dump = config.open_config(ns.config_dump)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf_dump, logger=_logger, lv=lv)

    from . import train
    output = ns.output
    if ns.train_file is None:
        from amulog import log_db
        d = parse_condition(ns.conditions)
        output_sampled = ns.output_sampled
        get_output_sampled = output_sampled is not None

        ld = log_db.LogData(conf_load)
        iterobj = [lm for lm in ld.iter_lines(**d)]
        if get_output_sampled:
            fn, l_train = train.make_crf_model(conf_dump, iterobj, output,
                                               return_sampled_messages=True)
            import pickle
            with open(output_sampled, 'wb') as f:
                pickle.dump(l_train, f)
            print("> {0}".format(output_sampled))
        else:
            fn = train.make_crf_model(conf_dump, iterobj, output)
    else:
        fn = train.make_crf_model_from_trainfile(conf_dump, ns.train_file, output)
    print("> {0}".format(fn))


def parse_condition(conditions):
    """
    Args:
        conditions (list)
    """
    d = {}
    for arg in conditions:
        if "=" not in arg:
            raise SyntaxError
        key = arg.partition("=")[0]
        if key == "ltid":
            d["ltid"] = int(arg.partition("=")[-1])
        elif key == "gid":
            d["ltgid"] = int(arg.partition("=")[-1])
        elif key == "top_date":
            date_string = arg.partition("=")[-1]
            d["top_dt"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        elif key == "end_date":
            date_string = arg.partition("=")[-1]
            d["end_dt"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
        elif key == "date":
            date_string = arg.partition("=")[-1]
            d["top_dt"] = datetime.datetime.strptime(date_string, "%Y-%m-%d")
            d["end_dt"] = d["top_dt"] + datetime.timedelta(days=1)
        elif key == "host":
            d["host"] = arg.partition("=")[-1]
        elif key == "area":
            d["area"] = arg.partition("=")[-1]
    return d


# common argument settings
OPT_DEBUG = [["--debug"],
             {"dest": "debug", "action": "store_true",
              "help": "set logging level to debug (default: info)"}]
ARG_CONFIG_LOAD = [["config_load"],
                   {"metavar": "CONFIG_LOAD",
                    "help": "Config for db to sample training data"}]
ARG_CONFIG_DUMP = [["config_dump"],
                   {"metavar": "CONFIG_DUMP",
                    "help": "Config for crf training model"}]
# OPT_CONFIG = [["-c", "--config"],
#               {"dest": "conf_path", "metavar": "CONFIG", "action": "store",
#                # "default": config.DEFAULT_CONFIG,
#                "default": None,
#                "help": "configuration file path for amulog"}]
OPT_RECUR = [["-r", "--recur"],
             {"dest": "recur", "action": "store_true",
              "help": "recursively search files to process"}]
OPT_TRAIN_SIZE = [["-n", "--train_size"],
                  {"dest": "train_size", "action": "store",
                   "type": int, "default": 1000,
                   "help": "number of training data to sample"}]
OPT_SAMPLE_METHOD = [["-m", "--method"],
                     {"dest": "method", "action": "store",
                      "default": "all",
                      "help": "train data sampling method name. "
                              "[all, random, ltgen, leak] is available."}]
OPT_MODEL_OUTPUT = [["-o", "--output"],
                    {"dest": "output", "action": "store",
                     "default": None,
                     "help": "output model filepath"}]
ARG_DBSEARCH = [["conditions"],
                {"metavar": "CONDITION", "nargs": "+",
                 "help": ("Conditions to search log messages. "
                          "Example: MODE gid=24 date=2012-10-10 ..., "
                          "Keys: ltid, gid, date, top_date, end_date, "
                          "host, area")}]


# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "make-crf-train": ["Generate CRF training file from DB.",
                       [OPT_DEBUG,
                        ARG_CONFIG_LOAD, ARG_CONFIG_DUMP, ARG_DBSEARCH],
                       make_crf_train],
    "make-crf-model": ["Generate CRF trained model from DB.",
                       [OPT_DEBUG, OPT_MODEL_OUTPUT,
                        ARG_CONFIG_LOAD, ARG_CONFIG_DUMP, ARG_DBSEARCH,
                        [["-t", "--train-file"],
                         {"dest": "train_file", "action": "store",
                          "default": None,
                          "help": "make model from training file"}],
                        [["-s", "--output-sampled-messages"],
                         {"dest": "output_sampled", "action": "store",
                          "default": None,
                          "help": ("output pickle of sampled LogMessage "
                                   "objects for training")}]],
                       make_crf_model],
}

USAGE_COMMANDS = "\n".join(["  {0}: {1}".format(key, val[0])
                            for key, val in sorted(DICT_ARGSET.items())])
USAGE = ("usage: {0} MODE [options and arguments] ...\n\n"
         "mode:\n".format(sys.argv[0])) + USAGE_COMMANDS


def _main():
    if len(sys.argv) < 2:
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
