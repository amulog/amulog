#!/usr/bin/env python
# coding: utf-8

import sys
import logging

from amulog import cli
from amulog import common
from amulog import config

_logger = logging.getLogger(__package__.partition(".")[0])


def get_targets_conf(conf):
    from amulog import __main__ as amulog_main
    return amulog_main.get_targets_conf(conf)


def get_targets_arg(ns):
    from amulog import __main__ as amulog_main
    return amulog_main.get_targets_arg(ns)


def get_targets_eval(conf):

    def _get_targets(input_path):
        if conf.getboolean("general", "src_recur"):
            return common.recur_dir(input_path)
        else:
            return common.rep_dir(input_path)

    l_path_src = config.getlist(conf, "general", "src_path")
    l_path_train = config.getlist(conf, "eval", "online_train_input")
    l_path_test = config.getlist(conf, "eval", "online_test_input")
    if len(l_path_train) == 0:
        ret_train = None
    else:
        ret_train = _get_targets(l_path_train)
    if len(l_path_test) == 0:
        ret_test = _get_targets(l_path_src)
    else:
        ret_test = _get_targets(l_path_test)
    return ret_train, ret_test


def measure_accuracy_answer(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    timer = common.Timer("measure-accuracy-answer", output=_logger)
    timer.start()
    _, targets_test = get_targets_eval(conf)
    maketpl.measure_accuracy_answer(conf, targets_test)
    timer.stop()


def measure_accuracy_trial(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    timer = common.Timer("measure-accuracy-trial", output=_logger)
    timer.start()
    targets_train, targets_test = get_targets_eval(conf)
    n_trial = int(conf["eval"]["n_trial_accuracy"])

    from amulog.__main__ import is_online
    if is_online(conf, False):
        maketpl.measure_accuracy_trial_online(conf, targets_train,
                                              targets_test, n_trial)
    else:
        maketpl.measure_accuracy_trial_offline(conf, targets_test,
                                               n_trial)
    timer.stop()


def measure_accuracy_trial_offline(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    timer = common.Timer("measure-accuracy-trial-offline", output=_logger)
    timer.start()
    _, targets_test = get_targets_eval(conf)
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    maketpl.measure_accuracy_trial_offline(conf, targets_test, n_trial)
    timer.stop()


def show_accuracy(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    mlt = maketpl.MeasureLTGen(conf, n_trial)
    mlt.load()

    functions = ["number_of_trial_clusters",
                 "word_accuracy",
                 "line_accuracy",
                 "tpl_accuracy",
                 "tpl_word_accuracy",
                 "rand_score",
                 "adjusted_rand_score",
                 "f1_score",
                 "parsing_accuracy",
                 "cluster_accuracy",
                 "v_measure_score"]

    if ns.partial:
        functions.extend(["tpl_description_accuracy",
                          "tpl_variable_accuracy",
                          "overdiv_ratio",
                          "overagg_ratio",
                          "homogeneity_score",
                          "completeness_score"])

    d_avg = maketpl.get_accuracy_average(conf, n_trial, functions)

    print("number of trials: {0}".format(n_trial))
    n_message = mlt.number_of_messages()
    print("number of messages: {0}".format(n_message))
    if n_message == 0:
        return

    print("number of clusters in answer: {0}".format(
        mlt.number_of_answer_clusters()))
    print("number of clusters in trial: {0}".format(
        d_avg["number_of_trial_clusters"]))
    print()

    print("word accuracy: {0}".format(d_avg["word_accuracy"]))
    print("line accuracy: {0}".format(d_avg["line_accuracy"]))
    print("tpl accuracy: {0}".format(d_avg["tpl_accuracy"]))
    print("tpl word accuracy: {0}".format(d_avg["tpl_word_accuracy"]))

    if ns.partial:
        print(" tpl description accuracy: {0}".format(
            d_avg["tpl_description_accuracy"]))
        print(" tpl variable accuracy: {0}".format(
            d_avg["tpl_variable_accuracy"]))

    print("rand score: {0}".format(d_avg["rand_score"]))
    print("adjusted rand score: {0}".format(d_avg["adjusted_rand_score"]))
    print("f1 score: {0}".format(d_avg["f1_score"]))
    print("parsing accuracy: {0}".format(d_avg["parsing_accuracy"]))
    print("cluster accuracy: {0}".format(d_avg["cluster_accuracy"]))

    if ns.partial:
        print(" over-division cluster ratio: {0}".format(
            d_avg["overdiv_ratio"]))
        print(" over-aggregation cluster ratio: {0}".format(
            d_avg["overagg_ratio"]))

    print("v-measure score: {0}".format(d_avg["v_measure_score"]))

    if ns.partial:
        print(" homogeneity score: {0}".format(
            d_avg["homogeneity_score"]))
        print(" completeness score: {0}".format(
            d_avg["completeness_score"]))


def show_accuracy_offline(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from amulog.__main__ import is_online
    if not is_online(conf, False):
        sys.exit("{0} is offline, use show_accuracy".format(ns.conf_path))

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    ret = maketpl.offline_structure_metrics(conf, n_trial, partial=ns.partial)
    print("word accuracy: {0}".format(ret[0]))
    print("line accuracy: {0}".format(ret[1]))
    print("tpl accuracy: {0}".format(ret[2]))
    print("tpl word accuracy: {0}".format(ret[3]))
    if ns.partial:
        print(" tpl description accuracy: {0}".format(ret[4]))
        print(" tpl variable accuracy: {0}".format(ret[5]))


def show_templates(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    for tpl in maketpl.get_templates(conf, n_trial, answer=ns.answer).values():
        print(" ".join(tpl))


def search_fail_template(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    try:
        maketpl.search_fail_template(conf, n_trial, pass_similar=(not ns.all))
    except KeyboardInterrupt:
        print("Keyboard Interrupt")


def search_diff_template(ns):
    conf1, conf2 = [config.open_config(confpath) for confpath in ns.confs]
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf1, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf1["eval"]["n_trial_accuracy"])
    try:
        maketpl.search_diff_template(conf1, conf2, n_trial,
                                     pass_similar=(not ns.all))
    except KeyboardInterrupt:
        print("Keyboard Interrupt")


def search_fail_cluster_overdiv(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    maketpl.search_fail_overdiv(conf, n_trial, n_samples=ns.samples)


def search_fail_cluster_overagg(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf["eval"]["n_trial_accuracy"])
    maketpl.search_fail_overagg(conf, n_trial, n_samples=ns.samples)


def search_diff_cluster_overdiv(ns):
    conf1, conf2 = [config.open_config(confpath) for confpath in ns.confs]
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf1, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf1["eval"]["n_trial_accuracy"])
    maketpl.search_diff_overdiv(conf1, conf2, n_trial, n_samples=ns.samples)


def search_diff_cluster_overagg(ns):
    conf1, conf2 = [config.open_config(confpath) for confpath in ns.confs]
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf1, logger=_logger, lv=lv)

    from . import maketpl
    n_trial = int(conf1["eval"]["n_trial_accuracy"])
    maketpl.search_diff_overagg(conf1, conf2, n_trial, n_samples=ns.samples)


def measure_parameters(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import param_searcher
    timer = common.Timer("measure-parameters", output=_logger)
    timer.start()
    targets = get_targets_arg(ns)
    param_searcher.measure_parameters(conf, targets, ns.method)
    timer.stop()


def measure_time_online(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    targets_train, targets_test = get_targets_eval(conf)
    d_time = maketpl.measure_time_online(conf, targets_train, targets_test)

    import numpy as np
    times = [str(t) for t in d_time.values()]
    avg = np.average(list(d_time.values()))
    print("Trials: {0}".format(", ".join(times)))
    print("Average: {0}".format(avg))


def measure_time_offline(ns):
    conf = config.open_config(ns.conf_path)
    lv = logging.DEBUG if ns.debug else logging.INFO
    config.set_common_logging(conf, logger=_logger, lv=lv)

    from . import maketpl
    _, targets_test = get_targets_eval(conf)
    d_time = maketpl.measure_time_offline(conf, targets_test)

    import numpy as np
    times = [str(t) for t in d_time.values()]
    avg = np.average(list(d_time.values()))
    print("Trials: {0}".format(", ".join(times)))
    print("Average: {0}".format(avg))


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
OPT_ALL = [["-a", "--all"],
           {"dest": "all", "action": "store_true",
            "help": "no omittion, show all components"}]
OPT_SAMPLES = [["-s", "--samples"],
               {"dest": "samples", "metavar": "SAMPLES",
                "action": "store", "type": int, "default": 1,
                "help": "number of samples for each cluster"}]
OPT_PARTIAL = [["-p", "--partial"],
               {"dest": "partial", "action": "store_true",
                "help": "show partial metrics (detailed output)"}]
ARG_FILES = [["files"],
             {"metavar": "PATH", "nargs": "+",
              "help": "files or directories as input"}]
ARG_FILES_OPT = [["files"],
                 {"metavar": "PATH", "nargs": "*",
                  "help": ("files or directories as input "
                           "(optional; defaultly read from config")}]
ARG_METHOD = [["method"],
              {"metavar": "METHOD",
               "help": "log template generation method to search"}]
ARG_2_CONFIG = [["confs"],
                {"metavar": "CONFIG", "nargs": 2,
                 "help": "2 config file path to compare"}]

# argument settings for each modes
# description, List[args, kwargs], func
# defined after functions because these settings use functions
DICT_ARGSET = {
    "measure-accuracy-answer": [("Make answer of log template generation. "
                                 "This works in online mode."),
                                [OPT_CONFIG, OPT_DEBUG],
                                measure_accuracy_answer],
    "measure-accuracy-trial": [("Measure accuracy of log template generation. "
                                "This works in online mode."),
                               [OPT_CONFIG, OPT_DEBUG],
                               measure_accuracy_trial],
    "measure-accuracy-trial-offline": [("Measure accuracy of "
                                        "log template generation. "
                                        "This works in offline mode."),
                                       [OPT_CONFIG, OPT_DEBUG],
                                       measure_accuracy_trial_offline],
    "show-accuracy": ["Load and show results of measuring accuracy",
                      [OPT_CONFIG, OPT_DEBUG, OPT_PARTIAL],
                      show_accuracy],
    "show-accuracy-offline": ["Show offline accuracy of online result",
                              [OPT_CONFIG, OPT_DEBUG, OPT_PARTIAL],
                              show_accuracy_offline],
    "show-templates": ["Show templates in accuracy measurement results",
                       [OPT_CONFIG, OPT_DEBUG,
                        [["-a", "--answer"],
                         {"dest": "answer", "action": "store_true",
                          "help": "show ground truth templates"}]],
                       show_templates],
    "search-fail-template": ["Show failed templates in template accuracy",
                             [OPT_CONFIG, OPT_DEBUG, OPT_ALL],
                             search_fail_template],
    "search-diff-template": ["Compare 2 results in template accuracy",
                             [OPT_DEBUG, OPT_ALL, ARG_2_CONFIG],
                             search_diff_template],
    "search-fail-cluster-overdiv": [("Show samples of clusters "
                                     "divided in surplus"),
                                    [OPT_CONFIG, OPT_DEBUG, OPT_SAMPLES],
                                    search_fail_cluster_overdiv],
    "search-fail-cluster-overagg": [("Show samples of clusters "
                                     "aggregated in surplus"),
                                    [OPT_CONFIG, OPT_DEBUG, OPT_SAMPLES],
                                    search_fail_cluster_overagg],
    "search-diff-cluster-overdiv": [("Show samples of clusters "
                                     "divided in surplus only in conf2"),
                                    [OPT_DEBUG, OPT_SAMPLES, ARG_2_CONFIG],
                                    search_diff_cluster_overdiv],
    "search-diff-cluster-overagg": [("Show samples of clusters "
                                     "divided in surplus only in conf2"),
                                    [OPT_DEBUG, OPT_SAMPLES, ARG_2_CONFIG],
                                    search_diff_cluster_overagg],
    "measure-parameter": [("Measure accuracy for parameter candidates"
                           "of log template generation"),
                          [OPT_DEBUG, OPT_CONFIG, OPT_RECUR, ARG_METHOD, ARG_FILES],
                          measure_parameters],
    "measure-time-online": [("Measure processing time of log template generation. "
                             "This works in online mode."),
                            [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES_OPT],
                            measure_time_online],
    "measure-time-offline": [("Measure processing time of log template generation. "
                              "This works in offline mode."),
                             [OPT_CONFIG, OPT_DEBUG, OPT_RECUR, ARG_FILES_OPT],
                             measure_time_offline],
}


def main():
    cli.main(DICT_ARGSET)


if __name__ == "__main__":
    main()
