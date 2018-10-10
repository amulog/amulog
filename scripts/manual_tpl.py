#!/usr/bin/env python
# coding: utf-8

import sys
import logging

from amulog import config
from amulog.crf import midlabel

_logger = logging.getLogger("manual_tpl")
POS_UNKNOWN = "unknown"


def merge_re(conf, lt_fn):
    middle = conf.get("log_template_crf", "middle_label_rule")
    lw = midlabel.LabelWord(conf, middle, pos_unknown = POS_UNKNOWN)

    l_tpl = set()
    with open(lt_fn, 'r') as f:
        for line in f:
            l_w = line.rstrip("\n").split(" ")
            tpl = []
            for w in l_w:
                label = lw.label(w)
                if label == POS_UNKNOWN:
                    rpl = w
                else:
                    _logger.debug(
                        "replace word <{0}>({1}) to **".format(w, label))
                    rpl = conf.get("log_template", "variable_symbol")
                tpl.append(rpl)
            l_tpl.add(tuple(tpl))

    for tpl in l_tpl:
        print(" ".join(tpl))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("usage: {0} mode conf input".format(sys.argv[0]))

    mode = sys.argv[1]
    conf = config.open_config(sys.argv[2])
    lv = logging.DEBUG
    #lv = logging.INFO
    config.set_common_logging(conf, logger = _logger, lv = lv)
    lt_fn = sys.argv[3]

    if mode == "merge-re":
        merge_re(conf, lt_fn)


