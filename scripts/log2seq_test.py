#!/usr/bin/env python
# coding: utf-8

import sys
from amulog import log_db
#from amulog import logparser
from amulog import config
from amulog import common

if len(sys.argv) < 3:
    sys.exit("usage: {0} config targets".format(sys.argv[0]))
conf = config.open_config(sys.argv[1])

lp = log_db.load_log2seq(conf)
for fp in common.rep_dir(sys.argv[2:]):
    with open(fp) as f:
        for line in f:
            d = lp.process_line(line)
            print("{0}|{1}|{2}|{3}".format(
                d["timestamp"], d["host"], d["words"], d["symbols"]))
            #print LP.process_line(line.rstrip("\n"))
            #print("{0}|{1}|{2}|{3}".format(*lp.process_line(line)))

