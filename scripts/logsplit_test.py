#!/usr/bin/env python
# coding: utf-8

import sys
from amulog import logsplit
from amulog import config
from amulog import common

if len(sys.argv) < 3:
    sys.exit("usage: {0} config targets".format(sys.argv[0]))
conf = config.open_config(sys.argv[1])

lp = logsplit.LogSplit(conf)
for fp in common.rep_dir(sys.argv[2:]):
    with open(fp) as f:
        for line in f:
            print(line.rstrip())
            #print LP.process_line(line.rstrip("\n"))
            print("{0}|{1}|{2}|{3}".format(*lp.process_line(line)))


