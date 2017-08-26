#!/usr/bin/env python
# coding: utf-8

# input
# - ed_threshold
# - d_ed1, d_ed2

import sys

from amulog import config
from amulog import lt_crf

_logger = logging.getLogger("amulog")

if len(sys.argv) < 4:
    sys.exit("usage: {0} CONFIG DISTFILE THRESHOLD".format(sys.argv[0]))

conf = config.open_config(sys.argv[1])
config.set_common_logging(conf, logger = _logger, lv = logging.INFO)

d_ed = {}
with open(sys.argv[2], "r") as f:
    for line in f:
        temp = line.rstrip("\n").split()
        ltid = int(temp[0])
        val = float(temp[1])
        assert not ltid in d_ed 
        d_ed[ltid] = val
threshold = float(sys.argv[3])

s_ltid = set()
for ltid, val in d_ed.items():
    if val <= threshold:
        s_ltid.add(ltid)

ma = lt_crf.MeasureAccuracy(conf, s_ltid)
if len(ma.results) == 0:
    raise ValueError("No measure results found")
print(ma.info())
print()
print(ma.result())






