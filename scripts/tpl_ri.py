#!/usr/bin/env python
# coding: utf-8

"""Measure adjusted rand index of 2 datasets.
Usually a ground truth dataset and an evaluated dataset
is compared with this metric.
This shows the similarity of 2 datasets in terms of clustering.

In this script, a dataset is defined with 1 config file,
and its DB which is labeled with its log templates by some algorithm.

This script requires package sklearn.
"""

import sys
import datetime
from sklearn.metrics.cluster import adjusted_rand_score

from amulog import config
from amulog import log_db
#from amulog.__main__ import parse_condition

if len(sys.argv) < 3:
    sys.exit("usage: {0} CONFIG1 CONFIG2".format(sys.argv[0]))

l_data = []
for conf in [config.open_config(confpath) for confpath in sys.argv[1:3]]:
    temp_data = []
    ld = log_db.LogData(conf)
    iterobj = ld.iter_lines(top_dt = datetime.datetime(2000, 1, 1))
    for lm in iterobj:
        temp_data.append(lm.lt.ltgid)
    l_data.append(temp_data)

assert len(l_data) == 2
assert len(l_data[0]) == len(l_data[1])

print(adjusted_rand_score(l_data[0], l_data[1]))


