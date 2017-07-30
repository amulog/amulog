#!/usr/bin/env python
# coding: utf-8

"""Measure minimum edit distance of 2 datasets.
The minimum edit distance of a log template in a dataset is defined as 
the minimum value of edit distances between the template and
all templates in another dataset..
This metric shows the similarity of log templates in 2 datasets.

In this script, the dataset is defined as a part of 1 DB.
The part definition is described in amulog.config.parse_condition format.
"""


import sys
from collections import defaultdict

from amulog import config
from amulog import log_db
from amulog.lt_shiso import edit_distance
from amulog.__main__ import parse_condition

RELATIVE = True
ACCEPT_SYM = False

if len(sys.argv) < 4:
    sys.exit("usage: {0} CONFIG RULE1 RULE2".format(sys.argv[0]))

conf = config.open_config(sys.argv[1])
if ACCEPT_SYM:
    sym = conf.get("log_template", "variable_symbol")
else:
    sym = None
d_rule1 = parse_condition(sys.argv[2].split(","))
d_rule2 = parse_condition(sys.argv[3].split(","))
s_ltid1 = set()
s_ltid2 = set()

ld = log_db.LogData(conf)
for lm in ld.iter_lines(**d_rule1):
    s_ltid1.add(lm.lt.ltid)

for lm in ld.iter_lines(**d_rule2):
    s_ltid2.add(lm.lt.ltid)

common = s_ltid1 & s_ltid2
print("{0} common log template found... ".format(len(common)))

d_ed1 = {}
d_ed2 = {}
for key in common:
    d_ed1[key] = 0
    d_ed2[key] = 0
    s_ltid1.remove(key)
    s_ltid2.remove(key)

for ltid1 in s_ltid1:
    for ltid2 in s_ltid2:
        lt1 = ld.lt(ltid1)
        lt2 = ld.lt(ltid2)
        ed = edit_distance(lt1.ltw, lt2.ltw, sym)
        if RELATIVE:
            ed = 1.0 * ed / max(len(lt1.ltw), len(lt2.ltw))

        if d_ed1.get(ltid1, sys.maxsize) > ed:
            d_ed1[ltid1] = ed
        if d_ed2.get(ltid2, sys.maxsize) > ed:
            d_ed2[ltid2] = ed

avg1 = 1.0 * sum(d_ed1.values()) / len(d_ed1)
avg2 = 1.0 * sum(d_ed2.values()) / len(d_ed2)
print("Average distance 1 : {0}".format(avg1))
print("Average distance 2 : {0}".format(avg2))
print()

print("group 1 ({0}):".format(sys.argv[2]))
for ltid, ed in sorted(d_ed1.items(), key = lambda x: x[1], reverse = True):
    print("ltid {0} : {1}".format(ltid, ed))

print("group 2 ({0}):".format(sys.argv[3]))
for ltid, ed in sorted(d_ed2.items(), key = lambda x: x[1], reverse = True):
    print("ltid {0} : {1}".format(ltid, ed))


