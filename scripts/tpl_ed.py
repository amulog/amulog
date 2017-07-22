#!/usr/bin/env python
# coding: utf-8

import sys
from collections import defaultdict

from amulog import config
from amulog import log_db
from amulog.lt_shiso import edit_distance
from amulog.__main__ import parse_condition

if len(sys.argv) < 4:
    sys.exit("usage: {0} CONFIG RULE1 RULE2".format(sys.argv[0]))

conf = config.open_config(sys.argv[1])
sym = conf.get("log_template", "variable_symbol")
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
print("{0} common log template found... "
      "these are ignored in following porcess".format(len(common)))
for key in common:
    s_ltid1.remove(key)
    s_ltid2.remove(key)

d_ed1 = {}
d_ed2 = {}
for ltid1 in s_ltid1:
    for ltid2 in s_ltid2:
        lt1 = ld.lt(ltid1)
        lt2 = ld.lt(ltid2)
        ed = edit_distance(lt1.ltw, lt2.ltw, None)

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


