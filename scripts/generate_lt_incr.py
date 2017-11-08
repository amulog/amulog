#!/usr/bin/env python
# coding: utf-8


from amulog import lt_common
from amulog import logparser


def generate_lt_file(conf, fp):
    lp = logparser.LogParser(conf)
    table = lt_common.TemplateTable()
    ltgen = lt_common.init_ltgen(conf, table, "crf")

    with open(fp, 'r') as f:
        for line in f:
            line = line.rstrip()
            dt, org_host, l_w, l_s = lp.process_line(line)
            tpl = ltgen.estimate_tpl(l_w, l_s)
            print(line)
            print(" ".join(tpl))
            print("")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: {0} config filepath".format(sys.argv[0]))
    generate_lt_file(sys.argv[1], sys.argv[2])
