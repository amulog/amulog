#!/usr/bin/env python
# coding: utf-8

from amulog.external import tpl_match
from amulog.external import mod_tplseq


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit("give me tpl filename and log filename")
    tpl_fp = sys.argv[1]
    data_fp = sys.argv[2]

    from . import tpl_match
    l_tpl = []
    l_regex = []
    with open(tpl_fp, 'r') as f:
        for line in f:
            tpl = line.rstrip()
            l_tpl.append(tpl)
            l_regex.append(tpl_match.generate_regex(tpl))

    p = log2seq.init_parser()
    with open(data_fp, 'r') as f:
        for line in f:
            parsed_line = p.process_line(line)
            ret = tpl_match.match_line(parsed_line, l_regex)
            if ret is None:
                pass
            else:
                print(parsed_line["message"])
                rid, mo = ret
                tpl = l_tpl[rid]
                print("{0}, {1}".format(rid, tpl))

                sym = "**"
                new_tpl = mod_tplseq.redefine_tpl(
                    tpl, parsed_line, sym, matchobj = mo)
                print(parsed_line["words"])
                print(new_tpl)
                print("")



