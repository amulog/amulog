#!/usr/bin/env python
# coding: utf-8


from amulog.external import tpl_match


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit("give me tpl filename and log filename")
    tpl_fp = sys.argv[1]
    data_fp = sys.argv[2]

    l_tpl = []
    l_regex = []
    with open(tpl_fp, 'r') as f:
        for line in f:
            tpl = line.rstrip()
            l_tpl.append(tpl)
            l_regex.append(tpl_match.generate_regex(tpl))

    import log2seq
    p = log2seq.init_parser()
    d_cnt = defaultdict(int)
    with open(data_fp, 'r') as f:
        for line in f:
            parsed_line = p.process_line(line)
            ret = tpl_match.match_line(parsed_line, l_regex)
            if ret is None:
                pass
            else:
                rid, mo = ret
                d_cnt[rid] += 1

    print("matched lines: {0}".format(sum(d_cnt.values())))
    for rid, cnt in d_cnt.items():
        print("{0},{1},{2}".format(rid, cnt, l_regex[rid]))
    for rid, tpl in enumerate(l_tpl):
        if rid in d_cnt:
            pass
        else:
            print("No match tpl: {0}, {1}".format(rid, tpl, l_regex[rid]))

