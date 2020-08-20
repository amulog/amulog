#!/usr/bin/env python
# coding: utf-8

"""Notice: after you edit log templates,
it is NOT reccommended to add other log messages,
especially with new log templates.
On current implementation, lt_edit will destroy
the structures of log template manager."""

import sys

import amulog.manager
from . import config
from . import log_db
from . import lt_common


def export(ld):
    for ltline in ld.table:
        print(ltline)


def show_all(ld):
    print(ld.show_all_ltgroup())


def show_lt(ld):
    print(ld.show_all_lt())


def show_ltg(ld, gid):
    if gid is None:
        print(ld.show_all_ltgroup())
    else:
        print(ld.show_ltgroup(gid))


def show_sort(ld):
    buf = []
    for ltline in ld.iter_lt():
        buf.append(str(ltline))
    buf.sort()
    print("\n".join(buf))


def breakdown_ltid(ld, ltid, limit):
    d_args = {}
    for line in ld.iter_lines(ltid = ltid):
        for vid, arg in enumerate(line.var()):
            d_var = d_args.setdefault(vid, {})
            d_var[arg] = d_var.get(arg, 0) + 1

    buf = []
    
    buf.append("LTID {0}> {1}".format(ltid, str(ld.lt(ltid))))
    buf.append(" ".join(ld.lt(ltid).ltw))
    buf.append("")
    for vid, loc in enumerate(ld.lt(ltid).var_location()):
        buf.append("Variable {0} (word location : {1})".format(vid, loc))
        items = sorted(d_args[vid].items(), key=lambda x: x[1], reverse=True)
        var_variety = len(d_args[vid].keys())
        if var_variety > limit:
            for item in items[:limit]:
                buf.append("{0} : {1}".format(item[0], item[1]))
            buf.append("... {0} kinds of variable".format(var_variety))
        else:
            for item in items:
                buf.append("{0} : {1}".format(item[0], item[1]))
        buf.append("")
    return "\n".join(buf)


def _str_lt(ltid, ld):
    return "ltid {0} : {1}".format(ltid, str(ld.lt(ltid)))


#def _update_tpl(ld, old_ltw, new_ltw):
#    table = ld.ltm.ltgen._table
#    if table.exists(old_ltw):
#        tid = table.get_tid(old_ltw)
#        state = ld.ltm.ltgen.update_table(new_ltw, tid, False)
#    else:
#        raise ValueError("No existing tpl, failed")


def merge_ltid(ld, ltid1, ltid2):
    amulog.manager.init_ltmanager()
    sym = ld.ltm.sym
    print("merge following log templates...")
    print(_str_lt(ltid1, ld))
    print(_str_lt(ltid2, ld))
    print()

    ltw1 = ld.lt(ltid1).ltw
    cnt1 = ld.lt(ltid1).cnt
    l_s = ld.lt(ltid1).lts
    ltw2 = ld.lt(ltid2).ltw
    cnt2 = ld.lt(ltid2).cnt
    if not len(ltw1) == len(ltw2):
        sys.exit("log template length is different, failed")
    new_ltw = lt_common.merged_template(ltw1, ltw2, sym)

    ld.ltm.replace_lt(ltid1, new_ltw, l_s, cnt1 + cnt2)
    ld.ltm.remove_lt(ltid2)
    ld.db.update_log({"ltid" : ltid2}, {"ltid" : ltid1})

    ld.commit_db()

    print("> new log template : ltid {0}".format(ltid1))
    print(_str_lt(ltid1, ld))


def separate_ltid(ld, ltid, vid, value):

    def remake_lt(ld, ltid):
        # use iter_words rather than iter_lines
        # because iter_lines needs ltline registered in table
        # but ltline on new_ltid is now on construction in this function...
        new_lt = None
        cnt = 0
        for l_w in ld.db.iter_words(ltid = ltid):
            if new_lt is None:
                new_lt = l_w
            else:
                new_lt = lt_common.merged_template(new_lt, l_w, sym)
            cnt += 1
        print("result : " + str(new_lt))
        print()
        return new_lt, cnt

    amulog.manager.init_ltmanager()
    sym = ld.ltm.sym
    print("separate following log template...")
    print(_str_lt(ltid, ld))
    print("new log template if variable {0} is {1}".format(vid, value))
    print()

    l_lid = [lm.lid for lm in ld.iter_lines(ltid = ltid)
            if lm.var()[vid] == value]
    new_ltid = ld.lttable.next_ltid()    
    for lid in l_lid:
        ld.db.update_log({"lid" : lid}, {"ltid" : new_ltid})

    l_s = ld.lt(ltid).lts
    ltw1, cnt1 = remake_lt(ld, ltid)
    ld.ltm.replace_lt(ltid, ltw1, l_s, cnt1)

    ltw2, cnt2 = remake_lt(ld, new_ltid)
    ltline = ld.ltm.add_lt(ltw2, l_s, cnt2)
    assert ltline.ltid == new_ltid

    ld.commit_db()

    print("> new log templates : ltid {0}, ltid {1}".format(ltid, new_ltid))
    print(_str_lt(ltid, ld))
    print(_str_lt(new_ltid, ld))


def split_ltid(ld, ltid, vid):
    amulog.manager.init_ltmanager()
    sym = ld.ltm.sym
    print("split following log template...")
    print(_str_lt(ltid, ld))
    
    d_lid = {}
    for lm in ld.iter_lines(ltid = ltid):
        lid = lm.lid
        w = lm.var()[vid] 
        d_lid.setdefault(w, []).append(lid)
    
    print("variable {0}: {1}".format(vid, d_lid.keys()))
    if len(d_lid) > 10:
        print("variable candidate is too large... are you sure?")
        import pdb; pdb.set_trace()

    args = d_lid.keys()
    d_word_ltid = {}

    ltobj = ld.lt(ltid)
    l_s = ltobj.lts
    vloc = ltobj.var_location()[vid]

    # Update a template on existing ltid
    new_ltid = ltid
    new_word = args.pop(0)
    new_ltw = ltobj.ltw[:]; new_ltw[vloc] = new_word
    cnt = len(d_lid[new_word])
    ld.ltm.replace_lt(new_ltid, new_ltw, l_s, cnt)
    d_word_ltid[new_word] = new_ltid
    print("> new log templates : ltid {0}".format(new_ltid))
    print(_str_lt(new_ltid, ld))

    # Add templates
    while len(args) > 0:
        new_ltid = ld.lttable.next_ltid()   
        new_word = args.pop(0)
        new_ltw = ltobj.ltw[:]
        new_ltw[vloc] = new_word
        cnt = len(d_lid[new_word])
        new_ltobj = ld.ltm.add_lt(new_ltw, l_s, cnt)
        assert new_ltid == new_ltobj.ltid
        d_word_ltid[new_word] = new_ltid
        print("> new log templates : ltid {0}".format(new_ltid))
        print(_str_lt(new_ltid, ld))
    
    # Update log lines
    for word, l_lid in d_lid.items():
        temp_ltid = d_word_ltid[word]
        for lid in l_lid:
            ld.db.update_log({"lid" : lid}, {"ltid" : temp_ltid})

    ld.commit_db()


def fix_ltid(ld, ltid, l_vid):
    amulog.manager.init_ltmanager()
    sym = ld.ltm.sym
    print("make variable (with no variety) into description word...")
    print(_str_lt(ltid, ld))
    
    d_variety = {vid : set() for vid in l_vid}
    for lm in ld.iter_lines(ltid = ltid):
        for vid in l_vid:
            d_variety[vid].add(lm.var()[vid])

    d_fixed = {}
    for vid in l_vid:
        variety = d_variety[vid]
        assert len(variety) > 0
        if len(variety) == 1:
            print("confirmed that variable {0} is stable".format(vid))
            d_fixed[vid] = variety.pop()
            print("fixed word : {0}".format(d_fixed[vid]))
        else:
            print("variable {0} is not stable, ignored".format(vid))
            print(variety)

    ltobj = ld.lt(ltid)
    new_ltw = ltobj.ltw[:]
    for vid in d_fixed.keys():
        vloc = ltobj.var_location()[vid]
        new_ltw[vloc] = d_fixed[vid]
    l_s = ltobj.lts
    cnt = ltobj.cnt

    ld.ltm.replace_lt(ltid, new_ltw, l_s, cnt)
    ld.commit_db()
    print("> new log templates : ltid {0}".format(ltid))
    print(_str_lt(ltid, ld))


def free_ltid(ld, ltid, l_wid):
    amulog.manager.init_ltmanager()
    sym = ld.ltm.sym
    print("make description word into variable (with no variety)...")
    print(_str_lt(ltid, ld))
 
    ltobj = ld.lt(ltid)
    new_ltw = ltobj.ltw[:]
    for wid in l_wid:
        if ltobj.ltw[wid] == sym:
            print("wid {0} seems a variable, ignored")
        else:
            print("confirmed that wid {0} is description word ({1})".format(
                wid, ltobj.ltw[wid]))
            new_ltw[wid] = sym
    l_s = ltobj.lts
    cnt = ltobj.cnt

    ld.ltm.replace_lt(ltid, new_ltw, l_s, cnt)
    ld.commit_db()
    print("> new log templates : ltid {0}".format(ltid))
    print(_str_lt(ltid, ld))


def search_stable_variable(ld, th = 1):
    #ld.init_ltmanager()

    for ltobj in ld.iter_lt():
        ltid = ltobj.ltid
        d_args = {}
        for lm in ld.iter_lines(ltid = ltid):
            for vid, arg in enumerate(lm.var()):
                d_var = d_args.setdefault(vid, {})
                d_var[arg] = d_var.get(arg, 0) + 1
        for vid, loc in enumerate(ld.lt(ltid).var_location()):
            var_variety = len(d_args[vid].keys())
            if var_variety <= th:
                print("{0} {1}".format(ltobj.ltid, ltobj))
                print("variable {0} (word location {1}): {2}".format(
                        vid, loc, d_args[vid]))


def search_stable_vrule(ld, restr, dry = False):
    import re
    reobj = re.compile(restr)
    for ltobj in ld.iter_lt():
        ltid = ltobj.ltid
        d_args = {}
        for lm in ld.iter_lines(ltid = ltid):
            for vid, arg in enumerate(lm.var()):
                d_var = d_args.setdefault(vid, {})
                d_var[arg] = d_var.get(arg, 0) + 1
        for vid, loc in enumerate(ld.lt(ltid).var_location()):
            if len(d_args[vid]) == 1 and reobj.match(
                    list(d_args[vid].keys())[0]):
                print("{0} {1}".format(ltobj.ltid, ltobj))
                print("variable {0} (word location {1}): {2}".format(
                        vid, loc, list(d_args[vid].keys())[0]))
                if not dry:
                    fix_ltid(ld, ltid, [vid])

    
def search_desc_free(ld, restr, dry = False):
    import re
    reobj = re.compile(restr)
    for ltobj in ld.iter_lt():
        ltid = ltobj.ltid
        for wid, w in enumerate(ltobj.ltw):
            if w == ltobj.sym:
                pass
            elif reobj.match(w):
                print("{0} {1}".format(ltobj.ltid, ltobj))
                print("word location {0}".format(wid))
                if not dry:
                    free_ltid(ld, ltid, wid)


#if __name__ == "__main__":
#    
#    usage = """
#usage: {0} [options] args...
#args:
#  show : show all ltgroups
#  show-lt : show all log template without grouping
#  show-group LTGID : show log template group which has given LTGID
#  breakdown LTID : show variables appeared in log instances of given LTID
#  merge LTID1 LTID2 : merge log data with LTID1 and LTID2
#  separate LTID VID VALUE : make new log template with log data
#                            that have given variable VALUE in place VID
#    """.format(sys.argv[0]).strip()
#    
#    op = optparse.OptionParser(usage)
#    op.add_option("-c", "--config", action="store",
#            dest="conf", type="string", default=config.DEFAULT_CONFIG_NAME,
#            help="configuration file")
#    op.add_option("-l", "--limit", action="store",
#            dest="show_limit", type="int", default=5,
#            help="Limitation rows to show source log data")
#    (options, args) = op.parse_args()
#    if len(args) == 0:
#        sys.exit(usage)
#    mode = args.pop(0)
#    conf = config.open_config(options.conf)
#
#    ld = log_db.LogData(conf, edit = True)
#    if mode == "export":
#        export(ld)
#    elif mode == "show":
#        show_all(ld)
#    elif mode == "show-lt":
#        show_lt(ld)
#    elif mode == "show-group":
#        if len(args) == 0:
#            show_ltg(ld, None)
#        else:
#            show_ltg(ld, int(args[1]))
#    elif mode == "show-sort":
#        show_sort(ld)
#    elif mode == "breakdown":
#        if len(args) == 0:
#            sys.exit("give me ltid, following \"{0}\"".format(mode))
#        ltid = int(args[0])
#        print(breakdown_ltid(ld, ltid, options.show_limit))
#    elif mode == "merge":
#        if len(args) < 2:
#            sys.exit("give me 2 ltid, following \"{0}\"".format(mode))
#        ltid1 = int(args[0])
#        ltid2 = int(args[1])
#        sym = conf.get("log_template", "variable_symbol")
#        merge_ltid(ld, ltid1, ltid2, sym)
#    elif mode == "separate":
#        if len(args) < 3:
#            sys.exit("give me ltid, variable id and value, "
#                    "following \"{0}\"".format(mode))
#        ltid = int(args[0])
#        vid = int(args[1])
#        val = args[2]
#        sym = conf.get("log_template", "variable_symbol")
#        separate_ltid(ld, ltid, vid, val, sym)
#    elif mode == "split":
#        if len(args) < 2:
#            sys.exit("give me ltid and variable id, "
#                    "following \"{0}\"".format(mode))
#        ltid = int(args[0])
#        vid = int(args[1])
#        sym = conf.get("log_template", "variable_symbol")
#        split_ltid(ld, ltid, vid, sym)
#    elif mode == "fix":
#        if len(args) < 2:
#            sys.exit("give me ltid and variable id to fix, "
#                    "following \"{0}\"".format(mode))
#        ltid = int(args[0])
#        l_vid = [int(i) for i in args[1:]]
#        sym = conf.get("log_template", "variable_symbol")
#        fix_ltid(ld, ltid, l_vid, sym)
#    elif mode == "free":
#        if len(args) < 2:
#            sys.exit("give me ltid and word id to free, "
#                    "following \"{0}\"".format(mode))
#        ltid = int(args[0])
#        l_wid = [int(i) for i in args[1:]]
#        sym = conf.get("log_template", "variable_symbol")
#        free_ltid(ld, ltid, l_wid, sym)
#    elif mode == "search-stable":
#        if len(args) >= 1:
#            th = int(args[0])
#        else:
#            th = 1
#        search_stable_variable(ld, th)

