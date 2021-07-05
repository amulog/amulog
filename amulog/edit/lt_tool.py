#!/usr/bin/env python
# coding: utf-8

"""Notice: after you edit log templates,
it is NOT reccommended to add other log messages,
especially with new log templates.
On current implementation, lt_edit will destroy
the structures of log template manager."""

import sys

from amulog import lt_common
from amulog import manager


def _str_lt(ltid, ld):
    return "ltid {0} : {1}".format(ltid, str(ld.lt(ltid)))


def merge_ltid(ld, ltid1, ltid2):
    ltm = manager.LTManager(ld.conf, ld.db, ld.lttable)
    print("merge following log templates...")
    print(_str_lt(ltid1, ld))
    print(_str_lt(ltid2, ld))
    print()

    ltw1 = ld.lt(ltid1).ltw
    cnt1 = ld.lt(ltid1).count
    l_s = ld.lt(ltid1).lts
    ltw2 = ld.lt(ltid2).ltw
    cnt2 = ld.lt(ltid2).count
    if not len(ltw1) == len(ltw2):
        sys.exit("log template length is different, failed")
    new_ltw = lt_common.merged_template(ltw1, ltw2)

    ltm.replace_lt(ltid1, new_ltw, l_s, cnt1 + cnt2)
    ltm.remove_lt(ltid2)
    ld.db.update_log({"ltid": ltid2}, ltid=ltid1)

    ld.commit_db()

    print("> new log template : ltid {0}".format(ltid1))
    print(_str_lt(ltid1, ld))


def separate_ltid(ld, ltid, vid, value):

    def _remake_lt(_ltid):
        # use iter_words rather than iter_lines
        # because iter_lines needs ltline registered in table
        # but ltline on new_ltid is now on construction in this function...
        new_lt = None
        cnt = 0
        for l_w in ld.db.iter_words(ltid=_ltid):
            if new_lt is None:
                new_lt = l_w
            else:
                new_lt = lt_common.merged_template(new_lt, l_w)
            cnt += 1
        print("result : " + str(new_lt))
        print()
        return new_lt, cnt

    ltm = manager.LTManager(ld.conf, ld.db, ld.lttable)
    print("separate following log template...")
    print(_str_lt(ltid, ld))
    print("new log template if variable {0} is {1}".format(vid, value))
    print()

    l_lid = [lm.lid for lm in ld.iter_lines(ltid=ltid)
             if lm.var()[vid] == value]
    new_ltid = ld.lttable.next_ltid()
    for lid in l_lid:
        ld.db.update_log({"lid": lid}, ltid=new_ltid)

    l_s = ld.lt(ltid).lts
    ltw1, cnt1 = _remake_lt(ltid)
    ltm.replace_lt(ltid, ltw1, l_s, cnt1)

    ltw2, cnt2 = _remake_lt(new_ltid)
    ltline = ltm.add_lt(ltw2, l_s, cnt2)
    assert ltline.ltid == new_ltid

    ld.commit_db()

    print("> new log templates : ltid {0}, ltid {1}".format(ltid, new_ltid))
    print(_str_lt(ltid, ld))
    print(_str_lt(new_ltid, ld))


def split_ltid(ld, ltid, vid):
    ltm = manager.LTManager(ld.conf, ld.db, ld.lttable)
    print("split following log template...")
    print(_str_lt(ltid, ld))

    d_lid = {}
    for lm in ld.iter_lines(ltid=ltid):
        lid = lm.lid
        w = lm.var()[vid]
        d_lid.setdefault(w, []).append(lid)

    print("variable {0}: {1}".format(vid, d_lid.keys()))
    if len(d_lid) > 10:
        ret = input("split into {0} new templates, ok? [y/n]: ").lower()
        if ret not in ("y", "yes"):
            return

    args = list(d_lid.keys())
    d_word_ltid = {}

    ltobj = ld.lt(ltid)
    l_s = ltobj.lts
    vloc = ltobj.var_location()[vid]

    # Update a template on existing ltid
    new_ltid = ltid
    new_word = args.pop(0)
    new_ltw = ltobj.ltw[:]
    new_ltw[vloc] = new_word
    cnt = len(d_lid[new_word])
    ltm.replace_lt(new_ltid, new_ltw, l_s, cnt)
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
        new_ltobj = ltm.add_lt(new_ltw, l_s, cnt)
        assert new_ltid == new_ltobj.ltid
        d_word_ltid[new_word] = new_ltid
        print("> new log templates : ltid {0}".format(new_ltid))
        print(_str_lt(new_ltid, ld))

    # Update log lines
    for word, l_lid in d_lid.items():
        temp_ltid = d_word_ltid[word]
        for lid in l_lid:
            ld.db.update_log({"lid": lid}, ltid=temp_ltid)

    ld.commit_db()


def fix_ltid(ld, ltid, l_vid):
    ltm = manager.LTManager(ld.conf, ld.db, ld.lttable)
    print("make variable (with no variety) into description word...")
    print(_str_lt(ltid, ld))

    d_variety = {vid: set() for vid in l_vid}
    for lm in ld.iter_lines(ltid=ltid):
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
    cnt = ltobj.count

    ltm.replace_lt(ltid, new_ltw, l_s, cnt)
    ld.commit_db()
    print("> new log templates : ltid {0}".format(ltid))
    print(_str_lt(ltid, ld))


def free_ltid(ld, ltid, l_wid):
    ltm = manager.LTManager(ld.conf, ld.db, ld.lttable)
    print("make description word into variable (with no variety)...")
    print(_str_lt(ltid, ld))

    ltobj = ld.lt(ltid)
    new_ltw = ltobj.ltw[:]
    for wid in l_wid:
        if ltobj.ltw[wid] == lt_common.REPLACER:
            print("wid {0} seems a variable, ignored")
        else:
            print("confirmed that wid {0} is description word ({1})".format(
                wid, ltobj.ltw[wid]))
            new_ltw[wid] = lt_common.REPLACER
    l_s = ltobj.lts
    cnt = ltobj.count

    ltm.replace_lt(ltid, new_ltw, l_s, cnt)
    ld.commit_db()
    print("> new log templates : ltid {0}".format(ltid))
    print(_str_lt(ltid, ld))
