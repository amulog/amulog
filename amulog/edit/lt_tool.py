#!/usr/bin/env python
# coding: utf-8

"""Notice: after you edit log templates,
it is NOT reccommended to add other log messages,
especially with new log templates.
On current implementation, lt_edit will destroy
the structures of log template manager."""

from amulog import lt_common


def merge_lt(ld, ltm, ltid1, ltid2, verbose=True):
    if verbose:
        print("merge following log templates...")
        print("{0}: {1}".format(ltid1, ld.lt(ltid1)))
        print("{0}: {1}".format(ltid2, ld.lt(ltid2)))
        print()

    ltw1 = ld.lt(ltid1).ltw
    cnt1 = ld.lt(ltid1).count
    l_s = ld.lt(ltid1).lts
    ltw2 = ld.lt(ltid2).ltw
    cnt2 = ld.lt(ltid2).count
    if not len(ltw1) == len(ltw2):
        print("log template length is different, failed")
        return
    new_ltw = lt_common.merged_template(ltw1, ltw2)

    if verbose:
        print("updated log template : ltid {0}".format(ltid1))
        print("> " + " ".join(new_ltw))

        ret = input("Is it ok to update and add templates? [y/n]: ").strip().lower()
        if ret not in ("y", "yes"):
            return

    ltm.replace_lt(ltid1, new_ltw, l_s, cnt1 + cnt2)
    ltm.remove_lt(ltid2)
    ld.commit_db()
    if verbose:
        print("templates updated")
        print("updating logs of changed templates...")

    ld.db.update_log({"ltid": ltid2}, ltid=ltid1)
    ld.commit_db()
    if verbose:
        print("logs updated")
    return ltid1


def separate_lt(ld, ltm, ltid, vid, value, verbose=True):
    if verbose:
        print("separate following log template...")
        print("{0}: {1}".format(ltid, ld.lt(ltid)))
        print("add new log template where variable{0} = {1}".format(vid, value))
        print()

    l_lm = [lm for lm in ld.iter_lines(ltid=ltid)]
    l_lm_separate = []
    l_lm_remain = []
    for lm in l_lm:
        if lm.var()[vid] == value:
            l_lm_separate.append(lm)
        else:
            l_lm_remain.append(lm)

    ltw1 = lt_common.template_from_messages(l_lm_remain)
    ltw2 = lt_common.template_from_messages(l_lm_separate)
    if verbose:
        print("updated log template : ltid {0}".format(ltid))
        print("> " + " ".join(ltw1))

        print("new log template :")
        print("> " + " ".join(ltw2))

        ret = input("Is it ok to update and add templates? [y/n]: ").strip().lower()
        if ret not in ("y", "yes"):
            return

    l_s = ld.lt(ltid).lts
    ltm.replace_lt(ltid, ltw1, l_s, len(l_lm_remain))
    new_ltobj = ltm.add_lt(ltw2, l_s, len(l_lm_separate))
    if verbose:
        print("templates updated")
        print("updating logs of changed templates...")

    new_ltid = new_ltobj.ltid
    for lm in l_lm_separate:
        ld.db.update_log({"lid": lm.lid}, ltid=new_ltid)
    ld.commit_db()
    if verbose:
        print("logs updated")


def split_lt(ld, ltm, ltid, vid, verbose=True):
    if verbose:
        print("split log template {0}...".format(ltid))
        print("< " + " ".join(ld.lt(ltid).ltw))
        print("searching log instances...")

    d_lid = {}
    for lm in ld.iter_lines(ltid=ltid):
        lid = lm.lid
        w = lm.var()[vid]
        d_lid.setdefault(w, []).append(lid)

    if verbose:
        print("variable {0}: {1}".format(vid, d_lid.keys()))

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
    count = len(d_lid[new_word])
    kwargs_updated = {"ltid": new_ltid,
                      "l_w": new_ltw,
                      "l_s": l_s,
                      "count": count}
    d_word_ltid[new_word] = new_ltid
    if verbose:
        print("updated log template : ltid {0}".format(new_ltid))
        print("> " + " ".join(new_ltw))

    # Add templates
    l_new_word = []
    l_kwargs_added = []
    while len(args) > 0:
        new_word = args.pop(0)
        new_ltw = ltobj.ltw[:]
        new_ltw[vloc] = new_word
        cnt = len(d_lid[new_word])
        kwargs = {"l_w": new_ltw,
                  "l_s": l_s,
                  "count": cnt}
        l_kwargs_added.append(kwargs)
        l_new_word.append(new_word)
        if verbose:
            print("new log template :")
            print("> " + " ".join(new_ltw))

    if verbose:
        ret = input("Is it ok to update and add templates? [y/n]: ").strip().lower()
        if ret not in ("y", "yes"):
            return

    ltm.replace_lt(**kwargs_updated)
    for kwargs, new_word in zip(l_kwargs_added, l_new_word):
        new_ltobj = ltm.add_lt(**kwargs)
        d_word_ltid[new_word] = new_ltobj.ltid

    if verbose:
        print("templates updated")
        print("updating logs of changed templates...")

    # Update log lines
    for word, l_lid in d_lid.items():
        tmp_ltid = d_word_ltid[word]
        for lid in l_lid:
            ld.db.update_log({"lid": lid}, ltid=tmp_ltid)
        if verbose:
            print("update logs of template {0}".format(tmp_ltid))

    ld.commit_db()


def fix_lt(ld, ltm, ltid, l_vid, verbose=True):
    if verbose:
        print("make variable (with no variety) into description word in following template...")
        print("log template {0}:")
        print("< " + " ".join(ld.lt(ltid).ltw))

    d_variety = {vid: set() for vid in l_vid}
    for lm in ld.iter_lines(ltid=ltid):
        for vid in l_vid:
            d_variety[vid].add(lm.var()[vid])

    d_fixed = {}
    for vid in l_vid:
        variety = d_variety[vid]
        assert len(variety) > 0
        if len(variety) == 1:
            d_fixed[vid] = variety.pop()
            if verbose:
                print("confirmed that variable {0} is stable".format(vid))
                print("fixed word : {0}".format(d_fixed[vid]))
        else:
            if verbose:
                print("variable {0} is not stable, ignored".format(vid))
                print(variety)
                print("NOTE: use lt-split to fix unstable variables")

    ltobj = ld.lt(ltid)
    new_ltw = ltobj.ltw[:]
    for vid in d_fixed.keys():
        vloc = ltobj.var_location()[vid]
        new_ltw[vloc] = d_fixed[vid]
    l_s = ltobj.lts
    cnt = ltobj.count

    if new_ltw == ltobj.ltw:
        if verbose:
            print("no changes")
        return

    if verbose:
        print("updated log template : ltid {0}".format(ltid))
        print("> " + " ".join(new_ltw))

        ret = input("Is it ok to update template? [y/n]: ").strip().lower()
        if ret not in ("y", "yes"):
            return

    ltm.replace_lt(ltid, new_ltw, l_s, cnt)
    ld.commit_db()
    if verbose:
        print("updated")
    return ltid


def free_lt(ld, ltm, ltid, l_wid, verbose=True):
    if verbose:
        print("make description word into variable (with no variety)...")
        print("log template {0}:")
        print("< " + " ".join(ld.lt(ltid).ltw))

    ltobj = ld.lt(ltid)
    new_ltw = ltobj.ltw[:]
    for wid in l_wid:
        if ltobj.ltw[wid] == lt_common.REPLACER:
            if verbose:
                print("wid {0} is not a description, ignored")
        else:
            new_ltw[wid] = lt_common.REPLACER
    l_s = ltobj.lts
    cnt = ltobj.count

    if new_ltw == ltobj.ltw:
        if verbose:
            print("no changes")
        return

    if verbose:
        print("updated log template : ltid {0}".format(ltid))
        print("> " + " ".join(new_ltw))

        ret = input("Is it ok to update template? [y/n]: ").lower()
        if ret not in ("y", "yes"):
            return

    ltm.replace_lt(ltid, new_ltw, l_s, cnt)
    ld.commit_db()
    if verbose:
        print("updated")
    return ltid


def merge_duplicated_lt(ld, ltm, verbose=True):
    from collections import defaultdict
    from itertools import combinations

    d_lt = defaultdict(list)
    for ltobj in ld.iter_lt():
        d_lt[tuple(ltobj.ltw)].append(ltobj)

    for key, l_ltobj in d_lt.items():
        if len(l_ltobj) > 1:
            if verbose:
                print("following {0} templates are duplicated.")
                for ltobj in l_ltobj:
                    print(" ".join(ltobj.ltw))

                ret = input("Is it ok to merge templates? [y/n]: ").lower()
                if ret not in ("y", "yes"):
                    print("passed as is")
                    continue
            for ltobj1, ltobj2 in combinations(l_ltobj, 2):
                merge_lt(ld, ltm, ltobj1.ltid, ltobj2.ltid,
                         verbose=False)
            print("merged")
