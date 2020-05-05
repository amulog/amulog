#!/usr/bin/env python
# coding: utf-8


def load_trainitems(fp):
    """Load labeled train data and output items"""

    with open(fp, 'r') as f:
        buf = []
        for line in f:
            if line.strip() == "":
                if len(buf) > 0:
                    yield buf
                    buf = []
            else:
                item = line.strip().split()
                assert len(item) == 3
                buf.append(item)
        else:
            if len(buf) > 0:
                yield buf


def dump_trainitems(iterable_items, fp):
    with open(fp, 'w') as f:
        f.write("\n\n".join([items2str(lineitems)
                             for lineitems in iterable_items]))


def items2str(lineitems):
    return "\n".join([" ".join(item) for item in lineitems])


def line2items(l_w, midlabel_func=None, dummy_label="N"):
    """Returns items of a line with dummy label, use only for tagging"""
    if midlabel_func is None:
        midlabel_func = lambda x: x
    return [(w, midlabel_func(w), dummy_label) for w in l_w]

#
# def make_trainitems(l_w, tpl, midlabel_func=None):
#     """Returns items of a line for training from log_db.LogMessage()"""
#     if midlabel_func is None:
#         midlabel_func = lambda x: x
#     ret = []
#     for line_w, tpl_w in zip(l_w, tpl):
#         if line_w == tpl_w:
#             label = "D"
#         else:
#             label = "V"
#         ret.append((line_w, midlabel_func(line_w), label))
#     return ret
#
#
# def lm2trainitems(lm: log_db.LogMessage, midlabel_func=None):
#     return make_trainitems(lm.l_w, lm.lt.ltw, midlabel_func)
