#!/usr/bin/env python
# coding: utf-8

import pycrfsuite


def iter_items_from_file(fp):
    """Yields items of a line for training from a file of items format."""

    def buf2items(buf):
        return [item.split() for item in buf]

    with open(fp, 'r') as f:
        buf = []
        for line in f:
            if line.strip() == "":
                if len(buf) > 0:
                    yield buf2items(buf)
                    buf = []
            else:
                item = line.strip().split()
                assert len(item) == 3
                buf.append(item)
        else:
            if len(buf) > 0:
                yield buf2items(buf)


def line2items(l_w, midlabel_func = None, dummy_label = "N"):
    """Returns items of a line only for tagging"""
    if midlabel_func is None:
        midlabel_func = lambda x: x
    #return [" ".join((w, midlabel_func(w), dummy_label)) for w in l_w]
    return [(w, midlabel_func(w), dummy_label) for w in l_w]


def line2train(line, midlabel_func = None):
    """Returns items of a line for training from log_db.LogMessage()"""
    if midlabel_func is None:
        midlabel_func = lambda x: x
    ret = []
    for line_w, tpl_w in zip(line.l_w, line.lt.ltw):
        if line_w == tpl_w:
            label = "D"
        else:
            label = "V"
        #ret.append(" ".join((line_w, midlabel_func(line_w), label)))
        ret.append((line_w, midlabel_func(line_w), label))
    return ret

def items2label(lineitems):
    return [item[-1] for item in lineitems]




