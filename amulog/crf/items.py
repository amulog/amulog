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
    """Returns items of a line for tagging from a word sequence of a line."""
    if midlabel_func is None:
        midlabel_func = lambda x: x
    return [" ".join((w, midlabel_func(w), dummy_label)) for w in l_w]



