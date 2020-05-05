#!/usr/bin/env python
# coding: utf-8


ONLINE_ALG = ["shiso", "drain", "fttree", "lenma"]
OFFLINE_ALG = ["va", "dlog"]
ANY_ALG = ["import", "import-ext", "re", "crf"]


def is_online(alg_name):
    if alg_name in ONLINE_ALG:
        return True
    elif alg_name in OFFLINE_ALG:
        return False
    elif alg_name in ANY_ALG:
        return True
    else:
        raise ValueError
