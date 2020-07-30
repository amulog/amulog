#!/usr/bin/env python
# coding: utf-8


ONLINE_ALG = ["shiso", "drain", "lenma", "fttree"]
OFFLINE_ALG = ["va", "dlog"]
ANY_ALG = ["import", "import-ext", "re", "crf"]


def is_online(mode, alg_names, multiproc):
    if multiproc:
        return False
    elif mode == "online":
        return True
    elif mode == "offline":
        return False
    else:
        for alg_name in alg_names:
            if alg_name in OFFLINE_ALG:
                return False
        else:
            return True
