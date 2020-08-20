
import os
import random
import logging
from collections import defaultdict

import amulog.manager
from amulog import config
from amulog import lt_common
from . import _items

_logger = logging.getLogger(__package__)


def _pop_fairly(d, size):
    ret = []
    keys = list(d.keys())
    random.shuffle(keys)
    while len(ret) < size:
        for key in keys:
            if len(d[key]) > 0:
                elm = d[key].pop()
                ret.append(elm)
            if len(ret) >= size:
                break
    return ret


def train_sample_random(l_lm, size):
    if len(l_lm) <= size:
        return l_lm
    else:
        return random.sample(l_lm, size)


def train_sample_ltgen(l_lm, size, ltgen):
    if len(l_lm) <= size:
        return l_lm

    plines = [{"words": lm.l_w, "symbols": lm.lt.lts} for lm in l_lm]
    d_ret = ltgen.process_offline(plines)
    d_lm = defaultdict(list)
    for mid, lm in enumerate(l_lm):
        tid = d_ret[mid]
        d_lm[tid].append(lm)
    for tid in d_lm:
        random.shuffle(d_lm[tid])

    return _pop_fairly(d_lm, size)


def train_sample_leak(l_lm, size):
    d_lm = defaultdict(list)
    for lm in l_lm:
        d_lm[lm.lt.ltid].append(lm)
    for ltid in d_lm:
        random.shuffle(d_lm[ltid])

    return _pop_fairly(d_lm, size)


def make_crf_train(conf, iterobj, return_ltidlist=False):
    method = conf["log_template_crf"]["sample_method"]
    size = conf.getint("log_template_crf", "n_sample")
    if method == "all":
        l_train = list(iterobj)
    elif method == "random":
        l_train = train_sample_random(iterobj, size)
    elif method == "ltgen":
        lt_methods = config.getlist(conf, "log_template_crf",
                                    "sample_lt_methods")
        use_mp = conf.getboolean("log_template_crf", "sample_lt_multiprocess")
        table = lt_common.TemplateTable()
        ltgen = amulog.manager.init_ltgen_methods(conf, table, lt_methods,
                                                  multiprocess=use_mp)
        l_train = train_sample_ltgen(iterobj, size, ltgen)
    elif method == "leak":
        l_train = train_sample_leak(iterobj, size)
    else:
        raise NotImplementedError(
            "Invalid sampling method name {0}".format(method))

    if return_ltidlist:
        train_ltidlist = [lm.lt.ltid for lm in l_train]
        return l_train, train_ltidlist
    else:
        return l_train


def crf_trainfile(conf, iterobj):
    from . import lt_crf
    table = lt_common.TemplateTable()
    ltgen = lt_crf.init_ltgen_crf(conf, table)
    l_train = make_crf_train(conf, iterobj)
    l_buf = [_items.items2str(ltgen.trainitems(lm)) for lm in l_train]
    return "\n\n".join(l_buf)


def make_crf_model(conf, iterobj, output=None,
                   return_sampled_messages=False):
    from . import lt_crf
    table = lt_common.TemplateTable()
    ltgen = lt_crf.init_ltgen_crf(conf, table)

    l_train = make_crf_train(conf, iterobj)
    ltgen.init_trainer()
    model_path = ltgen.train(l_train, output)
    assert os.path.exists(model_path)
    _logger.info("generate crf model {0}".format(model_path))

    if return_sampled_messages:
        return model_path, l_train
    else:
        return model_path


def make_crf_model_from_trainfile(conf, fp, output=None):
    from . import lt_crf
    table = lt_common.TemplateTable()
    ltgen = lt_crf.init_ltgen_crf(conf, table)
    ltgen.init_trainer()
    model_path = ltgen.train_from_file(fp, output)
    assert os.path.exists(model_path)
    _logger.info("generate crf model {0}".format(model_path))
    return model_path
