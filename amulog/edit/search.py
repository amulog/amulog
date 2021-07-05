from collections import defaultdict


def agg_words(ld, target="all"):
    """Return dict of words in all log messages and their counts.

    The behavior of this function depends on 'target'.
    - all: count all words in templates
    - description: count description words in templates
    - variable: count variable words in templates

    Args:
        ld (log_db.LogData)
        target (str): one of (all, description, variable)

    Returns:
        dict: key = word, val = counts
    """

    from .. import lt_common
    d_stats = defaultdict(int)
    for lm in ld.iter_all():
        if target == "all":
            for w in lm.l_w:
                d_stats[w] += 1
        elif target == "description":
            for w in lm.lt.ltw:
                if w != lt_common.REPLACER:
                    d_stats[w] += 1
        elif target == "variable":
            for w in lm.var():
                d_stats[w] += 1
        else:
            raise NotImplementedError

    return d_stats


def breakdown_lt(ld, ltid, limit):
    d_args = {}
    for line in ld.iter_lines(ltid=ltid):
        for vid, arg in enumerate(line.var()):
            d_var = d_args.setdefault(vid, {})
            d_var[arg] = d_var.get(arg, 0) + 1

    buf = ["LTID {0}> {1}".format(ltid, str(ld.lt(ltid))),
           " ".join(ld.lt(ltid).ltw),
           ""]
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


def stable_variables(ld, ltid=None, th=1):
    if ltid is None:
        for ltobj in ld.iter_lt():
            for ret in stable_variables(ld, ltid=ltobj.ltid, th=th):
                yield ret
    else:
        ltobj = ld.lt(ltid)
        d_var = defaultdict(lambda: defaultdict(int))
        for lm in ld.iter_lines(ltid=ltid):
            for vid, variable in enumerate(lm.var()):
                d_var[vid][variable] += 1

        for vid, loc in enumerate(ltobj.var_location()):
            d_count = d_var[vid]
            if len(d_count) <= th:
                yield {"ltid": ltid,
                       "vid": vid,
                       "vloc": loc,
                       "dict": d_count}
