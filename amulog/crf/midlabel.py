#!/usr/bin/env python
# coding: utf-8

import re
import ipaddress
import configparser

from amulog import host_alias


class LabelWord():

    def __init__(self, conf, fn = None, pos_unknown = "unknown"):
        self._ext = {}
        self._re = {}
        self.pos_unknown = pos_unknown
        
        self.conf = conf
        self._ha = host_alias.HostAlias(conf)
        if fn is not None:
            self._load_rule(fn)

    def _load_rule(self, fn):

        def gettuple(conf, sec, opt):
            s = conf[sec][opt]
            return tuple([w.strip() for w in s.split(",")])

        conf = configparser.ConfigParser()
        loaded = conf.read(fn)
        if len(loaded) == 0:
            sys.exit("opening LabelWord config {0} failed".format(fn))

        t_ext = gettuple(conf, "ext", "rules")
        for rule in t_ext:
            self._ext[rule] = getattr(self, conf["ext"][rule].strip())

        t_re_rules = gettuple(conf, "re", "rules")
        for rule in t_re_rules:
            temp = []
            t_re = gettuple(conf, "re", rule)
            for restr in t_re:
                temp.append(re.compile(restr))
            self._re[rule] = temp

    def label(self, word):
        for key, func in self._ext.items():
            if func(word):
                return key

        for key, l_reobj in self._re.items():
            for reobj in l_reobj:
                if reobj.match(word):
                    return key

        return self.pos_unknown


    def label_ipaddr(self, word):
        try:
            ret = ipaddress.ip_address(str(word))
            if isinstance(ret, ipaddress.IPv4Address):
                return "IPv4ADDR"
            elif isinstance(ret, ipaddress.IPv6Address):
                return "IPv6ADDR"
            else:
                raise TypeError("ip_address returns unknown type? {0}".format(
                        str(ret)))
        except ValueError:
            return None


    def label_ipnetwork(self, word):
        try:
            ret = ipaddress.ip_network(str(word), strict = False)
            if isinstance(ret, ipaddress.IPv4Network):
                return "IPv4NET"
            elif isinstance(ret, ipaddress.IPv6Network):
                return "IPv6NET"
            else:
                raise TypeError("ip_address returns unknown type? {0}".format(
                        str(ret)))
        except ValueError:
            return None


    def label_host(self, word):
        if self._ha.isknown(word):
            return "HOST"
        else:
            return None


if __name__ == "__main__":
    import sys
    from amulog import config
    from amulog import logparser

    if len(sys.argv) < 3:
        sys.exit("usage: {0} config filename".format(sys.argv[0]))
    conf = config.open_config(sys.argv[1])
    fn = sys.argv[2]

    lp = logparser.LogParser(conf)
    middle = conf.get("log_template_crf", "middle_label_rule")
    lw = LabelWord(conf, middle)

    with open(fn, "r") as f:
        for line in f:
            dt, host, l_w, l_s = lp.process_line(line.rstrip())
            for w in l_w:
                mid = lw.label(w)
                print(w, mid)
            print()






