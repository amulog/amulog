#!/usr/bin/env python
# coding: utf-8

import re
import logging
import ipaddress
import configparser

from amulog import lt_common
from amulog import host_alias

_logger = logging.getLogger(__package__)


class VariableRegex:

    def __init__(self, conf, fn=None, label_unknown="unknown"):
        self._ext = {}
        self._re = {}
        self.label_unknown = label_unknown

        self._ha = host_alias.init_hostalias(conf)
        if fn is not None and fn.strip() != "":
            self._load_rule(fn)

    def _load_rule(self, fn):

        def gettuple(conf, sec, opt):
            s = conf[sec][opt]
            if s.strip() == "":
                return tuple()
            else:
                return tuple([w.strip() for w in s.split(", ")])

        vre_conf = configparser.ConfigParser()
        loaded = vre_conf.read(fn)
        if len(loaded) == 0:
            mes = "VariableRegex config {0} is empty".format(fn)
            _logger.warning(mes)

        t_ext = gettuple(vre_conf, "ext", "rules")
        for rule in t_ext:
            self._ext[rule] = getattr(self, vre_conf["ext"][rule].strip())

        t_re_rules = gettuple(vre_conf, "re", "rules")
        for rule in t_re_rules:
            tmp = []
            t_re = gettuple(vre_conf, "re", rule)
            for restr in t_re:
                tmp.append(re.compile(restr))
            self._re[rule] = tmp

    def match(self, word):
        for key, func in self._ext.items():
            if func(word):
                return True

        for key, l_reobj in self._re.items():
            for reobj in l_reobj:
                if reobj.match(word):
                    return True
        return False

    def label(self, word):
        for key, func in self._ext.items():
            if func(word):
                return key

        for key, l_reobj in self._re.items():
            for reobj in l_reobj:
                if reobj.match(word):
                    return key

        return self.label_unknown

    @staticmethod
    def label_ipaddr(word):
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

    @staticmethod
    def label_ipnetwork(word):
        try:
            ret = ipaddress.ip_network(str(word), strict=False)
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


class LTGenRegularExpression(lt_common.LTGenStateless):

    def __init__(self, table, vreobj):
        super(LTGenRegularExpression, self).__init__(table)
        self._vre = vreobj

    def generate_tpl(self, pline):
        l_w = pline["words"]
        tpl = []
        for w in l_w:
            if self._vre.match(w):
                tpl.append(lt_common.REPLACER)
            else:
                tpl.append(w)
        return tpl


def init_ltgen_regex(conf, table, **_):
    fn = conf.get("log_template_re", "variable_rule")
    vreobj = VariableRegex(conf, fn)
    return LTGenRegularExpression(table, vreobj)
