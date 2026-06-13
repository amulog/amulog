#!/usr/bin/env python
# coding: utf-8

import ipaddress
from collections import defaultdict

from . import config


def _normalize_host(token):
    """Return the canonical string of an IP-address token, or the token
    unchanged if it is not a valid IP address (hostname, CIDR notation, ...).
    This lets equivalent IP notations (e.g. expanded IPv6) match by value.
    Subnet/CIDR containment is intentionally not supported: a CIDR token is
    kept as a plain literal string and only matches itself exactly."""
    try:
        return str(ipaddress.ip_address(token))
    except ValueError:
        return token


class HostAlias(object):
    """
    Note:
        1 host can not belong to multiple host groups, because
        group definition is used to label and classify variables
        in log templates.
    """

    def __init__(self, fn):
        self._fn = fn
        self._d_alias = defaultdict(list)  # key = alias, val = List[host]
        self._d_ralias = {}  # key = host, val = alias
        self._d_group = defaultdict(list)  # key = group, val = List[host]
        self._d_rgroup = {}  # key = host, val = group
        self._open(self._fn)

    def _open(self, fn):
        group = "default"
        if fn is None or fn == "":
            return
        with open(fn, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                line = line.partition("#")[0]
                if line == "" or line[0] == "#":
                    continue
                elif line[0] == "[" and "]" in line:
                    group = line.strip("[").partition("]")[0]
                elif line[0] == "<" and ">" in line:
                    l_temp = line.strip("<").partition(">")
                    alias = l_temp[0]
                    names = [alias] + l_temp[2].strip().rstrip("\n").split()
                    self._add_def(names, alias=alias, group=group)
                else:
                    names = line.rstrip("\n").split()
                    if len(names) == 0:
                        continue
                    self._add_def(names, group=group)

    def _add_def(self, l_name, alias=None, group=None):

        def add_alias(name, alias):
            if alias is not None:
                self._d_alias[alias].append(name)
                self._d_ralias[name] = alias
            else:
                self._d_alias[name].append(name)
                self._d_ralias[name] = name

        def add_groupdef(key, group):
            if group is not None:
                self._d_group[group].append(key)
                self._d_rgroup[key] = group

        for name in l_name:
            key = _normalize_host(name)
            add_alias(key, alias)
            add_groupdef(key, group)

    def keys(self):
        return self._d_alias.keys()

    def print_definitions(self):
        print("[aliases]")
        for k, val in self._d_alias.items():
            print("<{0}> ".format(k) + " ".join([str(v) for v in val]))
        print()
        print("[groups]")
        for k, val in self._d_group.items():
            print(k)
            print(" ".join([str(v) for v in val]))
            print()

    def isknown(self, string):
        try:
            # IP address: match by canonical form (no subnet containment).
            addr = str(ipaddress.ip_address(string))
            return addr in self._d_ralias
        except ValueError:
            # hostname: case-insensitive match.
            return string.lower() in self._d_ralias

    def resolve_host(self, string):
        try:
            # IP address: match by canonical form (no subnet containment).
            addr = str(ipaddress.ip_address(string))
            return self._d_ralias.get(addr)
        except ValueError:
            # hostname: case-insensitive match.
            return self._d_ralias.get(string.lower())

    def group(self, group):
        return self._d_group[group]

    def get_group(self, string):
        try:
            # IP address: match by canonical form (no subnet containment).
            addr = str(ipaddress.ip_address(string))
            return self._d_rgroup.get(addr)
        except ValueError:
            # hostname: case-insensitive match.
            return self._d_rgroup.get(string.lower())


def init_hostalias(conf):
    ha_fn = conf["manager"]["host_alias_filename"]
    ha = HostAlias(ha_fn)
    return ha


def test_hostalias(conf):
    names = ["192.168.0.1",
             "www.TEST.localdomain",
             "localhost",
             "www",
             "www3",
             "hoge",
             "10.100.1.254",
             "8.8.6.0"]
    # conf.set("database", "host_alias_filename", "host_alias_test.txt")
    ha = init_hostalias(conf)
    # ha = HostAlias(conf)
    ha.print_definitions()
    print()
    print("[test aliasing]")
    for name in names:
        print(name)
        print(ha.isknown(name))
        print(ha.resolve_host(name))
        print(ha.get_group(name))
        print()


#if __name__ == "__main__":
#    usage = ""
#    import optparse
#
#    op = optparse.OptionParser(usage)
#    op.add_option("-c", "--config", action="store",
#                  dest="conf", type="string", default=config.DEFAULT_CONFIG_NAME,
#                  help="configuration file path")
#    options, args = op.parse_args()
#    conf = config.open_config(options._conf)
#    test_hostalias(conf)
