#!/usr/bin/env python
# coding: utf-8


"""DEPRECATED"""

# Preprocessing for structuralizing log messages

import sys
import os
import re
import datetime
import ipaddress

from . import config
from . import strutil

DEFAULT_SYMDEF = "/".join((os.path.dirname(__file__), "data/symdef.txt.default"))


class LogParser():

    re_time = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d+)?$")
    re_mac = re.compile(r"^[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}$")
    #l_var_re = [re_time, re_mac, re_ipv6]
    l_var_re = [re_time, re_mac]

    re_datetime = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    re_year = re.compile(r"^[12]\d{3}$")
    month_name = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

    def __init__(self, conf, sep_variable = False):
        self.default_year = conf.get("database", "default_year")
        self.rmheader_fl = config.gettuple(conf, "database",
                                           "remove_header_filename")
        self.symdef = conf.get("log_template", "sym_filename")
        self.varsym = conf.get("log_template", "variable_symbol")
        if self.symdef is None or self.symdef == "":
            self.symdef = DEFAULT_SYMDEF
        self.sep_variable = sep_variable
        self.sym_ignore = conf.getboolean("log_template", "sym_ignore")
        hsize = conf.get("database", "header_size")
        if hsize == "" or hsize is None:
            self.header_size = None
        else:
            self.header_size = int(hsize)

        self.rm_header = []
        if self.rmheader_fl is not None and len(self.rmheader_fl) > 0:
            self._init_remove_header(self.rmheader_fl)
        self._init_splitter()

    def _init_remove_header(self, l_fn):
        for fn in l_fn:
            with open(fn, 'r'):
                for line in f:
                    line = line.rstrip("\n")
                    if not line == "":
                        self.rm_header.append(line.rstrip("\n"))

    def _init_splitter(self):
        buf = []
        with open(self.symdef, "r") as f:
            for line in f:
                buf.append(line.rstrip("\n").split("0"))
        self.spl = buf[0] # splitter
        self.cspl = buf[1] # conditional splitter (except variables in RE)

    def _is_removed(self, line):
        for description in self.rm_header:
            if description in line:
                return True
        else:
            return False

    def _set_year(self):
        if self.default_year is None or self.default_year == "":
            return datetime.datetime.today().year
        else:
            return int(self.default_year)
    
    def _re_cspl(self, string):
        # judge string is a variable or complex word to be parted
        try:
            r = ipaddress.ip_address(string)
        except ValueError:
            try:
                r = ipaddress.ip_network(string, strict = False)
            except ValueError:
                pass
            else:
                return True
        else:
            return True

        for restr in self.l_var_re:
            if restr.match(string):
                return True
        else:
            return False

    def _part(self, string):
        # partition if string have splitter
        for cnt, c in enumerate(string):
            if c in self.spl:
                return string[0:cnt], string[cnt], string[cnt+1:]
        # partition if string have variable string symbol
        # only for Non-symbol-ignoring mode
        if self.varsym in string and self.sep_variable:
            ind_s = string.find(self.varsym)
            ind_e = ind_s + len(self.varsym)
            if string == self.varsym:
                return string, None, None
            elif ind_s == 0:
                # top of string is equal to self.varsym
                return string[0:ind_e], "", string[ind_e:]
            else:
                return string[0:ind_s], "", string[ind_s:]
        # partition in reversed order if string is not one variable
        for cnt in reversed(range(len(string))):
            if string[cnt] in self.cspl:
                if self._re_cspl(string):
                    return string, None, None
                else:
                    return string[0:cnt], string[cnt], string[cnt+1:]
        return string, None, None

    def _split_word(self, string):
        # partition string recursively and label them
        w1, s, w2 = self._part(string)
        if w2 is None:
            assert s is None
            return [(w1, 'w')]
        else:
            ret1 = self._split_word(w1)
            ret2 = self._split_word(w2)
            return ret1 + [(s, 's')] + ret2

    @staticmethod
    def _merge_sym(l_elem):
        # merge continuous symbol strings
        # input : splited strings and labels(w or s)
        # output : sequence of words and symbol strings
        l_w = []
        l_s = []
        temp = []
        for e in l_elem:
            if e[1] == 'w':
                if len(e[0]) > 0:
                    l_s.append("".join(temp))
                    temp = []
                    l_w.append(e[0])
            elif e[1] == 's':
                temp.append(e[0])
            else:
                raise ValueError
        else:
            l_s.append("".join(temp))
        assert len(l_w) + 1 == len(l_s)
        return l_w, l_s

    def split_message(self, line):
        ret = self._split_word(line.rstrip("\n"))
        l_w, l_s = self._merge_sym(ret)
        if self.sym_ignore:
            return l_w, l_s
        else:
            # restore order of l_w and l_s, and return it as l_w (l_s is None)
            # s[0], w[0], s[1], w[1], ..., s[n], w[n], s[n+1]
            ret = []
            for w, s in zip(l_w + [""], l_s):
                for c in (s, w):
                    if not c == "":
                        ret.append(c)
            return ret, None
            #return [i[0] for i in ret if not i[0] == ""], None

    def pop_header(self, src_line):

        def pop_string(line):
            line = line.strip(" ")
            if " " in line:
                string, line = line.split(" ", 1)
                return string, line
            else:
                return line, ""

        def str2month(string):
            if not string in self.month_name:
                return None
            else:
                return self.month_name.index(string) + 1

        line = src_line[:]

        try:
            if self.re_datetime.match(line):
                # 2112-09-01 10:00:00 hostname mesasges...
                date_str, line = pop_string(line)
                time_str, line = pop_string(line)
                dt = datetime.datetime.strptime(" ".join((date_str, time_str)),
                        "%Y-%m-%d %H:%M:%S")
                host, line = pop_string(line)
                message = line
            else:
                # (2112) Sep 01 10:00:00 hostname messages...
                string, line = pop_string(line)
                if self.re_year.match(string):
                    year = int(string)
                    month_str, line = pop_string(line)
                    month = str2month(month_str)
                else:
                    year = self._set_year()
                    month = str2month(string)
                if month is None:
                    return None, None, None
                day_str, line = pop_string(line)
                day = int(day_str)
                time_str, line = pop_string(line)
                hour, minute, second = tuple(int(e) for e
                                             in time_str.split(":"))
                host, line = pop_string(line)
                message = line
                dt = datetime.datetime(year = year, month = month, day = day, 
                        hour = hour, minute = minute,
                        second = second, microsecond = 0)
        except:
            return None, None, None

        if self.header_size is not None:
            message = " ".join(src_line.split()[self.header_size:])
        return dt, host, message

    def process_line(self, line):
        line = line.rstrip("\n")
        if line == "":
            return None, None, None, None
        dt, host, message = self.pop_header(line)
        if message is None or message == "":
            return None, None, None, None
        elif self._is_removed(message):
            return None, None, None, None
        l_word, l_symbol = self.split_message(message)
        return dt, host, l_word, l_symbol


def iter_lines_log(conf, fn):
    lp = LogParser(conf)
    with open(fn, 'r') as f:
        for line in f:
            line = line.rstrip()
            dt, org_host, l_w, l_s = lp.process_line(line)
            if l_w is None: continue
            l_w = [strutil.add_esc(w) for w in l_w]
            yield line, l_w, l_s


def iter_lines_message(conf, fn):
    lp = LogParser(conf)
    with open(fn, 'r') as f:
        for line in f:
            line = line.rstrip()
            l_w, l_s = lp.split_message(line)
            if l_w is None: continue
            l_w = [strutil.add_esc(w) for w in l_w]
            yield line, l_w, l_s


def iter_lines(conf, fn, form):
    if form == "log":
        return iter_lines_log(conf, fn)
    elif form == "message":
        return iter_lines_message(conf, fn)
    else:
        raise NotImplementedError


def test_parse(conf):
    from . import common
    LP = LogParser(conf)
    ret = []
    if conf.getboolean("general", "src_recur"):
        l_fp = common.recur_dir(config.getlist(conf, "general", "src_path"))
    else:
        l_fp = common.rep_dir(config.getlist(conf, "general", "src_path"))
    for fp in l_fp:
        with open(fp, 'r') as f:
            for line in f:
                ret.append(LP.process_line(line.rstrip("\n")))
    return ret


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: {0} config targets".format(sys.argv[0]))
    conf = config.open_config(sys.argv[1])
    
    from . import common
    LP = LogParser(conf)
    for fp in common.rep_dir(sys.argv[2:]):
        with open(fp) as f:
            for line in f:
                #print LP.process_line(line.rstrip("\n"))
                print(" ".join(LP.process_line(line.rstrip("\n"))[2]))


