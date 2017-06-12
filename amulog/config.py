#!/usr/bin/env python
# coding: utf-8

import os
import datetime
import logging
import configparser

DEFAULT_CONFIG = "/".join((os.path.dirname(__file__),
                           "data/config.conf.default"))
IMPORT_SECTION = 'general'
IMPORT_OPTION = 'import'


class GroupDef():

    """
    Define grouping by external text
    Rules:
        description after # in a line will be recognized as comment
        line "[GROUP_NAME]" will change group to set
        other lines add elements in the group set with GROUP_NAME line
    """

    def __init__(self, fn, default_val = None):
        self.gdict = {}
        self.rgdict = {}
        self.default = default_val
        if fn is None or fn == "":
            pass
        else:
            self.open_def(fn)

    def open_def(self, fn):
        group = None
        with open(fn, 'r') as f:
            for line in f:
                # ignore after comment sygnal
                line = line.strip().partition("#")[0]
                if line == "":
                    continue
                elif line[0] == "#":
                    continue
                elif line[0] == "[" and line[-1] == "]":
                    group = line[1:].strip("[]")
                else:
                    if group is None:
                        raise ValueError("no group definition before value")
                    val = line 
                    self.gdict.setdefault(group, []).append(val)
                    self.rgdict.setdefault(val, []).append(group)

    def setdefault(self, group):
        self.default = group

    def groups(self):
        return self.gdict.keys()

    def values(self):
        return self.rgdict.keys()

    def ingroup(self, group, val):
        if self.rgdict(val):
            return val in self.rgdict[val]
        else:
            return False

    def get_group(self, val):
        if val in self.rgdict:
            return self.rgdict[val]
        else:
            return []

    def get_value(self, group):
        if group in self.gdict:
            return self.gdict[group]
        else:
            return []

    def iter_def(self):
        for group, l_val in self.gdict.iteritems():
            for val in l_val:
                yield group, val


def gettuple(conf, section, name):
    ret = conf.get(section, name)
    return tuple(e.strip() for e in ret.split(",")
                if not e.strip() == "")


def getlist(conf, section, name):
    ret = conf.get(section, name)
    if ret == "":
        return []
    else:
        return [e.strip() for e in ret.split(",")
                if not e.strip() == ""]


def getdt(conf, section, name):
    ret = conf.get(section, name)
    if ret == "":
        return None
    else:
        return datetime.datetime.strptime(ret.strip(), "%Y-%m-%d %H:%M:%S")


def getterm(conf, section, name):
    ret = conf.get(section, name)
    if ret == "":
        return None
    else:
        return tuple(datetime.datetime.strptime(e.strip(), "%Y-%m-%d %H:%M:%S")
                     for e in ret.split(","))


def getdur(conf, section, name):
    ret = conf.get(section, name)
    if ret == "":
        return None
    else:
        return str2dur(ret)


def str2dur(string):
    """
    Note:
        \d+s: \d seconds
        \d+m: \d minutes
        \d+h: \d hours
        \d+d: \d days
        \d+w: \d * 7 days
    """
    if "s" in string:
        num = int(string.partition("s")[0])
        return datetime.timedelta(seconds = num)
    elif "m" in string:
        num = int(string.partition("m")[0])
        return datetime.timedelta(minutes = num)
    elif "h" in string:
        num = int(string.partition("h")[0])
        return datetime.timedelta(hours = num)
    elif "d" in string:
        num = int(string.partition("d")[0])
        return datetime.timedelta(days = num)
    elif "w" in string:
        num = int(string.partition("w")[0])
        return datetime.timedelta(days = num * 7)
    else:
        raise ValueError("Duration string invalid")


def load_defaults(ex_conf = None):
    l_fn = [DEFAULT_CONFIG]
    if not ex_conf is None and len(ex_conf) > 0:
        l_fn = [DEFAULT_CONFIG] + ex_conf
    temp_conf = configparser.ConfigParser()
    temp_conf.read(l_fn)
    return temp_conf


def open_config(fn = None, ex_defaults = None):
    """
    Args:
        fn (str): Configuration file path.
        ex_defaults (List[str]): External default configurations.
                                 If you use other packages with amulog,
                                 use this to add its configurations.
        
    """
    conf = load_defaults(ex_defaults)
    if fn is not None:
        conf.read(fn)
    while conf.has_option(IMPORT_SECTION, IMPORT_OPTION):
        import_fn = conf.get(IMPORT_SECTION, IMPORT_OPTION)
        conf.remove_option(IMPORT_SECTION, IMPORT_OPTION)
        conf.read(import_fn)
    return conf


# common objects for logging
def set_common_logging(conf, logger = None, logger_name = None,
        lv = logging.INFO):
    """
    Args:
        conf
        logger (logging.Logger or list[logging.Logger])
        logger_name (str or list[str])
        lv (int): logging level
    Returns:
        logging.SomeHandler
    """
    fn = conf.get("general", "logging")
    fmt = logging.Formatter(
            fmt = "%(asctime)s %(levelname)s (%(processName)s) %(message)s", 
            datefmt = "%Y-%m-%d %H:%M:%S")
    #lv = logging.INFO
    if fn == "":
        ch = logging.StreamHandler()
    else:
        ch = logging.FileHandler(fn)
    ch.setFormatter(fmt)
    ch.setLevel(lv)

    temp_loggers = []
    if logger is None:
        pass
    elif isinstance(logger, list):
        temp_loggers += logger
    elif isinstance(logger, logging.Logger):
        temp_loggers.append(logger)
    else:
        raise TypeError
    if logger_name is None:
        pass
    elif isinstance(logger, list):
        temp_loggers += [logging.getLogger(ln) for ln in logger_name]
    elif isinstance(logger_name, str):
        temp_loggers.append(logging.getLogger(logger_name))
    else:
        raise TypeError

    for l in temp_loggers:
        l.setLevel(lv)
        l.addHandler(ch)
    return ch


def release_common_logging(ch, logger = None, logger_name = None):
    temp_loggers = []
    if logger is None:
        pass
    elif isinstance(logger, list):
        temp_loggers += logger
    elif isinstance(logger, logging.Logger):
        temp_loggers.append(logger)
    else:
        raise TypeError
    if logger_name is None:
        pass
    elif isinstance(logger, list):
        temp_loggers += [logging.getLogger(ln) for ln in logger_name]
    elif isinstance(logger_name, str):
        temp_loggers.append(logging.getLogger(logger_name))
    else:
        raise TypeError

    for l in temp_loggers:
        l.removeHandler(ch)


def test_config(conf_name):
    conf = open_config(conf_name)
    for section in conf.sections():
        print("[{0}]".format(section))
        for option, value in conf.items(section):
            print("{0} = {1}".format(option, value))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage : {0} config".format(sys.argv[0]))
    conf = sys.argv[1]
    test_config(conf)

