#!/usr/bin/env python
# coding: utf-8

import sys
import os
import datetime
import logging
import configparser
from collections import defaultdict
from dateutil.tz import tzlocal

CONFIG_ENV = "AMULOG_CONFIG"
DEFAULT_CONFIG = "/".join((os.path.dirname(os.path.abspath(__file__)),
                           "data/config.conf.default"))
LOAD_SECTION = 'general'
LOAD_OPTION = 'base_filename'
IMPORT_SECTION = 'general'
IMPORT_OPTION = 'import'


class GroupDef:
    """
    Define grouping by external text
    Rules:
        description after # in a line will be recognized as comment
        line "[GROUP_NAME]" will change group to set
        other lines add elements in the group set with GROUP_NAME line
    """

    def __init__(self, fn, default_val=None):
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
        return val in self.gdict[group]

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
        for group, l_val in self.gdict.items():
            for val in l_val:
                yield group, val


def gettuple(conf, section, name, sep=","):
    ret = conf.get(section, name)
    if ret.strip() == "":
        return tuple()
    else:
        return tuple(e.strip() for e in ret.split(sep)
                     if not len(e.strip()) == 0)


def getlist(conf, section, name, sep=","):
    ret = conf.get(section, name)
    if ret.strip() == "":
        return []
    else:
        return [e.strip() for e in ret.split(sep)
                if not len(e.strip()) == 0]


def getdict(conf, section, name):
    val = conf.get(section, name)
    ret = {}
    for e in val.split(","):
        e = e.strip()
        assert "=" in e
        k, _, v = e.partition("=")
        ret[k] = v
    return ret


def getdt(conf, section, name):
    ret = conf.get(section, name)
    if ret == "":
        return None
    else:
        dt = datetime.datetime.strptime(ret.strip(), "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=tzlocal())
        return dt


def getterm(conf, section, name):
    val = conf.get(section, name)
    if val == "":
        return None
    else:
        ret = []
        for e in val.split(","):
            dt = datetime.datetime.strptime(e.strip(), "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=tzlocal())
            ret.append(dt)
        return ret


def getdur(conf, section, name):
    ret = conf.get(section, name)
    if ret == "":
        return None
    else:
        return str2dur(ret)


def getname(conf):
    return conf.get(LOAD_SECTION, LOAD_OPTION)


def str2dur(string):
    """
    Note:
        \\d+s: \\d seconds
        \\d+m: \\d minutes
        \\d+h: \\d hours
        \\d+d: \\d days
        \\d+w: \\d * 7 days
    """
    if "s" in string:
        num = int(string.partition("s")[0])
        return datetime.timedelta(seconds=num)
    elif "m" in string:
        num = int(string.partition("m")[0])
        return datetime.timedelta(minutes=num)
    elif "h" in string:
        num = int(string.partition("h")[0])
        return datetime.timedelta(hours=num)
    elif "d" in string:
        num = int(string.partition("d")[0])
        return datetime.timedelta(days=num)
    elif "w" in string:
        num = int(string.partition("w")[0])
        return datetime.timedelta(days=num * 7)
    else:
        raise ValueError("Duration string invalid")


def dur2str(td):
    samples = [("m", 60), ("h", 60), ("d", 24), ("w", 7)]

    footer = "s"
    val = int(td.total_seconds())
    for tmp_footer, mod in samples:
        if val % mod == 0:
            footer = tmp_footer
            val = val // mod
        else:
            break
    return str(val) + footer


def merge_config(conf1, conf2):
    """Overwrite conf1 with conf2."""
    for sec in conf2.sections():
        for opt in conf2.options(sec):
            if conf1.has_option(sec, opt):
                pass
            else:
                if not conf1.has_section(sec):
                    conf1[sec] = {}
                conf1.set(sec, opt, conf2[sec][opt])
    return conf1


def load_defaults(iterable_conf_path=None):
    l_fn = iterable_conf_path
    temp_conf = configparser.ConfigParser()
    for fn in l_fn:
        ret = temp_conf.read(fn)
        if len(ret) == 0:
            raise IOError("config load error ({0})".format(fn))

    return temp_conf


def _load_imports(conf):
    while conf.has_option(IMPORT_SECTION, IMPORT_OPTION):
        if conf[IMPORT_SECTION][IMPORT_OPTION] == "":
            break
        import_fn = conf[IMPORT_SECTION][IMPORT_OPTION]
        conf.remove_option(IMPORT_SECTION, IMPORT_OPTION)
        import_conf = configparser.ConfigParser()
        if os.path.exists(import_fn):
            import_conf.read(import_fn)
        else:
            raise IOError("config load error ({0})".format(import_fn))
        conf = merge_config(conf, import_conf)

    return conf


def open_config(fn=None, env=CONFIG_ENV,
                ex_defaults=None,
                base_default=True, ignore_import=False,
                verbose=True):
    """
    Args:
        fn (str, optional): Configuration file path.
        env (str, optional): If fn is None, use Environment Variable of given name.
        ex_defaults (List[str], optional): External default configurations.
                                 If you use other packages with amulog,
                                 use this to add its configurations.
        base_default (bool, optional): Use amulog default values for missing options.
        ignore_import (bool, optional): Ignore general.import.
        
    """

#    def _import_config(_conf, _import_conf):
#        for sec in _import_conf.sections():
#            for opt in _import_conf.options(sec):
#                if _conf.has_option(sec, opt):
#                    pass
#                else:
#                    if not _conf.has_section(sec):
#                        _conf[sec] = {}
#                    _conf.set(sec, opt, _import_conf[sec][opt])
#        return _conf

    conf = configparser.ConfigParser()

    if fn is None and env is not None:
        fn = os.environ.get(env)

    if fn is None:
        if verbose:
            print("Processing with default configuration ...")
    else:
        if not os.path.exists(fn):
            raise IOError("{0} not found".format(fn))
        ret = conf.read(fn)
        if len(ret) == 0:
            raise IOError("config load error ({0})".format(fn))
        conf.set(LOAD_SECTION, LOAD_OPTION, fn)

    if not ignore_import:
        conf = _load_imports(conf)
#        while conf.has_option(IMPORT_SECTION, IMPORT_OPTION):
#            if conf[IMPORT_SECTION][IMPORT_OPTION] == "":
#                break
#            import_fn = conf.get(IMPORT_SECTION, IMPORT_OPTION)
#            conf.remove_option(IMPORT_SECTION, IMPORT_OPTION)
#            import_conf = configparser.ConfigParser()
#            ret = import_conf.read(import_fn)
#            if len(ret) == 0:
#                raise IOError("config load error ({0})".format(import_fn))
#            _import_config(conf, import_conf)

    l_defaults = []
    if base_default:
        l_defaults.append(DEFAULT_CONFIG)
    if ex_defaults:
        l_defaults += ex_defaults
    if l_defaults:
        default_conf = load_defaults(l_defaults)
        merge_config(conf, default_conf)

    return conf


def show_default_config(ex_defaults=None):
    conf = load_defaults(ex_defaults)
    for section in conf.sections():
        print("[{0}]".format(section))
        for option in conf.options(section):
            print("{0} = {1}".format(option, conf[section][option]))
        print()


def show_config_diff(l_conf_name, l_conf=None, ex_defaults=None):
    from . import common
    if l_conf is None:
        l_conf = [open_config(conf_path, ex_defaults)
                  for conf_path in l_conf_name]

    def iter_secopt(_conf):
        for _sec in _conf.sections():
            for _opt in _conf.options(_sec):
                yield _sec, _opt

    keys = set()
    for conf in l_conf:
        keys = keys | set(iter_secopt(conf))

    d = defaultdict(lambda: defaultdict(set))
    for conf in l_conf:
        for key in keys:
            sec, opt = key
            if conf.has_option(sec, opt):
                d[sec][opt].add(conf[sec][opt])

    d_keys = defaultdict(list)
    for sec in d:
        for opt in d[sec]:
            if len(d[sec][opt]) > 1:
                d_keys[sec].append(opt)

    for sec, l_opt in d_keys.items():
        print("[{0}]".format(sec))
        for opt in l_opt:
            print("{0} = ...".format(opt))
            buf = []
            for name, conf in zip(l_conf_name, l_conf):
                if conf.has_option(sec, opt):
                    buf.append([name, conf[sec][opt]])
            print(common.cli_table(buf, spl=": "))
        print()


def write(name, conf):
    with open(name, "w") as f:
        conf.write(f)


def minimize(conf, ex_defaults=None, ignore_import=False):
    # def add_opt(conf, sec, opt, val):
    #     if sec not in conf:
    #         conf[sec] = {}
    #     conf[sec][opt] = val

    new_conf = configparser.ConfigParser()
    default_conf = open_config(None, ex_defaults)
    if not ignore_import:
        conf = _load_imports(conf)

    # import_conf = configparser.ConfigParser()
    # if conf.has_option(IMPORT_SECTION, IMPORT_OPTION):
    #     import_fn = conf[IMPORT_SECTION][IMPORT_OPTION]
    #     if os.path.exists(import_fn):
    #         import_conf.read(import_fn)

    for sec in conf.sections():
        for opt in conf.options(sec):
            # if import_conf.has_option(sec, opt):
            #     if import_conf[sec][opt] == conf[sec][opt]:
            #         continue

            if not default_conf.has_option(sec, opt):
                # undefine key in defaults
                if sec not in new_conf:
                    new_conf[sec] = {}
                new_conf[sec][opt] = conf[sec][opt]
            elif conf[sec][opt] == default_conf[sec][opt]:
                # same value from defaults
                pass
            else:
                # different value from defaults
                if sec not in new_conf:
                    new_conf[sec] = {}
                new_conf[sec][opt] = conf[sec][opt]

    return new_conf


def config_minimum(fn, ex_defaults=None):
    conf = open_config(fn, ex_defaults)
    default_conf = open_config(None)

    for sec in conf.sections():
        l_diff = []
        for opt in conf.options(sec):
            if not default_conf.has_option(sec, opt):
                l_diff.append((opt, conf[sec][opt]))
            elif conf[sec][opt] == default_conf[sec][opt]:
                pass
            else:
                l_diff.append((opt, conf[sec][opt]))
        if len(l_diff) > 0:
            print("[{0}]".format(sec))
            for opt, val in l_diff:
                print("{0} = {1}".format(opt, val))
            print()


def config_group_edit(l_conf_name, d_rule, l_conf=None):
    if l_conf is None:
        l_conf = [open_config(conf_path, base_default=False, ignore_import=True)
                  for conf_path in l_conf_name]
    for key, l_val in d_rule.items():
        if isinstance(key, str):
            sec, opt = key.split(".")
        elif len(key) == 2:
            sec, opt = key
        else:
            raise ValueError
        assert len(l_val) == len(l_conf_name), "bad value length"
        for conf, val in zip(l_conf, l_val):
            if not conf.has_section(sec):
                conf[sec] = {}
            conf[sec][opt] = str(val)

    for name, conf in zip(l_conf_name, l_conf):
        write(name, conf)


def config_shadow(n=1, cond=None, incr=None, fn=None, output=None,
                  ignore_overwrite=False, ex_defaults=None):
    if cond is None:
        cond = {}
    if incr is None:
        incr = []

    l_ret = []
    for i in range(n):
        conf = open_config(fn, ex_defaults, ignore_import=True)
        for key, val in cond.items():
            sec, opt = key.split(".")
            conf[sec][opt] = val
        for key in incr:
            sec, opt = key.split(".")
            conf[sec][opt] = conf[sec][opt] + str(i)

        if n == 1:
            footer = ""
        else:
            footer = str(i)
        if output is None:
            if fn is None:
                temp_output = "copy.conf" + footer
            else:
                temp_output = fn + str(i)
        else:
            temp_output = output + footer

        if os.path.exists(temp_output) and not ignore_overwrite:
            raise IOError("{0} already exists, use -f to ignore".format(
                temp_output))

        write(temp_output, minimize(conf, ex_defaults, ignore_import=True))
        l_ret.append(temp_output)
        print("{0}".format(temp_output))
    return l_ret


def check_all_diff(l_conf_name, keys, l_conf=None):
    """Return True if all configs have different values in given keys."""
    if l_conf is None:
        l_conf = [open_config(conf_name) for conf_name in l_conf_name]
    ret = True
    for key in keys:
        if isinstance(key, tuple) or isinstance(key, list):
            sec, opt = key
        else:
            sec, opt = key.split(".")
        values = [conf[sec][opt] for conf in l_conf]
        if len(values) == len(set(values)):
            pass
        else:
            print("Notice: Same {0} settings found".format(key))
            for name, conf in zip(l_conf_name, l_conf):
                print("{0} : {1}".format(name, conf[sec][opt]))
            print()
            ret = False
    return ret


def read_config_group(cgroup_path):
    ret = []
    with open(cgroup_path, "r") as f:
        for line in f:
            fp = line.strip()
            if fp == "":
                pass
            else:
                ret.append(fp)
    return ret


def load_config_group(cgroup_path, ex_defaults=None):
    l_conf = []
    for fp in read_config_group(cgroup_path):
        if os.path.exists(fp):
            l_conf.append(open_config(fp, ex_defaults))
        else:
            sys.stderr.write(
                "Warning: Invalid config name {0}".format(fp))
    return l_conf


def dump_config_group(cgroup_name, l_conf_name):
    with open(cgroup_name, "w") as f:
        f.write("\n".join(l_conf_name))


def set_logging_stdio(logger=None, logger_name=None,
                      lv=logging.INFO):
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s (%(processName)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
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
    elif isinstance(logger_name, list):
        temp_loggers += [logging.getLogger(ln) for ln in logger_name]
    elif isinstance(logger_name, str):
        temp_loggers.append(logging.getLogger(logger_name))
    else:
        raise TypeError

    for logger in temp_loggers:
        logger.setLevel(lv)
        logger.addHandler(ch)
    return ch


# common objects for logging
def set_common_logging(conf, logger=None, logger_name=None,
                       lv=logging.INFO):
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
        fmt="%(asctime)s %(levelname)s (%(processName)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    # lv = logging.INFO
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
    elif isinstance(logger_name, list):
        temp_loggers += [logging.getLogger(ln) for ln in logger_name]
    elif isinstance(logger_name, str):
        temp_loggers.append(logging.getLogger(logger_name))
    else:
        raise TypeError

    for logger in temp_loggers:
        logger.setLevel(lv)
        logger.addHandler(ch)
        if lv <= logging.DEBUG:
            logger.debug("logger is on debug mode")

    return ch


def release_common_logging(ch, logger=None, logger_name=None):
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

    for logger in temp_loggers:
        logger.removeHandler(ch)
