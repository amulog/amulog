import os
import pickle
import sys
import logging
from typing import Optional
from importlib import import_module

import log2seq
from amulog import config
from amulog import strutil
from amulog import lt_common
from amulog import log_db
from amulog import host_alias

_logger = logging.getLogger(__package__)

ONLINE_COMMIT_INTERVAL = 1000
_MULTIPROCESS_LOCAL_OBJECTS = dict()


class LTManager(object):
    """A manager to generate log templates and their database.

    Attributes:
        #TODO

    """

    # adding lt to db (ltgen do not add)

    def __init__(self, conf, db, lttable, reset_db=False, parallel=False):
        self._conf = conf
        self._reset_db = reset_db
        self._filename = conf["manager"]["indata_filename"]
        self._fail_output = conf["manager"]["fail_output"]
        self._online_batchsize = conf.getint("manager", "online_batchsize")
        self._online_counter = 0
        self._offline_batchsize = conf.getint("manager", "offline_batchsize")
        self._drop_undefhost = conf.getboolean("manager", "undefined_host")
        self._shuffle_import = conf.getboolean("log_template_import", "shuffle")

        self._db = db
        self._lttable = lttable
        self._table = lt_common.TemplateTable()
        self._ltgen: Optional[lt_common.LTGen] = None

        self._pool = None
        if parallel:
            tmp_n_proc = conf["manager"]["n_process"]
            if tmp_n_proc.isdigit():
                n_proc = int(tmp_n_proc)
            else:
                n_proc = os.cpu_count()
            ltgen_kwargs = {"conf": conf,
                            "table": None,  # individual table for child process
                            "shuffle": self._shuffle_import}
            from multiprocessing import Pool
            self._pool = Pool(processes=n_proc,
                              initializer=self._init_pool,
                              initargs=(ltgen_kwargs,))
        else:
            self._lp = load_log2seq(self._conf)
            self._ha = host_alias.init_hostalias(self._conf)
            self._drop_undefhost = conf.getboolean("manager", "undefined_host")
            self._ltgen = init_ltgen_methods(self._conf, self._table)

        self._ltgroup = init_ltgroup(self._conf, self._lttable)
        if (not self._reset_db) and (self._ltgroup is not None):
            self._ltgroup.restore_ltg(self._db, self._lttable)

    @property
    def template_table(self):
        return self._table

    @staticmethod
    def _init_pool(ltgen_kwargs):
        objects = {}
        conf = ltgen_kwargs["conf"]
        objects["lp"] = load_log2seq(conf)
        objects["ha"] = host_alias.init_hostalias(conf)
        objects["drop_undefhost"] = conf.getboolean("manager", "undefined_host")

        ltgen = init_ltgen_methods(**ltgen_kwargs)
        assert not ltgen.is_stateful(), \
            "parallel processing is limited to stateless methods"
        objects["ltgen"] = ltgen

        global _MULTIPROCESS_LOCAL_OBJECTS
        _MULTIPROCESS_LOCAL_OBJECTS = objects

    @staticmethod
    def _pool_task(batch):
        lp = _MULTIPROCESS_LOCAL_OBJECTS["lp"]
        ha = _MULTIPROCESS_LOCAL_OBJECTS["ha"]
        ltgen = _MULTIPROCESS_LOCAL_OBJECTS["ltgen"]
        drop_undefhost = _MULTIPROCESS_LOCAL_OBJECTS["drop_undefhost"]

        ret = []
        for message_id, line in batch:
            pline = parse_line(strutil.add_esc(line), lp)
            pline = normalize_pline(pline, ha, drop_undefhost)
            if pline is None:
                ret.append([message_id, None, None])
            else:
                tpl = ltgen.generate_tpl(pline)
                ret.append([message_id, pline, tpl])
        return ret

    def _process_offline_parallel(self, iterable_lines):
        def _sigterm_handler():
            raise KeyboardInterrupt

        import signal
        signal.signal(signal.SIGTERM, _sigterm_handler)

        l_input = [(mid, line)
                   for mid, line in enumerate(iterable_lines)]
        iter_batch = (l_input[i::self._offline_batchsize]
                      for i in range(self._offline_batchsize))

        try:
            d_pline = {}
            d_tid = {}
            for ret in self._pool.imap_unordered(self._pool_task, iter_batch):
                for mid, pline, tpl in ret:
                    d_pline[mid] = pline
                    if tpl is None:
                        continue
                    if self._table.exists(tpl):
                        tid = self._table.get_tid(tpl)
                    else:
                        tid = self._table.add(tpl)
                    d_tid[mid] = tid
            self._pool.close()
        except KeyboardInterrupt:
            self._pool.terminate()
            exit()
        else:
            return d_pline, d_tid

    def _process_offline_single(self, iterable_lines):
        d_pline = {}
        for mid, line in enumerate(iterable_lines):
            pline = parse_line(strutil.add_esc(line), self._lp)
            pline = normalize_pline(pline, self._ha, self._drop_undefhost)
            d_pline[mid] = pline

        offline_input = {mid: pline for mid, pline in d_pline.items()
                         if pline is not None}
        d_tid = self._ltgen.process_offline(offline_input)
        return d_pline, d_tid

    def process_offline(self, list_lines):
        if self._pool is None:
            d_pline, d_tid = self._process_offline_single(list_lines)
        else:
            if (not self._reset_db) and self._ltgen.is_stateful():
                msg = ("offline additional change is limited "
                       "to stateless ltgen methods")
                raise ValueError(msg)
            d_pline, d_tid = self._process_offline_parallel(list_lines)

        s_added = set()
        for mid, pline in d_pline.items():
            if pline is None or mid not in d_tid:
                self.fail_dump(list_lines[mid])
                continue

            tid = d_tid[mid]
            if tid is None:
                self.fail_dump(list_lines[mid])
                continue
            elif tid in s_added:
                ltid = self._table.get_ltid(tid)
                self.count_lt(ltid)
                ltline = self._lttable[ltid]
            else:
                tpl = self._table.get_template(tid)
                ltline = self.add_lt(tpl, pline[log2seq.KEY_SYMBOLS])
                self._table.add_ltid(tid, ltline.ltid)
                s_added.add(tid)
            self.add_line(pline, ltline)

        self.commit_db()
        self.dump()

    def get_parsed_line(self, line):
        pline = parse_line(strutil.add_esc(line), self._lp)
        pline = normalize_pline(pline, self._ha, self._drop_undefhost)
        return pline

    def process_line(self, line):
        pline = self.get_parsed_line(line)
        if pline is None:
            self.fail_dump(line)
            return None

        tid, state = self._ltgen.process_line(pline)
        if tid is None:
            self.fail_dump(line)
            return None
        elif state == lt_common.LTGen.state_added:
            tpl = self._ltgen.get_tpl(tid)
            ltline = self.add_lt(tpl, pline[log2seq.KEY_SYMBOLS],
                                 add_group=True)
            self._table.add_ltid(tid, ltline.ltid)
        elif state == lt_common.LTGen.state_changed:
            tpl = self._ltgen.get_tpl(tid)
            ltid = self._table.get_ltid(tid)
            self.replace_and_count_lt(ltid, tpl)
            ltline = self._lttable[ltid]
        elif state == lt_common.LTGen.state_unchanged:
            ltid = self._table.get_ltid(tid)
            self.count_lt(ltid)
            ltline = self._lttable[ltid]
        else:
            raise AssertionError

        self.add_line(pline, ltline)
        self._online_counter += 1
        if self._online_counter >= self._online_batchsize:
            self.commit_db()
            self._online_counter = 0
        return ltline

    def process_online_end(self):
        if isinstance(self._ltgroup, lt_common.LTGroupOffline):
            self.remake_ltg()

    def add_line(self, pline, ltline):
        """Add a log message to DB.

        Args:
            ltline (lt_common.LogTemplate): A log template object.
            pline (dict): A parsed log message with log2seq.

        Returns:
            LogMessage: An annotated log message instance.
        """
        dt = pline[log2seq.KEY_TIMESTAMP]
        host = pline["host"]
        l_w = pline[log2seq.KEY_WORDS]
        kwargs = {"ltid": ltline.ltid,
                  "dt": dt,
                  "host": host,
                  "l_w": l_w}
        if "lid" in pline:
            kwargs["lid"] = pline["lid"]
        new_lid = self._db.add_line(**kwargs)
        return log_db.LogMessage(new_lid, ltline,
                                 dt, host, l_w)

    def add_lt(self, l_w, l_s, count=1, add_group=False):
        # add new lt to db and table
        ltid = self._lttable.next_ltid()
        ltline = lt_common.LogTemplate(ltid, None, l_w, l_s, count)
        # if self._ltgroup is not None:
        if add_group and isinstance(self._ltgroup, lt_common.LTGroupOnline):
            ltgid = self._ltgroup.add(ltline)
        else:
            ltgid = ltid
        ltline.ltgid = ltgid
        self._lttable.add_lt(ltline)
        self._db.add_lt(ltline)
        self._db.add_ltg(ltline.ltid, ltline.ltgid)
        return ltline

    def replace_lt(self, ltid, l_w, l_s=None, count=None):
        self._lttable[ltid].replace(l_w, l_s, count)
        self._db.update_lt(ltid, l_w, l_s, count)

    def replace_and_count_lt(self, ltid, l_w, l_s=None):
        cnt = self._lttable[ltid].increment()
        self._lttable[ltid].replace(l_w, l_s, None)
        self._db.update_lt(ltid, l_w, l_s, cnt)

    def count_lt(self, ltid):
        cnt = self._lttable[ltid].increment()
        self._db.update_lt(ltid, None, None, cnt)

    def remove_lt(self, ltid):
        self._lttable.remove_lt(ltid)
        self._db.remove_lt(ltid)

    def remake_ltg(self):
        self._db.reset_ltg()
        self._ltgroup.make()
        # self._ltgroup.update_lttable(self._lttable)
        for ltline in self._lttable:
            self._db.add_ltg(ltline.ltid, ltline.ltgid)

    def load_internal_data(self):
        if os.path.exists(self._filename) and not self._reset_db:
            self.load()

    def commit_db(self):
        """Commit requested changes in LogDB.
        """
        self._db.commit()

    def load(self):
        with open(self._filename, 'rb') as f:
            obj = pickle.load(f)
        table_data, ltgen_data, ltgroup_data = obj
        self._table.load(table_data)
        self._ltgen.load(ltgen_data)
        self._ltgroup.load(ltgroup_data)

    def dump(self):
        table_data = self._table.dumpobj()
        if self._ltgen is None:
            ltgen_data = None
        else:
            ltgen_data = self._ltgen.dumpobj()
        if self._ltgroup is None:
            ltgroup_data = None
        else:
            ltgroup_data = self._ltgroup.dumpobj()
        obj = (table_data, ltgen_data, ltgroup_data)
        with open(self._filename, 'wb') as f:
            pickle.dump(obj, f)

    def fail_dump(self, msg):
        with open(self._fail_output, 'a') as f:
            f.write(msg)


def init_manager(ld):
    return LTManager(ld.conf, ld.db, ld.lttable)


def init_ltgen_methods(conf, table, lt_methods=None, shuffle=None):
    if lt_methods is None:
        lt_methods = config.getlist(conf, "log_template", "lt_methods")
    if shuffle is None:
        shuffle = conf.getboolean("log_template_import", "shuffle")

    if len(lt_methods) > 1:
        l_ltgen = []
        import_index = None
        for index, method_name in enumerate(lt_methods):
            ltgen = init_ltgen(conf, table, method_name, shuffle)
            l_ltgen.append(ltgen)
            if method_name == "import":
                import_index = index
        return lt_common.LTGenJoint(table, l_ltgen, import_index)
    elif len(lt_methods) == 1:
        return init_ltgen(conf, table, lt_methods[0], shuffle)
    else:
        raise ValueError


def init_ltgen(conf, table, method, shuffle=False):
    kwargs = {"conf": conf,
              "table": table,
              "shuffle": shuffle}
    if method == "import":
        from . import lt_import
        return lt_import.init_ltgen_import(**kwargs)
    elif method == "import-ext":
        from . import lt_import_ext
        return lt_import_ext.init_ltgen_import_ext(**kwargs)
    elif method == "crf":
        from amulog.alg.crf import lt_crf
        return lt_crf.init_ltgen_crf(**kwargs)
    elif method == "re":
        from amulog import lt_regex
        return lt_regex.init_ltgen_regex(**kwargs)
    elif method == "va":
        from . import lt_va
        return lt_va.init_ltgen_va(**kwargs)
    else:
        modname = "amulog.alg." + method
        alg_module = import_module(modname)
        return alg_module.init_ltgen(**kwargs)


def init_ltgroup(conf, lttable):
    ltg_alg = conf.get("log_template", "ltgroup_alg")
    if ltg_alg == "shiso":
        from amulog.alg.shiso import shiso
        ltgroup = shiso.LTGroupSHISO(lttable,
                                     ngram_length=conf.getint(
                                            "log_template_shiso", "ltgroup_ngram_length"),
                                     th_lookup=conf.getfloat(
                                            "log_template_shiso", "ltgroup_th_lookup"),
                                     th_distance=conf.getfloat(
                                            "log_template_shiso", "ltgroup_th_distance"),
                                     mem_ngram=conf.getboolean(
                                            "log_template_shiso", "ltgroup_mem_ngram")
                                     )
    elif ltg_alg == "ssdeep":
        from . import lt_misc
        ltgroup = lt_misc.LTGroupFuzzyHash(lttable)
    elif ltg_alg == "semantics":
        from . import ltg_semantics
        ltgroup = ltg_semantics.init_ltgroup_semantics(conf, lttable)
    elif ltg_alg == "none":
        ltgroup = None
    else:
        raise ValueError("ltgroup_alg({0}) invalid".format(ltg_alg))

    return ltgroup


def load_log2seq(conf):
    """Return parser object log2seq.LogParser .

    Amulog accepts following additional keys extracted by log2seq.
    - host: device hostname, mandatory
    - lid: Log message identifier, optional

    The configuration is described in a python script file.
    The file is specified in database.parser_script option,
    and the "parser" module variable in the script is returned.
    Otherwise, this function returns log2seq default parser.

    Args:
        conf: config object.

    Returns:
        log2seq.LogParser

    """
    fp = conf.get("manager", "parser_script")
    if len(fp.strip()) == 0:
        return log2seq.init_parser()
    else:
        import sys
        from importlib import import_module
        path = os.path.dirname(fp)
        sys.path.append(os.path.abspath(path))
        libname = os.path.splitext(os.path.basename(fp))[0]
        script_mod = import_module(libname)
        lp = script_mod.parser
        return lp


def parse_line(line, lp):
    try:
        parsed_line = lp.process_line(line)
    except log2seq.LogParseFailure:
        return None
    else:
        if len(parsed_line[log2seq.KEY_WORDS]) == 0:
            msg = "pass empty message {0}".format(line.strip("\n"))
            _logger.debug(msg)
            return None
        else:
            return parsed_line


def normalize_pline(pline, ha, drop_undefhost=False):
    if pline is None:
        return None

    if "lid" in pline:
        pline["lid"] = int(pline["lid"])

    org_host = pline["host"]
    host = ha.resolve_host(org_host)
    if host is None:
        if drop_undefhost:
            return None
        else:
            host = org_host
    pline["host"] = host

    return pline


def _open_file(fp, **kwargs):
    ext = os.path.splitext(fp)[-1].lstrip(".")
    if ext == "bz2":
        import bz2
        open_func = bz2.open
    elif ext == "gz":
        import gzip
        open_func = gzip.open
    else:
        ext = "text"
        open_func = open

    _logger.info("processing {0} file {1}".format(ext, fp))
    return open_func(fp, 'rt', **kwargs)


def iter_lines(targets, encoding="utf-8", errors="ignore"):
    for fp in targets:
        if os.path.isdir(fp):
            sys.stderr.write(
                "{0} is a directory, fail to process\n".format(fp))
            sys.stderr.write(
                "Use -r if you need to search log data recursively\n")
        else:
            if not os.path.isfile(fp):
                raise IOError("File {0} not found".format(fp))
            with _open_file(fp, encoding=encoding, errors=errors) as f:
                for line in f:
                    yield line


def iter_plines(conf, targets, pass_none=True):
    lp = load_log2seq(conf)
    ha = host_alias.init_hostalias(conf)
    drop_undefhost = conf.getboolean("manager", "undefined_host")

    for line in iter_lines(targets):
        pline = parse_line(strutil.add_esc(line), lp)
        pline = normalize_pline(pline, ha, drop_undefhost)
        if pline is None and pass_none:
            pass
        else:
            yield pline


def process_files_online(conf, targets, reset_db):
    """Add log messages to DB from files.

    Args:
        conf (config.ExtendedConfigParser): A common configuration object.
        targets (List[str]): A sequence of filepaths to process.
        reset_db (bool): True if DB needs to reset before adding.

    Raises:
        IOError: If a file in targets not found.
    """
    def _sigterm_handler():
        raise KeyboardInterrupt

    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)

    msg = "amulog online processing"
    _logger.info(msg)

    ld = log_db.LogData(conf, edit=True, reset_db=reset_db)
    ltm = LTManager(conf, ld.db, ld.lttable, reset_db=reset_db)

    try:
        for line in iter_lines(targets):
            ltm.process_line(line)
    except KeyboardInterrupt:
        pass
    finally:
        ltm.process_online_end()
        ltm.commit_db()
        ltm.dump()


def process_files_offline(conf, targets, reset_db, parallel=False):
    """Add log messages to DB from files. This function do NOT process
    messages incrementally. Use this to avoid bad-start problem of
    log template generation with clustering or training methods.

    Note:
        This function needs large memory space.

    Args:
        conf (config.ExtendedConfigParser): A common configuration object.
        targets (List[str]): A sequence of filepaths to process.
        reset_db (bool): True if DB needs to reset before adding.
        parallel (bool, optional): Use multiprocessing.

    Raises:
        IOError: If a file in targets not found.
    """
    msg = "amulog offline processing"
    if parallel:
        msg += " in parallel"
    _logger.info(msg)

    ld = log_db.LogData(conf, edit=True, reset_db=reset_db)
    ltm = LTManager(conf, ld.db, ld.lttable, reset_db=reset_db,
                    parallel=parallel)

    l_line = [line for line in iter_lines(targets)]
    ltm.process_offline(l_line)


def data_from_data(conf, targets, dirname, method, reset):
    rod = log_db.RestoreOriginalData(dirname, method=method, reset=reset)
    lp = load_log2seq(conf)
    ha = host_alias.init_hostalias(conf)
    drop_undefhost = conf.getboolean("database", "undefined_host")
    for line in iter_lines(targets):
        pline = parse_line(line, lp)
        pline = normalize_pline(pline, ha, drop_undefhost)
        if pline is None:
            continue
        rod.add_str(pline[log2seq.KEY_TIMESTAMP], line)
    rod.commit()


def remake_ltgroup(conf):
    ld = log_db.LogData(conf, edit=True, reset_db=False)
    ltm = LTManager(conf, ld.db, ld.lttable, reset_db=False)
    ltm.remake_ltg()
    ltm.commit_db()
