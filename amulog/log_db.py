#!/usr/bin/env python
# coding: utf-8

"""
Construct and manage database of log messages, templates,
and grouping definitions.
"""

import datetime
import logging
from collections import defaultdict

from . import common
from . import strutil
from . import db_common
from . import lt_common

_logger = logging.getLogger(__package__)


class LogMessage:
    """An annotated log message.

    An instance have a set of information about 1 log message line,
    including timestamp, hostname, log template, and message contents.

    Attributes:
        lid (int): A message identifier in DB.
        lt (lt_common.LogTemplate): Log template of this message.
        dt (datetime.datetime): A timestamp for this message.
        host (str): A hostname that output this message.
        l_w (List(str)): A sequence of words in this message.

    """

    def __init__(self, lid, lt, dt, host, l_w):
        """
        Args:
            lid (int): A message identifier in DB.
            lt (lt_common.LogTemplate): Log template of this message.
            dt (datetime.datetime): A timestamp for this message.
            host (str): A hostname that output this message.
            l_w (List(str)): A sequence of words in this message.

        """
        self.lid = lid
        self.lt = lt
        self.dt = dt
        self.host = host
        self.l_w = l_w

    def __str__(self):
        """str: Show attributes in 1 string."""
        return " ".join((str(self.dt), self.host, str(self.lt.ltid),
                         str(self.l_w)))

    def get(self, name):
        """Return value for given attribute."""
        if name == "lid":
            return self.lid
        elif name == "ltid":
            return self.lt.ltid
        elif name == "ltgid":
            return self.lt.ltgid
        elif name == "host":
            return self.host
        else:
            raise NotImplementedError

    def var(self):
        """str: Get sequence of all variable words in this message.
        Variable words mean what is presented with mask
        (defaults **) in log template.
        """
        return self.lt.var(self.l_w)

    def restore_message(self):
        """str: Get pseudo original log message
        except headers (timestamp and hostname).
        """
        return self.lt.restore_message(self.l_w)

    def restore_line(self):
        """str: Get original log message contents
        including headers (timestamp and hostname).
        """
        return " ".join((str(self.dt), str(self.host),
                         self.restore_message()))


class LogData:
    """Interface to get, add or edit log messages in DB.

    Attributes:
        conf (config.ExtendedConfigParser): A common configuration object.
        lttable (lt_common.LTTable): Log template table object.
        db (LogDB): Log database instance.
    """

    def __init__(self, conf, edit=False, reset_db=False):
        """
        Args:
            conf (config.ExtendedConfigParser): A common configuration object.
            edit (Optional[bool]): Defaults to False.
                True if database will be added or edited.
                False if database is used in readonly mode.
            reset_db (Optional[bool]): Defaults to False.
                If True, database will be reset before following process.

        """
        self.conf = conf
        self._reset_db = reset_db

        self.db = LogDB(conf, edit, reset_db)
        self.lttable = None
        if edit:
            if reset_db:
                # use empty lttable
                self.lttable = lt_common.LTTable()
            else:
                # load existing lttable to edit
                self.lttable = self.db.restore_lttable()
        else:
            # load existing lttable to read
            self.lttable = self.db.restore_lttable()

    def _row_to_lm(self, d_line):
        ltid = d_line.pop("ltid")
        d_line["lt"] = self.lttable[ltid]
        return LogMessage(**d_line)

    def add_line(self, lid, ltid, dt, host, l_w):
        """Directly add LogMessage to DB.
        Used in Anonymize functions."""
        self.db.add_line(lid=lid, ltid=ltid, dt=dt, host=host, l_w=l_w)

    def add_lt(self, ltobj):
        """Directly add LogTemplate to DB.
        Used in Anonymize functions."""
        self.db.add_lt(ltobj)
        self.db.add_ltg(ltobj.ltid, ltobj.ltgid)

    def get_line(self, lid):
        return self._row_to_lm(self.db.get_line(lid))

    def iter_lines(self, **kwargs):
        """Generate log messages in DB that satisfy conditions
        given in arguments. All arguments are defaults to None, and ignored.

        Keyword Args:
            lid (int): A message identifier in DB.
            ltid (int): A log template identifier.
            ltgid (int): A log template grouping identifier.
            dts (datetime.datetime): Condition for timestamp.
                Messages after 'dts' will be yield.
            dte (datetime.datetime): Condition for timestamp.
                Messages before 'dte' will be yield.
            host (str): A source hostname of the message.

        Yields:
            LogMessage: An annotated log message instance
                which satisfies all given conditions.
        """
        _logger.debug("iter_lines called ({0})".format(" ".join(
            ["{0}:{1}".format(k, v) for k, v in kwargs.items()
             if v is not None])
        ))
        limit = None
        if "limit" in kwargs:
            limit = kwargs.pop("limit")
        assert len(kwargs) >= 1, "empty arguments"
        for d_line in self.db.iter_lines(kwargs, limit=limit):
            yield self._row_to_lm(d_line)

    def iter_all(self):
        for d_line in self.db.iter_all():
            yield self._row_to_lm(d_line)

    def get_tags(self, **kwargs):
        """Search tags for given template identifiers.
        One of the args should be given.

        Keyword Args:
            ltid (Optional[int]): Search tags for given log template identifier.
            ltgid (Optional[int]): Search tags for given group identifier.

        Yields:
            str
        """
        assert len(kwargs) >= 1, "empty arguments"
        return self.db.get_tags(**kwargs)

    def count_lines(self):
        """int: Number of all messages in DB."""
        return self.db.count_lines()

    def dt_term(self):
        """datetime.datetime, datetime.datetime:
            Period of all registered log messages in DB."""
        return self.db.dt_term()

    def whole_term(self):
        """datetime.datetime, datetime.datetime:
            Period of dates that include all log messages in DB.
        """
        # return date term that cover all log data
        top_dt, end_dt = self.db.dt_term()
        top_dt = datetime.datetime.combine(
            top_dt.date(), datetime.time()
        )
        end_dt = datetime.datetime.combine(
            end_dt.date(), datetime.time()
        ) + datetime.timedelta(days=1)
        return top_dt, end_dt

    def whole_host_lt(self, dts=None, dte=None):
        """List[str, str]: Sequence of all combinations of
        hostname and ltids in DB."""
        return self.db.whole_host_lt(dts=dts, dte=dte)

    def whole_host_ltg(self, dts=None, dte=None):
        """List[str, str]: Sequence of all combinations of
        hostname and ltgids in DB."""
        ret = set()
        for host, ltid in self.whole_host_lt(dts=dts, dte=dte):
            ret.add((host, self.ltgid_from_ltid(ltid)))
        return list(set(ret))

    def whole_host(self, dts=None, dte=None):
        """List[str]: Sequence of all source hostname in DB."""
        return self.db.whole_host(dts=dts, dte=dte)

    def count_lt(self):
        """int: Number of all log templates."""
        return self.db.count_lt()

    def count_ltg(self):
        """int: Number of all log template groups."""
        return self.db.count_ltg()

    def iter_lt(self):
        """Yields lt_common.LogTemplate: All log template instance in DB."""
        for ltline in self.lttable:
            yield ltline

    def lt(self, ltid):
        """Get log template instance of given log template identifier.

        Args:
            ltid (int): A log template identifier.

        Returns:
            lt_common.LogTemplate: A log template instance.

        """
        return self.lttable[ltid]

    def iter_gid(self, gid_name):
        if gid_name == "ltid":
            return [lt.ltid for lt in self.iter_lt()]
        elif gid_name == "ltgid":
            return self.iter_ltgid()

    def iter_ltgid(self):
        """Yields int: Get all identifiers of log template groups."""
        return self.db.iter_ltgid()

    def ltgid_from_ltid(self, ltid):
        lt = self.lttable[ltid]
        return lt.ltgid

    def ltg_members(self, ltgid):
        """Get all log templates in given log template group.

        Args:
            ltgid (int): A log template group identifier.

        Returns:
            List[lt_common.LogTemplates]: Sequence of all log template
                instances that belongs to given log template group.

        """
        return [self.lttable[ltid] for ltid in self.db.get_ltg_members(ltgid)]

    def show_lt_info(self, ltid):
        ltobj = self.lt(ltid)
        tags = [tag for tag in self.get_tags(ltid=ltobj.ltid)]
        if tags:
            tag_str = "|".join(tags)
            return " ".join((str(ltobj.ltid), "({0})".format(tag_str),
                             str(ltobj), "({0})".format(ltobj.count)))
        else:
            return " ".join((str(ltobj.ltid),
                             str(ltobj), "({0})".format(ltobj.count)))

    def show_ltg_info(self, ltgid):
        tags = [tag for tag in self.get_tags(ltgid=ltgid)]
        l_ltobj = self.ltg_members(ltgid)
        length = len(l_ltobj)
        count = sum(ltobj.count for ltobj in l_ltobj)

        header = "[ltgroup {0} ({1} tpls, {2} lines)]".format(
            ltgid, length, count,
        )
        if len(tags) > 0:
            header += " # tags: {0}".format(",".join(tags))
        buf = [header]
        for ltobj in l_ltobj:
            buf.append(self.show_lt_info(ltobj.ltid))
        return "\n".join(buf)

    def commit_db(self):
        """Commit requested changes in LogDB.
        """
        self.db.commit()

    def drop_all(self):
        self.db.drop_all()


class LogDB:
    """Interface of DB transaction for log data.

    Note:
        It is not recommended to use this class directly.
        Instead, use LogData.
    """

    tablename_log = "log"
    tablename_lt = "lt"
    tablename_ltg = "ltg"
    tablename_tag = "tag"
    table_names = (tablename_log, tablename_lt, tablename_ltg, tablename_tag)
    indexname_log = "log_index"
    indexname_ltg = "ltg_index"
    indexname_tag = "tag_index"
    index_names = (indexname_log, indexname_ltg, indexname_tag)
    _tablename_tmp_footer = "_tmp"

    def __init__(self, conf, edit, reset_db):
        self._line_cnt = 0
        self._splitter = conf.get("database", "split_symbol")
        self._table_switch = {}

        db_type = conf.get("database", "database")
        if db_type == "sqlite3":
            from . import db_sqlite
            self._db = db_sqlite.init_db_conn(conf)
        elif db_type == "mysql":
            from . import db_mysql
            self._db = db_mysql.init_db_conn(conf)
        else:
            raise ValueError("invalid database type ({0})".format(
                db_type))

        if self._db.db_exists():
            if edit:
                if reset_db:
                    # create mode
                    _logger.info("DB reset")
                    self._db.reset()
                    self._init_tables()
                else:
                    # append mode
                    # load _line_cnt as next lid
                    self._line_cnt = self.count_lines()
            else:
                # read-only mode
                pass
        else:
            if edit:
                # create mode
                if reset_db:
                    msg = "Requested to reset DB, but database not found"
                    _logger.warning(msg)
                self._init_tables()
            else:
                raise IOError("database not found")

    def _init_tables(self):
        self._init_table_log()
        self._init_table_lt()
        self._init_table_ltg()
        self._init_table_tag()
        self._init_index()

    def _init_table_log(self, table_name=None):
        if table_name is None:
            table_name = self.tablename_log

        l_key = [db_common.TableKey("lid", "integer",
                                    # ("primary_key", "auto_increment", "not_null")),
                                    ("primary_key", "not_null")),
                 db_common.TableKey("ltid", "integer", tuple()),
                 db_common.TableKey("dt", "datetime", tuple()),
                 db_common.TableKey("host", "text", tuple()),
                 db_common.TableKey("words", "text", tuple())]
        sql = self._db.create_table_sql(table_name, l_key)
        self._db.execute(sql)

    def _init_table_lt(self, table_name=None):
        if table_name is None:
            table_name = self.tablename_lt
        l_key = [db_common.TableKey("ltid", "integer", ("primary_key",)),
                 db_common.TableKey("ltw", "text", tuple()),
                 db_common.TableKey("lts", "text", tuple()),
                 db_common.TableKey("count", "integer", tuple())]
        sql = self._db.create_table_sql(table_name, l_key)
        self._db.execute(sql)

    def _init_table_ltg(self, table_name=None):
        if table_name is None:
            table_name = self.tablename_ltg
        l_key = [db_common.TableKey("ltid", "integer", ("primary_key",)),
                 db_common.TableKey("ltgid", "integer", tuple())]
        sql = self._db.create_table_sql(table_name, l_key)
        self._db.execute(sql)

    def _init_table_tag(self, table_name=None):
        if table_name is None:
            table_name = self.tablename_tag
        l_key = [db_common.TableKey("ltid", "integer", tuple()),
                 db_common.TableKey("tag", "text", tuple())]
        sql = self._db.create_table_sql(table_name, l_key)
        self._db.execute(sql)

    def _init_index(self):
        self._init_index_log()
        self._init_index_ltg()
        self._init_index_tag()

    def _init_index_log(self):
        index_name = self.indexname_log
        table_name = self.tablename_log
        l_key = [db_common.TableKey("lid", "integer", tuple()),
                 db_common.TableKey("ltid", "integer", tuple()),
                 db_common.TableKey("dt", "datetime", tuple()),
                 db_common.TableKey("host", "text", (100,))]
        sql = self._db.create_index_sql(table_name, index_name, l_key)
        self._db.execute(sql)

    def _init_index_ltg(self):
        index_name = self.indexname_ltg
        table_name = self.tablename_ltg
        l_key = [db_common.TableKey("ltgid", "integer", tuple()), ]
        sql = self._db.create_index_sql(table_name, index_name, l_key)
        self._db.execute(sql)

    def _init_index_tag(self):
        index_name = self.indexname_tag
        table_name = self.tablename_tag
        l_key = [db_common.TableKey("tag", "text", (100,)), ]
        sql = self._db.create_index_sql(table_name, index_name, l_key)
        self._db.execute(sql)

    def repair_tables(self):
        current_table_names = self._db.get_table_names()

        for name in self.table_names:
            if name not in current_table_names:
                if name == self.tablename_log:
                    raise ValueError("Empty database, cannot be repaired")
                elif name == self.tablename_lt:
                    raise ValueError("No log templates, cannot be repaired")
                elif name == self.tablename_ltg:
                    print("no ltg table, init table")
                    self._init_table_ltg()
                    print("NOTE: try \"\" db-remake-group if you need afterward")
                elif name == self.tablename_tag:
                    print("no tag table, init table")
                    self._init_table_tag()
                    print("NOTE: try \"\" db-tag if you need afterward")

        for name in self.index_names:
            if name not in current_table_names:
                if name == self.indexname_log:
                    print("no log index, init")
                    self._init_index_log()
                elif name == self.indexname_ltg:
                    print("no ltg index, init")
                    self._init_index_ltg()
                elif name == self.indexname_tag:
                    print("no tag index, init")
                    self._init_index_tag()

        for name in current_table_names:
            if self._tablename_tmp_footer in name:
                print("remove temporal table")
                sql = self._db.drop_table_sql(name)
                self._db.execute(sql)

        self._db.commit()
        current_table_names = self._db.get_table_names()
        print("now the db has {0}".format(current_table_names))

    def switch_temporal_table(self, table_name):
        """Switch valid table to temporal one.
        Used for large data update."""
        tmp_table_name = table_name + self._tablename_tmp_footer
        assert tmp_table_name not in self._db.get_table_names()

        if table_name == self.tablename_log:
            self._init_table_log(tmp_table_name)
        elif table_name == self.tablename_lt:
            self._init_table_lt(tmp_table_name)
        elif table_name == self.tablename_ltg:
            self._init_table_ltg(tmp_table_name)
        elif table_name == self.tablename_tag:
            self._init_table_tag(tmp_table_name)
        else:
            raise ValueError("invalid table name {0}".format(table_name))

        self._table_switch[table_name] = tmp_table_name

    def apply_temporal_table(self, table_name):
        """Use temporal table as the main table.
        This function will remove the old main table,
        and rename temporal one as the main table.
        Used for large data update."""
        tmp_table_name = table_name + self._tablename_tmp_footer
        assert tmp_table_name in self._db.get_table_names()

        if table_name == self.tablename_log:
            index_name = self.indexname_log
            index_func = self._init_index_log
        elif table_name == self.tablename_ltg:
            index_name = self.indexname_ltg
            index_func = self._init_index_ltg
        elif table_name == self.tablename_tag:
            index_name = self.indexname_tag
            index_func = self._init_index_tag
        else:
            index_name = None
            index_func = None

        if index_name:
            sql = self._db.drop_index_sql(index_name)
            self._db.execute(sql)

        sql = self._db.drop_table_sql(table_name)
        self._db.execute(sql)

        sql = self._db.alter_table_sql(tmp_table_name, table_name)
        self._db.execute(sql)

        if index_name:
            index_func()

        self._table_switch.pop(table_name)

    def _valid_table_name(self, table_name):
        # for write functions only
        if table_name in self._table_switch:
            return self._table_switch[table_name]
        else:
            return table_name

    def commit(self):
        self._db.commit()

    def _parse_input(self, **kwargs):
        d = {}
        for k, v in kwargs.items():
            if k == "dt":
                d[k] = self._db.strftime(v)
            elif k == "l_w":
                d["words"] = self._splitter.join(v)
            elif k in ("lid", "ltid", "host"):
                d[k] = v
            else:
                raise KeyError
        return d

    def _parse_row(self, row):
        d_line = {"lid": int(row[0]),
                  "ltid": int(row[1]),
                  "dt": self._db.datetime(row[2]),
                  "host": row[3]}
        if row[4] == "":
            d_line["l_w"] = []
        else:
            d_line["l_w"] = strutil.split_igesc(row[4], self._splitter)
        return d_line

    # def add_line(self, ltid, dt, host, l_w, lid=None):
    def add_line(self, **kwargs):
        assert "ltid" in kwargs
        assert "dt" in kwargs
        assert "host" in kwargs
        assert "l_w" in kwargs
        d_val = self._parse_input(**kwargs)

        self._line_cnt += 1
        if "lid" not in kwargs:
            d_val["lid"] = self._line_cnt

        table_name = self._valid_table_name(self.tablename_log)
        l_ss = [db_common.StateSet(k, k) for k in d_val.keys()]
        sql = self._db.insert_sql(table_name, l_ss)
        self._db.execute(sql, d_val)

        return d_val["lid"]

    def iter_all(self):
        l_order = [("lid", "asc")]
        for row in self._select_log({}, l_order=l_order):
            d_line = {"lid": int(row[0]),
                      "ltid": int(row[1]),
                      "dt": self._db.datetime(row[2]),
                      "host": row[3]}
            if row[4] == "":
                d_line["l_w"] = []
            else:
                d_line["l_w"] = strutil.split_igesc(row[4], self._splitter)
            yield d_line

    def iter_lines(self, conditions, limit=None):
        d_cond = {k: v for k, v in conditions.items()
                  if v is not None}
        if len(d_cond) == 0:
            raise ValueError("Use iter_all to get all lines")

        # for compatibility
        if "top_dt" in d_cond and "dts" not in d_cond:
            d_cond["dts"] = d_cond.pop("top_dt")
        if "end_dt" in d_cond and "dte" not in d_cond:
            d_cond["dte"] = d_cond.pop("end_dt")

        for row in self._select_log(d_cond, limit=limit):
            yield self._parse_row(row)

    def iter_words(self, conditions):
        d_cond = {k: v for k, v in conditions.items()
                  if v is not None}
        if len(d_cond) == 0:
            raise ValueError("Use iter_all to get all lines")

        # for compatibility
        if "top_dt" in d_cond and "dts" not in d_cond:
            d_cond["dts"] = d_cond.pop("top_dt")
        if "end_dt" in d_cond and "dte" not in d_cond:
            d_cond["dte"] = d_cond.pop("end_dt")

        for row in self._select_log(d_cond):
            if row[4] == "":
                yield []
            else:
                yield strutil.split_igesc(row[4], self._splitter)

    def _select_log(self, d_cond, l_order=None, limit=None):
        # if len(d_cond) == 0:
        #     raise ValueError("called select with empty condition")
        args = d_cond.copy()

        table_name = self.tablename_log
        l_key = ["lid", "ltid", "dt", "host", "words"]
        l_cond = []
        for c in d_cond.keys():
            if c == "ltgid":
                sql = self._db.select_sql("ltg", ["ltid"],
                                          [db_common.Condition(c, "=", c, True)])
                l_cond.append(db_common.Condition("ltid", "in", sql, False))
            elif c == "dts":
                l_cond.append(db_common.Condition("dt", ">=", c, True))
                args[c] = self._db.strftime(d_cond[c])
            elif c == "dte":
                l_cond.append(db_common.Condition("dt", "<", c, True))
                args[c] = self._db.strftime(d_cond[c])
            else:
                l_cond.append(db_common.Condition(c, "=", c, True))
        sql = self._db.select_sql(table_name, l_key, l_cond, l_order, limit)
        return self._db.execute(sql, args)

    def get_line(self, lid):
        table_name = self.tablename_log
        l_key = ["lid", "ltid", "dt", "host", "words"]
        l_cond = [db_common.Condition("lid", "=", "lid", True)]
        sql = self._db.select_sql(table_name, l_key, l_cond)
        args = {"lid": lid}

        ret = [self._parse_row(row) for row in self._db.execute(sql, args)]
        assert len(ret) > 0, "lid not found"
        assert len(ret) == 1, "lid duplicated"
        return ret[0]

    def update_log(self, d_cond, **kwargs):
        if len(d_cond) == 0:
            # _logger.warning("called update with empty condition")
            raise ValueError("called update with empty condition")

        d_update = self._parse_input(**kwargs)

        table_name = self._valid_table_name(self.tablename_log)
        args = d_cond.copy()
        l_ss = []
        for k, v in d_update.items():
            keyname = "update_" + k
            l_ss.append(db_common.StateSet(k, keyname))
            args[keyname] = v
        l_cond = []
        for c in d_cond.keys():
            if c == "ltgid":
                sql = self._db.select_sql("ltg", ["ltid"],
                                          [db_common.Condition(c, "=", c, True)])
                l_cond.append(db_common.Condition("ltid", "in", sql, False))
            elif c == "dts":
                l_cond.append(db_common.Condition("dt", ">=", c, True))
                args[c] = self._db.strftime(d_cond[c])
            elif c == "dte":
                l_cond.append(db_common.Condition("dt", "<", c, True))
                args[c] = self._db.strftime(d_cond[c])
            else:
                l_cond.append(db_common.Condition(c, "=", c, True))
        sql = self._db.update_sql(table_name, l_ss, l_cond)
        self._db.execute(sql, args)

    def count_lines(self):
        table_name = self.tablename_log
        l_key = ["max(lid)"]
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        tmp = cursor.fetchone()[0]
        if tmp is None:
            return 0
        else:
            return int(tmp)

    def dt_term(self):
        table_name = self.tablename_log
        l_key = ["min(dt)", "max(dt)"]
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        top_dtstr, end_dtstr = cursor.fetchone()
        if None in (top_dtstr, end_dtstr):
            raise ValueError("No data found in DB")
        return self._db.datetime(top_dtstr), self._db.datetime(end_dtstr)

    def whole_host_lt(self, dts=None, dte=None):
        table_name = "log"
        l_key = ["host", "ltid"]
        l_cond = []
        args = {}
        if dts is not None:
            l_cond.append(db_common.Condition("dt", ">=", "dts", True))
            args["dts"] = self._db.strftime(dts)
        if dte is not None:
            l_cond.append(db_common.Condition("dt", "<", "dte", True))
            args["dte"] = self._db.strftime(dte)

        sql = self._db.select_sql(table_name, l_key, l_cond, opt=["distinct"])
        cursor = self._db.execute(sql, args)
        return [(row[0], row[1]) for row in cursor]

    def whole_host(self, dts=None, dte=None):
        table_name = "log"
        l_key = ["host"]
        l_cond = []
        args = {}
        if dts is not None:
            l_cond.append(db_common.Condition("dt", ">=", "dts", True))
            args["dts"] = self._db.strftime(dts)
        if dte is not None:
            l_cond.append(db_common.Condition("dt", "<", "dte", True))
            args["dte"] = self._db.strftime(dte)
        sql = self._db.select_sql(table_name, l_key, l_cond, opt=["distinct"])
        cursor = self._db.execute(sql, args)
        return [row[0] for row in cursor]

    def add_lt(self, ltline):
        table_name = self._valid_table_name(self.tablename_lt)
        l_ss = [
            db_common.StateSet("ltid", "ltid"),
            db_common.StateSet("ltw", "ltw"),
            db_common.StateSet("lts", "lts"),
            db_common.StateSet("count", "count")
        ]
        if ltline.lts is None:
            lts = None
        else:
            lts = self._splitter.join(ltline.lts)
        args = {
            "ltid": ltline.ltid,
            "ltw": self._splitter.join(ltline.ltw),
            "lts": lts,
            "count": ltline.count,
        }
        sql = self._db.insert_sql(table_name, l_ss)
        self._db.execute(sql, args)

    def add_ltg(self, ltid, ltgid):
        table_name = self._valid_table_name(self.tablename_ltg)
        l_ss = [
            db_common.StateSet("ltid", "ltid"),
            db_common.StateSet("ltgid", "ltgid")
        ]
        args = {"ltid": ltid, "ltgid": ltgid}
        sql = self._db.insert_sql(table_name, l_ss)
        self._db.execute(sql, args)

    def update_lt(self, ltid, ltw, lts, count=None):
        table_name = self._valid_table_name(self.tablename_lt)
        l_ss = []
        args = {}
        if ltw is not None:
            l_ss.append(db_common.StateSet("ltw", "ltw"))
            args["ltw"] = self._splitter.join(ltw)
        if lts is not None:
            l_ss.append(db_common.StateSet("lts", "lts"))
            args["lts"] = self._splitter.join(lts)
        if count is not None:
            l_ss.append(db_common.StateSet("count", "count"))
            args["count"] = count
        l_cond = [db_common.Condition("ltid", "=", "ltid", True)]
        args["ltid"] = ltid

        sql = self._db.update_sql(table_name, l_ss, l_cond)
        self._db.execute(sql, args)

    def remove_lt(self, ltid):
        args = {"ltid": ltid}

        # remove from lt
        table_name = self._valid_table_name(self.tablename_lt)
        l_cond = [db_common.Condition("ltid", "=", "ltid", True)]
        sql = self._db.delete_sql(table_name, l_cond)
        self._db.execute(sql, args)

        # remove from ltg
        table_name = self._valid_table_name(self.tablename_ltg)
        l_cond = [db_common.Condition("ltid", "=", "ltid", True)]
        sql = self._db.delete_sql(table_name, l_cond)
        self._db.execute(sql, args)

    def restore_lttable(self):
        lttable = lt_common.LTTable()

        table_name = self._db.join_sql("left outer",
                                       self.tablename_lt,
                                       self.tablename_ltg,
                                       "ltid", "ltid")
        l_key = (self.tablename_lt + ".ltid",
                 self.tablename_ltg + ".ltgid",
                 self.tablename_lt + ".ltw",
                 self.tablename_lt + ".lts",
                 self.tablename_lt + ".count")
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        for row in cursor:
            ltid = int(row[0])
            if row[1]:
                ltgid = int(row[1])
            else:
                ltgid = ltid
            ltw = strutil.split_igesc(row[2], self._splitter)
            tmp = row[3]
            if tmp is None:
                lts = None
            else:
                lts = strutil.split_igesc(tmp, self._splitter)
            count = int(row[4])
            lttable.restore_lt(ltid, ltgid, ltw, lts, count)

        return lttable

    def count_lt(self):
        table_name = self.tablename_lt
        l_key = ["count(*)"]
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        return int(cursor.fetchone()[0])

    def count_ltg(self):
        table_name = self.tablename_ltg
        l_key = ["max(ltgid)"]
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        return int(cursor.fetchone()[0]) + 1

    def iter_ltg_def(self):
        table_name = self.tablename_ltg
        l_key = ["ltid", "ltgid"]
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        for row in cursor:
            ltid, ltgid = row
            yield int(ltid), int(ltgid)

    def iter_ltgid(self):
        table_name = self.tablename_ltg
        l_key = ["ltgid"]
        sql = self._db.select_sql(table_name, l_key, opt=["distinct"])
        cursor = self._db.execute(sql)
        for row in cursor:
            ltgid = row[0]
            yield int(ltgid)

    def get_ltg_members(self, ltgid):
        table_name = "ltg"
        l_key = ["ltid"]
        l_cond = [db_common.Condition("ltgid", "=", "ltgid", True)]
        args = {"ltgid": ltgid}
        sql = self._db.select_sql(table_name, l_key, l_cond)
        cursor = self._db.execute(sql, args)
        return [int(row[0]) for row in cursor]

    def add_tags(self, ltid, tags):
        table_name = self._valid_table_name(self.tablename_tag)
        for tag in tags:
            l_ss = [
                db_common.StateSet("ltid", "ltid"),
                db_common.StateSet("tag", "tag")
            ]
            args = {"ltid": ltid, "tag": tag}
            sql = self._db.insert_sql(table_name, l_ss)
            self._db.execute(sql, args)

    def get_tags(self, **kwargs):
        # compatibility
        l_table_name = self._db.get_table_names()
        if self.tablename_tag not in l_table_name:
            return

        if "ltgid" in kwargs:
            table_name = self._db.join_sql("left outer",
                                           self.tablename_ltg,
                                           self.tablename_tag,
                                           "ltid", "ltid")
            l_key = [self.tablename_tag + ".tag"]
            l_cond = [db_common.Condition(self.tablename_ltg + ".ltgid",
                                          "=", "ltgid", True)]
            args = {"ltgid": kwargs["ltgid"]}
            sql = self._db.select_sql(table_name, l_key, l_cond, opt=["distinct"])
            cursor = self._db.execute(sql, args)
        elif "ltid" in kwargs:
            table_name = self.tablename_tag
            l_key = ["tag"]
            l_cond = [db_common.Condition("ltid", "=", "ltid", True)]
            args = {"ltid": kwargs["ltid"]}
            sql = self._db.select_sql(table_name, l_key, l_cond)
            cursor = self._db.execute(sql, args)
        else:
            raise ValueError("No identifier given")

        for row in cursor:
            if row[0] is not None:
                yield row[0]

    def iter_tag_def(self):
        table_name = self.tablename_tag
        l_key = ["ltid", "tag"]
        sql = self._db.select_sql(table_name, l_key)
        cursor = self._db.execute(sql)
        for row in cursor:
            ltid, tag = row
            yield int(ltid), tag

    def reset_ltg(self):
        sql = self._db.delete_sql("ltg")
        self._db.execute(sql)

    def reset_tag(self):
        # compatibility
        l_table_name = self._db.get_table_names()
        if self.tablename_tag in l_table_name:
            sql = self._db.delete_sql(self.tablename_tag)
            self._db.execute(sql)
        else:
            self._init_table_tag()
            self._init_index_tag()

    def drop_all(self):
        self._db.reset()


class RestoreOriginalData(object):

    def __init__(self, dirname, style="date", method="commit",
                 reset=False):
        self.style = style
        self.method = method
        self.dirname = dirname
        common.mkdir(dirname)
        if reset:
            _logger.info("RestoreOriginalData was requested to reset files")
            common.rm_dirchild(dirname)

        assert self.style in ("date", "")
        assert self.method in ("incremental", "commit")
        if self.method == "commit":
            self.buf = defaultdict(list)

    def add(self, lm):
        if self.style == "date":
            fn = lm.dt.strftime("%Y%m%d")
        else:
            raise NotImplementedError

        linestr = lm.restore_line()
        if self.method == "incremental":
            self.write_line(linestr, fn)
        elif self.method == "commit":
            self.buf[fn].append(linestr)

    def add_str(self, dt, linestr):
        if self.style == "date":
            fn = dt.strftime("%Y%m%d")
        else:
            raise NotImplementedError

        if self.method == "incremental":
            self.write_line(linestr, fn)
        elif self.method == "commit":
            self.buf[fn].append(linestr)

    def commit(self):
        if self.method == "incremental":
            pass
        elif self.method == "commit":
            self.write_all()

    def write_line(self, linestr, fn):
        with open("/".join((self.dirname, fn)),
                  "a", encoding='utf-8') as f:
            f.write(linestr + "\n")

    def write_all(self):
        assert self.method == "commit"
        for fn, l_buf in self.buf.items():
            with open("/".join((self.dirname, fn)),
                      "a", encoding='utf-8') as f:
                f.write("\n".join(l_buf))


def info(conf):
    """Show abstruction of log messages registered in DB.

    Args:
        conf (config.ExtendedConfigParser): A common configuration object.

    """

    ld = LogData(conf)
    print("[DB status]")
    print("Registered log lines : {0}".format(ld.count_lines()))
    print("Term : {0[0]} - {0[1]}".format(ld.dt_term()))
    print("Log templates : {0}".format(ld.count_lt()))
    print("Log template groups : {0}".format(ld.count_ltg()))
    print("Hosts : {0}".format(len(ld.whole_host())))


def info_term(conf, top_dt, end_dt):
    cnt_line = 0
    s_ltid = set()
    s_gid = set()
    s_host = set()

    ld = LogData(conf)
    for line in ld.iter_lines(top_dt=top_dt, end_dt=end_dt):
        cnt_line += 1
        s_ltid.add(line.lt.ltid)
        s_gid.add(line.lt.ltgid)
        s_host.add(line.host)

    print("[DB status] in {0} - {1}".format(top_dt, end_dt))
    print("Registered log lines : {0}".format(cnt_line))
    print("Log templates : {0}".format(len(s_ltid)))
    print("Log template groups : {0}".format(len(s_gid)))
    print("Hosts : {0}".format(len(s_host)))


def show_all_lt(conf, **kwargs):
    buf = []
    ld = LogData(conf)
    if "simple" in kwargs and kwargs["simple"]:
        for ltobj in ld.iter_lt():
            buf.append(str(ltobj))
    else:
        for ltobj in ld.iter_lt():
            buf.append(ld.show_lt_info(ltobj.ltid))
    return "\n".join(buf)


def show_all_ltg(conf):
    buf = []
    ld = LogData(conf)
    for gid in ld.iter_ltgid():
        buf.append(ld.show_ltg_info(gid))
    return "\n\n".join(buf)


def show_tag(conf, tag=None):
    d_tag = defaultdict(list)
    ld = LogData(conf)
    for ltobj in ld.iter_lt():
        for tmp_tag in ld.get_tags(ltid=ltobj.ltid):
            if tag is None or tmp_tag == tag:
                d_tag[tmp_tag].append(ltobj.ltid)

    buf = []
    for tmp_tag, l_ltid in d_tag.items():
        buf.append("[tag {0} ({1} tpls)]".format(tmp_tag, len(l_ltid)))
        for ltid in l_ltid:
            buf.append(ld.show_lt_info(ltid))
    return "\n".join(buf)


def show_tag_stats(conf):
    d_tag = defaultdict(list)
    ld = LogData(conf)
    for ltobj in ld.iter_lt():
        for tmp_tag in ld.get_tags(ltid=ltobj.ltid):
            d_tag[tmp_tag].append(ltobj.ltid)

    table = [["tag", "templates", "lines"]]
    for tag, l_ltid in d_tag.items():
        n_line = sum(ld.lt(ltid).count for ltid in l_ltid)
        table.append([tag, len(l_ltid), n_line])
    return common.cli_table(table)


def show_lt_import(conf):
    ld = LogData(conf)
    for ltobj in ld.iter_lt():
        print(" ".join(ltobj.ltw))


def show_all_host(conf, dts=None, dte=None):
    ld = LogData(conf)
    for host in ld.whole_host(dts=dts, dte=dte):
        print(host)


def data_from_db(conf, dirname, method, reset):
    rod = RestoreOriginalData(dirname, method=method, reset=reset)
    ld = LogData(conf)
    top_dt, end_dt = ld.whole_term()
    for lm in ld.iter_lines(top_dt=top_dt, end_dt=end_dt):
        rod.add(lm)
    rod.commit()
