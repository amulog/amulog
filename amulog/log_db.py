#!/usr/bin/env python
# coding: utf-8

"""
Construct and manage database of log messages, templates,
and grouping definitions.
"""

import datetime
import logging
from collections import defaultdict

import amulog.manager

from . import config, common
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
        including headers (timestamp and hostname).
        """
        return self.lt.restore_message(self.l_w)

    def restore_line(self):
        """str: Get original log message contents
        except headers (timestamp and hostname).
        """
        return " ".join((str(self.dt), str(self.host),
                         self.restore_message()))

    # def restore_line_lid(self):
    #    """str: Get original log message contents
    #    except headers (timestamp and hostname).
    #    """
    #    return " ".join((str(self.lid), self.restore_line()))


class LogData:
    """Interface to get, add or edit log messages in DB.

    Attributes:
        conf (config.ExtendedConfigParser): A common configuration object.
        lttable (lt_common.LTTable): Log template table object.
        db (LogDB): Log database instance.
        ltm (amulog.manager.LTManager): Log template classifier object.
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
        self.lttable = lt_common.LTTable()  # lt_common.LTTable
        self.db = LogDB(conf, self.lttable, edit, reset_db)  # log_db.LogDB
        self.ltm = None  # lt_common.LTManager
        from . import lt_label
        self.ll = lt_label.init_ltlabel(conf)

    def get_line(self, lid):
        return self.db.get_line(lid)

    def iter_lines(self, **kargs):
        """Generate log messages in DB that satisfy conditions
        given in arguments. All arguments are defaults to None, and ignored.

        Args:
            lid (Optional[int]): A message identifier in DB.
            ltid (Optional[int]): A log template identifier.
            ltgid (Optional[int]): A log template grouping identifier.
            top_dt (Optional[datetime.datetime]): Condition for timestamp.
                Messages output after 'top_dt' will be yield.
            end_dt (Optional[datetime.datetime]): Condition for timestamp.
                Messages output before 'end_dt' will be yield.
            host (Optional[str]): A source hostname of the message.
            area (Optional[str]): An area name of source hostname.
                Some reserved area-name is available;
                \"all\" to use whole data for 1 area,
                \"host_*\" to use 1 hostname as 1 area.
                (\"each\" is only used for pc_log argument initialization)

        Yields:
            LogMessage: An annotated log message instance
                which satisfies all given conditions.
        """
        _logger.debug("iter_lines called ({0})".format(" ".join(
            ["{0}:{1}".format(k, v) for k, v in kargs.items()
             if v is not None])))
        return self.db.iter_lines(**kargs)

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

    def whole_host_lt(self, top_dt=None, end_dt=None, area=None):
        """List[str, str]: Sequence of all combinations of
        hostname and ltids in DB."""
        return self.db.whole_host_lt(top_dt=top_dt, end_dt=end_dt,
                                     area=area)

    def whole_host_ltg(self, top_dt=None, end_dt=None, area=None):
        """List[str, str]: Sequence of all combinations of
        hostname and ltgids in DB."""
        ret = set()
        for host, ltid in self.whole_host_lt(top_dt=top_dt, end_dt=end_dt,
                                             area=area):
            ret.add((host, self.ltgid_from_ltid(ltid)))
        return list(set(ret))

    def whole_host(self, top_dt=None, end_dt=None):
        """List[str]: Sequence of all source hostname in DB."""
        return self.db.whole_host(top_dt=top_dt, end_dt=end_dt)

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

    def host_area(self, host):
        """Get area names that given host belongs to.

        Args:
            host (str): A hostname.

        Returns:
            List[str]: A sequence of area names that given host belongs to.

        """
        return self.db.host_area(host)

    @staticmethod
    def _str_ltline(ltline):
        return " ".join((str(ltline.ltid), "({0})".format(ltline.ltgid),
                         str(ltline), "({0})".format(ltline.cnt)))

    def show_all_lt(self):
        """Show all log templates. Log template identifier
        and its template message will be output.

        Returns:
            str: Output message buffer.
        """
        buf = []
        for ltline in self.lttable:
            buf.append(self._str_ltline(ltline))
        return "\n".join(buf)

    def show_all_ltgroup(self):
        """Show all log template groups. Log template grouping identifier
        and its template candidates will be output.

        Returns:
            str: Output message buffer.
        """
        if self.db.count_ltg() == 0:
            self.show_all_lt()
        else:
            buf = []
            for ltgid in self.db.iter_ltgid():
                buf.append(self.show_ltgroup(ltgid))
            return "\n".join(buf)

    def show_ltgroup_cond(self, **kwargs):
        buf = []
        for gid in self.db.iter_ltgid():
            l_lt = self.ltg_members(gid)
            l_ltid = self.db.get_ltg_members(gid)
            label = self.ll.get_ltg_label(gid, l_lt)
            group = self.ll.get_ltg_group(gid, l_lt)
            if "group" in kwargs and not kwargs["group"] == group:
                continue
            if "label" in kwargs and not kwargs["label"] == label:
                continue
            buf.append(self.show_ltgroup(gid))
        return "\n".join(buf)

    def show_ltgroup(self, gid):
        """Show log template groups.

        Returns:
            str: Output message buffer.
        """
        buf = []
        l_lt = self.ltg_members(gid)
        l_ltid = self.db.get_ltg_members(gid)
        label = self.ll.get_ltg_label(gid, l_lt)
        group = self.ll.get_ltg_group(gid, l_lt)
        length = len(l_ltid)
        cnt = 0
        for ltid in l_ltid:
            ltline = self.lttable[ltid]
            cnt += ltline.cnt
            buf.append(self._str_ltline(ltline))

        buf = ["[ltgroup {0} ({1}, {2})] # {3}({4})".format(gid, length,
                                                            cnt, group,
                                                            label)] + buf
        return "\n".join(buf)

    def commit_db(self):
        """Commit requested changes in LogDB.
        """
        self.db.commit()
        if self.ltm is not None:
            self.ltm.dump()


class LogDB:
    """Interface of DB transaction for log data.

    Note:
        It is not recommended to use this class directly.
        Instead, use LogData.
    """

    def __init__(self, conf, lttable, edit, reset_db):
        self.lttable = lttable
        self._line_cnt = 0
        self._splitter = conf.get("database", "split_symbol")

        db_type = conf.get("database", "database")
        if db_type == "sqlite3":
            dbpath = conf.get("database", "sqlite3_filename")
            if dbpath is None:
                # for compatibility
                dbpath = conf.get("database", "db_filename")
            self.db = db_common.sqlite3(dbpath)
        elif db_type == "mysql":
            host = conf.get("database", "mysql_host")
            dbname = conf.get("database", "mysql_dbname")
            user = conf.get("database", "mysql_user")
            passwd = conf.get("database", "mysql_passwd")
            self.db = db_common.mysql(host, dbname, user, passwd)
        else:
            raise ValueError("invalid database type ({0})".format(
                db_type))

        if edit:
            if self.db.db_exists():
                if reset_db:
                    _logger.info("DB reset")
                    self.db.reset()
                    self._init_tables()
                else:
                    self._line_cnt = self.count_lines()
                    self._init_lttable()
            else:
                if reset_db:
                    _logger.warning(
                        "Requested to reset DB, but database not found")
                self._init_tables()
        else:
            if self.db.db_exists():
                self._line_cnt = self.count_lines()
                self._init_lttable()
            else:
                raise IOError("database not found")

    def _init_tables(self):
        table_name = "log"
        l_key = [db_common.tablekey("lid", "integer",
                                    # ("primary_key", "auto_increment", "not_null")),
                                    ("primary_key", "not_null")),
                 db_common.tablekey("ltid", "integer"),
                 db_common.tablekey("dt", "datetime"),
                 db_common.tablekey("host", "text"),
                 db_common.tablekey("words", "text")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        table_name = "lt"
        l_key = [db_common.tablekey("ltid", "integer", ("primary_key",)),
                 db_common.tablekey("ltw", "text"),
                 db_common.tablekey("lts", "text"),
                 db_common.tablekey("count", "integer")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        table_name = "ltg"
        l_key = [db_common.tablekey("ltid", "integer", ("primary_key",)),
                 db_common.tablekey("ltgid", "integer")]
        sql = self.db.create_table_sql(table_name, l_key)
        self.db.execute(sql)

        self._init_index()

    def _init_index(self):
        l_table_name = self.db.get_table_names()

        table_name = "log"
        index_name = "log_index"
        l_key = [db_common.tablekey("lid", "integer"),
                 db_common.tablekey("ltid", "integer"),
                 db_common.tablekey("dt", "datetime"),
                 db_common.tablekey("host", "text", (100,))]
        if index_name not in l_table_name:
            sql = self.db.create_index_sql(table_name, index_name, l_key)
            self.db.execute(sql)

        table_name = "ltg"
        index_name = "ltg_index"
        l_key = [db_common.tablekey("ltgid", "integer")]
        if index_name not in l_table_name:
            sql = self.db.create_index_sql(table_name, index_name, l_key)
            self.db.execute(sql)

    def commit(self):
        self.db.commit()

    def add_line(self, ltid, dt, host, l_w, lid=None):
        table_name = "log"
        d_val = {
            "ltid": ltid,
            "dt": self.db.strftime(dt),
            "host": host,
            "words": self._splitter.join(l_w),
        }

        self._line_cnt += 1
        if lid is None:
            d_val["lid"] = self._line_cnt
        else:
            d_val["lid"] = lid

        l_ss = [db_common.setstate(k, k) for k in d_val.keys()]
        sql = self.db.insert_sql(table_name, l_ss)
        self.db.execute(sql, d_val)

        return d_val["lid"]

    def iter_lines(self, lid=None, ltid=None, ltgid=None, top_dt=None,
                   end_dt=None, host=None):
        d_cond = {}
        if lid is not None: d_cond["lid"] = lid
        if ltid is not None: d_cond["ltid"] = ltid
        if ltgid is not None: d_cond["ltgid"] = ltgid
        if top_dt is not None: d_cond["top_dt"] = top_dt
        if end_dt is not None: d_cond["end_dt"] = end_dt
        if host is not None: d_cond["host"] = host

        if len(d_cond) == 0:
            raise ValueError("More than 1 argument should NOT be None")

        for row in self._select_log(d_cond):
            lid = int(row[0])
            ltid = int(row[1])
            dt = self.db.datetime(row[2])
            host = row[3]
            if row[4] == "":
                l_w = []
            else:
                l_w = strutil.split_igesc(row[4], self._splitter)
            yield LogMessage(lid, self.lttable[ltid], dt, host, l_w)

    def iter_words(self, lid=None, ltid=None, ltgid=None, top_dt=None,
                   end_dt=None, host=None):
        d_cond = {}
        if lid is not None: d_cond["lid"] = lid
        if ltid is not None: d_cond["ltid"] = ltid
        if ltgid is not None: d_cond["ltgid"] = ltgid
        if top_dt is not None: d_cond["top_dt"] = top_dt
        if end_dt is not None: d_cond["end_dt"] = end_dt
        if host is not None: d_cond["host"] = host
        if len(d_cond) == 0:
            raise ValueError("More than 1 argument should NOT be None")
        for row in self._select_log(d_cond):
            if row[4] == "":
                yield []
            else:
                yield strutil.split_igesc(row[4], self._splitter)

    def _select_log(self, d_cond):
        if len(d_cond) == 0:
            raise ValueError("called select with empty condition")
        args = d_cond.copy()

        table_name = "log"
        l_key = ["lid", "ltid", "dt", "host", "words"]
        l_cond = []
        for c in d_cond.keys():
            if c == "ltgid":
                sql = self.db.select_sql("ltg", ["ltid"],
                                         [db_common.cond(c, "=", c)])
                l_cond.append(db_common.cond("ltid", "in", sql, False))
            elif c == "area":
                sql = self.db.select_sql("area", ["host"],
                                         [db_common.cond(c, "=", c)])
                l_cond.append(db_common.cond("host", "in", sql, False))
            elif c == "top_dt":
                l_cond.append(db_common.cond("dt", ">=", c))
                args[c] = self.db.strftime(d_cond[c])
            elif c == "end_dt":
                l_cond.append(db_common.cond("dt", "<", c))
                args[c] = self.db.strftime(d_cond[c])
            else:
                l_cond.append(db_common.cond(c, "=", c))
        sql = self.db.select_sql(table_name, l_key, l_cond)
        return self.db.execute(sql, args)

    def get_line(self, lid):
        table_name = "log"
        l_key = ["lid", "ltid", "dt", "host", "words"]
        l_cond = [db_common.cond("lid", "=", "lid")]
        sql = self.db.select_sql(table_name, l_key, l_cond)
        args = {"lid": lid}

        ret = []
        for row in self.db.execute(sql, args):
            lid = int(row[0])
            ltid = int(row[1])
            dt = self.db.datetime(row[2])
            host = row[3]
            if row[4] == "":
                l_w = []
            else:
                l_w = strutil.split_igesc(row[4], self._splitter)
            lm = LogMessage(lid, self.lttable[ltid], dt, host, l_w)
            ret.append(lm)

        if len(ret) == 0:
            return None
        elif len(ret) == 1:
            return ret[0]
        else:
            raise ValueError("Duplicated messages for lid {0}".format(lid))

    def update_log(self, d_cond, d_update):
        if len(d_cond) == 0:
            _logger.warning("called update with empty condition")
            # raise ValueError("called update with empty condition")
        args = d_cond.copy()

        table_name = "log"
        l_ss = []
        for k, v in d_update.items():
            # assert k in ("ltid", "top_dt", "end_dt", "host")
            keyname = "update_" + k
            l_ss.append(db_common.setstate(k, keyname))
            args[keyname] = v
        l_cond = []
        for c in d_cond.keys():
            if c == "ltgid":
                sql = self.db.select_sql("ltg", ["ltid"],
                                         [db_common.cond(c, "=", c)])
                l_cond.append(db_common.cond("ltid", "in", sql, False))
            elif c == "top_dt":
                l_cond.append(db_common.cond("dt", ">=", c))
                args[c] = self.db.strftime(d_cond[c])
            elif c == "end_dt":
                l_cond.append(db_common.cond("dt", "<", c))
                args[c] = self.db.strftime(d_cond[c])
            else:
                l_cond.append(db_common.cond(c, "=", c))
        sql = self.db.update_sql(table_name, l_ss, l_cond)
        self.db.execute(sql, args)

    def count_lines(self):
        table_name = "log"
        l_key = ["max(lid)"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        tmp = cursor.fetchone()[0]
        if tmp is None:
            return 0
        else:
            return int(tmp)

    def dt_term(self):
        table_name = "log"
        l_key = ["min(dt)", "max(dt)"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        top_dtstr, end_dtstr = cursor.fetchone()
        if None in (top_dtstr, end_dtstr):
            raise ValueError("No data found in DB")
        return self.db.datetime(top_dtstr), self.db.datetime(end_dtstr)

    def whole_host_lt(self, top_dt=None, end_dt=None):
        table_name = "log"
        l_key = ["host", "ltid"]
        l_cond = []
        args = {}
        if top_dt is not None:
            l_cond.append(db_common.cond("dt", ">=", "top_dt"))
            # args["top_dt"] = top_dt
            args["top_dt"] = self.db.strftime(top_dt)
        if end_dt is not None:
            l_cond.append(db_common.cond("dt", "<", "end_dt"))
            # args["end_dt"] = end_dt
            args["end_dt"] = self.db.strftime(end_dt)

        sql = self.db.select_sql(table_name, l_key, l_cond, opt=["distinct"])
        cursor = self.db.execute(sql, args)
        return [(row[0], row[1]) for row in cursor]

    def whole_host(self, top_dt=None, end_dt=None):
        table_name = "log"
        l_key = ["host"]
        l_cond = []
        args = {}
        if top_dt is not None:
            l_cond.append(db_common.cond("dt", ">=", "top_dt"))
            args["top_dt"] = self.db.strftime(top_dt)
            # args["top_dt"] = top_dt
        if end_dt is not None:
            l_cond.append(db_common.cond("dt", "<", "end_dt"))
            # args["end_dt"] = end_dt
            args["end_dt"] = self.db.strftime(end_dt)
        sql = self.db.select_sql(table_name, l_key, l_cond, opt=["distinct"])
        cursor = self.db.execute(sql, args)
        return [row[0] for row in cursor]

    def add_lt(self, ltline):
        table_name = "lt"
        l_ss = []
        l_ss.append(db_common.setstate("ltid", "ltid"))
        l_ss.append(db_common.setstate("ltw", "ltw"))
        l_ss.append(db_common.setstate("lts", "lts"))
        l_ss.append(db_common.setstate("count", "count"))
        if ltline.lts is None:
            lts = None
        else:
            lts = self._splitter.join(ltline.lts)
        args = {
            "ltid": ltline.ltid,
            "ltw": self._splitter.join(ltline.ltw),
            "lts": lts,
            "count": ltline.cnt,
        }
        sql = self.db.insert_sql(table_name, l_ss)
        self.db.execute(sql, args)

        self.add_ltg(ltline.ltid, ltline.ltgid)

    def add_ltg(self, ltid, ltgid):
        table_name = "ltg"
        l_ss = []
        l_ss.append(db_common.setstate("ltid", "ltid"))
        l_ss.append(db_common.setstate("ltgid", "ltgid"))
        args = {"ltid": ltid, "ltgid": ltgid}
        sql = self.db.insert_sql(table_name, l_ss)
        self.db.execute(sql, args)

    def update_lt(self, ltid, ltw, lts, count):
        table_name = "lt"
        l_ss = []
        args = {}
        if ltw is not None:
            l_ss.append(db_common.setstate("ltw", "ltw"))
            args["ltw"] = self._splitter.join(ltw)
        if lts is not None:
            l_ss.append(db_common.setstate("lts", "lts"))
            args["lts"] = self._splitter.join(lts)
        if count is not None:
            l_ss.append(db_common.setstate("count", "count"))
            args["count"] = count
        l_cond = [db_common.cond("ltid", "=", "ltid")]
        args["ltid"] = ltid

        sql = self.db.update_sql(table_name, l_ss, l_cond)
        self.db.execute(sql, args)

    def remove_lt(self, ltid):
        args = {"ltid": ltid}

        # remove from lt
        table_name = "lt"
        l_cond = [db_common.cond("ltid", "=", "ltid")]
        sql = self.db.delete_sql(table_name, l_cond)
        self.db.execute(sql, args)

        # remove from ltg
        table_name = "ltg"
        l_cond = [db_common.cond("ltid", "=", "ltid")]
        sql = self.db.delete_sql(table_name, l_cond)
        self.db.execute(sql, args)

    def _init_lttable(self):
        table_name = self.db.join_sql("left outer",
                                      "lt", "ltg", "ltid", "ltid")
        l_key = ("lt.ltid", "ltg.ltgid", "lt.ltw", "lt.lts", "lt.count")
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        for row in cursor:
            ltid = int(row[0])
            ltgid = int(row[1])
            ltw = strutil.split_igesc(row[2], self._splitter)
            temp = row[3]
            if temp is None:
                lts = None
            else:
                lts = strutil.split_igesc(temp, self._splitter)
            count = int(row[4])
            self.lttable.restore_lt(ltid, ltgid, ltw, lts, count)

    def count_lt(self):
        table_name = "lt"
        l_key = ["count(*)"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        return int(cursor.fetchone()[0])

    def count_ltg(self):
        table_name = "ltg"
        l_key = ["max(ltgid)"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        return int(cursor.fetchone()[0]) + 1

    def iter_ltg_def(self):
        table_name = "ltg"
        l_key = ["ltid", "ltgid"]
        sql = self.db.select_sql(table_name, l_key)
        cursor = self.db.execute(sql)
        for row in cursor:
            ltid, ltgid = row
            yield int(ltid), int(ltgid)

    def iter_ltgid(self):
        table_name = "ltg"
        l_key = ["ltgid"]
        sql = self.db.select_sql(table_name, l_key, opt=["distinct"])
        cursor = self.db.execute(sql)
        for row in cursor:
            ltgid = row[0]
            yield int(ltgid)

    def get_ltg_members(self, ltgid):
        table_name = "ltg"
        l_key = ["ltid"]
        l_cond = [db_common.cond("ltgid", "=", "ltgid")]
        args = {"ltgid": ltgid}
        sql = self.db.select_sql(table_name, l_key, l_cond)
        cursor = self.db.execute(sql, args)
        return [int(row[0]) for row in cursor]

    def reset_ltg(self):
        sql = self.db.delete_sql("ltg")
        self.db.execute(sql)


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


def show_lt(conf, **kwargs):
    ld = LogData(conf)
    if "simple" in kwargs and kwargs["simple"]:
        for ltobj in ld.iter_lt():
            print(ltobj)
    else:
        print(ld.show_ltgroup_cond(**kwargs))


def dump_lt(conf):
    ld = LogData(conf)
    print(ld.show_all_lt())


def show_lt_import(conf):
    ld = LogData(conf)
    for ltobj in ld.iter_lt():
        print(" ".join(ltobj.ltw))


def show_all_host(conf, top_dt=None, end_dt=None):
    ld = LogData(conf)
    for host in ld.whole_host():
        print(host)


def agg_words(conf, target="all"):
    """Return dict of words in all log messages and their counts.

    Args:
        conf
        target (str): (all, description, variable) is available

    Returns:
        dict: key = word, val = counts
    """

    def getall(d, lm):
        for w in lm.l_w:
            d[w] += 1

    def getdesc(d, lm):
        for w in lm.lt.ltw:
            if w == lm.lt.sym:
                pass
            else:
                d[w] += 1

    def getvar(d, lm):
        for w in lm.var():
            d[w] += 1

    assert target in ("all", "description", "variable")
    if target == "all":
        func = getall
    elif target == "description":
        func = getdesc
    elif target == "variable":
        func = getvar
    else:
        raise NotImplementedError

    ld = LogData(conf)
    top_dt, end_dt = ld.dt_term()
    d = defaultdict(int)
    for lm in ld.iter_lines(top_dt=top_dt, end_dt=end_dt):
        func(d, lm)

    return d


def anonymize(conf):
    ld = LogData(conf, edit=True)
    d_cond = {}
    d_update = {"words": ""}
    ld.db.update_log(d_cond, d_update)
    ld.commit_db()


def data_from_db(conf, dirname, method, reset):
    rod = RestoreOriginalData(dirname, method=method, reset=reset)
    ld = LogData(conf)
    top_dt, end_dt = ld.whole_term()
    for lm in ld.iter_lines(top_dt=top_dt, end_dt=end_dt):
        rod.add(lm)
    rod.commit()
