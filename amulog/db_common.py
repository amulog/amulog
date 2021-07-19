import datetime
from typing import NamedTuple, Iterable
from abc import ABC, abstractmethod


class TableKey(NamedTuple):
    """A namedtuple to define a column in CREATE syntax.
    Used in CREATE TABLE and CREATE INDEX.
    """
    key: str  #: column name
    type: str  #: type name, in SQL data types
    attr: Iterable  #: attribute of the column, e.g. autoincrement


class Condition(NamedTuple):
    """A namedtuple to specify a condition in WHERE syntax.
    Used in SELECT and UPDATE.

    If :attr:`repl` is True, :attr:`val` is considered as a placeholder.
    It will be replaced with the corresponding item in the "args" argument of :meth:`Database.execute`.
    """
    key: str  #: column name
    opr: str  #: operand to compare key and val
    val: str  #: compared value (or placeholder key if repl is True)
    repl: bool  #: If true, val is considered as a placeholder.


class StateSet(NamedTuple):
    """A namedtuple to specify a value in VALUES syntax.
    Used in INSERT and UPDATE.
    """
    key: str  #: column name
    val: str  #: assigned value


# to be removed
# from collections import namedtuple
# # namedtuple defaults is available after python 3.7
# # TableKey = namedtuple("TableKey", ("key", "type", "attr"), defaults=(tuple(),))
# TableKey = namedtuple("TableKey", ("key", "type", "attr"))
# Condition = namedtuple("Condition", ("key", "opr", "val", "repl"))
# StateSet = namedtuple("StateSet", ("key", "val"))


class Database(ABC):

    # d_key in create_table : key = key_name, val = [type, attr, attr...]
    # d_opr : operand to use in compararizon
    # l_repl : values with given keys are replaced in sql_query
    #          to generate sql with subquery

    # database management methods
    @abstractmethod
    def db_exists(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def commit(self):
        raise NotImplementedError

    @abstractmethod
    def execute(self, sql, args):
        raise NotImplementedError

    @abstractmethod
    def executemany(self, sql, iter_args):
        raise NotImplementedError

    @abstractmethod
    def get_table_names(self):
        raise NotImplementedError

    def get_column_names(self, table_name):
        raise NotImplementedError

    # sql methods, basically classmethod or staticmethod
    @staticmethod
    def strftime(dt):
        if isinstance(dt, datetime.datetime):
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(dt, str):
            return dt
        else:
            raise TypeError

    @staticmethod
    def strptime(string):
        return datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

    @classmethod
    @abstractmethod
    def datetime(cls, ret):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _ph(varname):
        """place holder format"""
        raise NotImplementedError

    @classmethod
    def _set_state(cls, l_setstate):
        if len(l_setstate) == 0:
            raise ValueError("empty setstates")
        l_buf = []
        for ss in l_setstate:
            l_buf.append("{0} = {1}".format(ss.key, cls._ph(ss.val)))
        return ", ".join(l_buf)

    @classmethod
    def _cond_state(cls, l_cond):
        l_buf = []
        for cond in l_cond:
            if cond.repl:
                buf = "{0.key} {0.opr} {1}".format(cond, cls._ph(cond.val))
            else:
                buf = "{0.key} {0.opr} ({0.val})".format(cond)
            l_buf.append(buf)
        return " and ".join(l_buf)

    @staticmethod
    @abstractmethod
    def _table_key_type(type_str):
        # allowed typename : integer, text, datetime
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _table_key_attr(attr):
        # allowed attr : primary_key, auto_increment, not_null
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _index_key(tablekey):
        raise NotImplementedError

    @staticmethod
    def join_sql(join_opt, table_name1, table_name2, key1, key2):
        return "{1} {0} join {2} on {1}.{3} = {2}.{4}".format(
            join_opt, table_name1, table_name2, key1, key2)

    @classmethod
    def create_table_sql(cls, table_name, l_tablekey):
        l_def = []
        for key in l_tablekey:
            type_name = cls._table_key_type(key.type)
            if len(key.attr) > 0:
                l_attr = []
                for a in key.attr:
                    l_attr.append(cls._table_key_attr(a))
                l_def.append("{0} {1} {2}".format(key.key, type_name,
                                                  " ".join(l_attr)))
            else:
                l_def.append("{0} {1}".format(key.key, type_name))
        sql = "create table {0} ({1})".format(table_name, ", ".join(l_def))
        return sql

    @classmethod
    def create_index_sql(cls, table_name, index_name, l_tablekey):
        sql = "create index {0} on {1}({2})".format(
            index_name, table_name,
            ", ".join([cls._index_key(key) for key in l_tablekey])
        )
        return sql

    @classmethod
    def alter_table_sql(cls, old_table_name, new_table_name):
        sql = "alter table {0} rename to {1}".format(
            old_table_name, new_table_name
        )
        return sql

    @classmethod
    def select_sql(cls, table_name, l_key,
                   l_cond=None, l_order=None, opt=None, limit=None):
        # now only "distinct" is allowed for opt
        sql_header = "select"
        if opt is not None and "distinct" in opt:
            sql_header += " distinct"
        sql = "{0} {1} from {2}".format(sql_header, ", ".join(l_key),
                                        table_name)
        if l_cond is not None and len(l_cond) > 0:
            sql += " where {0}".format(cls._cond_state(l_cond))
        if l_order is not None and len(l_order) > 0:
            sql_order = ", ".join(["{0} {1}".format(col, order)
                                   for col, order in l_order])
            sql += " order by " + sql_order
        if limit is not None:
            sql += " limit " + str(limit)
        return sql

    @classmethod
    def insert_sql(cls, table_name, l_setstate):
        l_key, l_val = zip(*[(ss.key, cls._ph(ss.val)) for ss in l_setstate])
        sql = "insert into {0} ({1}) values ({2})".format(
            table_name, ", ".join(l_key), ", ".join(l_val))
        return sql

    @classmethod
    def update_sql(cls, table_name, l_setstate, l_cond=None):
        sql = "update {0} set {1}".format(table_name,
                                          cls._set_state(l_setstate))
        if l_cond is not None and len(l_cond) > 0:
            sql += " where {0}".format(cls._cond_state(l_cond))
        return sql

    @classmethod
    def delete_sql(cls, table_name, l_cond=None):
        sql = "delete from {0}".format(table_name)
        if l_cond is not None and len(l_cond) > 0:
            sql += " where {0}".format(cls._cond_state(l_cond))
        return sql

    @staticmethod
    def drop_table_sql(table_name):
        return "drop table {0}".format(table_name)

    @staticmethod
    def drop_index_sql(index_name):
        return "drop index {0}".format(index_name)
