import os

from . import db_common


class Sqlite3(db_common.Database):

    def __init__(self, dbpath):
        self._dbpath = dbpath
        self._connect = None

    def __del__(self):
        if self._connect is not None:
            self._connect.commit()
            self._connect.close()

    def _open(self):
        import sqlite3 as sqlite3_mod
        self._connect = sqlite3_mod.connect(self._dbpath)
        self._connect.text_factory = str

    def db_exists(self):
        if os.path.exists(self._dbpath):
            return True
        else:
            return False

    def reset(self):
        if os.path.exists(self._dbpath):
            os.remove(self._dbpath)

    def commit(self):
        if self._connect is not None:
            self._connect.commit()

    def execute(self, sql, args=None):
        # print(sql)
        # if args is not None:
        #     print(args)
        if self._connect is None:
            self._open()
        cursor = self._connect.cursor()
        if args is None or len(args) == 0:
            cursor.execute(sql)
        else:
            cursor.execute(sql, args)
        return cursor

    def executemany(self, sql, iter_args):
        if self._connect is None:
            self._open()
        cursor = self._connect.cursor()
        cursor.executemany(sql, iter_args)
        return cursor

    def get_table_names(self):
        sql = "select name from sqlite_master"
        cursor = self.execute(sql)
        return [row[0] for row in cursor]

    def get_column_names(self, table_name):
        sql = ("select sql from sqlite_master "
               "where type='table' and name='{0}'".format(table_name))
        cursor = self.execute(sql)
        row = cursor.__iter__().__next__()
        table_sql = row[0]

        tmp_columns = table_sql.partition("(")[-1].rstrip(")")
        column_names = [part.strip(" ").split(" ")[0]
                        for part in tmp_columns.split(",")]
        return column_names

    # sql methods, basically classmethod or staticmethod
    @classmethod
    def datetime(cls, ret):
        return cls.strptime(ret)

    @staticmethod
    def _ph(varname):
        return ":{0}".format(varname)

    @staticmethod
    def _table_key_type(type_str):
        if type_str == "datetime":
            return "text"
        else:
            return type_str

    @staticmethod
    def _table_key_attr(attr):
        if attr == "primary_key":
            return "primary key"
        elif attr == "auto_increment":
            return "autoincrement"
        elif attr == "not_null":
            return "not null"
        else:
            raise NotImplementedError

    @staticmethod
    def _index_key(tablekey):
        return tablekey.key


def init_db_conn(conf):
    dbpath = conf.get("database", "sqlite3_filename")
    return Sqlite3(dbpath)
