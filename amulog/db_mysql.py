import pymysql

from . import db_common

pymysql.install_as_MySQLdb()


class Mysql(db_common.Database):

    def __init__(self, host, dbname, user, passwd):
        self._host = host
        self._dbname = dbname
        self._user = user
        self._passwd = passwd
        self._connect = None

    def __del__(self):
        if self._connect is not None:
            self._connect.commit()
            self._connect.close()

    def _open(self):
        if not self.db_exists():
            self._init_database()
        self._connect = pymysql.connect(host=self._host, db=self._dbname,
                                        user=self._user, passwd=self._passwd)

    def _init_database(self):
        connect = self._connect_root()
        cursor = connect.cursor()
        cursor.execute("create database {0}".format(self._dbname))

    def _connect_root(self):
        return pymysql.connect(host=self._host, user=self._user,
                               passwd=self._passwd)

    def db_exists(self):
        connect = self._connect_root()
        cursor = connect.cursor()
        cursor.execute("show databases")
        return self._dbname in [row[0] for row in cursor]

    def reset(self):
        connect = self._connect_root()
        cursor = connect.cursor()
        sql = "drop database if exists {0}".format(self._dbname)
        cursor.execute(sql)
        connect.commit()

    def commit(self):
        if self._connect is not None:
            self._connect.commit()

    def is_internal_table(self, name):
        return False

    @classmethod
    def datetime(cls, ret):
        return ret

    @staticmethod
    def _ph(varname):
        return "%({0})s".format(varname)

    @staticmethod
    def _table_key_type(type_str):
        if type_str == "integer":
            return "int"
        else:
            return type_str

    @staticmethod
    def _table_key_attr(attr):
        if attr == "primary_key":
            return "primary key"
        elif attr == "auto_increment":
            # differ from sqlite
            return "auto_increment"
        elif attr == "not_null":
            return "not null"
        else:
            raise NotImplementedError

    @staticmethod
    def _index_key(tablekey):
        if tablekey.type == "text":
            return "{0}({1})".format(tablekey.key, tablekey.attr[0])
        else:
            return tablekey.key

    def execute(self, sql, args=None):
        # print(sql)
        # if args is not None and len(args) > 0:
        #     print(args)
        if self._connect is None:
            self._open()
        cursor = self._connect.cursor()
        if args is not None and len(args) == 0:
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
        sql = "show tables"
        cursor = self.execute(sql)
        return [row[0] for row in cursor]

    def get_column_names(self, table_name):
        raise NotImplementedError


def init_db_conn(conf):
    host = conf.get("database", "mysql_host")
    dbname = conf.get("database", "mysql_dbname")
    user = conf.get("database", "mysql_user")
    passwd = conf.get("database", "mysql_passwd")
    return Mysql(host, dbname, user, passwd)

