#!/usr/bin/env python
# coding: utf-8

import os
import time
import datetime
import logging
import subprocess  # for python3


# args
json_args = {"ensure_ascii": False,
             "indent": 4,
             "sort_keys": True}


# classes

#class singleton(object):
#
#    def __new__(clsObj, *args, **kwargs):
#        tmpInstance = None
#        if not hasattr(clsObj, "_instanceDict"):
#            clsObj._instanceDict = {}
#            clsObj._instanceDict[str(hash(clsObj))] = \
#                super(singleton, clsObj).__new__(clsObj, *args, **kwargs)
#            tmpInstance = clsObj._instanceDict[str(hash(clsObj))]
#        elif not hasattr(clsObj._instanceDict, str(hash(clsObj))):
#            clsObj._instanceDict[str(hash(clsObj))] = \
#                super(singleton, clsObj).__new__(clsObj, *args, **kwargs)
#            tmpInstance = clsObj._instanceDict[str(hash(clsObj))]
#        else:
#            tmpInstance = clsObj._instanceDict[str(hash(clsObj))]
#        return tmpInstance


class IDDict:

    def __init__(self, keyfunc=None):
        self._d_obj = {}
        self._d_id = {}
        if keyfunc is None:
            self.keyfunc = lambda x: x
        else:
            self.keyfunc = keyfunc

    def _next_id(self):
        next_id = len(self._d_obj)
        assert next_id not in self._d_obj
        return next_id

    def add(self, obj):
        if self.exists(obj):
            return self._d_id[self.keyfunc(obj)]
        else:
            key = self._next_id()
            self.set_item(key, obj)
            return key

    def set_item(self, key, obj):
        self._d_obj[key] = obj
        self._d_id[self.keyfunc(obj)] = key

    def exists(self, obj):
        return self.keyfunc(obj) in self._d_id

    def get(self, keyid):
        return self._d_obj[keyid]


# file managing

def is_empty(dirname):
    if os.path.isdir(dirname):
        if len(os.listdir(dirname)) > 1:
            return True
        else:
            return False
    else:
        return False


def rep_dir(args):
    if isinstance(args, list):
        ret = []
        for arg in args:
            if os.path.isdir(arg):
                ret.extend(["/".join((arg, fn)) \
                            for fn in sorted(os.listdir(arg))])
            else:
                ret.append(arg)
        return ret
    elif isinstance(args, str):
        arg = args
        if os.path.isdir(arg):
            return ["/".join((arg, fn)) for fn in sorted(os.listdir(arg))]
        else:
            return [arg]
    else:
        raise NotImplementedError


def recur_dir(args):
    def open_path(path):
        if os.path.isdir(path):
            ret = []
            for fn in sorted(os.listdir(path)):
                ret += open_path("/".join((path, fn)))
            return ret
        else:
            return [path]

    if isinstance(args, list):
        l_fn = []
        for arg in args:
            l_fn += open_path(arg)
    elif isinstance(args, str):
        l_fn = open_path(args)
    else:
        raise NotImplementedError
    return l_fn


def filepath(dn, fn):
    if len(dn) == 0:
        return fn
    else:
        return "/".join((dn, fn))


def filename(fp):
    if "/" in fp:
        return fp.split("/")[-1]
    else:
        return fp


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    elif not os.path.isdir(path):
        raise OSError("something already exists on given path, "
                      "and it is NOT a directory")
    else:
        pass


def rm(path):
    if os.path.exists(path):
        os.remove(path)
        return True
    else:
        return False


def rm_dirchild(dirpath):
    for fpath in rep_dir(dirpath):
        os.remove(fpath)


def filepath_local(current_path, fn):
    return os.path.dirname(os.path.abspath(current_path)) + "/" + fn


def last_modified(args, latest=False):
    """Get the last modified time of a file or a set of files.

    Args:
        args (str or list[str]): Files to investigate.
        latest (Optional[bool]): If true, return the latest datetime
                of timestamps. Otherwise, return the oldest timestamp.

    Returns:
        datetime.datetime

    """

    def file_timestamp(fn):
        stat = os.stat(fn)
        t = stat.st_mtime
        return datetime.datetime.fromtimestamp(t)

    if isinstance(args, list):
        l_dt = [file_timestamp(fn) for fn in args]
        if latest:
            return max(l_dt)
        else:
            return min(l_dt)
    elif isinstance(args, str):
        return file_timestamp(args)
    else:
        raise NotImplementedError


# subprocess
def call_process(cmd):
    """Call a subprocess and handle standard outputs.
    
    Args:
        cmd (list): A sequence of command strings.
    
    Returns:
        ret (int): Return code of the subprocess.
        stdout (str)
        stderr (str)
    """

    p = subprocess.Popen(cmd, shell=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    ret = p.returncode
    return ret, stdout, stderr


# parallel computing

def mprocess(l_process, pal):
    """
    Handle multiprocessing with an upper limit of working processes.
    This function incrementally fill working processes with remaining tasks.
    This function does not use memory sharing or process communications.
    If you need processe communications,
    use common.mprocess_queueing.

    Args:
        l_process (List(multiprocess.Process))
        pal (int): Maximum number of processes executed at once.
    """

    l_job = []
    while len(l_process) > 0:
        if len(l_job) < pal:
            job = l_process.pop(0)
            job.start()
            l_job.append(job)
        else:
            time.sleep(1)
            l_job = [j for j in l_job if j.is_alive()]
    else:
        for job in l_job:
            job.join()


def mprocess_queueing(l_pq, n_process):
    """
    Same as mprocess, but this function yields
    returned values of every processes with multiprocessing.Queue.

    Args:
        l_pq (List[multiprocessing.Process, multiprocessing.Queue])
        n_process (int): Maximum number of processes executed at once.
    """

    l_job = []
    while len(l_pq) > 0:
        if len(l_job) < n_process:
            process, queue = l_pq.pop(0)
            process.start()
            l_job.append((process, queue))
        else:
            time.sleep(1)
            l_temp = []
            for process, queue in l_job:
                if queue.empty():
                    l_temp.append((process, queue))
                else:
                    ret = queue.get()
                    yield ret
                    assert queue.empty()
                    process.join()
                    queue.close()
            l_job = l_temp
    else:
        while len(l_job) > 0:
            time.sleep(1)
            l_temp = []
            for process, queue in l_job:
                if queue.empty():
                    l_temp.append((process, queue))
                else:
                    ret = queue.get()
                    yield ret
                    assert queue.empty()
                    process.join()
                    queue.close()
            l_job = l_temp


# def mthread_queueing(l_thread, pal):
#    """
#    Args:
#        l_thread (List[threading.Thread]): A sequence of thread objects.
#        pal (int): Maximum number of threads executed at once.
#    """
#    l_job = []
#    while len(l_thread) > 0:
#        if len(l_job) < pal:
#            job = l_thread.pop(0)
#            job.start()
#            l_job.append(job)
#        else:
#            time.sleep(1)
#            l_job = [j for j in l_job if j.is_alive()]
#    else:
#        for job in l_job:
#            job.join()
#
#
# def mprocess_queueing(l_process, pal):
#    mthread_queueing(l_process, pal)


# measurement

class Timer:

    def __init__(self, header, output=None, timestr_func=None):
        self._dts = None
        self._dte = None
        self._lap_dt = []
        self.header = header
        self.output = output
        self._timestr_func = timestr_func

    def _output(self, string):
        if isinstance(self.output, logging.Logger):
            self.output.info(string)
        else:
            print(string)

    def timestr(self, td):
        if self._timestr_func is None:
            return td.total_seconds()
        else:
            return self._timestr_func(td)

    def start(self):
        self._dts = datetime.datetime.now()
        self._lap_dt.append(self._dts)
        self._output("{0} start".format(self.header))

    def lap(self, name):
        if self._dts is None:
            raise ValueError("call start() before lap()")
        lap_dt = datetime.datetime.now()
        t = self.timestr(lap_dt - self._dts)
        self._lap_dt.append(lap_dt)
        self._output("{0} lap({1}) ({2})".format(self.header, name, t))

    def lap_diff(self, name):
        if self._dts is None:
            raise ValueError("call start() before lap_diff()")
        lap_dt = datetime.datetime.now()
        t = self.timestr(lap_dt - self._lap_dt[-1])
        self._lap_dt.append(lap_dt)
        self._output("{0} diff({1}) ({2})".format(self.header, name, t))

    def stop(self):
        if self._dts is None:
            raise ValueError("call start() before stop()")
        self._dte = datetime.datetime.now()
        t = self.timestr(self._dte - self._dts)
        self._output("{0} done ({1})".format(self.header, t))

    def total_time(self):
        if self._dts is None or self._dte is None:
            raise ValueError("call start() and stop() before total_time()")
        else:
            return self._dte - self._dts

    def stat(self):
        if len(self._lap_dt) == 1:
            raise ValueError
        import numpy as np
        lap_times = np.diff(self._lap_dt)
        avg = np.average(lap_times)
        se = np.std(lap_times) / np.sqrt(len(lap_times))
        self._output("{0} lap times: {0}".format(self.header, lap_times))
        self._output("{0} average: {0}".format(self.header, avg))
        self._output("{0} standard error: {0}".format(self.header, se))


# visualization

def add_indent(buffer, indent):
    ret = []
    for line in buffer.splitlines():
        ret.append(" " * indent + line)
    return "\n".join(ret)


def show_repr(iterable, head, foot, indent=0, strfunc=str):
    data = list(iterable)
    if (head <= 0 and foot <= 0) or \
            (len(data) <= head + foot):
        buf = [strfunc(e) for e in data]
    else:
        buf = []
        if head > 0:
            buf += [strfunc(e) for e in data[:head]]
        buf += [" " * indent + "..."]
        if foot > 0:
            buf += [strfunc(e) for e in data[-foot:]]
        buf.append("({0})".format(len(data)))
    header = " " * indent
    return "\n".join([header + line for line in buf])


def cli_table(table, spl=" ", fill=" ", align="left"):
    """
    Args:
        table (List[List[str]]): input data
        spl (str): splitting string of columns
        fill (str): string of 1 byte, used to fill the space
        align (str): left and right is available
    """
    len_column = len(table[0])
    col_max = [0] * len_column

    for row in table:
        for cid, val in enumerate(row):
            if cid >= len_column:
                raise ValueError
            val = str(val)
            if len(val) > col_max[cid]:
                col_max[cid] = len(val)

    l_buf = []
    for row in table:
        line = []
        for cid, val in enumerate(row):
            cell = str(val)
            len_cell = col_max[cid]
            len_space = len_cell - len(cell)
            if align == "left":
                cell = cell + fill * len_space
            elif align == "right":
                cell = fill * len_space + cell
            else:
                raise NotImplementedError
            line.append(cell)
        l_buf.append(spl.join(line))

    return "\n".join(l_buf)


# compatibility

def pickle_comp_args(comp):
    if comp:
        d = {"encoding": "bytes"}
    else:
        d = {}
    return d
