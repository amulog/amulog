import queue
import logging
import multiprocessing

_logger = logging.getLogger(__package__)


class WorkerProcess(multiprocessing.Process):

    def __init__(self, task_queue, result_queue, timeout, *super_args, **super_kwargs):
        multiprocessing.Process.__init__(self, *super_args, **super_kwargs)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self._timeout = timeout
        _logger.debug("process {0} ready".format(self._name))

    def run_current_proc(self):
        _logger.info("mproc_queue runs in no multiprocessing mode")
        while True:
            try:
                next_task = self.task_queue.get_nowait()
                if next_task is None:
                     break
                else:
                    ret = self._target(next_task, *self._args, **self._kwargs)
                    self.result_queue.put_nowait(ret)
                    self.task_queue.task_done()
            except queue.Empty:
                break

    def run(self):
        try:
            while True:
                next_task = self.task_queue.get(True, self._timeout)
                if next_task is None:
                    _logger.debug("process {0} exiting".format(self._name))
                    self.task_queue.task_done()
                    break
                else:
                    _logger.debug("process {0} start next task".format(self._name))
                    ret = self._target(next_task, *self._args, **self._kwargs)
                    self.result_queue.put_nowait(ret)
                    _logger.debug("process {0} ended a task".format(self._name))
                    self.task_queue.task_done()
        except queue.Empty:
            _logger.debug("process {0} timeout".format(self._name))
        finally:
            return


class Manager(object):

    def __init__(self, target, n_proc, args=(), kwargs=None, namer=None,
                 timeout=3600):
        if kwargs is None:
            kwargs = {}
        if namer is None:
            namer = lambda x: None
        self._n_proc = n_proc
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.Queue()
        self._l_worker = [WorkerProcess(self.task_queue, self.result_queue,
                                        timeout, target=target, name=namer(i),
                                        args=args, kwargs=kwargs)
                          for i in range(max(n_proc, 1))]
        if self._n_proc > 0:
            for w in self._l_worker:
                w.start()

    def add(self, task):
        self.task_queue.put(task)

    def add_from(self, l_task):
        for task in l_task:
            self.task_queue.put(task)

    def join(self):
        if self._n_proc > 0:
            self.task_queue.join()
        else:
            # debug without multiprocessing
            self._l_worker[0].run_current_proc()

    def get(self, block=False, timeout=1):
        return self.result_queue.get(block, timeout)

    def get_from(self, l_task, block=False, timeout=1):
        for _ in l_task:
            yield self.result_queue.get(block, timeout)

    def get_all(self):
        while True:
            try:
                ret = self.result_queue.get_nowait()
            except queue.Empty:
                break
            else:
                yield ret

    def is_clean(self):
        assert self.task_queue.empty()
        assert self.result_queue.empty()

    def close(self):
        if self._n_proc > 0:
            for _ in self._l_worker:
                self.task_queue.put(None)
            self.task_queue.join()
        else:
            pass


