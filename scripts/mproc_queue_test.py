import time
import logging

from amulog import config
from amulog import mproc_queue

_logger = logging.getLogger(__package__)


def target_func(val, sq=2):
    time.sleep(0.3)
    return val + sq


if __name__ == "__main__":
    logging_lv = logging.DEBUG
    config.set_logging_stdio(logger_name=["amulog",], lv=logging_lv)
    m = mproc_queue.Manager(target=target_func, n_proc=10, kwargs={"sq":3})

    l_task = list(range(100))
    m.add_from(l_task)
    m.join()
    #for j in m.get_from(l_task):
    #    print(j)
    for j in m.get_all():
        print(j)
    m.is_clean()

    l_task2 = list(range(50))
    m.add_from(l_task2)
    m.join()
    for j in m.get_all():
        print(j)
    m.is_clean()

    del m
    print("done")
