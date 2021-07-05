import logging
from collections import defaultdict

import amulog.manager
from amulog import common
from amulog import lt_common
from . import maketpl

_logger = logging.getLogger(__package__.partition(".")[0])


class ParameterSearcher(maketpl.MeasureLTGen):

    FILEPATH_DIGIT_LENGTH = 4

    def __init__(self, conf, n_trial):
        super().__init__(conf, n_trial)
        self._params = None

    def _init_trial_info(self):
        common.mkdir(self._output_dir_trial(self._conf))
        if self._n_trial is None:
            return
        assert self._n_trial < 10 ** self.FILEPATH_DIGIT_LENGTH
        self._d_trial = {"params": self._params,
                         "l_tid": list(),
                         "n_c_lines": int(),
                         "d_n_c_lines": defaultdict(int),
                         "n_c_words": int(),
                         "d_n_c_words": defaultdict(int),
                         }

    @staticmethod
    def _output_dir_trial(conf):
        return conf["eval"]["ltgen_param_dir"]

    def init_trial(self, trial_id, params=None):
        common.rm(self._trial_label_path(trial_id))
        self._current_trial = trial_id
        self._params = params
        self._init_trial_info()

    def load_trial(self, trial_id):
        self._current_trial = trial_id
        self._load_trial_info()
        self._params = self._d_trial["params"]

    def get_params(self):
        return self._params


def _get_param_candidates(method):
    from importlib import import_module
    modname = "amulog.alg.{0}".format(method)
    alg_module = import_module(modname)
    return alg_module.get_param_candidates()


def _init_ltgen_with_params(conf, table, method, params):
    from importlib import import_module
    modname = "amulog.alg." + method
    alg_module = import_module(modname)
    return alg_module.init_ltgen_with_params(conf, table, params)


def measure_parameters(conf, targets, method):
    param_candidates = list(_get_param_candidates(method))
    n_trial = len(param_candidates)
    ps = ParameterSearcher(conf, n_trial)
    ps.load()

    from amulog import log_db
    for trial_id, params in enumerate(param_candidates):
        timer = common.Timer("measure-parameters trial{0}".format(
            trial_id), output=_logger)
        timer.start()
        ps.init_trial(trial_id, params)
        table = lt_common.TemplateTable()
        ltgen = _init_ltgen_with_params(conf, table, method, params)

        l_pline = list(amulog.manager.iter_plines(conf, targets))
        d_plines = {mid: pline for mid, pline in enumerate(l_pline)}
        d_tid = ltgen.process_offline(d_plines)
        iterobj = zip(l_pline,
                      ps.tid_list_answer(),
                      ps.iter_tpl_answer())
        for mid, (pline, tid_answer, tpl_answer) in enumerate(iterobj):
            if tid_answer is None:
                tid_trial = None
                tpl_trial = None
            else:
                tid_trial = d_tid[mid]
                if tid_trial is None:
                    tpl_trial = None
                else:
                    tpl_trial = ltgen.get_tpl(tid_trial)
            ps.add_trial(tid_trial, tpl_trial,
                         tid_answer, tpl_answer, pline["words"])
        ps.dump_trial()
        timer.stop()

    return ps
