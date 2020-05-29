from amulog import log_db
from amulog import lt_common
from amulog import lt_import
from amulog.eval import cluster_metrics


class ParameterSearcher(object):

    metrics_candidates = [cluster_metrics.cluster_accuracy,
                          cluster_metrics.over_division_cluster_ratio,
                          cluster_metrics.over_aggregation_cluster_ratio]

    def __init__(self, conf, param_candidates, metrics_candidates=None):
        self._conf = conf
        self._param_candidates = param_candidates
        if metrics_candidates is None:
            self._metrics_candidates = self.metrics_candidates
        else:
            self._metrics_candidates = metrics_candidates

    def _get_ltgen(self, table, params):
        raise NotImplementedError

    def measure(self, plines):
        table_answer = lt_common.TemplateTable()
        ltgen_answer = lt_import.init_ltgen_import(self._conf, table_answer)
        d_tid_answer = ltgen_answer.process_offline(plines)
        l_tid_answer = [tid for mid, tid
                        in sorted(d_tid_answer.items(), key=lambda x: x[0])
                        if tid is not None]

        for params in self._param_candidates:
            table = lt_common.TemplateTable()
            ltgen = self._get_ltgen(table, params)
            d_tid_trial = ltgen.process_offline(plines)
            l_tid_trial = [tid for mid, tid
                           in sorted(d_tid_trial.items(), key=lambda x: x[0])
                           if d_tid_answer[mid] is not None]

            d_metrics = {}
            for metrics_func in self.metrics_candidates:
                kwargs = {"labels_true": l_tid_answer,
                          "labels_pred": l_tid_trial}
                val = metrics_func(**kwargs)
                d_metrics[metrics_func.__name__] = val

            del table
            del ltgen
            del d_tid_trial
            del l_tid_trial
            yield params, d_metrics


def compare_parameters(conf, targets, method):
    try:
        from importlib import import_module
        modname = "amulog.alg." + method
        alg_module = import_module(modname)
        ps = alg_module.ParameterSearcher(conf)
    except ImportError as e:
        raise e
    else:
        plines = list(log_db.iter_plines(conf, targets))
        results = ps.measure(plines)

    return results
