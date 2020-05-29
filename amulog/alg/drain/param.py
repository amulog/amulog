from itertools import product
import numpy as np

from amulog.eval import param_searcher


class ParameterSearcher(param_searcher.ParameterSearcher):

    def __init__(self, conf):
        from amulog.eval import cluster_metrics
        param_candidates = product(np.arange(0.1, 1.0, 0.1), (3, 4, 5))
        metrics_candidates = [cluster_metrics.cluster_accuracy,
                              cluster_metrics.over_division_cluster_ratio,
                              cluster_metrics.over_aggregation_cluster_ratio]
        super().__init__(conf, param_candidates, metrics_candidates)

    def _get_ltgen(self, table, params):
        from amulog.alg.drain import LTGen
        from amulog.lt_regex import VariableRegex
        preprocess_fn = self._conf.get("log_template_drain", "preprocess_rule")
        if preprocess_fn.strip() == "":
            from amulog.alg.drain.drain import DEFAULT_REGEX_CONFIG
            preprocess_fn = DEFAULT_REGEX_CONFIG
        vreobj = VariableRegex(self._conf, preprocess_fn)
        kwargs = {"table": table,
                  "threshold": params[0],
                  "depth": params[1],
                  "vreobj": vreobj}
        return LTGen(**kwargs)
