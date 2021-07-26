"""Currently, required libraries are not ready in OSS."""

from . import lt_common
import log_normalizer as lognorm
from amsemantics import TopicClustering


class LTGroupSemantics(lt_common.LTGroupOffline):

    def __init__(self, lttable, normalizer,
                 lda_modelname, stop_words=None, random_seed=None,
                 lda_n_topics=None, cluster_eps=None):
        super().__init__(lttable)
        self._lognorm = normalizer
        # self._lognorm = lognorm.LogNormalizer(preprocess_conf_path)
        # self._lognorm_method = preprocess_func_name
        self._lda_modelname = lda_modelname
        self._stop_words = stop_words
        self._random_seed = random_seed
        self._lda_n_topics = lda_n_topics
        self._cluster_eps = cluster_eps

        self._sc = None
        self._tuning_rules = None

    def _tokenize(self, ltobj):
        return self._lognorm.process_line(ltobj.desc())
        # text = " ".join(input_text)
        # return getattr(self._lognorm, self._lognorm_method)(text=text)

    def set_tuning_rules(self, *args, **kwargs):
        self._tuning_rules = (args, kwargs)

    def make(self, verbose=False):
        l_ltid = [ltobj.ltid for ltobj in self.lttable]
        l_input = [self._tokenize(ltobj)
                   for ltobj in self.lttable]

        self._sc = TopicClustering(model=self._lda_modelname,
                                   redistribute=True,
                                   stop_words=self._stop_words,
                                   random_seed=self._random_seed,
                                   lda_n_topics=self._lda_n_topics,
                                   cluster_eps=self._cluster_eps,
                                   verbose=verbose)
        if self._tuning_rules is not None:
            args, kwargs = self._tuning_rules
            self._sc.set_tuning_rules(*args, **kwargs)
        clusters = self._sc.fit(l_input)

        for cls, l_idx in clusters.items():
            for idx in l_idx:
                ltid = l_ltid[idx]
                self.add_lt(int(cls), self.lttable[ltid])

    def get_inspection(self, topn=10):
        assert self._sc is not None
        return self._sc.inspection(topn=topn)


def init_nlp_normalizer(conf):
    from . import config
    from . import host_alias
    from amsemantics import Normalizer
    filters = config.getlist(conf, "nlp_preprocess", "filters")
    mreplacer_sources = config.getlist(conf, "nlp_preprocess",
                                       "replacer_sources")
    vreplacer_source = conf["nlp_preprocess"]["variable_rule"]
    ha = host_alias.init_hostalias(conf)
    lemma_exception = config.getlist(conf, "nlp_preprocess",
                                     "lemma_exception")
    return Normalizer(filters, mreplacer_sources,
                      vreplacer_source, ha, lemma_exception)


def init_ltgroup_semantics(conf, lttable):
    from . import config
    normalizer = init_nlp_normalizer(conf)
    # preprocess_conf_path = conf["log_template_group_semantics"]["conf_path"]
    # preprocess_func_name = conf["log_template_group_semantics"]["func_name"]
    lda_model = conf["log_template_group_semantics"]["lda_model"]
    stop_words = config.getlist(conf, "log_template_group_semantics",
                                "lda_stop_words")
    random_seed_str = conf["log_template_group_semantics"]["lda_seed"]
    if random_seed_str.isdigit():
        random_seed = int(random_seed_str)
    else:
        random_seed = None
    lda_n_topics_str = conf["log_template_group_semantics"]["lda_n_topics"]
    if lda_n_topics_str == "":
        lda_n_topics = None
    else:
        lda_n_topics = int(lda_n_topics_str)
    cluster_eps_str = conf["log_template_group_semantics"]["cluster_eps"]
    if cluster_eps_str == "":
        cluster_eps = None
    else:
        cluster_eps = float(cluster_eps_str)

    ltgroup = LTGroupSemantics(lttable, normalizer,
                               # preprocess_conf_path, preprocess_func_name,
                               lda_model,
                               stop_words=stop_words,
                               random_seed=random_seed,
                               lda_n_topics=lda_n_topics,
                               cluster_eps=cluster_eps)
    if lda_n_topics is None or cluster_eps_str is None:
        union_rules_str = config.getlist(conf,
                                         "log_template_group_semantics",
                                         "tuning_union_rules")
        union_rules = [rulestr.split("|")
                       for rulestr in union_rules_str]
        separation_rules_str = config.getlist(conf,
                                              "log_template_group_semantics",
                                              "tuning_separation_rules")
        separation_rules = [rulestr.split("|")
                            for rulestr in separation_rules_str]
        if len(union_rules) > 0 or len(separation_rules) > 0:
            term_class = conf["log_template_group_semantics"]["tuning_term_class"]
            tuning_topn = conf.getint("log_template_group_semantics", "tuning_topwords")
            ltgroup.set_tuning_rules(union_rules, separation_rules,
                                     term_class=term_class, topn=tuning_topn)

    return ltgroup
