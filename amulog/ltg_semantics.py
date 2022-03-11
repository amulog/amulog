import logging

from . import strutil
from . import lt_common
from amsemantics import SemanticClassifier

_logger = logging.getLogger(__package__)


class LTGroupSemantics(lt_common.LTGroupOffline):

    def __init__(self, lttable, normalizer,
                 lda_knowledge_sources=None,
                 **kwargs):
        super().__init__(lttable)
        self._lognorm = normalizer
        self._sc_kwargs = kwargs

        kwargs["normalizer"] = normalizer
        kwargs["training_sources"] = [
            source_name for source_name in lda_knowledge_sources
            if source_name != "self"
        ]
        kwargs["input_sources"] = None
        self._use_input_as_training = "self" in lda_knowledge_sources

        self._sc = SemanticClassifier(**kwargs)

    @property
    def classifier(self):
        return self._sc

    def _tokenize(self, ltobj):
        return self._lognorm.process_line(
            [strutil.restore_esc(w) for w in ltobj.desc()]
        )

    def get_input_documents(self):
        documents = []
        annotations = []
        for ltobj in self.lttable:
            documents.append(self._tokenize(ltobj))
            annotations.append(ltobj.ltid)
        return documents, annotations

    def set_input_documents(self):
        documents, annotations = self.get_input_documents()
        if self._use_input_as_training:
            self._sc.add_training_documents(documents)
        self._sc.add_input_documents(documents, annotations=annotations)

    def _restore_clustering_results(self):
        self._n_groups = len(self._sc.clusters())

        labels = self._sc.cluster_labels()
        l_ltid = self._sc.input_annotations()
        for cls, ltid in zip(labels, l_ltid):
            self.add_lt(int(cls), self.lttable[ltid])

    def make(self, verbose=False):
        self.set_input_documents()
        self._sc.make(verbose=verbose)
        self._restore_clustering_results()
        return self.update_lttable(self.lttable)


class LTGroupSemanticsMultiData(LTGroupSemantics):

    def __init__(self, lttables, normalizer, **kwargs):
        super().__init__(lttables[0], normalizer, **kwargs)
        self._l_lttable = lttables

        self._l_domain = None

    def get_input_documents(self):
        documents = []
        annotations = []
        for domain, lttable in enumerate(self._l_lttable):
            for ltobj in lttable:
                documents.append(self._tokenize(ltobj))
                annotations.append((domain, ltobj.ltid))
        return documents, annotations

    def _restore_clustering_results(self):
        self._n_groups = len(self._sc.clusters())

        labels = self._sc.cluster_labels()
        annotations = self._sc.input_annotations()
        for cls, (domain, ltid) in zip(labels, annotations):
            self._l_lttable[domain][ltid].ltgid = cls


def init_nlp_normalizer(conf):
    from . import config
    from amsemantics import Normalizer
    filters = config.getlist(conf, "nlp_preprocess", "filters")
    mreplacer_sources = config.getlist(conf, "nlp_preprocess",
                                       "replacer_sources")
    vreplacer_source = conf["nlp_preprocess"]["variable_rule"]
    ha_source = conf["manager"]["host_alias_filename"]
    if ha_source == "":
        ha_source = None
    lemma_exception = config.getlist(conf, "nlp_preprocess",
                                     "lemma_exception")
    th_word_length = conf.getint("nlp_preprocess",
                                 "remove_short_word_length")
    return Normalizer(filters, mreplacer_sources,
                      vreplacer_source, ha_source,
                      lemma_exception, th_word_length)


def _load_guidedlda_seed_topic_list(filepath):
    if filepath is None or filepath == "":
        return None

    seed_topic_list = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                linestr = line.rstrip()
                if linestr != "":
                    items = linestr.split()
                    seed_topic_list.append(items)
    except IOError:
        raise
    return seed_topic_list


def _get_ltgroup_semantic_params(conf):
    from . import config

    lda_library = conf["log_template_group_semantics"]["lda_library"]
    lda_knowledge_sources = config.getlist(
        conf, "log_template_group_semantics", "lda_knowledge_sources"
    )

    cache_dir = conf["general"]["cache_dir"]
    use_cache = conf.getboolean("log_template_group_semantics", "use_cache")
    rfc_document_unit = conf["log_template_group_semantics"]["rfc_document_unit"]
    ltjunos_filepath = conf["log_template_group_semantics"]["ltjunos_filepath"]
    use_template_replacer = conf.getboolean(
        "log_template_group_semantics", "use_template_replacer"
    )
    stop_words = config.getlist(conf, "log_template_group_semantics",
                                "lda_stop_words")
    use_nltk_stopwords = conf.getboolean(
        "log_template_group_semantics", "lda_use_nltk_stopwords"
    )
    use_sklearn_stopwords = conf.getboolean(
        "log_template_group_semantics", "lda_use_sklearn_stopwords"
    )
    random_seed_str = conf["log_template_group_semantics"]["lda_seed"]
    if random_seed_str.isdigit():
        random_seed = int(random_seed_str)
    else:
        random_seed = None
    lda_use_zscore = conf.getboolean("log_template_group_semantics", "lda_use_zscore")
    lda_n_topics_str = conf["log_template_group_semantics"]["lda_n_topics"]
    if lda_n_topics_str == "":
        lda_n_topics = None
    else:
        lda_n_topics = int(lda_n_topics_str)
    lda_cachename = conf["log_template_group_semantics"]["lda_cachename"]
    guidedlda_seed_topic_list = _load_guidedlda_seed_topic_list(
        conf["log_template_group_semantics"]["guidedlda_seed_topic_list_file"]
    )
    guidedlda_seed_confidence = conf.getfloat(
        "log_template_group_semantics", "guidedlda_seed_confidence"
    )
    cluster_method = conf["log_template_group_semantics"]["cluster_method"]
    dbscan_eps_str = conf["log_template_group_semantics"]["dbscan_eps"]
    if dbscan_eps_str == "":
        dbscan_eps = None
    else:
        dbscan_eps = float(dbscan_eps_str)
    dbscan_cluster_size_min = conf.getint("log_template_group_semantics",
                                          "dbscan_cluster_size_min")
    dbscan_tuning_metrics = conf["log_template_group_semantics"]["dbscan_tuning_metrics"]
    rdbscan_cluster_size_max = conf.getint("log_template_group_semantics",
                                           "rdbscan_cluster_size_max")

    return {
        "lda_library": lda_library,
        "lda_knowledge_sources": lda_knowledge_sources,
        "use_cache": use_cache,
        "cache_dir": cache_dir,
        "rfc_document_unit": rfc_document_unit,
        "ltjunos_filepath": ltjunos_filepath,
        "use_template_replacer": use_template_replacer,
        "random_seed": random_seed,
        "stop_words": stop_words,
        "use_nltk_stopwords": use_nltk_stopwords,
        "use_sklearn_stopwords": use_sklearn_stopwords,
        "lda_n_topics": lda_n_topics,
        "lda_use_zscore": lda_use_zscore,
        "lda_cachename": lda_cachename,
        "guidedlda_seed_topic_list": guidedlda_seed_topic_list,
        "guidedlda_seed_confidence": guidedlda_seed_confidence,
        "cluster_method": cluster_method,
        "dbscan_eps": dbscan_eps,
        "dbscan_cluster_size_min": dbscan_cluster_size_min,
        "dbscan_tuning_metrics": dbscan_tuning_metrics,
        "rdbscan_cluster_size_max": rdbscan_cluster_size_max
    }


def _ltgroup_set_tuning_rules(conf, ltgroup):
    from . import config
    union_rules_str = config.getlist(
        conf, "log_template_group_semantics", "tuning_union_rules"
    )
    union_rules = [rulestr.split("|")
                   for rulestr in union_rules_str]
    separation_rules_str = config.getlist(
        conf, "log_template_group_semantics", "tuning_separation_rules"
    )
    separation_rules = [rulestr.split("|")
                        for rulestr in separation_rules_str]
    if len(union_rules) > 0 or len(separation_rules) > 0:
        term_class = conf["log_template_group_semantics"]["tuning_term_class"]
        tuning_topn = conf.getint("log_template_group_semantics", "tuning_topwords")
        ltgroup.set_tuning_rules(union_rules, separation_rules,
                                 term_class=term_class, topn=tuning_topn)
    return ltgroup


def init_ltgroup_semantics(conf, lttable):
    normalizer = init_nlp_normalizer(conf)
    kwargs = _get_ltgroup_semantic_params(conf)
    ltgroup = LTGroupSemantics(lttable, normalizer, **kwargs)
    ltgroup = _ltgroup_set_tuning_rules(conf, ltgroup)
    return ltgroup


def init_ltgroup_semantics_multi_domain(l_conf, l_lttable):
    conf = l_conf[0]
    normalizer = init_nlp_normalizer(conf)
    kwargs = _get_ltgroup_semantic_params(conf)
    ltgroup = LTGroupSemanticsMultiData(l_lttable, normalizer, **kwargs,)
    ltgroup = _ltgroup_set_tuning_rules(conf, ltgroup)
    return ltgroup
