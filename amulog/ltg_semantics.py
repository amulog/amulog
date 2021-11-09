import logging
import pandas as pd

from . import strutil
from . import lt_common
from amsemantics import TopicClustering

_logger = logging.getLogger(__package__)


class LTGroupSemantics(lt_common.LTGroupOffline):

    def __init__(self, lttable, normalizer,
                 lda_modelname="gensim",
                 lda_knowledge_sources=None,
                 cache_dir="/tmp",
                 rfc_use_cache=True, rfc_document_unit="rfc",
                 random_seed=None,
                 stop_words=None,
                 use_nltk_stopwords=False,
                 use_sklearn_stopwords=False,
                 lda_n_topics=None, cluster_eps=None,
                 cluster_size_min=5,
                 tuning_metrics="cluster"):
        super().__init__(lttable)
        self._lognorm = normalizer
        self._lda_modelname = lda_modelname
        self._lda_knowledge_sources = lda_knowledge_sources
        self._cache_dir = cache_dir
        self._rfc_use_cache = rfc_use_cache
        self._rfc_document_unit = rfc_document_unit
        self._stop_words = stop_words
        self._use_nltk_stopwords = use_nltk_stopwords,
        self._use_sklearn_stopwords = use_sklearn_stopwords,
        self._random_seed = random_seed
        self._lda_n_topics = lda_n_topics
        self._cluster_eps = cluster_eps
        self._cluster_size_min = cluster_size_min
        self._tuning_metrics = tuning_metrics

        self._sc = None
        self._tuning_rules = None
        self._l_ltid = None
        self._training_docs = None
        self._input_docs = None

    def _tokenize(self, ltobj):
        return self._lognorm.process_line(
            [strutil.restore_esc(w) for w in ltobj.desc()]
        )

    def set_tuning_rules(self, *args, **kwargs):
        self._tuning_rules = (args, kwargs)

    def set_documents(self, verbose=False):
        self._l_ltid = [ltobj.ltid for ltobj in self.lttable]
        self._input_docs = [self._tokenize(ltobj)
                            for ltobj in self.lttable]
        self._training_docs = self.get_training_data(
            self._input_docs, verbose=verbose
        )

    def make(self, verbose=False):
        if self._input_docs is None:
            self.set_documents(verbose=verbose)

        self._sc = TopicClustering(
            model=self._lda_modelname,
            redistribute=True,
            random_seed=self._random_seed,
            stop_words=self._stop_words,
            use_nltk_stopwords=self._use_nltk_stopwords,
            use_sklearn_stopwords=self._use_sklearn_stopwords,
            lda_n_topics=self._lda_n_topics,
            cluster_eps=self._cluster_eps,
            cluster_size_min=self._cluster_size_min,
            tuning_metrics=self._tuning_metrics,
            verbose=verbose
        )
        if self._tuning_rules is not None:
            args, kwargs = self._tuning_rules
            self._sc.set_tuning_rules(*args, **kwargs)
        clusters = self._sc.fit(self._input_docs, training_docs=self._training_docs)

        self._n_groups = len(clusters)
        for cls, l_idx in clusters.items():
            for idx in l_idx:
                ltid = self._l_ltid[idx]
                self.add_lt(int(cls), self.lttable[ltid])
        return self.update_lttable(self.lttable)

    def get_inspection(self, topn=10):
        assert self._sc is not None
        return self._sc.inspection(topn=topn)

    def get_training_data(self, docs, verbose=False):
        """
        Generate training data based on the knowledge_sources parameter.

        Args:
            docs (List[List[str]]): Input documents.
            verbose (boolean)
        """
        if self._lda_knowledge_sources is None:
            knowledge_sources = ["self"]
        else:
            knowledge_sources = self._lda_knowledge_sources

        training_docs = []
        for source_name in knowledge_sources:
            if source_name == "self":
                training_docs += docs
            elif source_name == "rfc":
                from amsemantics.source import rfcdoc
                kwargs = {"cache_dir": self._cache_dir,
                          "use_cache": self._rfc_use_cache,
                          "normalizer": self._lognorm}
                if self._rfc_document_unit == "rfc":
                    loader = rfcdoc.RFCLoader(**kwargs)
                elif self._rfc_document_unit == "section":
                    loader = rfcdoc.RFCSectionLoader(**kwargs)
                elif self._rfc_document_unit == "line":
                    loader = rfcdoc.RFCLinesLoader(**kwargs)
                else:
                    raise ValueError("invalid rfc_document_unit")
                for rfc in loader.iter_all():
                    try:
                        if verbose:
                            print("loading RFC {0}".format(rfc.n))
                        docs = loader.get_document(rfc)
                        training_docs += docs
                    except FileNotFoundError:
                        mes = "skip RFC {0}: line file not found".format(rfc.n)
                        if verbose:
                            print(mes)
                        _logger.warning(mes)

        return training_docs

    def lda_search_param(self, verbose=False):
        if self._training_docs is None:
            self.set_documents(verbose)
        results = [(n, p, c) for n, p, c
                   in self._sc.lda_search_param(self._training_docs)]
        return pd.DataFrame(results,
                            columns=["n_topics", "log_perplexity", "coherence"])

    def get_visual_data(self, dim=2, with_mean=True, with_std=True):
        return self._sc.get_principal_components(
            input_docs=self._input_docs,
            n_components=dim,
            with_mean=with_mean, with_std=with_std
        )

    def show_pyldavis(self, mds="pcoa"):
        return self._sc.show_pyldavis(mds=mds)


class LTGroupSemanticsMultiData(LTGroupSemantics):

    def __init__(self, lttables, normalizer, **kwargs):
        super().__init__(lttables[0], normalizer, **kwargs)
        self._l_lttable = lttables

        # used afeter make
        self._l_ltid = None
        self._l_domain = None

    def set_documents(self, verbose=False):
        self._l_ltid = [ltobj.ltid for ltobj in self.lttable]
        self._l_domain = []
        self._input_docs = []
        for domain, lttable in enumerate(self._l_lttable):
            for ltobj in lttable:
                self._l_ltid.append(ltobj.ltid)
                self._l_domain.append(domain)
                self._input_docs.append(self._tokenize(ltobj))
        self._training_docs = self.get_training_data(
            self._input_docs, verbose=verbose
        )

    def make(self, verbose=False):
        if self._input_docs is None:
            self.set_documents(verbose=verbose)

        self._sc = TopicClustering(
            model=self._lda_modelname,
            redistribute=True,
            stop_words=self._stop_words,
            random_seed=self._random_seed,
            lda_n_topics=self._lda_n_topics,
            cluster_eps=self._cluster_eps,
            cluster_size_min=self._cluster_size_min,
            tuning_metrics=self._tuning_metrics,
            verbose=verbose
        )
        if self._tuning_rules is not None:
            args, kwargs = self._tuning_rules
            self._sc.set_tuning_rules(*args, **kwargs)
        clusters = self._sc.fit(self._input_docs, training_docs=self._training_docs)

        self._n_groups = len(clusters)
        for cls, l_idx in clusters.items():
            for idx in l_idx:
                ltid = self._l_ltid[idx]
                domain = self._l_domain[idx]
                self._l_lttable[domain][ltid].ltgid = cls

        return self._l_lttable

    def get_labels(self):
        import numpy as np
        return np.array(self._l_domain), np.array(self._l_ltid), self._sc.cluster_labels


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


def _get_ltgroup_semantic_params(conf):
    from . import config

    lda_model = conf["log_template_group_semantics"]["lda_model"]
    lda_knowledge_sources = config.getlist(
        conf, "log_template_group_semantics", "lda_knowledge_sources"
    )

    cache_dir = conf["general"]["cache_dir"]
    rfc_use_cache = conf["log_template_group_semantics"]["rfc_use_cache"]
    rfc_document_unit = conf["log_template_group_semantics"]["rfc_document_unit"]
    stop_words = config.getlist(conf, "log_template_group_semantics",
                                "lda_stop_words")
    use_nltk_stopwords = conf["log_template_group_semantics"]["lda_use_nltk_stopwords"]
    use_sklearn_stopwords = conf["log_template_group_semantics"]["lda_use_sklearn_stopwords"]
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
    cluster_size_min = conf.getint("log_template_group_semantics",
                                   "cluster_size_min")
    tuning_metrics = conf["log_template_group_semantics"]["tuning_metrics"]

    return {
        "lda_modelname": lda_model,
        "lda_knowledge_sources": lda_knowledge_sources,
        "cache_dir": cache_dir,
        "rfc_use_cache": rfc_use_cache,
        "rfc_document_unit": rfc_document_unit,
        "random_seed": random_seed,
        "stop_words": stop_words,
        "use_nltk_stopwords": use_nltk_stopwords,
        "use_sklearn_stopwords": use_sklearn_stopwords,
        "lda_n_topics": lda_n_topics,
        "cluster_eps": cluster_eps,
        "cluster_size_min": cluster_size_min,
        "tuning_metrics": tuning_metrics,
    }


def init_ltgroup_semantics(conf, lttable):
    from . import config
    normalizer = init_nlp_normalizer(conf)
    kwargs = _get_ltgroup_semantic_params(conf)
    ltgroup = LTGroupSemantics(lttable, normalizer, **kwargs)

    if kwargs["lda_n_topics"] is None or kwargs["cluster_eps"] is None:
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


def init_ltgroup_semantics_multi_domain(l_conf, l_lttable):
    from . import config
    conf = l_conf[0]
    normalizer = init_nlp_normalizer(conf)
    kwargs = _get_ltgroup_semantic_params(conf)
    ltgroup = LTGroupSemanticsMultiData(l_lttable, normalizer, **kwargs,)

    if kwargs["lda_n_topics"] is None or kwargs["cluster_eps"] is None:
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
