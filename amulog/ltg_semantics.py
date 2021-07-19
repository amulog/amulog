"""Currently, required libraries are not ready in OSS."""

from collections import defaultdict
import numpy as np

from . import lt_common
import log_normalizer as lognorm
import log_vectorizer as logvec


class SemanticClustering:

    def __init__(self, model="gensim", redistribute=True,
                 stop_words=None, random_seed=None):
        self._model = model
        self._redistribute = redistribute
        self._random_seed = random_seed
        if stop_words is None:
            self._stop_words = []
        else:
            self._stop_words = stop_words

        # used after fit
        self._loglda = None
        self._l_topicv = None
        self._eps = None
        self._clustering = None
        self._clusters = None

    @staticmethod
    def _split_and_redistribute_clusters(clustering, l_topicv):
        # split clusters and outliers
        outliers = []
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                outliers.append(idx)
            else:
                clusters[label].append(idx)

        # average distribution (centers) of clusters
        l_cluster_centers = []
        for cid, cluster in clusters.items():
            l_distance = [l_topicv[idx] for idx in cluster]
            avg = np.array(l_distance).mean(axis=0)
            l_cluster_centers.append(avg)

        # re-distribute outliers to nearest clusters
        from scipy.spatial.distance import cityblock
        for outlier in outliers:
            outlier_v = l_topicv[outlier]
            l_distance = [(cid, cityblock(center_v, outlier_v))
                          for cid, center_v in enumerate(l_cluster_centers)]
            min_cls, min_distance = min(l_distance, key=lambda x: x[1])
            clusters[min_cls].append(outlier)

        return clusters

    @staticmethod
    def _split_clusters(clustering):
        # outliers will be an additional cluster
        outliers = []
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label == -1:
                outliers.append(idx)
            else:
                clusters[label].append(idx)

        clusters[max(clusters.keys()) + 1] = outliers
        return clusters

    def fit(self, lda_input):
        self._loglda = logvec.LogLDA(lda_input, lda_input,
                                     stop_words=self._stop_words,
                                     random_seed=self._random_seed)

        if self._model == "gensim":
            self._loglda.fit_gensim()
            self._l_topicv = [topicv for idx, topicv
                              in self._loglda.predict_all(self._model,
                                                          return_dist=True,
                                                          return_id=True)]
        elif self._model == "gibbs":
            raise NotImplementedError
            # loglda.fit_gibbs(l_input, alpha)
            # l_topicv = loglda.predict_all(model, return_dist=True, return_id=True)
        else:
            raise ValueError("invalid model name".format(self._model))

        from sklearn.cluster import DBSCAN
        x_input = np.array(self._l_topicv)
        best_eps = logvec.LogClustering.find_best_eps(x_input)
        self._eps = best_eps
        self._clustering = DBSCAN(eps=best_eps, min_samples=5,
                                  metric="cityblock").fit(x_input)
        if self._redistribute:
            self._clusters = self._split_and_redistribute_clusters(
                self._clustering, self._l_topicv
            )
        else:
            self._clusters = self._split_clusters(self._clustering)

        return self._clusters

    def inspection(self, topn=10):
        from gensim import matutils
        # TODO: depends on gensim
        ret = {}

        # topic terms
        key = "topic_terms"
        ret[key] = {}
        for topic in range(self._loglda.lda.num_topics):
            iterobj = self._loglda.lda.get_topic_terms(topic, topn=topn)
            ret[key][topic] = [(self._loglda.id2word[widx], val)
                               for widx, val in iterobj]

        # cluster topicv
        key = "cluster_centers"
        l_avg_v = []
        for cid, cluster in self._clusters.items():
            l_cls_topicv = [self._l_topicv[idx] for idx in cluster]
            avg_v = np.array(l_cls_topicv).mean(axis=0)
            l_avg_v.append(avg_v)
        ret[key] = np.array(l_avg_v)

        # cluster terms
        key = "cluster_terms"
        ret[key] = {}
        cluster_voc_matrix = np.dot(ret["cluster_centers"],
                                    self._loglda.lda.get_topics())
        for cid in self._clusters:
            topicv = cluster_voc_matrix[cid]
            topicv = topicv / topicv.sum()
            bestn = matutils.argsort(topicv, topn, reverse=True)
            ret[key][cid] = [(self._loglda.id2word[widx], topicv[widx])
                             for widx in bestn]

        return ret


class LTGroupSemantics(lt_common.LTGroupOffline):

    def __init__(self, lttable, preprocess_conf_path, preprocess_func_name,
                 lda_modelname, stop_words, random_seed):
        super().__init__(lttable)
        self._lognorm = lognorm.LogNormalizer(preprocess_conf_path)
        self._lognorm_method = preprocess_func_name
        self._lda_modelname = lda_modelname
        self._stop_words = stop_words
        self._random_seed = random_seed

        self._sc = None

    def _tokenize(self, text):
        return getattr(self._lognorm, self._lognorm_method)(text=text)

    def make(self):
        l_ltid = [ltobj.ltid for ltobj in self.lttable]
        l_input = [self._tokenize(" ".join(ltobj.ltw))
                   for ltobj in self.lttable]

        self._sc = SemanticClustering(model=self._lda_modelname,
                                      redistribute=True,
                                      stop_words=self._stop_words,
                                      random_seed=self._random_seed)
        clusters = self._sc.fit(l_input)

        for cls, l_idx in clusters.items():
            for idx in l_idx:
                ltid = l_ltid[idx]
                self.add_lt(int(cls), self.lttable[ltid])

    def get_inspection(self, topn=10):
        assert self._sc is not None
        return self._sc.inspection(topn=topn)


def init_ltgroup_semantics(conf, lttable):
    from . import config
    preprocess_conf_path = conf["log_template_group_semantics"]["conf_path"]
    preprocess_func_name = conf["log_template_group_semantics"]["func_name"]
    lda_model = conf["log_template_group_semantics"]["lda_model"]
    stop_words = config.getlist(conf, "log_template_group_semantics",
                                "lda_stop_words")
    random_seed_str = conf["log_template_group_semantics"]["lda_seed"]
    if random_seed_str.isdigit():
        random_seed = int(random_seed_str)
    else:
        random_seed = None
    return LTGroupSemantics(lttable,
                            preprocess_conf_path, preprocess_func_name,
                            lda_model, stop_words, random_seed)
